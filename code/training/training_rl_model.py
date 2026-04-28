"""
MMBoundary Stage-2: PPO Training for Multimodal Confidence Calibration.

From-scratch PPO implementation for VLMs, referencing Vision-LLM-Alignment.
Does NOT depend on trl — uses pure PyTorch + DeepSpeed + transformers.

Architecture:
  Actor  — LLaVA (SFT checkpoint, trainable with LoRA)
  Ref    — LLaVA (frozen SFT copy, for KL divergence)
  Critic — LLaVA + v_head (nn.Linear → scalar value per token)

Reward: MMBoundary's three functions (no learned reward model):
  R_KA — Knowledge Accuracy: CrossEncoder match against GT chain (Eq.10)
  R_EC — Expected Calibration: |correctness - expressed_confidence|² (Eq.11)
  R_CS — Confidence Self-Consistency: 1 - 2·(IV - EV)² (Eq.12)
  R    = α·R_KA + β·R_EC + γ·R_CS (Eq.13)

Usage:
  deepspeed --num_gpus=N code/training/training_rl_model.py \\
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \\
    --train_file annotated_data/aokvqa/train_mmb.json \\
    --image_dir datasets/coco \\
    --output_dir output/mmb_rl
"""

import argparse
import copy
import json
import logging
import math
import os
import re
import sys
import random
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from PIL import Image

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from sentence_transformers import CrossEncoder, SentenceTransformer

try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.process_score_similarity import unordered_row_score
from evaluation.inverse_mapping import prepare_statements_embeddings, classify_batch

logger = logging.getLogger(__name__)


# ============================================================
# 1. Critic Model: VLM + Value Head
# ============================================================

class VLMCritic(nn.Module):
    """VLM wrapped with a linear value head for PPO critic.
    References: Vision-LLM-Alignment/modeling_reward.py ViRewardModel"""

    def __init__(self, vlm, hidden_size=None):
        super().__init__()
        self.vlm = vlm
        if hidden_size is None:
            hidden_size = vlm.config.text_config.hidden_size
        self.v_head = nn.Linear(hidden_size, 1, bias=False)
        nn.init.zeros_(self.v_head.weight)  # critic starts at zero

    def forward_value(self, input_ids, pixel_values=None, attention_mask=None,
                      image_sizes=None):
        """Get per-token values from critic.
        Returns dict with 'values' [B, seq_len-1] and 'end_scores' [B]."""
        outputs = self.vlm(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]  # [B, seq_len, hidden]
        values = self.v_head(hidden_states).squeeze(-1)  # [B, seq_len]

        # End-of-sequence value
        pad_id = getattr(self.vlm.config, 'pad_token_id', 0) or 0
        bs = values.size(0)
        end_scores = []
        for i in range(bs):
            non_pad = (input_ids[i] != pad_id).nonzero(as_tuple=True)[0]
            end_idx = non_pad[-1].item() if len(non_pad) > 0 else values.size(1) - 1
            end_scores.append(values[i, end_idx])

        return {
            "values": values[:, :-1],
            "end_scores": torch.stack(end_scores).detach(),
        }


# ============================================================
# 2. PPO Utilities (from Vision-LLM-Alignment/ppo_training_utils.py)
# ============================================================

def gather_log_probs(logits, labels):
    """Gather log probabilities of the chosen tokens."""
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def get_advantages_and_returns(values, rewards, start, gamma=0.99, lam=0.95):
    """GAE (Generalized Advantage Estimation)."""
    lastgaelam = 0
    advantages_reversed = []
    length = rewards.size(-1)
    for t in reversed(range(start, length)):
        nextvalues = values[:, t + 1] if t < length - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values[:, start:]
    return advantages.detach(), returns


def actor_loss_fn(logprobs, old_logprobs, advantages, mask, cliprange=0.2):
    """PPO clipped policy gradient loss."""
    log_ratio = (logprobs - old_logprobs) * mask
    ratio = torch.exp(log_ratio)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    return torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()


def critic_loss_fn(values, old_values, returns, mask, cliprange=0.2):
    """PPO clipped value function loss."""
    values_clipped = torch.clamp(values, old_values - cliprange, old_values + cliprange)
    vf_loss1 = (values - returns) ** 2
    vf_loss2 = (values_clipped - returns) ** 2
    return 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()


# ============================================================
# 3. Sampling (generate responses from actor with images)
# ============================================================

def sampling_llava(actor_model, pixel_values, input_ids, attention_mask,
                   processor, image_sizes=None,
                   max_new_tokens=256, temperature=0.7,
                   top_k=50, top_p=0.95, do_sample=True):
    """Generate responses from LLaVA actor, one sample at a time.
    Returns list of (response_ids, response_text) tuples."""
    generation_kwargs = dict(
        top_k=top_k, top_p=top_p, do_sample=do_sample, temperature=temperature,
    )
    batch_size = input_ids.size(0)
    all_res = []

    actor_model.eval()
    with torch.no_grad():
        for idx in range(batch_size):
            # Remove padding from query
            mask = attention_mask[idx]
            query_ids = input_ids[idx][mask == 1].unsqueeze(0)

            # Image for this sample
            if pixel_values is not None and pixel_values.size(0) >= batch_size:
                img = pixel_values[idx].unsqueeze(0)
            elif pixel_values is not None:
                img = pixel_values.unsqueeze(0)
            else:
                img = None

            kwargs = dict(input_ids=query_ids, max_new_tokens=max_new_tokens,
                          **generation_kwargs)
            if img is not None:
                kwargs['pixel_values'] = img
            if image_sizes is not None:
                kwargs['image_sizes'] = image_sizes[idx].unsqueeze(0)

            output_ids = actor_model.generate(**kwargs)[0]
            # Response is the part after the query
            response_ids = output_ids[query_ids.shape[1]:]
            response_text = processor.decode(response_ids, skip_special_tokens=True)
            all_res.append((response_ids, response_text))

    actor_model.train()
    return all_res


# ============================================================
# 4. MMBoundary Reward Functions (Equations 10-13)
# ============================================================

def parse_mmb_response(response_text):
    """Parse 'sentence, [confidence_statement] sentence, [statement]...'"""
    pattern = r'([^\[\]]+?)\s*\[([^\]]+)\]'
    matches = re.findall(pattern, response_text)
    if not matches:
        return [response_text.strip()], []
    sentences = [m[0].strip().rstrip(',').strip() for m in matches if m[0].strip()]
    statements = [m[1].strip() for m in matches if m[0].strip()]
    return sentences, statements


def compute_knowledge_accuracy(pred_sentences, ref_sentences, cross_encoder,
                               threshold=0.5):
    """R_KA (Eq.10): per-sentence match indicators + mean."""
    if not pred_sentences or not ref_sentences:
        return [0] * max(len(pred_sentences), 1), 0.0
    result = unordered_row_score(
        preds=pred_sentences, refs=ref_sentences,
        model=cross_encoder, match_threshold=threshold,
    )
    matched = set(result.get('match_pred', []))
    indicators = [1 if i in matched else 0 for i in range(len(pred_sentences))]
    r_ka = sum(indicators) / len(indicators)
    return indicators, r_ka


def compute_internal_confidence(logprobs_per_token, token_texts):
    """Compute per-sentence internal confidence from token log-probs.
    Uses LNLP (Length-Normalized Log Probability): sigmoid(avg_logprob)."""
    sentence_ends = {'.', '!', '?'}
    confidences = []
    current_lps = []

    for lp, tok in zip(logprobs_per_token, token_texts):
        current_lps.append(lp)
        if tok.strip() in sentence_ends and current_lps:
            avg_lp = sum(current_lps) / len(current_lps)
            conf = 1.0 / (1.0 + math.exp(-10.0 * (avg_lp + 0.5)))
            confidences.append(max(0.0, min(1.0, conf)))
            current_lps = []

    if current_lps:
        avg_lp = sum(current_lps) / len(current_lps)
        conf = 1.0 / (1.0 + math.exp(-10.0 * (avg_lp + 0.5)))
        confidences.append(max(0.0, min(1.0, conf)))

    return confidences


def compute_mmb_reward(response_text, response_ids, actor_logits,
                       ref_sentences, tokenizer, cross_encoder,
                       statement_embeddings, statement_model,
                       alpha=1.0, beta=1.0, gamma=1.0, threshold=0.5):
    """Compute MMBoundary combined reward R = α·R_KA + β·R_EC + γ·R_CS.

    Args:
        response_text: decoded response string
        response_ids: token ids of response
        actor_logits: logits from actor forward pass [seq_len, vocab]
        ref_sentences: GT reference chain (Image_level + Reasoning_level)
        Other args: reward model components

    Returns:
        reward: scalar float
        info: dict with r_ka, r_ec, r_cs
    """
    pred_sentences, statements = parse_mmb_response(response_text)
    T = len(pred_sentences)
    if T == 0:
        return -1.0, {"r_ka": 0.0, "r_ec": 0.0, "r_cs": 0.0}

    # --- R_KA (Eq.10) ---
    indicators, r_ka = compute_knowledge_accuracy(
        pred_sentences, ref_sentences, cross_encoder, threshold
    )

    # --- Expressed confidence: inverse-map statements → [0,1] ---
    if statements:
        try:
            _, expressed_values = classify_batch(
                statements, statement_embeddings, statement_model
            )
            ev = [v / 5.0 for v in expressed_values]
        except (ValueError, TypeError, RuntimeError):
            ev = [0.5] * T
    else:
        ev = [0.5] * T

    # --- Internal confidence from logprobs (Section 3.1.1) ---
    if actor_logits is not None and len(response_ids) > 0:
        log_softmax = F.log_softmax(actor_logits.float(), dim=-1)
        token_lps = []
        token_texts = []
        for i, tid in enumerate(response_ids):
            if i < log_softmax.size(0):
                token_lps.append(log_softmax[i, tid].item())
                token_texts.append(tokenizer.decode([tid]))
        iv = compute_internal_confidence(token_lps, token_texts)
    else:
        iv = [0.5] * T

    # Align lengths
    n = min(len(indicators), len(ev), len(iv))
    indicators = indicators[:n]
    ev = ev[:n]
    iv = iv[:n]

    # --- R_EC (Eq.11): (1/T) * Σ[1 - 2·(I - EV)²] ---
    r_ec = sum(1.0 - 2.0 * (ind - e) ** 2 for ind, e in zip(indicators, ev)) / n

    # --- R_CS (Eq.12): (1/T) * Σ[1 - 2·(IV - EV)²] ---
    r_cs = sum(1.0 - 2.0 * (i - e) ** 2 for i, e in zip(iv, ev)) / n

    # --- Combined (Eq.13) ---
    reward = alpha * r_ka + beta * r_ec + gamma * r_cs

    return reward, {"r_ka": r_ka, "r_ec": r_ec, "r_cs": r_cs}


# ============================================================
# 5. Dataset
# ============================================================

class MMBPPODataset(Dataset):
    """Dataset for MMBoundary PPO training.
    Each item: image + question + reference reasoning chain."""

    def __init__(self, data_path, image_dir, processor, tokenizer,
                 max_seq_length=512, max_samples=None):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        if max_samples:
            self.data = self.data[:max_samples]
        self.image_dir = image_dir
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.prompt_template = (
            "[INST] <image>\n{question}. "
            "Answer the question based on the image, explain your reasoning concisely:\n"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        prompt = self.prompt_template.format(question=question)

        # Image
        image_id = item.get('image_id', 0)
        split = item.get('split', 'train')
        image_path = item.get('image_path',
                              os.path.join(self.image_dir, f"{split}2017",
                                           f"{image_id:012}.jpg"))
        try:
            image = Image.open(image_path).convert('RGB')
        except (FileNotFoundError, OSError):
            image = Image.new('RGB', (336, 336), color=(128, 128, 128))

        # Process
        inputs = self.processor(images=image, text=prompt, return_tensors='pt',
                                padding='max_length', max_length=self.max_seq_length,
                                truncation=True)

        # Reference chain
        ref = item.get('keys_filtered', item.get('keys', {}))
        ref_sentences = ref.get('Image_level', []) + ref.get('Reasoning_level', [])

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'image_sizes': inputs.get('image_sizes', torch.tensor([336, 336])),
            'reference': json.dumps(ref_sentences),
            'question': question,
        }


def collate_fn(batch):
    """Collate with padding for variable-length sequences."""
    out = {}
    out['input_ids'] = torch.stack([b['input_ids'] for b in batch])
    out['attention_mask'] = torch.stack([b['attention_mask'] for b in batch])
    out['pixel_values'] = torch.stack([b['pixel_values'] for b in batch])
    out['reference'] = [b['reference'] for b in batch]
    out['question'] = [b['question'] for b in batch]
    return out


# ============================================================
# 6. Argument Parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="MMBoundary PPO Training")

    # Model
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--sft_checkpoint", type=str, default=None,
                        help="Path to SFT LoRA checkpoint (Stage 1 output)")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=float, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # Data
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--image_dir", type=str, default="./datasets/coco")
    parser.add_argument("--confidence_statement_path", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "confidence_statement.json"))
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--num_train_examples", type=int, default=20000)

    # Reward models
    parser.add_argument("--cross_encoder_model", type=str,
                        default="cross-encoder/stsb-roberta-large")
    parser.add_argument("--statement_model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--match_threshold", type=float, default=0.5)

    # PPO hyperparameters
    parser.add_argument("--actor_lr", type=float, default=1e-5)
    parser.add_argument("--critic_lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--ppo_epochs", type=int, default=2,
                        help="Inner PPO update epochs per batch")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--kl_coef", type=float, default=0.1)
    parser.add_argument("--clip_reward_value", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=0.99, help="GAE discount")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # Reward weights (Eq.13)
    parser.add_argument("--alpha", type=float, default=1.0, help="R_KA weight")
    parser.add_argument("--beta", type=float, default=1.0, help="R_EC weight")
    parser.add_argument("--gamma_reward", type=float, default=1.0, help="R_CS weight")

    # Output
    parser.add_argument("--output_dir", type=str, default="output/mmb_rl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--bf16", action="store_true")

    # DeepSpeed
    parser = deepspeed.add_config_arguments(parser) if HAS_DEEPSPEED else parser

    return parser.parse_args()


# ============================================================
# 7. Main Training
# ============================================================

def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    os.makedirs(args.output_dir, exist_ok=True)

    # ================================================================
    # A. Load reward scoring models (CPU, no grad)
    # ================================================================
    logger.info("Loading reward scoring models...")
    cross_encoder = CrossEncoder(args.cross_encoder_model)
    statement_model = SentenceTransformer(args.statement_model)
    with open(args.confidence_statement_path, 'r') as f:
        confidence_statements = json.load(f)
    statement_embeddings = prepare_statements_embeddings(
        confidence_statements, statement_model
    )

    # ================================================================
    # B. Load processor & tokenizer
    # ================================================================
    logger.info("Loading processor & tokenizer...")
    processor = LlavaNextProcessor.from_pretrained(args.model_name_or_path)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # ================================================================
    # C. Load Actor (LLaVA, trainable)
    # ================================================================
    logger.info("Loading Actor model...")
    actor = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_name_or_path, torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    if args.sft_checkpoint:
        logger.info(f"Loading SFT LoRA adapter from {args.sft_checkpoint}")
        actor = PeftModel.from_pretrained(actor, args.sft_checkpoint,
                                          is_trainable=True)
    elif args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=args.lora_rank, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "o_proj", "v_proj", "k_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        actor = get_peft_model(actor, peft_config)
        actor.print_trainable_parameters()

    # Freeze vision tower
    for p in actor.vision_tower.parameters():
        p.requires_grad = False
    actor.to(device)

    # ================================================================
    # D. Load Reference model (frozen copy)
    # ================================================================
    logger.info("Loading Reference model (frozen)...")
    ref_model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_name_or_path, torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    if args.sft_checkpoint:
        ref_model = PeftModel.from_pretrained(ref_model, args.sft_checkpoint,
                                              is_trainable=False)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.to(device)

    # ================================================================
    # E. Load Critic (LLaVA + v_head)
    # ================================================================
    logger.info("Loading Critic model...")
    critic_vlm = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_name_or_path, torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    if args.sft_checkpoint:
        critic_vlm = PeftModel.from_pretrained(critic_vlm, args.sft_checkpoint,
                                               is_trainable=True)
    # Freeze vision tower in critic too
    for p in critic_vlm.vision_tower.parameters():
        p.requires_grad = False

    hidden_size = critic_vlm.config.text_config.hidden_size
    critic = VLMCritic(critic_vlm, hidden_size).to(device)

    # ================================================================
    # F. Optimizers
    # ================================================================
    actor_optimizer = torch.optim.AdamW(
        [p for p in actor.parameters() if p.requires_grad], lr=args.actor_lr
    )
    critic_optimizer = torch.optim.AdamW(
        [p for p in critic.parameters() if p.requires_grad], lr=args.critic_lr
    )

    # ================================================================
    # G. Dataset & DataLoader
    # ================================================================
    logger.info("Loading dataset...")
    train_dataset = MMBPPODataset(
        args.train_file, args.image_dir, processor, tokenizer,
        args.max_seq_length, args.num_train_examples,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    # ================================================================
    # H. Training Loop
    # ================================================================
    logger.info("=" * 60)
    logger.info("Starting PPO Training")
    logger.info(f"  Actor LR: {args.actor_lr}, Critic LR: {args.critic_lr}")
    logger.info(f"  Epochs: {args.num_epochs}, PPO epochs: {args.ppo_epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Reward weights: α={args.alpha} β={args.beta} γ={args.gamma_reward}")
    logger.info(f"  KL coef: {args.kl_coef}")
    logger.info("=" * 60)

    global_step = 0
    reward_queue = []  # For reward normalization

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                pixel_values = batch['pixel_values'].to(device, dtype=dtype)
                references = batch['reference']

                bs = input_ids.size(0)
                prompt_len = input_ids.size(1)

                # ---- STEP 1: Generate responses from Actor ----
                sampling_results = sampling_llava(
                    actor, pixel_values, input_ids, attention_mask,
                    processor=processor,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )

                # ---- STEP 2: Compute MMBoundary rewards ----
                # Build combined sequence: [prompt | response]
                max_resp_len = max(len(r[0]) for r in sampling_results)
                response_pad = tokenizer.pad_token_id or 0

                all_response_ids = []
                all_response_texts = []
                combined_ids_list = []
                combined_mask_list = []
                reward_scores = []

                for idx, (resp_ids, resp_text) in enumerate(sampling_results):
                    all_response_ids.append(resp_ids)
                    all_response_texts.append(resp_text)

                    # Pad response
                    pad_len = max_resp_len - len(resp_ids)
                    padded_resp = F.pad(resp_ids, (0, pad_len), value=response_pad)

                    # Combined: [query | response]
                    query = input_ids[idx][attention_mask[idx] == 1]
                    combined = torch.cat([query, padded_resp], dim=0)
                    cmask = torch.cat([
                        torch.ones(len(query), device=device),
                        torch.cat([torch.ones(len(resp_ids), device=device),
                                   torch.zeros(pad_len, device=device)]),
                    ], dim=0)
                    combined_ids_list.append(combined)
                    combined_mask_list.append(cmask)

                    # Compute reward using actor logits for internal confidence
                    ref_sents = json.loads(references[idx])

                    # Quick forward for logits (to get internal confidence)
                    with torch.no_grad():
                        actor_out = actor(
                            input_ids=combined.unsqueeze(0),
                            attention_mask=cmask.unsqueeze(0),
                            pixel_values=pixel_values[idx].unsqueeze(0),
                        )
                    resp_logits = actor_out.logits[0, len(query):len(query)+len(resp_ids)]

                    reward, info = compute_mmb_reward(
                        resp_text, resp_ids, resp_logits, ref_sents,
                        tokenizer, cross_encoder,
                        statement_embeddings, statement_model,
                        alpha=args.alpha, beta=args.beta,
                        gamma=args.gamma_reward, threshold=args.match_threshold,
                    )
                    reward_scores.append(reward)

                    if step % args.logging_steps == 0 and idx == 0:
                        logger.info(
                            f"[E{epoch} S{step}] R_KA={info['r_ka']:.3f} "
                            f"R_EC={info['r_ec']:.3f} R_CS={info['r_cs']:.3f} "
                            f"Total={reward:.3f}"
                        )
                        logger.info(f"  Resp: {resp_text[:150]}...")

                # Normalize rewards (running mean/std)
                reward_scores_t = torch.tensor(reward_scores, device=device, dtype=dtype)
                reward_queue.extend(reward_scores)
                if len(reward_queue) > 1000:
                    reward_queue = reward_queue[-1000:]
                if len(reward_queue) > 10:
                    r_mean = np.mean(reward_queue)
                    r_std = max(np.std(reward_queue), 1e-6)
                    reward_scores_t = (reward_scores_t - r_mean) / r_std
                reward_scores_t = torch.clamp(reward_scores_t,
                                              -args.clip_reward_value,
                                              args.clip_reward_value)

                # ---- STEP 3: Pad combined sequences to same length ----
                max_combined = max(c.size(0) for c in combined_ids_list)
                critic_input_ids = torch.full((bs, max_combined), response_pad,
                                             device=device, dtype=torch.long)
                critic_attention_mask = torch.zeros((bs, max_combined), device=device)
                for i in range(bs):
                    L = combined_ids_list[i].size(0)
                    critic_input_ids[i, :L] = combined_ids_list[i]
                    critic_attention_mask[i, :L] = combined_mask_list[i]

                # ---- STEP 4: Compute log-probs from actor & ref ----
                with torch.no_grad():
                    actor_outputs = actor(
                        input_ids=critic_input_ids,
                        pixel_values=pixel_values,
                        attention_mask=critic_attention_mask,
                        output_hidden_states=True,
                    )
                    ref_outputs = ref_model(
                        input_ids=critic_input_ids,
                        pixel_values=pixel_values,
                        attention_mask=critic_attention_mask,
                    )

                actor_logits = actor_outputs.logits
                ref_logits = ref_outputs.logits

                logprobs = gather_log_probs(actor_logits[:, :-1, :],
                                           critic_input_ids[:, 1:])
                ref_logprobs = gather_log_probs(ref_logits[:, :-1, :],
                                               critic_input_ids[:, 1:])

                # ---- STEP 5: Get old values from critic ----
                with torch.no_grad():
                    old_value_out = critic.forward_value(
                        input_ids=critic_input_ids,
                        pixel_values=pixel_values,
                        attention_mask=critic_attention_mask,
                    )
                    old_values = old_value_out['values']

                # ---- STEP 6: KL-adjusted rewards ----
                kl_divergence = -args.kl_coef * (logprobs - ref_logprobs)
                kl_rewards = kl_divergence.clone()

                # Add final reward at end of each response
                response_mask = critic_attention_mask[:, 1:].clone()
                for i in range(bs):
                    non_pad = (critic_attention_mask[i] == 1).nonzero(as_tuple=True)[0]
                    if len(non_pad) > 0:
                        end_pos = min(non_pad[-1].item() - 1, kl_rewards.size(1) - 1)
                        kl_rewards[i, end_pos] += reward_scores_t[i]

                # ---- STEP 7: GAE advantages ----
                advantages, returns = get_advantages_and_returns(
                    old_values, kl_rewards, start=prompt_len - 1,
                    gamma=args.gamma, lam=args.lam,
                )

                # ---- STEP 8: PPO update (inner loop) ----
                for ppo_ep in range(args.ppo_epochs):
                    # Critic update
                    value_out = critic.forward_value(
                        input_ids=critic_input_ids,
                        pixel_values=pixel_values,
                        attention_mask=critic_attention_mask,
                    )
                    new_values = value_out['values'][:, prompt_len - 1:]
                    old_vals_slice = old_values[:, prompt_len - 1:]
                    resp_mask = response_mask[:, prompt_len - 1:]

                    c_loss = critic_loss_fn(new_values, old_vals_slice, returns,
                                           resp_mask, args.cliprange)
                    critic_optimizer.zero_grad()
                    c_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                    critic_optimizer.step()

                    # Actor update
                    actor_out_new = actor(
                        input_ids=critic_input_ids,
                        pixel_values=pixel_values,
                        attention_mask=critic_attention_mask,
                    )
                    new_logprobs = gather_log_probs(
                        actor_out_new.logits[:, :-1, :], critic_input_ids[:, 1:]
                    )
                    new_lp_slice = new_logprobs[:, prompt_len - 1:]
                    old_lp_slice = logprobs[:, prompt_len - 1:]

                    a_loss = actor_loss_fn(new_lp_slice, old_lp_slice, advantages,
                                          resp_mask, args.cliprange)
                    actor_optimizer.zero_grad()
                    a_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in actor.parameters() if p.requires_grad], 1.0
                    )
                    actor_optimizer.step()

                global_step += 1

                if global_step % args.logging_steps == 0:
                    kl_dist = torch.abs(logprobs - ref_logprobs).mean().item()
                    avg_reward = np.mean(reward_scores)
                    logger.info(
                        f"[Step {global_step}] actor_loss={a_loss.item():.4f} "
                        f"critic_loss={c_loss.item():.4f} "
                        f"kl={kl_dist:.4f} avg_reward={avg_reward:.4f}"
                    )

                # ---- Save ----
                if global_step % args.save_steps == 0:
                    save_dir = os.path.join(args.output_dir, f"step_{global_step}")
                    os.makedirs(save_dir, exist_ok=True)
                    actor.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
                    torch.save(critic.v_head.state_dict(),
                               os.path.join(save_dir, "v_head.pt"))
                    logger.info(f"Saved checkpoint → {save_dir}")

                torch.cuda.empty_cache()

            except RuntimeError as e:
                logger.error(f"Error at step {step}: {e}")
                logger.error(traceback.format_exc())
                if 'out of memory' in str(e).lower():
                    logger.warning("OOM — clearing cache and skipping batch.")
                    for p in list(actor.parameters()) + list(critic.parameters()):
                        if p.grad is not None:
                            del p.grad
                    torch.cuda.empty_cache()
                    continue
                raise

        # End-of-epoch save
        save_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        actor.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        torch.save(critic.v_head.state_dict(),
                   os.path.join(save_dir, "v_head.pt"))
        logger.info(f"Epoch {epoch} done → {save_dir}")

    # Final save
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    actor.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Training complete → {final_dir}")


if __name__ == "__main__":
    main()
