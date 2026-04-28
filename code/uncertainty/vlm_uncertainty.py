"""
Unified VLM uncertainty estimation pipeline.

Merges the dataset-specific scripts (aokvqa, sciencevqa, culturalvqa) into a
single entry point.  Uses vLLM for generation and computes four sentence-level
uncertainty signals: LNLP, MTE, TokenSAR, and CLIP-score.

Usage examples:
    python vlm_uncertainty.py \
        --model_path /path/to/LLaVA-1.6-hf \
        --dataset aokvqa \
        --data_path /data/aokvqa/val_mmb.json \
        --image_dir /data/coco \
        --output_path ./output/aokvqa_val_mmb.json \
        --split val \
        --prompt_id 8

    python vlm_uncertainty.py \
        --model_path /path/to/LLaVA-1.6-hf \
        --dataset sciencevqa \
        --data_path /data/scvqa/problems.json \
        --image_dir /data/scvqa/images \
        --output_path ./output/scvqa_val_mmb.json \
        --split val \
        --prompt_id 8

    python vlm_uncertainty.py \
        --model_path /path/to/LLaVA-1.6-hf \
        --dataset culturalvqa \
        --data_path /data/cvqa/cvqa_sc_mmb.json \
        --image_dir /data/cvqa/cvqa_images \
        --output_path ./output/cvqa_mmb.json \
        --prompt_id 8
"""

import argparse
import copy
import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams

from token_SAR import extract_by_sentences, calculate_token_sar, calculate_token_similarity
from normalized_logprob import get_token_logprob, calculate_sentences_length_normalized_logprob
from mean_token_entropy import calculate_sentence_entropy, calculate_mean_token_entropy
from clip_score import get_clip_score


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

QUESTION_PROMPTS = {
    0: """
[INST] <image>\n{question}.
""",
    1: """
[INST] <image>\n{question}.
Follow these steps to generate your analysis (reasoning chain):
Your output should include concise, interconnected sentences, with each sentence conveying a single piece of knowledge.
Ensure the reasoning chain covers all necessary knowledge points concisely, with each sentence in this reasoning chain is essential and avoid adding redundant or irrelevant sentences.
Provide a clear conclusion with the answer finally, like: "So, the answer is ...".[/INST]
""",
    2: """
[INST] <image>\n{question}. Please analyze the image and the question step by step, and conclude with "So, the answer is xxx".
""",
    3: """
[INST] <image>\n{question}. Please analyze the image and the question, and conclude with "So, the answer is xxx".
""",
    4: """
[INST] <image>\n{question}. Answer the question based on the image with your concise reasoning chain.
""",
    5: """
Answer the question based on the image step by step.
[INST] <image>\n{question}.
""",
    6: """
Give your analysis and answer the questions:
[INST] <image>\n{question}.
""",
    7: """
[INST] <image>\n{question}.
Analyze the image and question, explain your reasoning concisely:
""",
    8: """
[INST] <image>\n{question}.
Answer the question based on the image, explain your reasoning concisely:
""",
}


# ---------------------------------------------------------------------------
# Core uncertainty computation
# ---------------------------------------------------------------------------

def uncertainty_compute(raw_data):
    """Compute LNLP, MTE, and TokenSAR from raw token log-probabilities."""
    token_logprob = get_token_logprob(raw_data)
    sentence_li, sentence_lnlp = calculate_sentences_length_normalized_logprob(token_logprob)
    _, sentence_MTE = calculate_sentence_entropy(token_logprob)
    sentence_list, tokens_list, token_logprob_list = extract_by_sentences(token_logprob)

    sentence_tokensar_list = []
    for sentence, tokens, token_lp in zip(sentence_list, tokens_list, token_logprob_list):
        similarity_matrix = calculate_token_similarity(sentence, tokens)
        token_sar = calculate_token_sar(tokens, token_lp, similarity_matrix)
        sentence_tokensar_list.append(token_sar)

    # Remove empty / whitespace-only sentences
    meaningless_indices = {idx for idx, s in enumerate(sentence_li) if not s.strip()}

    def remove_indices(lists, indices):
        return [[item for idx, item in enumerate(lst) if idx not in indices] for lst in lists]

    sentence_li, sentence_lnlp, sentence_MTE, sentence_tokensar_list = remove_indices(
        [sentence_li, sentence_lnlp, sentence_MTE, sentence_tokensar_list],
        meaningless_indices,
    )

    return sentence_li, sentence_lnlp, sentence_MTE, sentence_tokensar_list


def image_response_compute(response_sents, image):
    """Compute per-sentence CLIP scores between each sentence and the image."""
    sentence_clip_score = []
    for sent in response_sents:
        score = get_clip_score(image, sent)
        sentence_clip_score.append(score)
    return sentence_clip_score


def run_llava_next(llm, text_input, image_path, compute_clip=True):
    """Run LLaVA-NeXT inference and return uncertainty signals.

    Args:
        llm: A vLLM LLM instance.
        text_input: The formatted prompt string.
        image_path: Path to the input image.
        compute_clip: Whether to compute CLIP scores (default True).

    Returns:
        A dict with generated_text, sentences, and per-sentence uncertainty scores.
    """
    result_info = {}
    image = Image.open(image_path)
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.95,
        max_tokens=500,
        logprobs=10,
    )

    outputs = llm.generate(
        {
            "prompt": text_input,
            "multi_modal_data": {"image": image},
        },
        sampling_params=sampling_params,
    )

    generated_text = ""
    for o in outputs:
        generated_text += o.outputs[0].text

    raw_token_logprob = outputs[0].outputs[0].logprobs

    sentence_li, sentence_lnlp, sentence_MTE, sentence_tokensar_list = uncertainty_compute(raw_token_logprob)

    result_info['generated_text'] = generated_text
    result_info['sentences'] = sentence_li
    result_info['sentence_lnlp'] = sentence_lnlp
    result_info['sentence_MTE'] = sentence_MTE
    result_info['sentence_tokensar'] = sentence_tokensar_list

    if compute_clip:
        sentence_clip_score = image_response_compute(sentence_li, image)
        result_info['sentence_clipscore'] = sentence_clip_score

    return result_info


# ---------------------------------------------------------------------------
# Dataset-specific helpers
# ---------------------------------------------------------------------------

def get_coco_path(split, image_id, coco_dir):
    """Build COCO image path from split name and numeric image id."""
    return os.path.join(coco_dir, f"{split}2017", f"{image_id:012}.jpg")


def process_aokvqa(args, llm, prompt_template):
    """Process A-OKVQA dataset."""
    with open(args.data_path, 'r', encoding='utf-8') as f:
        val_mmb = json.load(f)

    success_path = os.path.join(os.path.dirname(args.output_path), "success_ids.json")
    if os.path.exists(success_path):
        with open(success_path, 'r', encoding='utf-8') as f:
            success_list = json.load(f)
    else:
        success_list = []

    if os.path.exists(args.output_path):
        with open(args.output_path, 'r', encoding='utf-8') as f:
            result_list = json.load(f)
    else:
        result_list = []

    for question_info in tqdm(val_mmb, desc="aokvqa"):
        image_id = question_info['image_id']
        if image_id in success_list:
            continue

        question_str = question_info['question']
        image_path = get_coco_path(args.split, image_id, args.image_dir)
        text_prompt = prompt_template.format(question=question_str)

        result_info_dict = _run_with_retry(llm, text_prompt, image_path, image_id, args.max_retry)
        if result_info_dict is None:
            continue

        success_list.append(image_id)
        new_item = copy.deepcopy(question_info)
        new_item['llava_reslut'] = result_info_dict
        result_list.append(new_item)

        _save_checkpoint(success_path, success_list, args.output_path, result_list)


def process_sciencevqa(args, llm, prompt_template):
    """Process ScienceQA dataset."""
    with open(args.data_path, encoding='utf-8') as f:
        data = json.load(f)

    splits = [args.split] if args.split else ['val']

    success_path = os.path.join(os.path.dirname(args.output_path), "success_ids.json")
    if os.path.exists(success_path):
        with open(success_path, 'r', encoding='utf-8') as f:
            success_list = json.load(f)
    else:
        success_list = []

    if os.path.exists(args.output_path):
        with open(args.output_path, 'r', encoding='utf-8') as f:
            result_list = json.load(f)
    else:
        result_list = []

    for image_id, row in tqdm(data.items(), desc="sciencevqa"):
        if image_id in success_list:
            continue
        if row['split'] not in splits:
            continue
        if not row.get('image'):
            continue

        image_path = os.path.join(args.image_dir, row['split'], image_id, 'image.png')

        question = row['question']
        choices = row['choices']
        context = row.get('hint', '')
        question_str = context + '\n' + question + '\nSelect the correct answer from: ' + str(choices)

        text_prompt = prompt_template.format(question=question_str)

        new_item = copy.deepcopy(row)
        new_item['image_id'] = image_id

        result_info_dict = _run_with_retry(llm, text_prompt, image_path, image_id, args.max_retry)
        if result_info_dict is None:
            continue

        new_item['llava_reslut'] = result_info_dict

        success_list.append(image_id)
        result_list.append(new_item)

        _save_checkpoint(success_path, success_list, args.output_path, result_list)


def process_culturalvqa(args, llm, prompt_template):
    """Process CulturalVQA dataset."""
    with open(args.data_path, 'r', encoding='utf-8') as f:
        cvqa_data = json.load(f)

    success_path = os.path.join(os.path.dirname(args.output_path), "success_ids.json")
    if os.path.exists(success_path):
        with open(success_path, 'r', encoding='utf-8') as f:
            success_list = json.load(f)
    else:
        success_list = []

    if os.path.exists(args.output_path):
        with open(args.output_path, 'r', encoding='utf-8') as f:
            result_list = json.load(f)
    else:
        result_list = []

    for question_info in tqdm(cvqa_data, desc="culturalvqa"):
        question_str = question_info['question']
        image_id = question_info['id']
        if image_id in success_list:
            continue

        image_name = f"image_{image_id}.jpg"
        image_path = os.path.join(args.image_dir, image_name)
        text_prompt = prompt_template.format(question=question_str)

        new_item = copy.deepcopy(question_info)

        result_info_dict = _run_with_retry(llm, text_prompt, image_path, image_id, args.max_retry)
        if result_info_dict is None:
            continue

        new_item['llava_reslut'] = result_info_dict

        success_list.append(image_id)
        result_list.append(new_item)

        _save_checkpoint(success_path, success_list, args.output_path, result_list)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _run_with_retry(llm, text_prompt, image_path, image_id, max_retry):
    """Run inference with retry logic. Returns result dict or None on failure."""
    for attempt in range(max_retry):
        try:
            return run_llava_next(llm, text_prompt, image_path)
        except Exception as e:
            print(f"LLaVA call failed on attempt {attempt + 1}/{max_retry} "
                  f"for image_id {image_id}: {e}")
    return None


def _save_checkpoint(success_path, success_list, output_path, result_list):
    """Persist progress to disk after each successful sample."""
    os.makedirs(os.path.dirname(success_path), exist_ok=True)
    with open(success_path, 'w', encoding='utf-8') as f:
        json.dump(success_list, f, ensure_ascii=False, indent=4)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_list, f, ensure_ascii=False, indent=4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASET_PROCESSORS = {
    'aokvqa': process_aokvqa,
    'sciencevqa': process_sciencevqa,
    'culturalvqa': process_culturalvqa,
}


def parse_args():
    parser = argparse.ArgumentParser(description="VLM uncertainty estimation pipeline")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the LLaVA model (local or HuggingFace)')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(DATASET_PROCESSORS.keys()),
                        help='Dataset name')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the dataset JSON file')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Root directory of images (COCO dir for aokvqa, '
                             'image dir for sciencevqa/culturalvqa)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path for the output JSON file')
    parser.add_argument('--split', type=str, default='val',
                        help='Data split to process (default: val)')
    parser.add_argument('--prompt_id', type=int, default=8,
                        choices=list(QUESTION_PROMPTS.keys()),
                        help='Prompt template ID (default: 8)')
    parser.add_argument('--max_retry', type=int, default=3,
                        help='Max retries per sample on failure (default: 3)')
    parser.add_argument('--max_model_len', type=int, default=4096,
                        help='vLLM max model length (default: 4096)')
    parser.add_argument('--gpu_id', type=str, default=None,
                        help='CUDA_VISIBLE_DEVICES value (e.g. "0" or "0,1")')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    llm = LLM(model=args.model_path, max_model_len=args.max_model_len)

    prompt_template = QUESTION_PROMPTS[args.prompt_id]

    processor = DATASET_PROCESSORS[args.dataset]
    processor(args, llm, prompt_template)

    print("Done.")


if __name__ == '__main__':
    main()
