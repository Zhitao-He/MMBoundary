"""
Construct SFT training data for MMBoundary.
Maps uncertainty-estimated confidence levels to natural language statements,
then builds instruction-tuning format data for LLaMA-Factory.
"""

import os
import json
import random
import argparse
from tqdm import tqdm
import copy


def construct_confidence_answer(sentences, confidence_levels, statement_dict):
    """
    For each sentence, pick a random confidence statement matching its level,
    then concatenate: 'sentence, [statement]'
    """
    sentence_with_statement = []
    for sent, level in zip(sentences, confidence_levels):
        for entry in statement_dict:
            if entry["confidence_level"] == level:
                statement = random.choice(entry["statement"])
                new_sent = sent.replace('.', ',') + ' [' + statement + ']'
                sentence_with_statement.append(new_sent)
                break
    return ' '.join(sentence_with_statement).strip()


def build_sft_data(confidence_data, statement_dict, coco_dir, question_prompt, split='train'):
    """Build LLaMA-Factory SFT format data."""
    sft_data = []
    for item in tqdm(confidence_data, desc="Building SFT data"):
        image_id = item['image_id']
        question = item['question']
        llava_result = item.get('llava_reslut', {})
        sentences = llava_result.get('sentences', [])
        confidence_levels = item.get('sentence_confidence_level', [])

        if not sentences or not confidence_levels:
            continue

        confidence_answer = construct_confidence_answer(sentences, confidence_levels, statement_dict)
        image_path = os.path.join(coco_dir, f"{split}2017", f"{image_id:012}.jpg")
        text_prompt = question_prompt.format(question=question)

        sft_data.append({
            "messages": [
                {"content": text_prompt, "role": "user"},
                {"content": confidence_answer, "role": "assistant"}
            ],
            "images": [image_path]
        })
    return sft_data


def parse_args():
    parser = argparse.ArgumentParser(description="Construct MMBoundary SFT training data")
    parser.add_argument("--confidence_data_path", type=str, required=True,
                        help="Path to confidence level data (output of uncertainty_normalize.py)")
    parser.add_argument("--confidence_statement_path", type=str,
                        default=os.path.join(os.path.dirname(__file__), "confidence_statement.json"),
                        help="Path to confidence_statement.json")
    parser.add_argument("--coco_dir", type=str, default="./datasets/coco",
                        help="Path to COCO dataset images")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for SFT data JSON")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    return parser.parse_args()


QUESTION_PROMPT = """[INST] <image>\n{question}.
Answer the question based on the image, explain your reasoning concisely:
"""


if __name__ == "__main__":
    args = parse_args()

    with open(args.confidence_data_path, 'r', encoding='utf-8') as f:
        confidence_data = json.load(f)

    with open(args.confidence_statement_path, 'r', encoding='utf-8') as f:
        statement_dict = json.load(f)

    sft_data = build_sft_data(
        confidence_data, statement_dict, args.coco_dir, QUESTION_PROMPT, args.split
    )

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=4)

    print(f"Built {len(sft_data)} SFT training examples -> {args.output_path}")
