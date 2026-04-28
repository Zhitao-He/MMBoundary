"""
Construct MMBoundary reasoning-chain dataset for A-OKVQA.

Uses GPT-4o to generate two-level reasoning chains (Image_level + Reasoning_level)
for each question in the A-OKVQA dataset, then filters them for conciseness.
Intermediate results are saved incrementally so the process can be resumed.
"""

import os
import json
import base64
import ast
import re
import argparse
from tqdm import tqdm
from openai import OpenAI


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def extract_and_convert(text_output):
    """Extract a dictionary from GPT output text and convert it to a Python dict."""
    match = re.search(r'{(.*)}', text_output, re.DOTALL)
    if match:
        dict_content = match.group(0).strip()

        temp_content = re.sub(r'"', '$$', dict_content)
        formatted_text = temp_content.replace("\'s", '##s')
        temp_content = temp_content.replace("\\'", "@@")
        formatted_text = temp_content.replace("'", '"')
        formatted_text = formatted_text.replace('$$', "'")
        formatted_text = formatted_text.replace("@@", "\\'")
        formatted_text = formatted_text.replace("##s", "\'s")

        try:
            result_dict = ast.literal_eval(formatted_text)
            return result_dict
        except (SyntaxError, ValueError) as e:
            print("Error converting text to dictionary:", e)
            return None
    else:
        print("No dictionary found in the text.")
        return None


def encode_image(image_path):
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def GPT4(text_input):
    """Call GPT-4o with a text-only prompt."""
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text_input},
        ],
    )
    return completion.choices[0].message.content


def GPT4v(text_input, image_input):
    """Call GPT-4o with a text + image prompt."""
    client = OpenAI()
    base64_image = encode_image(image_input)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_input},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_aokvqa(aokvqa_dir, split, version='v1p0'):
    assert split in ['train', 'val', 'test', 'test_w_ans']
    dataset = json.load(open(
        os.path.join(aokvqa_dir, f"aokvqa_{version}_{split}.json")
    ))
    return dataset


def get_coco_path(split, image_id, coco_dir):
    return os.path.join(coco_dir, f"{split}2017", f"{image_id:012}.jpg")


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

prompt_chain = '''
    You will receive a question, an accompanying image, the correct answer, and the corresponding rationales. Follow these steps to generate your analysis (reasoning chain) structured in two levels.
    Each level should include concise, interconnected sentences, with each sentence conveying a single piece of knowledge. Ensure the reasoning chain covers all necessary knowledge points concisely, with each sentence in this reasoning chain is essential and avoid adding redundant or irrelevant sentences.
    The levels are as follows:
    **Image level**: Identify key visual elements in the image that are mostly relevant to the question and answer. Format these sentences in JSON, like: ['sentence 1', ..., 'sentence i'].
    **Reasoning level**: Based on the extracted visual elements, provide logical reasoning that justifies why the answer is correct. Format these in JSON as well: ['sentence i+1', 'sentence i+2', ..., 'So, the answer is ...'].
    The sentences in both levels together should form a coherent chain of reasoning that clearly explains why the answer is correct. Ensure that each sentence builds upon the previous one to complete the reasoning chain.
    In the final sentence of the reasoning level, provide a clear conclusion with the answer, like: "So, the answer is ...".

    Refer to the example below:
    Please answer the following question:
    Image: (Three people in traditional clothing holding drums, performing a form of the 'whirling dervishes' ritual.)
    Question: Which city in Turkey is the origin of the performers depicted in the image?
    Answer: Konya
    Analysis: {{'Image_level': [], 'Reasoning_level': []}}
    Your output:
    Analysis: {{
                'Image_level': [
                    'The image shows a group of performers in traditional Turkish attire',
                    'The performers are engaged in a traditional dance involving drums'
                ],
                'Reasoning_level': [
                    'This dance style is associated with the Whirling Dervishes',
                    'Whirling Dervishes are followers of the Mevlevi Sufi order',
                    'The Mevlevi order originated in Konya, Turkey',
                    'So the answer is Konya'
                ]
                }}

    Now, please answer the following question:
    Image: image
    Question: {question}
    Answer: {answer}
    Analysis:{{
                'Image_level': [],
                'Reasoning_level': []
                }}

    Your output:
    Analysis:
'''

prompt_concise = '''
    Now, the following analysis (reasoning chain) is structured.
    Image_level and Reasoning_level together form a complete reasoning chain. Please filter out any irrelevant sentences to maintain a concise reasoning chain, including only the essential sentences.

    Refer to the example below:
    Image: (image)
    Question: What is the brown bear in the grey and black jacket doing?
    Answer: playing hockey
    Analysis:{{'Image_level': [ 'The image shows a brown teddy bear wearing a grey and black jacket.', 'The teddy bear is holding a hockey stick.', 'There is a hockey puck positioned near the teddy bear's stick.', 'The setting includes artificial snow, adding a winter or ice hockey theme.' ], 'Reasoning_level': [ 'The presence of the hockey stick and puck suggests an activity related to hockey.', 'The teddy bear's posture and positioning of the stick indicate it is actively involved in a hockey action.', 'Hockey is typically associated with actions like skating, passing, or shooting the puck.', 'Given the setup, it is reasonable to conclude that the bear is mimicking playing hockey.', 'So, the answer is playing hockey.' ] }}
    Your output:
    Analysis:{{
                'Image_level': [
                    'The teddy bear is holding a hockey stick.',
                    'There is a hockey puck positioned near the teddy bear's stick.'
                ],
                'Reasoning_level': [
                    'The presence of the hockey stick and puck suggests an activity related to hockey.',
                    'The teddy bear's posture and positioning of the stick indicate it is actively involved in a hockey action.',
                    'So, the answer is playing hockey.'
                ]
            }}

    Now:
    Image: (image)
    Question: {question}
    Answer: {answer}
    Analysis: {analysis}
    Your output:
    Analysis:
'''


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    # Load dataset
    data = load_aokvqa(args.aokvqa_dir, args.split)

    # Resume state
    output_path = os.path.join(args.output_dir, f"{args.split}_mmb.json")
    success_path = os.path.join(args.output_dir, f"{args.split}_success.json")
    error_path = os.path.join(args.output_dir, f"{args.split}_error.json")

    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            existed_augmented_data = json.load(f)
    else:
        existed_augmented_data = []

    if os.path.exists(success_path):
        with open(success_path, 'r', encoding='utf-8') as f:
            success_id = json.load(f)
    else:
        success_id = []

    if os.path.exists(error_path):
        with open(error_path, 'r', encoding='utf-8') as f:
            error_id = json.load(f)
    else:
        error_id = []

    max_retry = 3

    for question_info in tqdm(data):
        image_id = question_info['image_id']

        if image_id in success_id:
            continue

        question_str = question_info['question']
        answer_choice = question_info['choices']
        answer_str = answer_choice[question_info['correct_choice_idx']]
        image_path = get_coco_path(args.split, image_id, args.coco_dir)

        GPT4v_prompt = prompt_chain.format(question=question_str, answer=answer_str)

        # Call GPT4v with retry
        for attempt in range(max_retry):
            try:
                GPT4v_output = GPT4v(GPT4v_prompt, image_path)
                break
            except Exception as e:
                print(f"GPT4v failed attempt {attempt + 1}/{max_retry} for image_id {image_id}: {e}")
                if attempt == max_retry - 1:
                    error_id.append(image_id)
                    with open(error_path, 'w', encoding='utf-8') as f:
                        json.dump(error_id, f, ensure_ascii=False, indent=4)
                    continue
        else:
            continue

        print('--GPT4v_output is OK')
        question_analysis = GPT4v_output
        GPT4_prompt = prompt_concise.format(
            question=question_str, answer=answer_str, analysis=question_analysis
        )

        # Call GPT4 with retry
        for attempt in range(max_retry):
            try:
                GPT4_output = GPT4(GPT4_prompt)
                break
            except Exception as e:
                print(f"GPT4 failed attempt {attempt + 1}/{max_retry} for image_id {image_id}: {e}")
                if attempt == max_retry - 1:
                    error_id.append(image_id)
                    with open(error_path, 'w', encoding='utf-8') as f:
                        json.dump(error_id, f, ensure_ascii=False, indent=4)
                    continue
        else:
            continue

        print('--GPT4_output is OK\n')

        # Parse outputs
        print('GPT4v_transfer......')
        dict_GPT4v_output = extract_and_convert(GPT4v_output)
        print('GPT4_transfer.......')
        dict_GPT4_output = extract_and_convert(GPT4_output)

        if dict_GPT4v_output and dict_GPT4_output:
            success_id.append(image_id)
            question_info['keys'] = dict_GPT4v_output
            question_info['keys_filtered'] = dict_GPT4_output
            existed_augmented_data.append(question_info)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(existed_augmented_data, f, ensure_ascii=False, indent=4)
            with open(success_path, 'w', encoding='utf-8') as f:
                json.dump(success_id, f, ensure_ascii=False, indent=4)
        else:
            error_id.append(image_id)
            with open(error_path, 'w', encoding='utf-8') as f:
                json.dump(error_id, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate GPT-4o reasoning chains for A-OKVQA."
    )
    parser.add_argument("--aokvqa_dir", type=str, required=True,
                        help="Path to A-OKVQA dataset directory (contains aokvqa_v1p0_*.json)")
    parser.add_argument("--coco_dir", type=str, required=True,
                        help="Path to COCO images root directory (contains train2017/, val2017/)")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test", "test_w_ans"],
                        help="Dataset split to process (default: train)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save output JSON files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
