"""
Construct MMBoundary reasoning-chain dataset for ScienceQA.

Uses GPT-4o to generate two-level reasoning chains (Image_level + Reasoning_level)
for each image-bearing question in the ScienceQA dataset, then filters them for
conciseness.  Intermediate results are saved incrementally so the process can be
resumed.
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
# Prompt templates
# ---------------------------------------------------------------------------

prompt_chain = '''
    You will receive a question with context, an accompanying image, the lecture about question, the solution about answering question, and the correct answer. Follow these steps to generate your analysis (reasoning chain) structured in two levels.
    Each level should include concise, interconnected sentences, with each sentence conveying a single piece of knowledge. Ensure the reasoning chain covers all necessary knowledge points concisely, with each sentence in this reasoning chain is essential and avoid adding redundant or irrelevant sentences.
    The levels are as follows:
    **Image level**: Identify key visual elements in the image that are mostly relevant to the question and answer. Format these sentences in JSON, like: ['sentence 1', ..., 'sentence i'].
    **Reasoning level**: Based on the extracted visual elements, provide logical reasoning that justifies why the answer is correct. Format these in JSON as well: ['sentence i+1', 'sentence i+2', ..., 'So, the answer is ...'].
    The sentences in both levels together should form a coherent chain of reasoning that clearly explains why the answer is correct. Ensure that each sentence builds upon the previous one to complete the reasoning chain.
    In the final sentence of the reasoning level, provide a clear conclusion with the answer, like: "So, the answer is ...".

    Refer to the example below:
    Please answer the following question:
    Image: (Black-bellied whistling duck)
    Question: Black-bellied whistling ducks live near lakes and streams in North America. They find most of their food in shallow water. The feet of the are adapted for swimming. Which animal's feet are also adapted for swimming?
    Choice: ['purple gallinule', 'lava gull']
    Lecture: An adaptation is an inherited trait that helps an organism survive or reproduce. Adaptations can include both body parts and behaviors. The shape of an animal's feet is one example of an adaptation. Animals' feet can be adapted in different ways. For example, webbed feet might help an animal swim. Feet with thick fur might help an animal walk on cold, snowy ground.
    Solution: Look at the picture of the black-bellied whistling duck. The black-bellied whistling duck has webbed feet. Its feet are adapted for swimming. As it swims, the black-bellied whistling duck uses its webbed feet to push itself through water. Now look at each animal. Figure out which animal has a similar adaptation. The lava gull has webbed feet. Its feet are adapted for swimming. The purple gallinule has long, thin toes on its feet. Its feet are not adapted for swimming. The purple gallinule uses its feet to walk on large leaves floating on the surface of the water.
    Answer: lava gull
    Analysis: {{'Image_level': [], 'Reasoning_level': []}}
    Your output:
    Analysis: {{
                'Image_level': [
                    'The image shows the feet of a black-bellied whistling duck',
                    'The feet are webbed, with a broad structure'
                ],
                'Reasoning_level': [
                    'Webbed feet help animals push through water, aiding in swimming',
                    'Among the given options, the lava gull also has webbed feet',
                    'The purple gallinule has long, thin toes, which are not adapted for swimming',
                    'So, the answer is lava gull'
                ]
                }}

    Now, please answer the following question:
    Image: (image)
    Question: {question}
    Choice: {choice}
    Lecture: {lecture}
    Solution: {solution}
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
    Question: Black-bellied whistling ducks live near lakes and streams in North America. They find most of their food in shallow water. The feet of the are adapted for swimming. Which animal's feet are also adapted for swimming?
    Choice: ['purple gallinule', 'lava gull']
    Answer: lava gull
    Analysis:{{'Image_level': [ 'The image shows the feet of a black-bellied whistling duck.', 'The feet are webbed, with a broad structure ideal for swimming.'], 'Reasoning_level': [ 'The black-bellied whistling duck has webbed feet that are adapted for swimming.', 'Webbed feet help animals push through water, aiding in swimming.', 'The lecture mentions that webbed feet are a specific adaptation for aquatic environments.', 'Among the given options, the lava gull also has webbed feet.', 'The purple gallinule has long, thin toes, which are not adapted for swimming but for walking on floating vegetation.', 'So, the answer is lava gull.'] }}
    Your output:
    Analysis:{{
                'Image_level': [
                    'The image shows the feet of a black-bellied whistling duck',
                    'The feet are webbed, with a broad structure'
                ],
                'Reasoning_level': [
                    'Webbed feet help animals push through water, aiding in swimming',
                    'Among the given options, the lava gull also has webbed feet',
                    'The purple gallinule has long, thin toes, which are not adapted for swimming',
                    'So, the answer is lava gull'
                ]
            }}

    Now:
    Image: (image)
    Question: {question}
    Choice: {choice}
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
    with open(args.data_path, encoding="utf-8") as f:
        data = json.load(f)

    target_splits = [s.strip() for s in args.split.split(",")]

    # Resume state
    output_path = os.path.join(args.output_dir, f"{'_'.join(target_splits)}_mmb.json")
    success_path = os.path.join(args.output_dir, f"{'_'.join(target_splits)}_success.json")

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

    max_retry = 3

    for image_id, row in tqdm(data.items()):
        if image_id in success_id:
            continue
        if row['split'] not in target_splits:
            continue
        if not row['image']:
            continue

        image_path = os.path.join(args.image_dir, row['split'], image_id, 'image.png')

        question = row['question']
        choices = row['choices']
        choice_index = row['answer']
        context = row['hint']
        lecture = row['lecture']
        solution = row['solution']
        answer_str = choices[choice_index]

        question_str = context + ' ' + question
        GPT4v_prompt = prompt_chain.format(
            question=question_str, choice=choices, lecture=lecture,
            solution=solution, answer=answer_str,
        )

        # Call GPT4v with retry
        for attempt in range(max_retry):
            try:
                GPT4v_output = GPT4v(GPT4v_prompt, image_path)
                break
            except Exception as e:
                print(f"GPT4v failed attempt {attempt + 1}/{max_retry} for image_id {image_id}: {e}")
                if attempt == max_retry - 1:
                    continue
        else:
            continue

        print('--GPT4v_output is OK')
        question_analysis = GPT4v_output
        GPT4_prompt = prompt_concise.format(
            question=question_str, choice=choices,
            answer=answer_str, analysis=question_analysis,
        )

        # Call GPT4 with retry
        for attempt in range(max_retry):
            try:
                GPT4_output = GPT4(GPT4_prompt)
                break
            except Exception as e:
                print(f"GPT4 failed attempt {attempt + 1}/{max_retry} for image_id {image_id}: {e}")
                if attempt == max_retry - 1:
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
            row['keys'] = dict_GPT4v_output
            row['keys_filtered'] = dict_GPT4_output
            existed_augmented_data.append(row)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(existed_augmented_data, f, ensure_ascii=False, indent=4)
            with open(success_path, 'w', encoding='utf-8') as f:
                json.dump(success_id, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate GPT-4o reasoning chains for ScienceQA."
    )
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to problems.json")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Root directory of ScienceQA images (contains train/, val/, test/ subdirs)")
    parser.add_argument("--split", type=str, default="val",
                        help="Comma-separated splits to process, e.g. 'train,val' (default: val)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save output JSON files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
