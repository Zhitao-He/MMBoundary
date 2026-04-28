"""
Construct MMBoundary reasoning-chain dataset for CulturalVQA.

Uses GPT-4o to (1) generate a general thought+answer, (2) produce a two-level
reasoning chain (Image_level + Reasoning_level), and (3) filter the chain for
conciseness.  Images come from HuggingFace Arrow files via the `datasets`
library and are PIL Image objects.  Intermediate results are saved incrementally
so the process can be resumed.
"""

import os
import io
import json
import base64
import ast
import re
import argparse
from tqdm import tqdm
from datasets import Dataset
from PIL import Image
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


def encode_image(pil_image):
    """Convert a PIL Image to a base64-encoded JPEG string."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


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
    """Call GPT-4o with a text + image prompt.  image_input is a PIL Image."""
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
    You will receive a question, an accompanying image. Follow these steps to generate your analysis (reasoning chain) structured in two levels.
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
    Analysis: {analysis}
    Your output:
    Analysis:
'''

general_prompt = '''
    Answer the question based on the given image. Your response should follow the specified format. Ensure your thought process is logical, step-by-step, and your answer is concise and accurate.

    Output Format
    Thought: (Describe your reasoning clearly)
    Answer: So the answer is (a short and precise answer)

    Example:
    Image: (Three people in traditional clothing holding drums, performing a form of the 'whirling dervishes' ritual.)
    Question: Which city in Turkey is the origin of the performers depicted in the image?
    Your Output:
    Thought: The image shows a group of performers in traditional Turkish attire. This dance style is associated with the Whirling Dervishes. Whirling Dervishes are followers of the Mevlevi Sufi order. The Mevlevi order originated in Konya, Turkey.
    Answer: So the answer is Konya.

    Now Answer the Following:
    Image: image
    Question: {question}
    Your Output:

'''


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    # Load dataset from Arrow file
    dataset = Dataset.from_file(args.data_path)

    # Resume state
    output_path = os.path.join(args.output_dir, "cvqa_mmb.json")
    success_path = os.path.join(args.output_dir, "cvqa_success.json")
    error_path = os.path.join(args.output_dir, "cvqa_error.json")
    error_general_path = os.path.join(args.output_dir, "cvqa_general_error.json")

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

    error_general = []
    max_retry = 3

    for idx, question_info in tqdm(enumerate(dataset), total=len(dataset)):
        image_id = idx
        current_data = {}

        if image_id in success_id:
            continue

        print('---Current:', image_id)
        question_str = question_info["question"]
        pil_image = question_info["image"]

        # Save image locally as JPEG
        image_name = f"image_{image_id}.jpg"
        file_path = os.path.join(args.image_save_dir, image_name)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        pil_image.save(file_path, format="JPEG")

        current_data['question'] = question_str
        current_data['id'] = image_id
        current_data['image_info'] = str(pil_image)

        # --- Step 1: General prompt ---
        GPT4v_general_prompt = general_prompt.format(question=question_str)
        for attempt in range(max_retry):
            try:
                GPT4v_general_output = GPT4v(GPT4v_general_prompt, pil_image)
                break
            except Exception as e:
                print(f"GPT4v_general failed attempt {attempt + 1}/{max_retry} for image_id {image_id}: {e}")
                if attempt == max_retry - 1:
                    error_general.append(image_id)
                    with open(error_general_path, 'w', encoding='utf-8') as f:
                        json.dump(error_general, f, ensure_ascii=False, indent=4)
                    continue
        else:
            continue

        print('-----GPT4v_general_output is OK')
        current_data['general_output'] = GPT4v_general_output

        # --- Step 2: Chain prompt ---
        GPT4v_prompt = prompt_chain.format(question=question_str)
        for attempt in range(max_retry):
            try:
                GPT4v_output = GPT4v(GPT4v_prompt, pil_image)
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

        print('-------GPT4v_output is OK')
        question_analysis = GPT4v_output
        GPT4_prompt = prompt_concise.format(
            question=question_str, analysis=question_analysis,
        )

        # --- Step 3: Concise prompt ---
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

        print('---------GPT4_output is OK')

        # Parse outputs
        print('-----------GPT4v_transfer......OK')
        dict_GPT4v_output = extract_and_convert(GPT4v_output)
        print('-------------GPT4_transfer.......OK')
        dict_GPT4_output = extract_and_convert(GPT4_output)

        if dict_GPT4v_output and dict_GPT4_output:
            success_id.append(image_id)
            current_data['keys'] = dict_GPT4v_output
            current_data['keys_filtered'] = dict_GPT4_output
            existed_augmented_data.append(current_data)
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
        description="Generate GPT-4o reasoning chains for CulturalVQA."
    )
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the CulturalVQA Arrow file (e.g. cultural_vqa-test-00000-of-00002.arrow)")
    parser.add_argument("--image_save_dir", type=str, required=True,
                        help="Directory to save extracted JPEG images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save output JSON files")
    args = parser.parse_args()

    os.makedirs(args.image_save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
