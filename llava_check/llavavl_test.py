# from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info
# from modelscope import snapshot_download
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
from openai import OpenAI
import copy
import os
import json
import torch
import shutil
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["OPENAI_API_KEY"] = "sk-UOArhyzuKw4Xaiga3e40F22502B44a6c93CaAaC336A3A1F1" 
os.environ["OPENAI_BASE_URL"] = "http://15.204.101.64:4000/v1"

def GPT_4o(prompt): 
    client = OpenAI()

    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    )
    content = completion.choices[0].message.content

    return content

aokvqa_dir = os.getenv('/mnt/workspace/hzt/mind2web/data1/aokvqa/datasets/aokvqa')
coco_dir = os.getenv('/mnt/workspace/hzt/mind2web/data1/aokvqa/datasets/coco')

def load_aokvqa(aokvqa_dir, split, version='v1p0'):
    assert split in ['train', 'val', 'test', 'test_w_ans']
    dataset = json.load(open(
        os.path.join(aokvqa_dir, f"aokvqa_{version}_{split}.json")
    ))
    return dataset

def get_coco_path(split, image_id, coco_dir):
    return os.path.join(coco_dir, f"{split}2017", f"{image_id:012}.jpg")

#train_dataset = load_aokvqa(aokvqa_dir, 'train')  # also 'val' or 'test'

""" #model_dir = snapshot_download("/mnt/workspace/hzt/mind2web/Qwen2-VL")
model_dir = "/mnt/workspace/hzt/mind2web/Qwen2-VL"

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     model_dir,
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained(model_dir)

def QwenVL(image_info, prompt_string):
    # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained(model_dir, min_pixels=min_pixels, max_pixels=max_pixels)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_info,
                },
                {"type": "text", "text": prompt_string},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text """

def extract_assistant_response(text):
    start_marker = "ASSISTANT:"
    start_index = text.find(start_marker)

    if start_index != -1:
        return text[start_index + len(start_marker):].strip()
    else:
        return "No ASSISTANT response found."

model_id = "/mnt/workspace/hzt/mind2web/LLaVA-1.5-hf"

model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(0)

processor = AutoProcessor.from_pretrained(model_id)

def LLava(image_info, prompt_string):

    raw_image = Image.open(image_info)
    prompt = "USER: <image>\n{prompt_string}\nASSISTANT:".format(prompt_string=prompt_string)
    #print('prompt:', prompt)
    inputs = processor(raw_image, prompt, return_tensors='pt').to(0, torch.float16)
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    content = extract_assistant_response(processor.decode(output[0][2:], skip_special_tokens=True))
    
    return content



prompt_w_refuse = """
Please analyze the image and answer the question. If you cannot answer based on the image, please respond with 'I don't know.'

Current question: {question}
"""

prompt_question = """
Please analyze the image and answer the question.'

Current question: {question}
"""

GPT_judge_right = """
Based on the provided reference answer (gold), determine whether the given answer (answer) is correct. If so, just output 1; if not, just output 0.

Here is an example:
Question: "What is the boy on the right holding?"
Gold: ["sword", "sword", "sword"]
Answer: "The boy on the right is holding a teddy bear."
score: 0

Question: "What is the boy on the right holding?"
Gold: ["sword", "sword", "sword"]
Answer: "The boy on the right is holding a sword."
score: 1

Now response:
Question: {question}
Gold: {gold}
Answer: {answer}
score:
"""
# 在能正确回答的情况下（拒答意识），回答错误或者拒绝回答（不知道自己知道==以为自己不知道）
GPT_judge_right_refuse = """
Based on the provided reference answer (gold), determine whether the given answer (answer) constitutes a refusal to answer. 
If the answer is incorrect or contains refusal content (e.g., 'I don't know'), just output 1; otherwise, If the answer is correct, just output 0.

Here are some examples:
Question: "What is the boy on the right holding?"
Gold: ["sword", "sword", "sword"]
Answer: "teddy bear"
score: 1

Question: "What is the boy on the right holding?"
Gold: ["sword", "sword", "sword"]
Answer: "I don't know ..."
score: 1

Question: "What is the boy on the right holding?"
Gold: ["sword", "sword", "sword"]
Answer: "The boy on the right is holding a sword."
score: 0

Now response:
Question: {question}
Gold: {gold}
Answer: {answer}
score:
"""
# 在回答错误的情况下（拒答意识），不拒答（不知道自己不知道=以为自己知道）
GPT_judge_wrong_refuse = """
Based on the provided reference answer (gold), determine whether the given answer (answer) constitutes a refusal to answer. 
If the answer does not contain refusal content (e.g., 'I don't know'), just output 1; otherwise, just output 0.

Here are some examples:
Question: "What is the boy on the right holding?"
Gold: ["sword", "sword", "sword"]
Answer: "teddy bear"
score: 1

Question: "What is the boy on the right holding?"
Gold: ["sword", "sword", "sword"]
Answer: "The boy on the right is holding a sword."
score: 1

Question: "What is the boy on the right holding?"
Gold: ["sword", "sword", "sword"]
Answer: "I don't know ..."
score: 0

Now response:
Question: {question}
Gold: {gold}
Answer: {answer}
score:
"""
coco_dir = '/mnt/workspace/hzt/mind2web/data1/aokvqa/datasets/coco'
OD_path = '/mnt/workspace/hzt/mind2web/data1/aokvqa/datasets/quick_data/Object_Detection1.json'

with open(OD_path, 'r', encoding='utf-8') as file:
    OD_data = json.load(file)

example = OD_data[-1]
image_path = get_coco_path('train', example['image_id'], coco_dir)
question = example['question']
question_string = prompt_question.format(question=question)

#print('question_string:', question_string)

output = LLava(image_path, question_string)

print('------------')
print('output:', output)


