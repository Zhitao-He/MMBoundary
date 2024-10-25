# from transformers import llava2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from llava_vl_utils import process_vision_info
# from modelscope import snapshot_download
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
from openai import OpenAI
import copy
import os
import json
import torch
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

def extract_assistant_response(text):
    start_marker = "ASSISTANT:"
    start_index = text.find(start_marker)

    if start_index != -1:
        return text[start_index + len(start_marker):].strip()
    else:
        return "No ASSISTANT response found."

def load_aokvqa(aokvqa_dir, split, version='v1p0'):
    assert split in ['train', 'val', 'test', 'test_w_ans']
    dataset = json.load(open(
        os.path.join(aokvqa_dir, f"aokvqa_{version}_{split}.json")
    ))
    return dataset

def get_coco_path(split, image_id, coco_dir):
    return os.path.join(coco_dir, f"{split}2017", f"{image_id:012}.jpg")

#train_dataset = load_aokvqa(aokvqa_dir, 'train')  # also 'val' or 'test'

""" #model_dir = snapshot_download("/mnt/workspace/hzt/mind2web/llava2-VL")
model_dir = "/mnt/workspace/hzt/mind2web/llava2-VL"

# default: Load the model on the available device(s)
model = llava2VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = llava2VLForConditionalGeneration.from_pretrained(
#     model_dir,
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained(model_dir)

def LLava(image_info, prompt_string):
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

OD_path = '/mnt/workspace/hzt/mind2web/data1/aokvqa/datasets/quick_data/Object_Detection.json'
AR_path = '/mnt/workspace/hzt/mind2web/data1/aokvqa/datasets/quick_data/Attribute_Recognition.json'
OR_path = '/mnt/workspace/hzt/mind2web/data1/aokvqa/datasets/quick_data/Object_Relationships.json'
AC_path = '/mnt/workspace/hzt/mind2web/data1/aokvqa/datasets/quick_data/Action_Recognition.json'
SC_path = '/mnt/workspace/hzt/mind2web/data1/aokvqa/datasets/quick_data/Scene_Context.json'

with open(OD_path, 'r', encoding='utf-8') as file:
    OD_data = json.load(file)

with open(AR_path, 'r', encoding='utf-8') as file:
    AR_data = json.load(file)

with open(OR_path, 'r', encoding='utf-8') as file:
    OR_data = json.load(file)

with open(AC_path, 'r', encoding='utf-8') as file:
    AC_data = json.load(file)

with open(SC_path, 'r', encoding='utf-8') as file:
    SC_data = json.load(file)

idx = 0
data_result = []
OD_data = []
AR_data = []
for data in [OD_data, AR_data, OR_data, AC_data, SC_data]:
    # 记录不同的数量
    right = 0
    wrong = 0
    right_refuse = 0
    right_right = 0
    wrong_non_refuse = 0
    wrong_refuse = 0
    coco_dir = '/mnt/workspace/hzt/mind2web/data1/aokvqa/datasets/coco'
    data_new = []
    for sample in data:
        image_path = get_coco_path('train', sample['image_id'], coco_dir)
        #iamge_save_path = f"/mnt/workspace/hzt/mind2web/py_file/llava_check/images/{sample['image_id']:012}.jpg"
        #shutil.copyfile(image_path, iamge_save_path)
        question = sample['question']
        question_answer = str(sample["direct_answers"]).replace('"', "'")
        question_id = sample['question_id']
        question_string = prompt_question.format(question=question)
        llava_answer = LLava(image_path, question_string)

        #print('---question_prompt:', question_string)
        #print('Current image_path:', image_path)
        #print('Current llava_answer:', llava_answer)

        # 保存llava的回答以备人工检查
        sample['llava_answer'] = llava_answer
        
        
        try:
            GPT_judge_right_prompt = GPT_judge_right.format(question=question, gold=question_answer, answer=llava_answer) 
            sample['GPT_judge_prompt'] = GPT_judge_right_prompt
            #print('---GPT_judge_prompt:', GPT_judge_right_prompt)
            right_score = GPT_4o(GPT_judge_right_prompt)
            sample['GPT_judge'] = right_score
            print('---right_score:', right_score)

            judge_right = float(right_score)
            #print('---judge_right:', judge_right)

            if judge_right: 
                right += 1
                # 错误拒答（以为不知道）
                # (确实知道)
                try:
                    question_string = prompt_w_refuse.format(question=question)
                    llava_answer = LLava(image_path, question_string)
                    sample['llava_right_prompt_w_refuse'] = llava_answer
                    #print('---llava_right_prompt_w_refuse:', question_string)
                    GPT_judge_right_refuse_prompt = GPT_judge_right_refuse.format(question=question, gold=question_answer, answer=llava_answer) 
                    sample['GPT_judge_right_refuse_prompt'] = GPT_judge_right_refuse_prompt
                    #print('---GPT_judge_right_refuse_prompt:', GPT_judge_right_refuse_prompt)
                    refuse_score = GPT_4o(GPT_judge_right_refuse_prompt)
                    sample['GPT_judge_right_refuse'] = refuse_score 
                    judge_right_refuse = float(refuse_score)
                    print('---judge_right_refuse:', judge_right_refuse)
                    if judge_right_refuse:
                        right_refuse += 1
                    else:
                        right_right += 1
                except Exception as e:
                    print('E_judge_right_refuse:', e)
                    print('error_id:', question_id)
                    print("judge_right_refuse:", judge_right_refuse)
            else:
                wrong += 1 
                # （以为知道）
                # （确实不知道）
                try:
                    question_string = prompt_w_refuse.format(question=question)
                    llava_answer = LLava(image_path, question_string)
                    sample['llava_wrong_prompt_w_refuse'] = llava_answer
                    #print('---llava_wrong_prompt_w_refuse:', question_string)
                    GPT_judge_wrong_refuse_prompt = GPT_judge_wrong_refuse.format(question=question, gold=question_answer, answer=llava_answer) 
                    sample['GPT_judge_wrong_refuse_prompt'] = GPT_judge_wrong_refuse_prompt
                    #print('---GPT_judge_wrong_refuse_prompt:', GPT_judge_wrong_refuse_prompt)
                    wrong_refuse_score = GPT_4o(GPT_judge_wrong_refuse_prompt)
                    sample['GPT_judge_wrong_refuse'] = wrong_refuse_score 
                    judge_wrong_refuse = float(wrong_refuse_score)
                    print('---judge_wrong_refuse:', judge_wrong_refuse)
                    if judge_wrong_refuse:
                        wrong_non_refuse += 1
                    else:
                        wrong_refuse += 1
                except Exception as e:
                    print('E_judge_wrong_refuse:', e)
                    print('error_id:', question_id)
                    print("judge_wrong_refuse:", judge_wrong_refuse)
        except Exception as e:
            print('E:', e)
            print('error_id:', question_id)
            #print("judge_right:", judge_right)
        
        sample_new = copy.deepcopy(sample)
        data_new.append(sample_new)
        
        # if idx == 0:
        #     print('---Current: Object_Detection')
        #     with open('/mnt/workspace/hzt/mind2web/py_file/llava_check/Object_Detection.json', 'w', encoding='utf-8') as f:
        #         json.dump(data_new, f, ensure_ascii=False, indent=4)
        # if idx == 1:
        #     print('---Current: Attribute_Recognition')
        #     with open('/mnt/workspace/hzt/mind2web/py_file/llava_check/Attribute_Recognition.json', 'w', encoding='utf-8') as f:
        #         json.dump(data_new, f, ensure_ascii=False, indent=4)
        if idx == 2:
            print('---Current: Object_Relationships')
            with open('/mnt/workspace/hzt/mind2web/py_file/llava_check/Object_Relationships.json', 'w', encoding='utf-8') as f:
                json.dump(data_new, f, ensure_ascii=False, indent=4)
        if idx == 3:
            print('---Current: Action_Recognition')
            with open('/mnt/workspace/hzt/mind2web/py_file/llava_check/Action_Recognition.json', 'w', encoding='utf-8') as f:
                json.dump(data_new, f, ensure_ascii=False, indent=4)
        if idx == 4:
            print('---Current: Scene_Context')
            with open('/mnt/workspace/hzt/mind2web/py_file/llava_check/Scene_Context.json', 'w', encoding='utf-8') as f:
                json.dump(data_new, f, ensure_ascii=False, indent=4)


        print(f"right: {right}, "
        f"right_refuse: {right_refuse}, "
        f"right_right: {right_right}, "
        f"\nwrong: {wrong}, "
        f"wrong_non_refuse: {wrong_non_refuse},"
        f"wrong_refuse: {wrong_refuse}")
        print('-----------------\n')

    print('====================================')

    data = {
        "right": right,
        "wrong": wrong,
        "right_refuse": right_refuse,
        "right_right": right_right,
        "wrong_non_refuse": wrong_non_refuse,
        "wrong_refuse": wrong_refuse
    }
    data_result.append(data)
    
    with open("/mnt/workspace/hzt/mind2web/py_file/llava_check/data_result.json", "w") as json_file:
        json.dump(data_result, json_file, indent=4)
     
    idx += 1 

 
# 先用无拒答的提示让VLM作答 得到 正确回答和错误回答
# 用有拒答的提示让VLM作答 得到 正确拒答和错误拒答
# 保存VLM答案
# GPT4判断是否正确
# 记录 1. 正确回答：VLM给出了正确的答案，且与预期相符 （确实知道）
#      2. 错误回答：VLM给出了错误的答案，但未回答"不知道" （以为知道）
# 记录 1. 正确拒答：VLM回答了"不知道"且问题确实超出它的能力 （确实不知道）
#     2. 错误拒答：VLM错误地回答了"不知道"，实际上它应该能给出正确答案 （以为不知道）
# 错误回答，错误拒答







