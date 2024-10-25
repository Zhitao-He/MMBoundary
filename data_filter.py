import os
from openai import OpenAI
import json
import random

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

prompt_string = """
    You will receive VQA question information (image data is missing). Based on the given "question", corresponding "answer", and provided "rationales", 
    determine whether the question belongs to any of the following categories (note that not all questions may fit these categories, and a question can only belong to the closest category.). 
    If the question belongs to a category, then just output the category name.

    Question Categories: ['Object Detection', 'Attribute Recognition', 'Object Relationships', 'Action Recognition', 'Scene Context']

    Here is an example:

    Question:
    {{'split': 'xxxxx',
    'image_id': xxxxx,
    'question_id': 'xxxxxx',
    'question': 'What does it look like the girl is holding?',
    'choices': [ 'baby', 'cat', 'egg', 'tin foil' ],
    'difficult_direct_answer': false
    ...
    }}

    Question Category: 'Object_Detection'

    Please respond:
    Question: {question}

    Question Category:
"""

Object_Detection = []
Attribute_Recognition = []
Object_Relationships = []
Action_Recognition = []
Scene_Context = []

train_data = '/mnt/workspace/hzt/mind2web/data1/aokvqa/datasets/aokvqa/aokvqa_v1p0_train.json'
val_data = '/mnt/workspace/hzt/mind2web/data1/aokvqa/datasets/aokvqa/aokvqa_v1p0_val.json'

with open(train_data, 'r', encoding='utf-8') as file:
    data = json.load(file)

for question_info in data:
    
    random_float = random.random()

    if random_float > 0.6:
        question_info_str = str(question_info)
        question_info_str = '{' + question_info_str.replace('"', "'") + '}'

        prompt = prompt_string.format(question=question_info_str)
        Category = GPT_4o(prompt)

        print('Question:', question_info["question"])
        print('Current category:', Category)

        if Category == 'Object Detection' and len(Object_Detection) < 200:
            Object_Detection.append(question_info)
        elif Category == 'Attribute Recognition' and len(Attribute_Recognition) < 200:
            Attribute_Recognition.append(question_info)
        elif Category == 'Object Relationships' and len(Object_Relationships) < 200:
            Object_Relationships.append(question_info)
        elif Category == 'Action Recognition' and len(Action_Recognition) < 200:
            Action_Recognition.append(question_info)
        elif Category == 'Scene Context' and len(Scene_Context) < 200:
            Scene_Context.append(question_info)
        else:
            continue

        print(f"OD: {len(Object_Detection)}, "
        f"AR: {len(Attribute_Recognition)}, "
        f"OR: {len(Object_Relationships)}, "
        f"AC: {len(Action_Recognition)}, "
        f"SC: {len(Scene_Context)}")
        print('---------------------------------\n')

        with open('/mnt/workspace/hzt/mind2web/data1/aokvqa/datasets/quick_data/Object_Detection.json', 'w', encoding='utf-8') as f:
            json.dump(Object_Detection, f, ensure_ascii=False, indent=4)

        with open('/mnt/workspace/hzt/mind2web/data1/aokvqa/datasets/quick_data/Attribute_Recognition.json', 'w', encoding='utf-8') as f:
            json.dump(Attribute_Recognition, f, ensure_ascii=False, indent=4)

        with open('/mnt/workspace/hzt/mind2web/data1/aokvqa/datasets/quick_data/Object_Relationships.json', 'w', encoding='utf-8') as f:
            json.dump(Object_Relationships, f, ensure_ascii=False, indent=4)

        with open('/mnt/workspace/hzt/mind2web/data1/aokvqa/datasets/quick_data/Action_Recognition.json', 'w', encoding='utf-8') as f:
            json.dump(Action_Recognition, f, ensure_ascii=False, indent=4)

        with open('/mnt/workspace/hzt/mind2web/data1/aokvqa/datasets/quick_data/Scene_Context.json', 'w', encoding='utf-8') as f:
            json.dump(Scene_Context, f, ensure_ascii=False, indent=4)

    if len(Object_Detection) == 200 and len(Attribute_Recognition) == 200 and len(Object_Relationships) == 200 and len(Action_Recognition) == 200 and len(Scene_Context) == 200:
        break

    
    
