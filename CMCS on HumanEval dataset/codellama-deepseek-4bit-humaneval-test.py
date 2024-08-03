from human_eval.data import write_jsonl, read_problems
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
import subprocess
from subprocess import TimeoutExpired
import os
import uuid
from tqdm import tqdm
from torch.nn.parallel import DataParallel
import objgraph
import gc
from array import array
import resource
import json
from human_eval.data import HUMAN_EVAL
from human_eval.execution import check_correctness
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
problems = read_problems()

problems_numbers = len(problems)

# print(problems)

# problem = {
#     'task_id': 'HumanEval/0',
#     'prompt': 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
#     'entry_point': 'has_close_elements',
#     'canonical_solution': '    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n',
#     'test': "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n"
# }

# output_file = "/root/Work/human-eval-master/human-eval-master/failed_test_problems/codellama-8bit-humaneval-repair-test.py/failed_test_problems.txt"

model_id = "/root/autodl-tmp/deepseek-coder-7b-instruct-v1.5"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
)

model_id2 = "/root/autodl-tmp/CodeLlama-7b-Instruct-hf"

tokenizer2 = AutoTokenizer.from_pretrained(model_id2)
model2 = AutoModelForCausalLM.from_pretrained(
    model_id2,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
)


def extract_python_code(text):
    # 正则表达式用于匹配```python```标记之间的内容
    pattern = re.compile(r'```python(.*?)```', re.DOTALL)
    matches = pattern.findall(text)

    if matches:
        # 初始化变量以存储最大行数的代码块
        max_lines = 0
        largest_code_block = ""

        for match in matches:
            code_block = match.strip()
            # 计算代码块的行数
            lines = code_block.split('\n')
            if len(lines) > max_lines:
                max_lines = len(lines)
                largest_code_block = code_block

        # 去除 `# Example usage或# Test cases` 及其后的部分
        example_usage_pattern = re.compile(r'(.*?)# (Example usage|Test cases)', re.DOTALL)
        example_usage_match = example_usage_pattern.match(largest_code_block)

        if example_usage_match:
            cleaned_code = example_usage_match.group(1).strip()
        else:
            cleaned_code = largest_code_block

        return True, cleaned_code
    else:
        return False, None


def custom_dialogs_creator(failed_info):
    """
    Creates a list of lists, each containing one dialog dictionary, based on the error information provided in failed_info.

    Parameters:
    - failed_info: List of tuples, each containing a pseudocode snippet, the type of error,
      the C++ code, error message, and any additional feedback.

    Returns:
    - A list of lists, each containing one dialog dictionary.
    """
    dialog = []
    for entry in failed_info:
        prompt, error_type, python_code, error_message, test = entry

        if error_type == "test_failed":
            # if error_message[1] is None:
            #     generated_output = "Process timed out"
            # else:
            #     generated_output = error_message[1][:100]
            dialog = [{
                "role": "user",
                "content": f"""
Assuming you are a professional coding expert.

The following is Python code description and incorrect Python program translation and feedback. 
```Python code description
{prompt}
```
```Incorrect Python program translation
{python_code}
```
```Feedback:
Got the wrong result : "{error_message}". 
```

You should check the feedback, analyze and explain and correct errors in Incorrect Python program translation based on Python code descriptions and provide the complete modified Python code within triple backticks ```python ``` in the end!
"""
            }]
        elif error_type == "generation":
            dialog = [{
                "role": "user",
                "content": f"""
Assuming you are a professional coding expert. 

Python code description:
```
{prompt}
```

You should generate Python code based on above Python code description and provide the complete Python code within triple backticks ```python ``` in the end!
"""
            }]

    return dialog


def convert_dialogs(dialog):
    converted_dialog = ""
    for message in dialog:
        if message["role"] == "user":
            content = message["content"].strip()
            converted_dialog += content + " "  # 将内容追加到字符串中，使用空格分隔
    return converted_dialog.strip()  # 去除最后多余的空格


def using_model(failed_info):
    """
    Using a pre-trained model to convert pseudocode to C++ code or handle errors.

    Parameters:
    - mode: The mode to specify the type of dialog creation.
    - model: Pre-trained model
    - tokenizer: Pre-trained tokenizer
    - pseudocodes: List of pseudocode strings
    - codes: Corresponding C++ codes (used in specific modes)
    - error_messages: Error messages from compiling the codes
    - failed_test_cases: Details about failed test cases
    - max_length: Maximum length for the tokenizer
    - device: Device to run the model on ('cuda:0' for GPU, 'cpu' for CPU)

    Returns:
    - A list of generated C++ code responses
    """
    # Ensure the tokenizer has a pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare dialogues for each pseudocode using a custom dialog creator
    dialogs = custom_dialogs_creator(failed_info)

    # Encode each dialogue, ensuring all sequences are of uniform length
    text = tokenizer.apply_chat_template(
        dialogs,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # Concatenate all encoded inputs into a batch

    try:
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=4096
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        content = dialogs[0]["content"]

        output = content + "\n" + response
        print(output)
        return output
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def using_model2(failed_info):
    """
    Using a pre-trained model to convert pseudocode to C++ code or handle errors.

    Parameters:
    - mode: The mode to specify the type of dialog creation.
    - model: Pre-trained model
    - tokenizer: Pre-trained tokenizer
    - pseudocodes: List of pseudocode strings
    - codes: Corresponding C++ codes (used in specific modes)
    - error_messages: Error messages from compiling the codes
    - failed_test_cases: Details about failed test cases
    - max_length: Maximum length for the tokenizer
    - device: Device to run the model on ('cuda:0' for GPU, 'cpu' for CPU)

    Returns:
    - A list of generated C++ code responses
    """
    # Ensure the tokenizer has a pad_token
    if tokenizer2.pad_token is None:
        tokenizer2.pad_token = tokenizer2.eos_token

    # Prepare dialogues for each pseudocode using a custom dialog creator
    dialogs = custom_dialogs_creator(failed_info)

    # Encode each dialogue, ensuring all sequences are of uniform length
    text = tokenizer2.apply_chat_template(
        dialogs,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer2([text], return_tensors="pt").to(device)

    # Concatenate all encoded inputs into a batch

    try:
        generated_ids = model2.generate(
            model_inputs.input_ids,
            max_new_tokens=2048
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer2.batch_decode(generated_ids, skip_special_tokens=True)[0]
        content = dialogs[0]["content"]

        output = content + "\n" + response
        # print(output)
        return output
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def update_failed_info(failed_info, prompt, error_type, python_code, error_info, test):
    # 检查是否应该删除条目，无论是因为测试通过或维修次数过多
    if error_type == "test_pass":
        # 使用列表推导来过滤掉所有匹配的条目
        failed_info[:] = [item for item in failed_info if item[0].strip() != prompt.strip()]
    else:
        found = False
        for item in failed_info:
            # print(f"pseudocode:\n{item[0]}")
            if item[0].strip() == prompt.strip():
                index = failed_info.index(item)
                # 更新已有条目
                failed_info[index] = (prompt, error_type, python_code, error_info, test)
                found = True
                break
        # 如果条目不存在并且操作不是因为成功修复，则添加新条目
        if not found:
            print(f"Have no found!")


def add_problem_to_file(output_file, new_problem, idx):
    """
    将新的 problem 添加到指定的文件中，新的 problem 的键名为 'HumanEval/{idx}'。

    参数:
    output_file (str): 存储 problems 的文件名
    new_problem (dict): 新的 problem 字典
    idx (int): 新的 problem 的索引

    返回:
    None
    """
    try:
        # 从文件中读取现有的 problems 数据
        with open(output_file, 'r') as file:
            try:
                problems = json.load(file)
            except json.JSONDecodeError:
                problems = {}  # 如果文件是空的或不是有效的 JSON 格式，初始化为空字典

        # 将新的 problem 加入到 problems 中
        problems[f'HumanEval/{idx}'] = new_problem

        # 将更新后的 problems 写回到文件中
        with open(output_file, 'w') as file:
            json.dump(problems, file, indent=4)

        print("新的 problem 已成功添加到文件中。")

    except FileNotFoundError:
        print(f"文件 {output_file} 不存在。")
    except Exception as e:
        print(f"发生错误: {e}")


# 设置从第几个伪代码开始生成
start_index = 0
passed_tests = 0
success_num = 0
total_tests = 0

batch_size = 1
repair_num = 5

success_count = 0
failure_indices = []
generate_succeed_num = 0
repair_succeed_within_1_num = 0
repair_succeed_within_2_num = 0
repair_succeed_within_3_num = 0
repair_succeed_within_4_num = 0
repair_succeed_within_5_num = 0

progress_bar = tqdm(total=problems_numbers, desc="Generating Programs", initial=start_index)

idx = start_index  # 初始化索引

while idx < len(problems):
    timeout = 5.0
    print(f"#####正在处理第{(idx + batch_size) / batch_size}批代码！#####")
    failed_info = []
    failed_info2 = []
    problem = {}

    # 收集特定索引位置的四个伪代码字符串到列表中

    for i in range(batch_size):
        if idx + i < len(problems) - 1:  # 确保索引有效
            problem_key = list(problems.keys())[idx]
            problem = problems[problem_key]
            failed_info.append((problem['prompt'], "generation", None, None, None))
            failed_info2.append((problem['prompt'], "generation", None, None, None))
        else:
            break  # 如果 idx + i 超出了 pseudocodes 的范围，提前终止循环

    success = False
    compile_pass = False
    test_pass = False

    python_code = ""

    # 创建一部字典来存储每个 pseudocode 的修复次数计数器
    repair_counters = {}
    consecutive_failures = 0  # 记录连续失败次数
    while failed_info and consecutive_failures < repair_num + 1:
        print(f"\n#############consecutive_failuresA:{consecutive_failures}################\n")
        # 打印 failed_info 列表中的所有条目
        for index, item in enumerate(failed_info):
            print(f"Entry {index + 1}:")
            # print(f"Pseudocode: {item[0]}")
            print(f"Error Type: {item[1]}")
            # print(f"C++ Code: {item[2]}")
            # print(f"Error Info: {item[3]}\n")

        success = False

        while not success:
            response = using_model(failed_info)
            print(response)
            success, python_code = extract_python_code(response)

        completion_id = None
        result = check_correctness(problem, python_code, timeout, completion_id)
        print(result)

        if result['passed']:
            test_pass = True
            success_num += 1
            print("test YES!!!!!!!!!!!!!")
            # 如果测试成功，增加成功计数
            if consecutive_failures == 0:
                print("生成通过！")
                generate_succeed_num += 1
            else:
                print(f"第{consecutive_failures}次修复通过")
            if 0 < consecutive_failures <= 1:
                repair_succeed_within_1_num += 1
            if 0 < consecutive_failures <= 2:
                repair_succeed_within_2_num += 1
            if 0 < consecutive_failures <= 3:
                repair_succeed_within_3_num += 1
            if 0 < consecutive_failures <= 4:
                repair_succeed_within_4_num += 1
            if 0 < consecutive_failures <= 5:
                repair_succeed_within_5_num += 1
            break
        else:
            print("test NO!!!!!!!!!!!!!")
            # if consecutive_failures == 5:
            #     add_problem_to_file(output_file, problem, idx)
            #     print("已将失败problem添加至文件中")
            # 如果测试失败，更新失败信息
            update_failed_info(failed_info, problem['prompt'], "test_failed", python_code, result['result'],
                               problem['test'])

        consecutive_failures += 1

    consecutive_failures = 0
    if not test_pass:
        while failed_info2 and consecutive_failures < repair_num + 1:
            print(f"\n#############consecutive_failuresB:{consecutive_failures}################\n")
            # 打印 failed_info 列表中的所有条目
            for index, item in enumerate(failed_info2):
                print(f"Entry {index + 1}:")
                # print(f"Pseudocode: {item[0]}")
                print(f"Error Type: {item[1]}")
                # print(f"C++ Code: {item[2]}")
                # print(f"Error Info: {item[3]}\n")

            success = False
            result = ''
            while not success:
                response = using_model2(failed_info2)
                print(response)
                success, python_code = extract_python_code(response)

            completion_id = None
            result = check_correctness(problem, python_code, timeout, completion_id)
            print(result)

            if result['passed']:
                test_pass = True
                success_num += 1
                print("test YES!!!!!!!!!!!!!")
                # 如果测试成功，增加成功计数
                if consecutive_failures == 0:
                    print("生成通过！")
                    generate_succeed_num += 1
                else:
                    print(f"第{consecutive_failures}次修复通过")
                if 0 < consecutive_failures <= 1:
                    repair_succeed_within_1_num += 1
                if 0 < consecutive_failures <= 2:
                    repair_succeed_within_2_num += 1
                if 0 < consecutive_failures <= 3:
                    repair_succeed_within_3_num += 1
                if 0 < consecutive_failures <= 4:
                    repair_succeed_within_4_num += 1
                if 0 < consecutive_failures <= 5:
                    repair_succeed_within_5_num += 1
                break
            else:
                print("test NO!!!!!!!!!!!!!")
                # if consecutive_failures == 5:
                #     add_problem_to_file(output_file, problem, idx)
                #     print("已将失败problem添加至文件中")
                # 如果测试失败，更新失败信息
                update_failed_info(failed_info2, problem['prompt'], "test_failed", python_code, result['result'],
                                   problem['test'])

            consecutive_failures += 1

    idx += batch_size

    if success:
        success_count += batch_size
    else:
        failure_indices.append(idx)

    total_tests += batch_size

    passed_tests = generate_succeed_num + repair_succeed_within_5_num

    print(f"success_count:{success_count}, passed_tests:{passed_tests}, total_tests:{total_tests}")
    passed_rate = (passed_tests / total_tests) * 100
    success_rate = (success_count / total_tests) * 100
    # 在需要记录的地方调用这些函数
    current_memory = torch.cuda.memory_allocated()

    progress_bar.update(batch_size)
    progress_bar.set_postfix({"生成成功率": f"{success_rate:.2f}%",
                              "总成功数": f"{success_num}",
                              "总成功率": f"{passed_rate:.2f}%",
                              "生成成功数": f"{generate_succeed_num}",
                              "1次内修复数": f"{repair_succeed_within_1_num}",
                              "2次内修复数": f"{repair_succeed_within_2_num}",
                              "3次内修复数": f"{repair_succeed_within_3_num}",
                              "4次内修复数": f"{repair_succeed_within_4_num}",
                              "5次内修复数": f"{repair_succeed_within_5_num}",
                              "current_memory": f"{current_memory}",
                              })
    objgraph.show_most_common_types()

max_memory = torch.cuda.max_memory_allocated()

# 关闭进度条
progress_bar.close()

print(f"成功提取的 C++ 代码块数量：{success_count}/{num_pseudocodes}")
print(f"提取失败的索引：{failure_indices}")

# return output
# completion = """from typing import List
#
#
# def has_close_elements(numbers: List[float], threshold: float) -> bool:
#
#     for i in range(len(numbers)):
#         for j in range(i + 1, len(numbers)):
#             if abs(numbers[i] - numbers[j]) < threshold:
#                 return True
#     return False"""

# # 输出结果

# num_samples_per_task = 200
# samples = [
#     dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
#     for task_id in problems
#     for _ in range(num_samples_per_task)
# ]
# write_jsonl("samples.jsonl", samples)