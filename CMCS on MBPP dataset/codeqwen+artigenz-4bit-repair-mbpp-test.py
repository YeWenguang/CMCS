import json
import sys
import subprocess
from subprocess import TimeoutExpired, CalledProcessError
import os
import shutil
from tempfile import mkdtemp
from contextlib import contextmanager
import multiprocessing
import platform
import numpy as np
import resource
from tqdm import tqdm
import uuid
import objgraph
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from execution import check_correctness
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_file = "/root/Work/human-eval-master/mbpp/failed_test_problems/codellama-4bit-repair-mbpp-test.py/failed_test_problems.txt"

file_path = "/root/Work/human-eval-master/mbpp/full/test-00000-of-00001.parquet"

problems = {}

try:
    # 读取 Parquet 文件
    df = pd.read_parquet(file_path)
    # 将 DataFrame 转换为字典
    problems = df.to_dict(orient='records')
except (FileNotFoundError, Exception) as e:
    print(f"An error occurred: {e}")

problems_numbers = len(problems)
print(f"problems:{len(problems)}")
print(problems[1])

# problem = problems[2]

# print(problem)

# code = problem['code']
#
# test_result = check_correctness(problem, code)
# print(test_result)

model_id = "Qwen/CodeQwen1.5-7B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
)

model_id2 = "/root/autodl-tmp/Artigenz-Coder-DS-6.7B"

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
    global result
    dialog = []
    for entry in failed_info:
        prompt, error_type, python_code, error_message, test = entry
        if error_message is not None:
            lines = error_message.split('\n')
            # 保留前五行
            first_five_lines = lines[:10]
            # 将保留的行重新拼接为一个字符串
            result = '\n'.join(first_five_lines)

        if error_type == "test_failed":
            # if error_message[1] is None:
            #     generated_output = "Process timed out"
            # else:
            #     generated_output = error_message[1][:100]
            formatted_tests = "\n".join(test)
            dialog = [{
                "role": "user",
                "content": f"""
### Instruction:
Assuming you are a professional coding expert.
The following is code task description, test list and incorrect Python program translation.
```code task description
{prompt}
```
```test 
{formatted_tests}
```
```incorrect python program
{python_code}
```
```feedback:
{result}
```
You should analyze and correct errors in Python programs through code task description and provide feedback, and meet the testing requirements of the test list
Finally, enclose the provided Python code snippet within triple backticks ```python ```, test cases snippet within triple backticks ```test cases ```.
"""
            }]
        elif error_type == "generation":
            formatted_tests = "\n".join(test)
            dialog = [{
                "role": "user",
                "content": f"""
### Instruction:
Assuming you are a professional coding expert.
{prompt} 
Analyze and ensure the correctness of Python code and ensure that it satisfy the following tests:
```test
{formatted_tests}
```
Finally, enclose the provided Python code snippet within triple backticks ```python ```, test cases snippet within triple backticks ```test cases ```.
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
            max_new_tokens=2048
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        content = dialogs[0]["content"]

        output = content + "\n" + response
        # print(output)
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
    try:
        with open(output_file, 'a') as file:
            problem_str = f"Task ID: {new_problem['task_id']}\n"
            problem_str += f"Text: {new_problem['text']}\n"
            problem_str += f"Code: {new_problem['code']}\n"
            problem_str += f"Test List: {new_problem['test_list']}\n"
            problem_str += f"Test Setup Code: {new_problem['test_setup_code']}\n"

            file.write(problem_str + '\n\n\n')
    except Exception as e:
        print(f"An error occurred: {e}")


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

    consecutive_failures = 0  # 记录连续失败次数
    # 收集特定索引位置的四个伪代码字符串到列表中

    for i in range(batch_size):
        if idx + i < len(problems) - 1:  # 确保索引有效
            problem = problems[idx + i]
            failed_info.append((problem['text'], "generation", None, None, problem['test_list']))
            failed_info2.append((problem['text'], "generation", None, None, problem['test_list']))
        else:
            break  # 如果 idx + i 超出了 pseudocodes 的范围，提前终止循环

    success = False
    compile_pass = False
    test_pass = False

    python_code = ""

    # 创建一部字典来存储每个 pseudocode 的修复次数计数器
    repair_counters = {}

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
        result = {}
        while not success:
            response = using_model(failed_info)
            print(response)
            success, python_code = extract_python_code(response)
            # print(python_code)

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
            update_failed_info(failed_info, problem['text'], "test_failed", python_code, result['result'], problem['test_list'])

        consecutive_failures += 1
    if not test_pass:
        consecutive_failures = 0
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
            result = {}
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
                update_failed_info(failed_info2, problem['text'], "test_failed", python_code, result['result'], problem['test_list'])

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

print(f"成功提取的 C++ 代码块数量：{success_count}/{problems_numbers}")
print(f"提取失败的索引：{failure_indices}")
