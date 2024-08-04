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
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


file_path = "Your_Path/mbpp/full/test-00000-of-00001.parquet"

problems = {}

try:
    # Read Parquet file
    df = pd.read_parquet(file_path)
    # Convert DataFrame to dictionary
    problems = df.to_dict(orient='records')
except (FileNotFoundError, Exception) as e:
    print(f"An error occurred: {e}")

problems_numbers = len(problems)
print(f"problems: {len(problems)}")
print(problems[1])

# problem = problems[2]

# print(problem)

# code = problem['code']
#
# test_result = check_correctness(problem, code)
# print(test_result)

model_id = "path/CodeQwen1.5-7B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
)

model_id2 = "path/Artigenz-Coder-DS-6.7B"

tokenizer2 = AutoTokenizer.from_pretrained(model_id2)
model2 = AutoModelForCausalLM.from_pretrained(
    model_id2,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
)


def extract_python_code(text):
    # Regular expression to match content between ```python``` markers
    pattern = re.compile(r'```python(.*?)```', re.DOTALL)
    matches = pattern.findall(text)

    if matches:
        # Initialize variable to store the code block with the maximum number of lines
        max_lines = 0
        largest_code_block = ""

        for match in matches:
            code_block = match.strip()
            # Count the number of lines in the code block
            lines = code_block.split('\n')
            if len(lines) > max_lines:
                max_lines = len(lines)
                largest_code_block = code_block

        # Remove `# Example usage or # Test cases` and everything after it
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
            first_five_lines = lines[:10]
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
            converted_dialog += content + " "
    return converted_dialog.strip()


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

    if error_type == "test_pass":
        failed_info[:] = [item for item in failed_info if item[0].strip() != prompt.strip()]
    else:
        found = False
        for item in failed_info:
            # print(f"pseudocode:\n{item[0]}")
            if item[0].strip() == prompt.strip():
                index = failed_info.index(item)

                failed_info[index] = (prompt, error_type, python_code, error_info, test)
                found = True
                break
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



start_index = 0
passed_tests = 0
success_num = 0
success_numA = 0
success_numB = 0
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

idx = start_index

while idx < len(problems):
    timeout = 5.0
    print(f"##### Processing batch {(idx + batch_size) / batch_size} #####")
    failed_info = []
    failed_info2 = []
    problem = {}

    consecutive_failures = 0

    for i in range(batch_size):
        if idx + i < len(problems) - 1:
            problem = problems[idx + i]
            failed_info.append((problem['text'], "generation", None, None, problem['test_list']))
            failed_info2.append((problem['text'], "generation", None, None, problem['test_list']))
        else:
            break

    success = False
    compile_pass = False
    test_pass = False

    python_code = ""

    repair_counters = {}

    while consecutive_failures < repair_num + 1:
        print(f"\n############# consecutive_failuresA: {consecutive_failures} ################\n")
        for index, item in enumerate(failed_info):
            print(f"Entry {index + 1}:")
            print(f"Error Type: {item[1]}")

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
            success_numA += 1
            print("test YES!!!!!!!!!!!!!")
            if consecutive_failures == 0:
                print("Generation passed!")
                generate_succeed_num += 1
            else:
                print(f"Passed on repair attempt {consecutive_failures}")
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
            test_pass = False
            update_failed_info(failed_info, problem['text'], "test_failed", python_code, result['result'], problem['test_list'])

        consecutive_failures += 1
    if not test_pass:
        consecutive_failures = 0
        while consecutive_failures < repair_num + 1:
            print(f"\n############# consecutive_failuresB: {consecutive_failures} ################\n")
            for index, item in enumerate(failed_info2):
                print(f"Entry {index + 1}:")
                print(f"Error Type: {item[1]}")

            success = False
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
                success_numB += 1
                print("test YES!!!!!!!!!!!!!")
                if consecutive_failures == 0:
                    print("Generation passed!")
                    generate_succeed_num += 1
                else:
                    print(f"Passed on repair attempt {consecutive_failures}")
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
                test_pass = False
                update_failed_info(failed_info2, problem['text'], "test_failed", python_code, result['result'], problem['test_list'])

            consecutive_failures += 1

    idx += batch_size

    if success:
        success_count += batch_size
    else:
        failure_indices.append(idx)

    total_tests += batch_size

    passed_tests = generate_succeed_num + repair_succeed_within_5_num

    print(f"success_count: {success_count}, passed_tests: {passed_tests}, total_tests: {total_tests}")
    passed_rate = (passed_tests / total_tests) * 100
    success_rate = (success_count / total_tests) * 100
    current_memory = torch.cuda.memory_allocated()

    progress_bar.update(batch_size)
    print(f"Model A success count: {success_numA}")
    print(f"Model B success count: {success_numB}")
    progress_bar.set_postfix({"Generation success rate": f"{success_rate:.2f}%",
                              "Total successes": f"{success_num}",
                              "Overall success rate": f"{passed_rate:.2f}%",
                              "Generation successes": f"{generate_succeed_num}",
                              "Repairs within 1 attempt": f"{repair_succeed_within_1_num}",
                              "Repairs within 2 attempts": f"{repair_succeed_within_2_num}",
                              "Repairs within 3 attempts": f"{repair_succeed_within_3_num}",
                              "Repairs within 4 attempts": f"{repair_succeed_within_4_num}",
                              "Repairs within 5 attempts": f"{repair_succeed_within_5_num}",
                              "Current memory": f"{current_memory}",
                              })
    objgraph.show_most_common_types()

max_memory = torch.cuda.max_memory_allocated()

progress_bar.close()

print(f"Number of successfully extracted C++ code blocks: {success_count}/{problems_numbers}")
print(f"Failed indices: {failure_indices}")
