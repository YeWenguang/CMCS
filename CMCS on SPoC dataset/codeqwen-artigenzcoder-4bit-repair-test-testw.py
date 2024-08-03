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

# if torch.cuda.is_available():
#     # 获取 GPU 的总显存（以字节为单位）
#     total_memory = torch.cuda.get_device_properties(0).total_memory
#
#     # 计算显存限制比例，设定最大显存为 40GB
#     memory_limit_fraction = 36864 * 1024 * 1024 / total_memory  # 转换为字节后计算比例
#
#     # 设置每个进程的显存限制比例
#     torch.cuda.set_per_process_memory_fraction(memory_limit_fraction, 0)  # 设备ID 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "Qwen/CodeQwen1.5-7B-Chat"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model_id2 = "/root/autodl-tmp/Artigenz-Coder-DS-6.7B"

model2 = AutoModelForCausalLM.from_pretrained(
    model_id2,
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
)
tokenizer2 = AutoTokenizer.from_pretrained(model_id2)

file_path = "/root/Work/SPoC/test-testw.txt"
# output_file = "/root/Work/failed-test/codeqwen-4bit-repair-test.py/repaired_5_failed.txt"
# output_folder = "/home/yewenguang/work/Mistral-7B-Instruct-v0.2/Code/C++/test/testp/split1"  # 请将此路径替换为你想保存生成的C++代码的文件夹路径
testcases_base_path = '/root/Work/SPoC/testcases'

with open(file_path, "r", encoding="utf-8") as file:
    pseudocode_content = file.read()

# 将伪代码拆分为每个probid的部分
pseudocodes = pseudocode_content.split("\n\n")

# 设置从第几个伪代码开始生成
start_index = 71
passed_tests = 68
total_tests = 71

batch_size = 1
repair_num = 5

# 统计伪代码的数量
num_pseudocodes = len(pseudocodes) - 1
print(f"文件中一共有 {num_pseudocodes} 个伪代码。")

# 使用 tqdm 创建进度条，设置 initial 参数
progress_bar = tqdm(total=num_pseudocodes, desc="Generating Programs", initial=start_index)


def extract_cpp_code(terminal_output, index):
    """
        Extracts the last C++ code block and pseudo program code block from the terminal output,
        that includes specific headers or markers, along with probid information.

        Parameters:
        - terminal_output: The string output from which to extract the code.
        - index: An additional index for specific processing or logging (not used here).

        Returns:
        - Tuple of (success flag, extracted C++ code, extracted pseudo code, probid content)
        """
    # 匹配 probid 行
    pattern_probid = r'probid:\s*(\w+)'
    match_probid = re.search(pattern_probid, terminal_output)

    probid_content = match_probid.group(1).strip() if match_probid else None
    if not match_probid:
        print("未找到匹配的 probid: 行.")
        return False, None, None, None

    # 匹配所有代码块
    pattern_code = r'```(.*?)```'
    matches_code = re.findall(pattern_code, terminal_output, re.DOTALL)

    cpp_code = None
    pseudo_code = None

    # 从最后一个代码块开始遍历
    for code_block in reversed(matches_code):
        if '#include <iostream>' in code_block or '#include<iostream>' in code_block:
            lines = code_block.split('\n')
            if all('```' not in line for line in lines[:-1]):  # 排除最后一行含有```的情况
                code_lines = code_block.split('\n')
                if code_lines[0].strip().startswith(('cpp', 'c++')):
                    code_block = '\n'.join(code_lines[1:])
                cpp_code = code_block.strip()
                break

    # 匹配所有 "pseudo program" 代码块
    pattern_pseudo_code = r'pseudo program:\s*```(.*?)```'
    matches_pseudo_code = re.findall(pattern_pseudo_code, terminal_output, re.DOTALL)
    if probid_content and matches_pseudo_code:
        pseudo_code = "probid: " + probid_content + "\n" + "pseudo program:\n```\n" + matches_pseudo_code[
            -1].strip() + "\n```"

    success = bool(cpp_code and pseudo_code)
    return success, cpp_code, pseudo_code, probid_content


def compile_and_run_cpp(cpp_code):
    """
        Compiles C++ code and generates an executable, handling file operations securely.

        Parameters:
        - cpp_code: The C++ code as a string to be compiled.

        Returns:
        - Tuple of (success flag, error message if compilation fails, executable filename if successful)
    """
    compile_process = None  # 在 try 块开始处初始化为 None
    unique_id = uuid.uuid4()
    cpp_filename = f'temp_{unique_id}.cpp'
    executable_filename = f'temp_{unique_id}'

    try:
        with open(cpp_filename, 'w') as file:
            file.write(cpp_code)

        compile_process = subprocess.run(['g++', cpp_filename, '-o', executable_filename], capture_output=True,
                                         text=True)

        if compile_process.returncode != 0:
            return False, compile_process.stderr, None
        return True, None, executable_filename
    finally:
        # 删除源代码文件
        os.remove(cpp_filename)
        # 确保 compile_process 是一个有效的 CompletedProcess 对象再检查 returncode
        if compile_process and compile_process.returncode != 0 and os.path.exists(executable_filename):
            os.remove(executable_filename)


def read_test_cases(probid):
    """
        Reads test cases for a specific problem ID from predefined file paths.

        Parameters:
        - probid: The problem ID for which to read test cases.

        Returns:
        - List of tuples, each containing input and expected output for a test case.
    """
    test_cases = []
    # testcases_base_path = '/home/yewenguang/work/Code-Llama/spoc/testcases'
    testcases_path = f"{testcases_base_path}/{probid}/{probid}_testcases.txt"
    # print("正在进行测试用例测试：")
    # print(f"testcases_path: {testcases_path}")
    with open(testcases_path, 'r') as file:
        lines = file.readlines()
        input_part = []
        output_part = []
        reading_input = True

        for line in lines:
            if line.strip() == "###ENDINPUT###":
                reading_input = False
            elif line.strip() == "###ENDOUTPUT###":
                test_cases.append((input_part, output_part))
                input_part = []
                output_part = []
                reading_input = True
            elif reading_input:
                input_part.append(line.strip())
            else:
                output_part.append(line.strip())
    return test_cases


def run_test_cases(executable_filename, probid):
    """
    Runs the compiled C++ executable against predefined test cases and checks for correctness.

    Parameters:
    - executable_filename: Filename of the compiled C++ executable.
    - probid: Problem ID to fetch the appropriate test cases.

    Returns:
    - Tuple of (success flag, details of the first failed test case if any).
    """
    global passed_tests, total_tests

    # 读取测试用例
    test_cases = read_test_cases(probid)
    success = True
    failed_test_cases = []

    try:
        for idx, (input_data, expected_output) in enumerate(test_cases):
            output = run_cpp_with_input(executable_filename, input_data)

            if compare_output(output, expected_output):
                continue  # 测试通过，继续下一个测试
            else:
                # 记录失败的测试用例
                input_data_str = ''.join(input_data)
                expected_output_str = ''.join(expected_output)
                failed_test_cases.append((input_data_str, output, expected_output_str))
                success = False
                break  # 只要有一个测试失败，立即终止测试

    finally:
        # 清理生成的可执行文件
        os.remove(executable_filename)

    if success:
        return True, None
    else:
        # print(f"failed_test_cases:\n{failed_test_cases[0]}")
        return False, failed_test_cases[0] if failed_test_cases else None


def run_cpp_with_input(executable_filename, input_data):
    """
        Runs a compiled C++ executable with given input data and captures its output.

        Parameters:
        - executable_filename: Filename of the C++ executable to run.
        - input_data: List of strings representing the input for the C++ program.

        Returns:
        - The output from the executable as a string.
    """
    process = None
    try:
        process = subprocess.Popen([f'./{executable_filename}'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True)
        stdout, stderr = process.communicate(input='\n'.join(input_data), timeout=5)  # 将输入数据转换为字符串形式并传递给子进程
        # print(f"input_data: {input_data}")
        # 检查输出中是否存在冒号
        last_colon_index = stdout.rfind(":")  # 找到最后一个冒号的位置
        if last_colon_index != -1:
            return stdout[last_colon_index + 1:].strip()  # 返回最后一个冒号后的部分
        else:
            return stdout.strip()  # 返回完整输出
    except TimeoutExpired:
        print("Process timed out. Terminating...")
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=2)
        except TimeoutExpired:
            print("Process did not terminate in time. Killing...")
            process.kill()
        except Exception as e:
            print(f"Unexpected error: {e}")
    except Exception as e:
        print(f"Unexpected error when running the subprocess: {e}")
    finally:
        # Try to terminate the process gracefully
        if process.poll() is None:  # 检查进程是否还在运行
            process.terminate()  # 尝试正常终止
            try:
                process.communicate(timeout=2)  # 给它一点时间来清理资源
            except TimeoutExpired:
                process.kill()  # 如果它没有及时终止，强制结束
            except Exception as e:
                print(f"Unexpected error during termination: {e}")


def compare_output(output, expected_output):
    """
        Compares the output of a C++ program with the expected output.

        Parameters:
        - output: The output from the C++ program as a string.
        - expected_output: The expected output as a list of strings.

        Returns:
        - Boolean indicating if the output matches the expected output.
    """
    return output == '\n'.join(expected_output)


def extract_error_info(error_messages, source_code):
    """
        Extracts and formats error information from compilation error messages.

        Parameters:
        - error_messages: String containing all error messages from the compiler.
        - source_code: The source code that was compiled, used for context in error messages.

        Returns:
        - Formatted string containing detailed error information.
    """
    error_info = ""
    line_count = 0
    max_lines = 10  # 设定最大行数为50

    # 使用正则表达式匹配错误信息中的行号、错误内容以及错误指示符位置
    pattern = re.compile(r'([^:]+\.cpp):(\d+):(\d+):\s(error|note):\s(.+)')
    matches = pattern.findall(error_messages)

    for match in matches:
        line_num = int(match[1])
        error_desc = match[3]

        if error_desc == 'error':
            error_content = source_code.split('\n')[line_num - 1].strip()
            error_info += f'error: Line: {line_num}, Line_content: "{error_content}", error_msg: "{match[4]}"\n'
        else:
            error_info += f'note: Line: {line_num}, note_msg: "{match[4]}"\n'

        line_count += 1
        if line_count >= max_lines:
            break  # 达到50行时停止添加更多错误信息

    return error_info


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
        if len(entry) < 4:
            continue  # 跳过不完整的条目

        pseudocode, error_type, cpp_code, error_message = entry

        if error_type == "compile_failed":
            lines = error_message.split('\n')
            # 取前10行内容
            first_10_lines = lines[:10]
            error_message_string = '\n'.join(first_10_lines)
            dialog = [{
                "role": "user",
                "content": f"""
### Instruction:
Assuming you are a professional coding expert.
There were some errors during the compilation of the C++ program. 
First, you should review the error messages corresponding to the C++ program and fix the bugs. 
And then you should provide the complete modified C++ code in the end.
Enclose the provided C++ code snippet within triple backticks ``` ``` to properly format the code block.

### Information Provided: 
C++ program:
{cpp_code}

{pseudocode}

error_message:
{error_message_string}
Fix the bug.

### Request:
You should provide the complete modified C++ code in the end!
                """
            }]
        elif error_type == "test_failed":
            if error_message[1] is None:
                generated_output = "Process timed out"
            else:
                generated_output = error_message[1][:100]
            dialog = [{
                "role": "user",
                "content": f"""
### Instruction:
Assuming you are a professional coding expert.
The following is pseudocode and incorrect C++ program translation. 
Correct errors in C++programs through pseudocode and provided feedback.
Enclose the provided C++ code snippet within triple backticks ``` ``` to properly format the code block.

### Information Provided:
{pseudocode}

C++ program:
{cpp_code}

Feedback:
Wrong Answer with input: "{error_message[0]}". 
Expected output is "{error_message[2]}", but the generated output is "{generated_output}". 

### Request:
You should provide the complete modified C++ code in the end!
            """
            }]
        elif error_type == "generation":
            dialog = [{
                "role": "user",
                "content": f"{pseudocode}\nAssuming you are a professional coding expert. Convert the pseudocode to C++ code, and ensure the presence of header file '# include<iostream>' in C++code.\n"
            }]
    return dialog


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
    dialogs = custom_dialogs_creator(failed_info)
    input_ids = tokenizer.apply_chat_template(dialogs, add_generation_prompt=True, return_tensors="pt").to(
        model.device)

    outputs = model.generate(input_ids,
                             do_sample=True,
                             temperature=0.5,
                             max_new_tokens=4096
                             )
    response = outputs[0][input_ids.shape[-1]:]
    # print(tokenizer.decode(response, skip_special_tokens=True))
    content = dialogs[0]["content"]

    output = content + "\n" + tokenizer.decode(response, skip_special_tokens=True)
    # print(output)
    return output

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
    dialogs = custom_dialogs_creator(failed_info)
    input_ids = tokenizer2.apply_chat_template(dialogs, add_generation_prompt=True, return_tensors="pt").to(
        model2.device)

    outputs = model2.generate(input_ids,
                             do_sample=True,
                             temperature=0.5,
                             max_new_tokens=4096
                             )
    response = outputs[0][input_ids.shape[-1]:]
    # print(tokenizer.decode(response, skip_special_tokens=True))
    content = dialogs[0]["content"]

    output2 = content + "\n" + tokenizer2.decode(response, skip_special_tokens=True)
    # print(output)
    return output2


def update_failed_info(failed_info, pseudocode, error_type, cpp_code, error_info):
    """
    更新或删除失败信息列表中的条目。

    参数:
    failed_info: list - 存储所有失败信息的列表，每个条目包括(pseudocode, error_type, cpp_code, error_info)。
    pseudocode: str - 用于唯一标识失败条目的伪代码字符串。
    error_type: str - 错误类型，可以是'compile_failed', 'test_failed'或'repair_succeed'。
    cpp_code: str - 与失败条目关联的C++代码。
    error_info: str - 错误详情或为None，当error_type为'repair_succeed'时使用None。
    repair_count: int - 维修尝试的次数。

    功能描述:
    - 如果error_type为'test_pass'或'repair_count'超过3，从列表中删除对应的pseudocode条目。
    - 对于其他类型的错误，如果列表中存在对应的pseudocode，则更新该条目。
    - 如果没有找到匹配的条目，并且操作不是因为成功修复，就向列表中添加新的错误记录。
    """
    # 检查是否应该删除条目，无论是因为测试通过或维修次数过多
    if error_type == "test_pass":
        # 使用列表推导来过滤掉所有匹配的条目
        failed_info[:] = [item for item in failed_info if item[0].strip() != pseudocode.strip()]
    else:
        found = False
        for item in failed_info:
            # print(f"pseudocode:\n{item[0]}")
            if item[0].strip() == pseudocode.strip():
                index = failed_info.index(item)
                # 更新已有条目
                failed_info[index] = (pseudocode, error_type, cpp_code, error_info)
                found = True
                break
        # 如果条目不存在并且操作不是因为成功修复，则添加新条目
        if not found:
            print(f"Have no found!")


def convert_dialogs(dialogs):
    converted_dialogs = []
    for dialog in dialogs:
        converted_dialog = []
        for message in dialog:
            if message["role"] == "user":
                content = message["content"].strip()
                converted_dialog.append(content)
        converted_dialogs.append("\n".join(converted_dialog))
    return converted_dialogs


def write_info_to_file(info, output_file_path):
    """
    将信息写入指定文件，并在末尾添加两个空格。

    参数:
    info (str): 信息字符串。
    output_file_path (str): 目标文件的路径。
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # pseudo_info = info[0]
    #
    # probid_start = pseudo_info[0].find('probid: ') + len('probid: ')
    # probid_end = pseudo_info[0].find('\n', probid_start)
    # probid = pseudo_info[0][probid_start:probid_end]
    #
    # pseudo_start = pseudo_info[0].find('pseudo program:\n```') + len('pseudo program:\n```\n')
    # pseudo_end = pseudo_info[0].find('```', pseudo_start)
    # pseudo_program = pseudo_info[0][pseudo_start:pseudo_end]
    #
    # # 保存到一起
    # combined_info = f"probid: {probid}\npseudo program:\n{pseudo_program}"

    # 打开文件（如果文件不存在则会创建）
    with open(output_file_path, 'a', encoding='utf-8') as file:
        # 将数据写入文件，并在末尾添加两个空格
        file.write(info + "\n\n\n")

    print("数据已成功写入文件:", output_file_path)


success_count = 71
failure_indices = []
generate_succeed_num = 68
succeed_num = 68

modelA_succeed_num = 0
modelB_succeed_num = 0
modelA_and_modelB_succeed_num = 0
modelA_or_modelB_succeed_num = 0

idx = start_index  # 初始化索引

while idx < len(pseudocodes) - 1:
    print(f"#####正在处理第{(idx + batch_size) / batch_size}批代码！#####")
    failed_info = []
    failed_info2 = []

    # 收集特定索引位置的四个伪代码字符串到列表中
    for i in range(batch_size):
        if idx + i < len(pseudocodes) - 1:  # 确保索引有效
            if pseudocodes[idx + i] != "":
                failed_info.append((pseudocodes[idx + i], "generation", None, None))
                failed_info2.append((pseudocodes[idx + i], "generation", None, None))
        else:
            break  # 如果 idx + i 超出了 pseudocodes 的范围，提前终止循环

    success = False
    compile_pass = False
    test_pass = False

    cpp_codes = []
    probid_contents = []
    # modelA_pass_flag = array('i', [0] * batch_size)
    # modelB_pass_flag = array('i', [0] * batch_size)

    # 创建一部字典来存储每个 pseudocode 的修复次数计数器

    consecutive_failures = 0  # 记录连续失败次数
    while failed_info and consecutive_failures < repair_num + 1:
        print(f"\n#############consecutive_failures:{consecutive_failures}################\n")
        # 打印 failed_info 列表中的所有条目
        for index, item in enumerate(failed_info):
            print(f"Entry {index + 1}:")
            # print(f"Pseudocode: {item[0]}")
            print(f"Error Type: {item[1]}")
            # print(f"C++ Code: {item[2]}")
            # print(f"Error Info: {item[3]}\n")

        success = False
        cpp_code = ''
        pseudocode = ''
        probid_content = ''
        while not success:
            decoded_responses = using_model(failed_info)

            # for response in decoded_responses:
            # print(decoded_responses + "\n##########<next>##########\n")
            success, cpp_code, pseudocode, probid_content = extract_cpp_code(decoded_responses, start_index)

        compile_pass, compile_stderr, executable_filename = compile_and_run_cpp(cpp_code)

        if compile_pass:
            # 如果编译成功，则进行测试
            print("compiles YES!!!!!!!!!!!!!")
            test_pass, failed_test_cases = run_test_cases(executable_filename, probid_content)
            if test_pass:
                succeed_num += 1
                print("test YES!!!!!!!!!!!!!")
                # 如果测试成功，增加成功计数
                if consecutive_failures == 0:
                    print("生成通过！")
                else:
                    print(f"第{consecutive_failures}次修复通过")
                break
            else:
                print("test NO!!!!!!!!!!!!!")
                # if consecutive_failures == 5:
                #     write_info_to_file(pseudocode, output_file)
                # 如果测试失败，更新失败信息
                update_failed_info(failed_info, pseudocode, "test_failed", cpp_code, failed_test_cases)
        else:
            print("compiles NO!!!!!!!!!!!!!")
            # if consecutive_failures == 5:
            #     write_info_to_file(pseudocode, output_file)
            error_info = extract_error_info(compile_stderr, cpp_code)
            # 如果编译失败，更新失败信息
            update_failed_info(failed_info, pseudocode, "compile_failed", cpp_code, error_info)

        consecutive_failures += 1
    if not test_pass:
        print("第一个模型失败，正在使用第二个模型")
        consecutive_failures = 0
        while failed_info2 and consecutive_failures < repair_num + 1:
            print(f"\n#############consecutive_failures:{consecutive_failures}################\n")
            # 打印 failed_info 列表中的所有条目
            for index, item in enumerate(failed_info2):
                print(f"Entry {index + 1}:")
                # print(f"Pseudocode: {item[0]}")
                print(f"Error Type: {item[1]}")
                # print(f"C++ Code: {item[2]}")
                # print(f"Error Info: {item[3]}\n")

            success = False
            cpp_code = ''
            pseudocode = ''
            probid_content = ''
            while not success:
                decoded_responses = using_model2(failed_info2)
                # for response in decoded_responses:
                # print(decoded_responses + "\n##########<next>##########\n")
                success, cpp_code, pseudocode, probid_content = extract_cpp_code(decoded_responses, start_index)

            compile_pass, compile_stderr, executable_filename = compile_and_run_cpp(cpp_code)
            if compile_pass:
                # 如果编译成功，则进行测试
                print("compiles YES!!!!!!!!!!!!!")
                test_pass, failed_test_cases = run_test_cases(executable_filename, probid_content)
                if test_pass:
                    print("test YES!!!!!!!!!!!!!")
                    # 如果测试成功，增加成功计数
                    succeed_num += 1
                    if consecutive_failures == 0:
                        print("生成通过！")
                    else:
                        print(f"第{consecutive_failures}次修复通过")
                    break
                else:
                    print("test NO!!!!!!!!!!!!!")
                    # if consecutive_failures == 5:
                    #     write_info_to_file(pseudocode, output_file)
                    # 如果测试失败，更新失败信息
                    update_failed_info(failed_info2, pseudocode, "test_failed", cpp_code, failed_test_cases)
            else:
                print("compiles NO!!!!!!!!!!!!!")
                # if consecutive_failures == 5:
                #     write_info_to_file(pseudocode, output_file)
                error_info = extract_error_info(compile_stderr, cpp_code)
                # 如果编译失败，更新失败信息
                update_failed_info(failed_info2, pseudocode, "compile_failed", cpp_code, error_info)

            consecutive_failures += 1

    idx += batch_size

    if success:
        success_count += batch_size
    else:
        failure_indices.append(idx)

    total_tests += batch_size

    passed_tests = succeed_num

    print(f"success_count:{success_count}, passed_tests:{passed_tests}, total_tests:{total_tests}")
    passed_rate = (passed_tests / total_tests) * 100
    success_rate = (success_count / total_tests) * 100
    # 在需要记录的地方调用这些函数
    current_memory = torch.cuda.memory_allocated()

    progress_bar.update(batch_size)
    progress_bar.set_postfix({"生成成功率": f"{success_rate:.2f}%",
                              "总成功数": f"{passed_tests}",
                              "总成功率": f"{passed_rate:.2f}%",
                              "current_memory": f"{current_memory}",
                              })
    objgraph.show_most_common_types()

max_memory = torch.cuda.max_memory_allocated()

# 关闭进度条
progress_bar.close()

print(f"成功提取的 C++ 代码块数量：{success_count}/{num_pseudocodes}")
print(f"提取失败的索引：{failure_indices}")