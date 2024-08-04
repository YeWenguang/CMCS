
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
#     # Get the total GPU memory (in bytes)
#     total_memory = torch.cuda.get_device_properties(0).total_memory
#
#     # Calculate memory limit ratio, set maximum memory to 40GB
#     memory_limit_fraction = 36864 * 1024 * 1024 / total_memory  # Convert to bytes and calculate ratio
#
#     # Set the memory limit ratio per process
#     torch.cuda.set_per_process_memory_fraction(memory_limit_fraction, 0)  # Device ID 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "Your_file_path/deepseek-coder-7b-instruct-v1.5"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model_id2 = "Your_file_path/CodeLlama-7b-Instruct-hf"

model2 = AutoModelForCausalLM.from_pretrained(
    model_id2,
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
)
tokenizer2 = AutoTokenizer.from_pretrained(model_id2)

file_path = "/Your_file_path/test-testw.txt"

testcases_base_path = '/Your_file_path/SPoC/testcases'

with open(file_path, "r", encoding="utf-8") as file:
    pseudocode_content = file.read()

# Split the pseudocode into parts for each probid
pseudocodes = pseudocode_content.split("\n\n")

# Set the starting index for generating
start_index = 144
passed_tests = 125
total_tests = 144

batch_size = 1
repair_num = 5

# Count the number of pseudocode snippets
num_pseudocodes = len(pseudocodes) - 1
print(f"There are {num_pseudocodes} pseudocode snippets in the file.")

# Create a progress bar using tqdm, setting the initial parameter
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
    # Match probid line
    pattern_probid = r'probid:\s*(\w+)'
    match_probid = re.search(pattern_probid, terminal_output)

    probid_content = match_probid.group(1).strip() if match_probid else None
    if not match_probid:
        print("No matching probid: line found.")
        return False, None, None, None

    # Match all code blocks
    pattern_code = r'```(.*?)```'
    matches_code = re.findall(pattern_code, terminal_output, re.DOTALL)

    cpp_code = None
    pseudo_code = None

    # Iterate from the last code block
    for code_block in reversed(matches_code):
        if '#include <iostream>' in code_block or '#include<iostream>' in code_block:
            lines = code_block.split('\n')
            if all('```' not in line for line in lines[:-1]):  # Exclude the last line containing ```
                code_lines = code_block.split('\n')
                if code_lines[0].strip().startswith(('cpp', 'c++')):
                    code_block = '\n'.join(code_lines[1:])
                cpp_code = code_block.strip()
                break

    # Match all "pseudo program" code blocks
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
    compile_process = None  # Initialize as None at the start of try block
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
        # Remove source code file
        os.remove(cpp_filename)
        # Ensure compile_process is a valid CompletedProcess object before checking returncode
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
    testcases_path = f"{testcases_base_path}/{probid}/{probid}_testcases.txt"
    # print("Running test cases:")
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

    # Read test cases
    test_cases = read_test_cases(probid)
    success = True
    failed_test_cases = []

    try:
        for idx, (input_data, expected_output) in enumerate(test_cases):
            output = run_cpp_with_input(executable_filename, input_data)

            if compare_output(output, expected_output):
                continue  # Test passed, continue to next test
            else:
                # Record failed test case
                input_data_str = ''.join(input_data)
                expected_output_str = ''.join(expected_output)
                failed_test_cases.append((input_data_str, output, expected_output_str))
                success = False
                break  # Terminate testing as soon as one test fails

    finally:
        # Clean up the generated executable file
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
        stdout, stderr = process.communicate(input='\n'.join(input_data), timeout=5)  # Convert input data to string form and pass to subprocess
        last_colon_index = stdout.rfind(":")  # Find the position of the last colon
        if last_colon_index != -1:
            return stdout[last_colon_index + 1:].strip()  # Return the part after the last colon
        else:
            return stdout.strip()  # Return the full output
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
        if process.poll() is None:  # Check if the process is still running
            process.terminate()  # Try to terminate normally
            try:
                process.communicate(timeout=2)  # Give it some time to clean up resources
            except TimeoutExpired:
                process.kill()  # Force terminate if it does not terminate in time
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
    max_lines = 10  # Set the maximum number of lines to 10

    # Use regular expression to match the line number, error content, and error indicator position in the error messages
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
            break  # Stop adding more error information after 10 lines

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
            continue  # Skip incomplete entries

        pseudocode, error_type, cpp_code, error_message = entry

        if error_type == "compile_failed":
            lines = error_message.split('\n')
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
    Update or delete entries in the failed info list.

    Parameters:
    failed_info: list - A list that stores all failed information, with each entry including (pseudocode, error_type, cpp_code, error_info).
    pseudocode: str - The pseudocode string used to uniquely identify the failed entry.
    error_type: str - The type of error, which can be 'compile_failed', 'test_failed', or 'repair_succeed'.
    cpp_code: str - The C++ code associated with the failed entry.
    error_info: str - Error details or None, used as None when error_type is 'repair_succeed'.
    repair_count: int - The number of repair attempts.

    Function description:
    - If the error_type is 'test_pass' or the repair_count exceeds 3, remove the corresponding pseudocode entry from the list.
    - For other types of errors, if the corresponding pseudocode exists in the list, update that entry.
    - If no matching entry is found and the operation is not due to successful repair, add a new error record to the list.
    """
    # Check if the entry should be deleted, either because the test passed or the repair count is too high
    if error_type == "test_pass":
        # Use list comprehension to filter out all matching entries
        failed_info[:] = [item for item in failed_info if item[0].strip() != pseudocode.strip()]
    else:
        found = False
        for item in failed_info:
            # print(f"pseudocode:\n{item[0]}")
            if item[0].strip() == pseudocode.strip():
                index = failed_info.index(item)
                # Update the existing entry
                failed_info[index] = (pseudocode, error_type, cpp_code, error_info)
                found = True
                break
        # If the entry does not exist and the operation is not due to successful repair, add a new entry
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
    Write the information to the specified file and add two blank lines at the end.

    Parameters:
    info (str): Information string.
    output_file_path (str): Path to the target file.
    """
    # Ensure the directory exists
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
    # # Save together
    # combined_info = f"probid: {probid}\npseudo program:\n{pseudo_program}"

    # Open the file (it will be created if it doesn't exist)
    with open(output_file_path, 'a', encoding='utf-8') as file:
        # Write data to the file and add two blank lines at the end
        file.write(info + "\n\n\n")

    print("Data has been successfully written to the file:", output_file_path)


success_count = 144
failure_indices = []
generate_succeed_num = 125
succeed_num = 125

modelA_succeed_num = 0
modelB_succeed_num = 0
modelA_and_modelB_succeed_num = 0
modelA_or_modelB_succeed_num = 0

idx = start_index

while idx < len(pseudocodes) - 1:
    print(f"#####Processing batch {int((idx + batch_size) / batch_size)}!#####")
    failed_info = []
    failed_info2 = []

    # Collect four pseudocode strings at specific index positions into a list
    for i in range(batch_size):
        if idx + i < len(pseudocodes) - 1:  # Ensure the index is valid
            if pseudocodes[idx + i] != "":
                failed_info.append((pseudocodes[idx + i], "generation", None, None))
                failed_info2.append((pseudocodes[idx + i], "generation", None, None))
        else:
            break  # If idx + i exceeds the range of pseudocodes, terminate the loop early

    success = False
    compile_pass = False
    test_pass = False

    cpp_codes = []
    probid_contents = []
    # modelA_pass_flag = array('i', [0] * batch_size)
    # modelB_pass_flag = array('i', [0] * batch_size)

    # Create a dictionary to store the repair count counter for each pseudocode

    consecutive_failures = 0  # Record consecutive failures
    while failed_info and consecutive_failures < repair_num + 1:
        print(f"\n#############consecutive_failures:{consecutive_failures}################\n")
        # Print all entries in the failed_info list
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
            # If compilation succeeds, proceed to testing
            # print("compiles YES!!!!!!!!!!!!!")
            test_pass, failed_test_cases = run_test_cases(executable_filename, probid_content)
            if test_pass:
                succeed_num += 1
                print("test YES!!!!!!!!!!!!!")
                # If testing succeeds, increment the success count
                if consecutive_failures == 0:
                    print("Generation passed!")
                else:
                    print(f"Repair passed on attempt {consecutive_failures}")
                break
            else:
                print("test NO!!!!!!!!!!!!!")
                # if consecutive_failures == 5:
                #     write_info_to_file(pseudocode, output_file)
                # If testing fails, update the failure information
                update_failed_info(failed_info, pseudocode, "test_failed", cpp_code, failed_test_cases)
        else:
            print("compiles NO!!!!!!!!!!!!!")
            # if consecutive_failures == 5:
            #     write_info_to_file(pseudocode, output_file)
            error_info = extract_error_info(compile_stderr, cpp_code)
            # If compilation fails, update the failure information
            update_failed_info(failed_info, pseudocode, "compile_failed", cpp_code, error_info)

        consecutive_failures += 1
    if not test_pass:
        print("First model failed, switching to the second model")
        consecutive_failures = 0
        while failed_info2 and consecutive_failures < repair_num + 1:
            print(f"\n#############consecutive_failures:{consecutive_failures}################\n")
            # Print all entries in the failed_info list
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
                #     print(response + "\n##########<next>##########\n")
                success, cpp_code, pseudocode, probid_content = extract_cpp_code(decoded_responses, start_index)

            compile_pass, compile_stderr, executable_filename = compile_and_run_cpp(cpp_code)
            if compile_pass:
                # If compilation succeeds, proceed to testing
                print("compiles YES!!!!!!!!!!!!!")
                test_pass, failed_test_cases = run_test_cases(executable_filename, probid_content)
                if test_pass:
                    print("test YES!!!!!!!!!!!!!")
                    # If testing succeeds, increment the success count
                    succeed_num += 1
                    if consecutive_failures == 0:
                        print("Generation passed!")
                    else:
                        print(f"Repair passed on attempt {consecutive_failures}")
                    break
                else:
                    print("test NO!!!!!!!!!!!!!")
                    # if consecutive_failures == 5:
                    #     write_info_to_file(pseudocode, output_file)
                    # If testing fails, update the failure information
                    update_failed_info(failed_info2, pseudocode, "test_failed", cpp_code, failed_test_cases)
            else:
                print("compiles NO!!!!!!!!!!!!!")
                # if consecutive_failures == 5:
                #     write_info_to_file(pseudocode, output_file)
                error_info = extract_error_info(compile_stderr, cpp_code)
                # If compilation fails, update the failure information
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
    # Call these functions where recording is needed
    current_memory = torch.cuda.memory_allocated()

    progress_bar.update(batch_size)
    progress_bar.set_postfix({
        "Success Rate": f"{success_rate:.2f}%",
        "Total Successes": f"{passed_tests}",
        "Overall Success Rate": f"{passed_rate:.2f}%",
        "Current Memory": f"{current_memory}",
    })
    objgraph.show_most_common_types()

max_memory = torch.cuda.max_memory_allocated()

# Close the progress bar
progress_bar.close()

print(f"Number of successfully extracted C++ code blocks: {success_count}/{num_pseudocodes}")
print(f"Indices of failed extractions: {failure_indices}")
