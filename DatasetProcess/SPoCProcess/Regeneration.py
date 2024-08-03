# -*- coding: utf-8 -*-
import subprocess
import sys
import os
import json
import re
from datetime import datetime
from pathlib import Path

config = {
    'MAX_RETRIES': 3,
    'CUDA': 1,
    'error_message_lines': 20,
    'log_folder': '/home/yewenguang/work/Code-Llama/spoc/logs/code_test_logs/train/split16/compile_failure_log',
    'log_folder_success': 'path_to_success_logs',
    'log_folder_failure': 'path_to_failure_logs',
    'python_file_path': '/home/yewenguang/work/Code-Llama/example_instructions.py',
    'output_folder': '/home/yewenguang/work/Code-Llama/spoc/code/fixed/split16_3',
    'summary_log_path': 'path_to_summary_log',
    'command': "sudo CUDA_VISIBLE_DEVICES={CUDA} /home/anaconda3/envs/codellama/bin/python /home/yewenguang/work/Code-Llama/example_instructions.py",
    'custom_requirements': """
Please complete the following 6 steps step by step：
Step 1: View error messages and locate the error location
Step 2: Check the surrounding code to gain a deeper understanding of the cause of the error
Step 3: Contact the context and find a solution
Step 4: Modify the code based on error messages and solutions
Step 5: Save the modified code and recompile to ensure there are no new issues
Step 6: Provide the final modification code

Answer: let's fix C++code step by step,
""",
    'example': """Question: 
C++code:
#include <iostream>

int main() {
    int n, m, a, b, count = 0;
    std::cin >> n >> m;
    for (a = 0; a * a <= n; a++) {
        for (b = 0; a + b * b <= m; b++) {
            if (a * a + b == n && a + b * b == m) {
                count++;
            }
        }
    }
    std::cout << count << "
    ";
    return 0;
}

error_message:
/home/yewenguang/work/Code-Llama/spoc/code/train/split16/cpp_code_0092_214A.cpp:13:27: warning: missing terminating " character
   13 |     std::cout << count << "
      |                           ^
/home/yewenguang/work/Code-Llama/spoc/code/train/split16/cpp_code_0092_214A.cpp:13:27: error: missing terminating " character
/home/yewenguang/work/Code-Llama/spoc/code/train/split16/cpp_code_0092_214A.cpp:14:1: warning: missing terminating " character
   14 | ";
      | ^
/home/yewenguang/work/Code-Llama/spoc/code/train/split16/cpp_code_0092_214A.cpp:14:1: error: missing terminating " character
   14 | ";
      | ^~
/home/yewenguang/work/Code-Llama/spoc/code/train/split16/cpp_code_0092_214A.cpp: In function ‘int main()’:
/home/yewenguang/work/Code-Llama/spoc/code/train/split16/cpp_code_0092_214A.cpp:15:5: error: expected primary-expression before ‘return’
   15 |     return 0;
      |     ^~~~~~

Fix C++code based on error information and provide the final fixed complete C++code step by step.


Answer: OK, let's fix C++code step by step,


> Assistant:  
Here's a step-by-step breakdown of the changes:

1. The first error message indicates that there is a missing terminating " character in the std::cout statement. To fix this, we need to add a closing quotation mark at the end of the std::cout statement.
2. The second error message indicates that there is a missing terminating " character in the std::cout statement. To fix this, we need to add a closing quotation mark at the end of the std::cout statement.
3. The third error message indicates that there is a missing primary-expression before 'return' in the return statement. To fix this, we need to remove the extraneous semicolon in the std::cout statement.

By applying these fixes, we get the corrected C++ code that compiles and runs without any errors.

Here's the corrected C++ code:
```
#include <iostream>

int main() {
    int n, m, a, b, count = 0;
    std::cin >> n >> m;
    for (a = 0; a * a <= n; a++) {
        for (b = 0; a + b * b <= m; b++) {
            if (a * a + b == n && a + b * b == m) {
                count++;
            }
        }
    }
    std::cout << count << " ";
    return 0;
}

```

Question: 
C++code:
#include <iostream>
#include <string>

const long long INF = 2000000000000000000LL;
const int MOD = 10000007;

int a[1000006];

bool ispal(string s) {
    string r = s;
    reverse(r.begin(), r.end());
    if (s == r) {
        return true;
    } else {
        return false;
    }
}

int main() {
    string s;
    cin >> s;
    int n = s.size();
    int cnt = 0;
    for (int i = 0; i < n / 2; i++) {
        char samne = s[i];
        char pichone = s[n - (i + 1)];
        if (samne != pichone) {
            cnt++;
        }
    }
    if (cnt == 0 && ispal(s) && n % 2 == 0) {
        cout << "NO" << endl;
        cout << "
" << endl;
    } else if (cnt > 1) {
        cout << "NO" << endl;
        cout << "
" << endl;
    } else {
        cout << "YES" << endl;
        cout << "
" << endl;
    }
    return 0;
}

error_message:
/home/yewenguang/work/Code-Llama/spoc/code/train/split16/cpp_code_0248_798A.cpp:33:17: error: missing terminating " character
/home/yewenguang/work/Code-Llama/spoc/code/train/split16/cpp_code_0248_798A.cpp:34:1: error: missing terminating " character
   34 | " << endl;
      | ^~~~~~~~~~
/home/yewenguang/work/Code-Llama/spoc/code/train/split16/cpp_code_0248_798A.cpp:37:17: error: missing terminating " character
/home/yewenguang/work/Code-Llama/spoc/code/train/split16/cpp_code_0248_798A.cpp:38:1: error: missing terminating " character
   38 | " << endl;
      | ^~~~~~~~~~

Fix C++code based on error information and provide the final fixed complete C++code step by step.
Answer: OK, let's fix C++code step by step,


> Assistant:  
Here's a step-by-step breakdown of the changes:

1. The first error message indicates that there is a missing terminating " character in the cout statement. To fix this, we need to add a closing quotation mark at the end of the cout statement.
2. The second error message indicates that there is a missing terminating " character in the cout statement. To fix this, we need to add a closing quotation mark at the end of the cout statement.
3. The third error message indicates that there is a missing primary-expression before 'return' in the return statement. To fix this, we need to remove the extraneous semicolon in the cout statement.

By applying these fixes, we get the corrected C++ code that compiles and runs without any errors.

Here's the corrected C++ code:
```
#include <iostream>
#include <string>

const long long INF = 2000000000000000000LL;
const int MOD = 10000007;

int a[1000006];

bool ispal(string s) {
    string r = s;
    reverse(r.begin(), r.end());
    if (s == r) {
        return true;
    } else {
        return false;
    }
}

int main() {
    string s;
    cin >> s;
    int n = s.size();
    int cnt = 0;
    for (int i = 0; i < n / 2; i++) {
        char samne = s[i];
        char pichone = s[n - (i + 1)];
        if (samne != pichone) {
            cnt++;
        }
    }
    if (cnt == 0 && ispal(s) && n % 2 == 0) {
        cout << "NO" << endl;
        cout << " " << endl;
    } else if (cnt > 1) {
        cout << "NO" << endl;
        cout << " " << endl;
    } else {
        cout << "YES" << endl;
        cout << " " << endl;
    }
    return 0;
}


Question: """
}

MAX_RETRIES = config["MAX_RETRIES"]
python_file_path = config["python_file_path"]
# txt_file_path = config["txt_file_path"]
output_folder = config["output_folder"]
log_folder = config["log_folder"]
summary_log_path = config["summary_log_path"]
txt_files = [os.path.join(log_folder, file) for file in os.listdir(log_folder) if file.endswith('.txt')]


# 用于更新 Python 文件的特定部分，包括用户自定义的额外要求
def update_python_file(index, cpp_content, error_message, probid_content,
                       custom_requirements=config["custom_requirements"], example=config["example"]):
    with open(python_file_path, 'r', encoding='utf-8') as python_file:
        python_code = python_file.read()

    search_text = 'instructions = ['
    content_start = python_code.find(search_text)
    if content_start != -1:
        content_end = python_code.find('"""', content_start + len(search_text))
        if content_end != -1:
            updated_content = (
                    python_code[:content_end + 3] +
                    # (f'{example}' if example else '') +
                    f'\nprobid:' +
                    f'{probid_content}\n' +
                    f'C++code:\n' +
                    f'{cpp_content}\n\n'
                    f'error_message:\n' +
                    f'{error_message}\n'
                    + (f'{custom_requirements}\n' if custom_requirements else '') +
                    '"""' + python_code[content_end + 3:].split('"""', 1)[1]
            )

            with open(python_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(updated_content)


# 用于从终端输出提取 C++ 代码并保存到文件
def extract_cpp_code(terminal_output, index):
    pattern = r'probid:\s*(.*)'
    match = re.search(pattern, terminal_output)

    if match:
        probid_content = match.group(1).strip()

        pattern_code = r'```(.*?)```'

        matches_code = re.findall(pattern_code, terminal_output, re.DOTALL)
        last_code_block_with_iostream = None

        code_blocks = matches_code  # 所有匹配的代码块

        for code_block in code_blocks[::-1]:  # 逆序遍历匹配的代码块
            if '#include <iostream>' in code_block:
                lines = code_block.split('\n')
                if all('```' not in line for line in lines[:-1]):  # 排除最后一行含有```的情况
                    last_code_block_with_iostream = code_block.strip()
                    break

        if last_code_block_with_iostream:
            iostream_index = last_code_block_with_iostream.find('#include <iostream>')
            if iostream_index != -1:
                cpp_code = last_code_block_with_iostream[iostream_index:]

                cpp_file_path = os.path.join(output_folder, f"cpp_code_{index}_{probid_content}.cpp")
                with open(cpp_file_path, 'w', encoding='utf-8') as cpp_file:
                    cpp_file.write(cpp_code)

                # print(f"已存储提取的 C++ 代码到文件：{cpp_file_path}")
            else:
                print("未在代码块中找到 #include <iostream>。")
        else:
            print("未找到符合条件的 C++ 代码段。")
    else:
        print("未找到匹配的 probid: 行。")


def process_code(n):
    global success, terminal_output, attempt, success_count, failure_count, error_output
    start_time = datetime.now()
    # log_folder_success = config["log_folder_success"]
    # log_folder_failure = config["log_folder_failure"]

    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            content = f.read()

            program_path_match = re.search(r'C\+\+程序:\s*(.+)', content)
            error_message_match = re.search(r'失败信息:\s*([\s\S]+?)###################################', content)

            if program_path_match:
                program_path = program_path_match.group(1)
                with open(program_path, 'r') as cpp_file:
                    cpp_content = cpp_file.read()
                    # print(f'文件: {txt_file}\nC++程序路径: {program_path}\nC++程序内容:\n{cpp_content}')

            if error_message_match:
                error_message = error_message_match.group(1).strip()
                # 提取以失败信息:开头后的前 10 行代码
                error_message_lines = error_message.split('\n')[:config["error_message_lines"]]
                error_message_extracted = '\n'.join(error_message_lines)
                # print(f'提取的失败信息前 10 行代码:\n{error_message_extracted}')

        probid_content = re.search(r'cpp_code_\d+_(\w+)\.cpp$', program_path)
        index_content = re.search(r'cpp_code_(\d+)_(\w+)\.cpp$', program_path)
        index = index_content.group(1)
        # print(probid_content.group(1))
        # print(index)

        update_python_file(n, cpp_content, error_message_extracted, probid_content.group(1))

        attempt = 0
        success_count = 0
        failure_count = 0
        success = False

        while not success and attempt < MAX_RETRIES:
            print(f"正在进行第 {attempt + 1} 次代码生成尝试，最多尝试 {MAX_RETRIES} 次：")
            try:
                command = config["command"].format(CUDA=config["CUDA"])
                terminal_output = subprocess.run(command, shell=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                error_output = e.stderr

            if terminal_output.returncode == 0:
                print(terminal_output.stdout)
                success = True
                print(f"第 {attempt + 1} 次尝试成功")
                content = terminal_output.stdout
                break
            else:
                print(f"第 {attempt + 1} 次尝试失败")

            attempt += 1

        if success:
            success_count += 1
            extract_cpp_code(terminal_output.stdout, index)

        else:
            print(f"第{index}个程序生成失败")
            failure_count += 1
            # content = error_output  # 更改为适合的错误消息


process_code(1)
