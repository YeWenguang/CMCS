# -*- coding: utf-8 -*-
import subprocess
import os
from tqdm import tqdm
from datetime import datetime
import unicodedata

cpp_files_directory = '/home/yewenguang/work/Code-Llama/spoc/code/test/testp/testp-split'
testcases_base_path ='/home/yewenguang/work/Code-Llama/spoc/testcases'
log_directory = '/home/yewenguang/work/Code-Llama/spoc/logs/code_test_logs/test/testp/testp-split'
compile_failure_log = '/home/yewenguang/work/Code-Llama/spoc/logs/code_test_logs/test/testp/testp-split/compile_failure_log'
run_failure_log = '/home/yewenguang/work/Code-Llama/spoc/logs/code_test_logs/test/testp/testp-split/run_failure_log'
run_success_log = '/home/yewenguang/work/Code-Llama/spoc/logs/code_test_logs/test/testp/testp-split/run_success_log'
testcases_summary_path = '/home/yewenguang/work/Code-Llama/spoc/logs/code_test_logs/test/testp/testp-split/summary_log/summary_log.txt'


def log_compile_failure(file_path, compile_output):
    log_file = os.path.join(compile_failure_log, f'compile_logs_{os.path.basename(file_path)}.txt')
    with open(log_file, 'a') as log:
        log.write(
            f"C++程序: {file_path}\n编译时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n编译结果: 失败\n失败信息:\n{compile_output}\n###################################\n\n")


def log_run_success(file_path, output, expected_output):
    log_file = os.path.join(run_success_log, f'compile_logs_{os.path.basename(file_path)}.txt')
    with open(log_file, 'a') as log:
        log.write(
            f"C++程序: {file_path}\n测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n测试结果: 运行成功\n输出结果:\n{output}\n预期输出:\n{expected_output}\n###################################\n\n")


def log_run_failure(file_path, output, expected_output, error_msg):
    log_file = os.path.join(run_failure_log, f'compile_logs_{os.path.basename(file_path)}.txt')
    #print("log_file:" +log_file)
    with open(log_file, 'a') as log:
        log.write(
            f"C++程序: {file_path}\n测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n测试结果: 运行失败\n输出结果:\n{output}\n预期输出:\n{expected_output}\n失败信息: {error_msg}\n###################################\n\n")


def compile_cpp(file_path):
    compile_process = subprocess.run(['g++', file_path, '-o', 'example'], capture_output=True, text=True)
    if compile_process.returncode == 0:
        return True
    else:
        compile_output = compile_process.stderr
        log_compile_failure(file_path, compile_output)
        #print(compile_process.stderr)
        return False


def normalize_text(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").strip()


def run_cpp_with_input(code, input_text):
    try:
        import subprocess

        # 保存C++代码到文件
        with open('temp.cpp', 'w') as file:
            file.write(code)

        # 编译C++代码
        compile_process = subprocess.run(['g++', 'temp.cpp', '-o', 'temp'], capture_output=True, text=True)
        if compile_process.returncode == 0:
            # 启动可执行文件并传入输入信息
            process = subprocess.Popen('./temp', stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(input=input_text, timeout=15)  # 传入输入并等待程序执行完毕

            # 检查输出中是否存在冒号
            last_colon_index = stdout.rfind(":")  # 找到最后一个冒号的位置
            if last_colon_index != -1:
                return stdout[last_colon_index + 1:].strip()  # 返回最后一个冒号后的部分
            else:
                return stdout.strip()  # 返回完整输出

        else:
            return compile_process.stderr.strip()

    except Exception as e:
        return f"运行时出错: {e}"


def extract_cpp_code(file_path):
    cpp_code = ""
    try:
        with open(file_path, 'r') as file:
            cpp_code = file.read()  # 读取整个文件内容作为代码
    except FileNotFoundError:
        print("文件不存在")
    return cpp_code


def compile_and_run_multiple_cpp_files(directory):
    global test_folder_name, test_failed
    success_count = 0
    failure_count = 0
    start_time = datetime.now()  # 记录测试开始时间

    # 统计文件数量
    total_files = sum(1 for filename in os.listdir(directory) if filename.endswith(".cpp"))

    # 使用 tqdm 创建进度条
    progress_bar = tqdm(total=total_files, desc="Compiling and Running")

    for idx, filename in enumerate(os.listdir(directory)):
        if filename.endswith(".cpp"):
            # 提取测试文件夹名称
            parts = filename.split('_')
            if len(parts) >= 3:
                test_folder_name = parts[-1].split('.')[0]  # 获取最后一个元素，并去掉文件扩展名
                #print(f"测试文件夹名称为: {test_folder_name}")

            # 读取测试文件内容
            testcases_path = f"{testcases_base_path}/{test_folder_name}/{test_folder_name}_testcases_public.txt"
            #print(testcases_path)
            try:
                with open(testcases_path, 'r') as file:
                    content = file.read()
                #print(content)

                testcases = []
                current_input = []
                current_output = []
                input_mode = True  # 初始状态为输入模式

                # 假设 content 是您测试文件的内容
                lines = content.split('\n')

                for line in lines:
                    if line.startswith("###ENDINPUT###"):
                        input_mode = False  # 切换到输出模式
                    elif line.startswith("###ENDOUTPUT###"):
                        input_mode = True  # 切换到输入模式
                        if len(current_output) > 0:
                            testcases.append((current_input, current_output))
                            current_input = []
                            current_output = []
                    elif input_mode:
                        current_input.append(line)
                    else:
                        current_output.append(line)

                file_path = os.path.join(directory, filename)
                cpp_code = extract_cpp_code(file_path)

                # 按顺序逐组处理测试数据
                for input_data, expected_output in testcases:
                    if compile_cpp(file_path):
                        # 运行可执行文件并传入输入
                        input_text = '\n'.join(input_data)
                        #print(f"input_text:\n{input_text}")
                        output = run_cpp_with_input(cpp_code, input_text)
                        output_normalized = normalize_text(output)
                        #print(f"output_normalized:\n{output_normalized}")
                        # 检查输出是否与期望输出匹配
                        expected_output_text = '\n'.join(expected_output)
                        expected_output_normalized = normalize_text(expected_output_text)
                        #print(f"expected_output_normalized:\n{expected_output_normalized}")
                        # 更改为标记为运行成功的逻辑
                        if output_normalized == expected_output_normalized:
                            test_failed = 2
                            log_run_success(file_path, output, expected_output_text)
                            #print("输出与预期一致")
                        else:
                            #print("输出与预期不匹配")
                            #print("0")
                            log_run_failure(file_path, output, expected_output_text, "输出与预期不匹配")
                            #print("1")
                            test_failed = 1  # 标记测试失败
                            break  # 中止循环，整个程序运行失败
                    else:
                        test_failed = 0  # 标记编译失败
                        break  # 中止循环，整个程序运行失败

                # 检查整个程序运行是否失败
                if test_failed == 1 or test_failed == 0:
                    failure_count += 1
                    #print("程序运行失败")
                    # 可以在这里执行失败时的处理
                else:
                    success_count += 1
                    #print("程序运行成功")
                    # 可以在这里执行成功时的处理
            except FileNotFoundError:
                #print("找不到测试文件")
                #print("尝试使用绝对路径:", os.path.abspath(testcases_path))
                #print("当前工作目录:", os.getcwd())
                failure_count += 1
                progress_bar.update(1)

            end_time = datetime.now()  # 记录测试结束时间
            test_duration = end_time - start_time  # 计算测试持续时间
            progress_bar.update(1)
            success_rate = (success_count / (idx + 1)) * 100 if (idx + 1) > 0 else 0  # 计算当前成功率
            progress_bar.set_postfix({"Success Rate": f"{success_rate:.2f}%"})
            print(f"当前测试程序运行时间: {test_duration}、运行测试进度: {idx + 1}/{total_files}、当前测试程序的成功率: {success_rate:.2f}%")

    progress_bar.close()
    return success_count, failure_count


# 记录运行结果统计信息和信息到日志文件
def log_statistics(total_files, success, failure, log_directory, testcases_summary_path):
    success_rate = (success / total_files) * 100 if total_files > 0 else 0

    print(f"运行程序的数量: {total_files}")
    print(f"运行成功的程序数量: {success}")
    print(f"运行失败的程序数量: {failure}")
    print(f"成功率: {success_rate:.2f}%\n")

    log_file_stats = os.path.join(log_directory, testcases_summary_path)
    with open(log_file_stats, 'w') as log_stats:
        log_stats.write(f"运行程序数量: {total_files}\n")
        log_stats.write(f"运行成功的程序数量: {success}\n")
        log_stats.write(f"运行失败的程序数量: {failure}\n")
        log_stats.write(f"成功率: {success_rate:.2f}%\n")


# 执行编译并运行函数
def execute_compile_and_run():
    success, failure = compile_and_run_multiple_cpp_files(cpp_files_directory)
    total_files = success + failure

    # 调用记录统计信息的函数
    log_statistics(total_files, success, failure, log_directory, testcases_summary_path)


# 执行函数
execute_compile_and_run()
