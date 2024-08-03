import os
import subprocess
import re
from Kill_process import release_port_for_user

python_file_path = '/home/yewenguang/work/Code-Llama/example_instructions.py'
txt_file_path = '/home/yewenguang/work/Code-Llama/spoc/pseudocode/train/split/spoc-train-test_extracted_pseudocode.txt'
output_folder = '/home/yewenguang/work/Code-Llama/spoc/code/train/split'  # 文件夹路径

# 调用函数释放特定用户占用的端口的所有相关进程
release_port_for_user(29500, 'yewenguang')

# 读取 extracted_pseudocode.txt 中的程序内容
with open(txt_file_path, 'r', encoding='utf-8') as programs_file:
    programs_content = programs_file.read().split('\n\n')

# 从第 n 个程序开始处理
n = 19
for index, program in enumerate(programs_content[n - 1:], start=n):
    # 构建要填充的内容字符串
    content_to_fill = '\n'.join(program.split('\n'))

    # 读取 Python 文件内容
    with open(python_file_path, 'r', encoding='utf-8') as python_file:
        python_code = python_file.read()

    # 寻找并替换 Python 文件中的内容字段
    search_text = 'instructions = ['
    content_start = python_code.find(search_text)
    if content_start != -1:
        content_end = python_code.find('"""', content_start + len(search_text))
        if content_end != -1:
            # 填充程序到 content 字段中的三引号字符串中
            updated_content = (
                python_code[:content_end + 3] + f'\n{content_to_fill}\n"""' + python_code[content_end + 3:].split('"""', 1)[1]
            )

            # 将更新后的内容写回到文件中
            with open(python_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(updated_content)
            print(f"已将第 {index} 个程序写入到 {python_file_path} 文件中的 instructions 变量中。")

            # 设置工作目录和环境变量
            os.chdir('/home/yewenguang/work/Code-Llama')  # 替换为实际路径
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'

            command = [
                "torchrun",
                "--nproc_per_node",
                "1",
                "example_instructions.py",
                "--ckpt_dir",
                "CodeLlama-7b-Instruct/",
                "--tokenizer_path",
                "CodeLlama-7b-Instruct/tokenizer.model",
                "--max_seq_len",
                "1024",  # 更小的值
                "--max_batch_size",
                "1" 
            ]

            try:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                terminal_output, error_output = process.communicate()

                # 检查是否有输出
                if terminal_output:
                    print(terminal_output)
                else:
                    print(error_output)

            except subprocess.CalledProcessError as e:
                print(f"命令执行出错: {e}")
                terminal_output = e.stdout

            release_port_for_user(29500, 'yewenguang')

            # 使用正则表达式匹配终端输出中的 C++ 代码内容
            pattern = r'```([^`]*)```'  # 匹配 ``` 之间的内容
            matches = re.findall(pattern, terminal_output, re.DOTALL)
            if not matches:
                pattern = r'```([^`]*)=================================='  # 匹配 ``` ================================== 之间的内容
                matches = re.findall(pattern, terminal_output, re.DOTALL)

            if matches:
                cpp_code = matches[0].strip()  # 获取匹配的 C++ 代码内容

                # 创建文件夹（如果不存在）
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                # 写入到单独的文件中
                cpp_file_path = os.path.join(output_folder, f"cpp_code_{index-1}.cpp")
                with open(cpp_file_path, 'w', encoding='utf-8') as cpp_file:
                    cpp_file.write(cpp_code)
                    print(f"已存储提取的 C++ 代码到文件：{cpp_file_path}")
            else:
                print("未找到匹配的 C++ 代码内容。")
        else:
            print("找不到要填充的位置。")
    else:
        print("未找到 instructions 变量的定义。")
