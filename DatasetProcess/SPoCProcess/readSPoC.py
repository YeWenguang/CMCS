import os
import re

def readSPoC(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        pseudocode_content = file.read()

    # 将伪代码拆分为每个probid的部分
    pseudocodes = pseudocode_content.split("\n\n")

    # 设置从第几个伪代码开始生成
    start_index = 1210
    # 统计伪代码的数量
    num_pseudocodes = len(pseudocodes)
    if not pseudocodes[-1].strip():
        num_pseudocodes -= 1
    print(f"文件中一共有 {num_pseudocodes} 个伪代码。")

    # 确定要提取的伪代码的范围
    end_index = start_index + 605
    pseudocodes_to_extract = pseudocodes[start_index:end_index]
    return pseudocodes_to_extract