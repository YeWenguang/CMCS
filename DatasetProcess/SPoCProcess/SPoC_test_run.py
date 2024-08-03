# -*- coding: utf-8 -*-
import datetime
import json
import os

# file_path = "/home/yewenguang/work/Code-Llama/spoc/test/spoc-testw.tsv"
#
# # 打开.tsv文件
# with open(file_path, 'r') as file:
#     count = 0
#     # 逐行读取文件内容
#     for line in file:
#         # 按制表符分割每一行
#         columns = line.strip().split('\t')
#         # 检查倒数第二列是否为"0"
#         if len(columns) >= 2 and columns[-2] == "10":
#             count += 1
#
# print("倒数第二列中包含字符串'0'的次数为:", count)


file_path = "/home/yewenguang/work/codellama-main/SPoC/test-testp-2.txt"

# 打开.tsv文件
with open(file_path, "r", encoding="utf-8") as file:
    pseudocode_content = file.read()

# 将伪代码拆分为每个probid的部分
pseudocodes = pseudocode_content.split("\n\n")

# 统计伪代码的数量
num_pseudocodes = len(pseudocodes)
if not pseudocodes[-1].strip():
    num_pseudocodes -= 1

print(f"文件中一共有 {num_pseudocodes} 个伪代码。")


