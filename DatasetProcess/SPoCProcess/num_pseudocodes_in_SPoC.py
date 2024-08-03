# -*- coding: utf-8 -*-
import datetime
import json
import os

file_path = " "

# Open the .tsv file
with open(file_path, "r", encoding="utf-8") as file:
    pseudocode_content = file.read()

# Split the pseudocode into parts for each probid
pseudocodes = pseudocode_content.split("\n\n")

# Count the number of pseudocode parts
num_pseudocodes = len(pseudocodes)
if not pseudocodes[-1].strip():
    num_pseudocodes -= 1

print(f"There are {num_pseudocodes} pseudocode parts in the file.")



