import os
import re

def readSPoC(file_path, start_index, num):
    with open(file_path, "r", encoding="utf-8") as file:
        pseudocode_content = file.read()

    # Split the pseudocode into parts by each probid
    pseudocodes = pseudocode_content.split("\n\n")

    # Set the starting index for generating pseudocode
    # start_index = 1210
    # Count the number of pseudocode sections
    num_pseudocodes = len(pseudocodes)
    if not pseudocodes[-1].strip():
        num_pseudocodes -= 1
    print(f"There are {num_pseudocodes} pseudocode sections in the file.")

    # Determine the range of pseudocode to extract
    end_index = start_index + num
    pseudocodes_to_extract = pseudocodes[start_index:end_index]
    return pseudocodes_to_extract

