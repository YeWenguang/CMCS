# -*- coding: utf-8 -*-
import os
import re

input_file_path = '../spoc/test/spoc-testw.tsv'
output_file_path = '../SPoC/test-testw.txt'


def extract_pseudocode():
    global probid
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    programs = {}  # Initialize a dictionary to store all programs

    current_program = []  # Initialize current_program list
    prev_line_number = None  # Initialize prev_line_number to None
    prev_probid = None

    for line in lines[1:]:
        columns = line.strip().split('\t')
        pseudocode = columns[0]
        probid = columns[-4]
        subid = columns[-3]
        line_number = int(columns[-2])
        indent_level = int(columns[-1])

        indented_pseudocode = '\t' * indent_level + pseudocode

        if prev_line_number is not None and line_number == 0:
            # If the current line's line_number is 0, indicating the start of a new pseudocode
            programs.setdefault(prev_probid, []).append('\n'.join(current_program))  # Append current_program as a string
            current_program = []  # Initialize a new current_program list

        current_program.append(indented_pseudocode)
        prev_line_number = line_number
        prev_probid = probid

    # Append the last program to programs
    if current_program:
        programs.setdefault(probid, []).append('\n'.join(current_program))

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for probid, program_list in programs.items():
            for program in program_list:
                output_file.write(f"probid: {probid}\n")  # Write probid at the beginning of each program
                output_file.write("pseudo program:\n```\n")  # Indicate the start of pseudo program
                output_file.write(program + '\n')  # Write the program
                output_file.write("```\n\n\n")

    total_programs = sum(len(program_list) for program_list in programs.values())
    print(f"Total programs extracted: {total_programs}")

extract_pseudocode()


