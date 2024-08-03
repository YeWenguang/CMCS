def find_line_in_file(content, file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            if content in line:
                return line_number
    return None



if __name__ == "__main__":
    file_path = "/home/yewenguang/work/Code-Llama/spoc/pseudocode/test/spoc-testp_extracted_pseudocode.txt"

    while True:
        content_to_find = input("请输入要查找的内容：")
        if content_to_find == "z":
            break

        line_number = find_line_in_file(content_to_find, file_path)
        if line_number:
            print(f"内容 '{content_to_find}' 在文件中的行数为：{line_number}")
        else:
            print(f"内容 '{content_to_find}' 未在文件中找到。")