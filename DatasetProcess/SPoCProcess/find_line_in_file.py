def find_line_in_file(content, file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            if content in line:
                return line_number
    return None

if __name__ == "__main__":
    file_path = "....txt"

    while True:
        content_to_find = input("Please enter the content to find: ")
        if content_to_find == "z":
            break

        line_number = find_line_in_file(content_to_find, file_path)
        if line_number:
            print(f"Content '{content_to_find}' is found at line number: {line_number}")
        else:
            print(f"Content '{content_to_find}' was not found in the file.")
