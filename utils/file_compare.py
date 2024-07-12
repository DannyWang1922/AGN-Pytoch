def read_file(file_path):
    """读取文件内容并返回为字符串"""
    with open(file_path, 'r') as file:
        return file.read()


def compare_four_files(file1_path, file2_path, file3_path, file4_path):
    """比较两个文件的内容是否相同"""
    content1 = read_file(file1_path)
    content2 = read_file(file2_path)
    content3 = read_file(file3_path)
    content4 = read_file(file4_path)
    return content1 == content2 == content3 == content4

def compare_files(file1_path, file2_path):
    """比较两个文件的内容是否相同"""
    content1 = read_file(file1_path)
    content2 = read_file(file2_path)
    return content1 == content2


# 文件路径
file1_path = "bert_inputs_tcol1.txt"
# file2_path = "bert_inputs_tcol2.txt"
file3_path = "bert_inputs_tfidf1.txt"
# file4_path = "bert_inputs_tfidf2.txt"

file5_path = "bert_outputs_tcol1.txt"
# file6_path = "bert_outputs_tcol2.txt"
file7_path = "bert_outputs_tfidf1.txt"
# file8_path = "bert_outputs_tfidf2.txt"


# 比较文件内容
# are_files_identical = compare_four_files(file5_path, file6_path, file7_path, file8_path)
# if are_files_identical:
#     print("The files are identical.")
# else:
#     print("The files are different.")

are_files_identical2 = compare_files(file5_path, file7_path)
if are_files_identical2:
    print("The files are identical.")
else:
    print("The files are different.")