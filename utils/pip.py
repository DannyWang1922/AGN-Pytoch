import subprocess

# 执行 pip freeze 并获取输出
output = subprocess.run(['pip', 'freeze'], stdout=subprocess.PIPE).stdout.decode('utf-8')

# 分割输出到单独的行
lines = output.split('\n')

# 过滤掉包含 'file://' 的行
filtered_lines = [line for line in lines if 'file://' not in line]

# 将过滤后的内容写入文件
with open('requirements.txt', 'w') as f:
    for line in filtered_lines:
        if line.strip():  # 确保不写入空行
            f.write(line + '\n')