import numpy as np


'''
将双线性插值计算为arg_max结果
'''

input_file_path = "./bili.coe"   # 替换为实际文件路径
output_file_path = "output_arg_max.coe"  # 替换为输出文件路径

max_list = ''
hex_str = ''
count = 16
with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
    for line_number, line in enumerate(input_file, 1):
        # 删除偶数行数据
        # if line_number % 2 != 0:
        #     output_file.write(line)



        # second_part = line[16:]
        # print(second_part)
        # output_file.write(second_part)

        result = [line[i:i + 2] for i in range(0, len(line), 2)]
        list = [int(result[3],16),int(result[4],16),int(result[5],16),int(result[6],16),int(result[7],16)]
        list = np.array(list)
        max = np.argmax(list)
        if max==4:
            pos=1
        else:
            pos=0
        max_list = max_list + str(pos)
        if len(max_list)==64:
            result = [max_list[i:i + 4] for i in range(0, len(max_list), 4)]
            hex_list = [int(result[0], 2), int(result[1], 2), int(result[2], 2), int(result[3], 2),
                    int(result[4], 2), int(result[5], 2), int(result[6], 2), int(result[7], 2),
                    int(result[8], 2), int(result[9], 2), int(result[10], 2), int(result[11], 2),
                    int(result[12], 2), int(result[13], 2), int(result[14], 2), int(result[15], 2)]
            for i in range(16):
                hex_str =hex_str+ str(hex(hex_list[i]))[2:]
            output_file.write(hex_str+"\n")
            hex_str = ''
            max_list=''
