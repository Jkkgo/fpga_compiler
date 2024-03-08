import os

import cv2
import numpy as np
from tqdm import tqdm

'''
批量测试可视化
'''

folder_path = 'D:\WorkStation\BiSeNet/tools/val/img/'  # 替换为实际的文件夹路径
write_path = "./output/"

# 使用 os.listdir() 获取文件夹中所有文件的列表
files_in_folder = os.listdir(folder_path)

for file_name in files_in_folder:
    file_path = file_name  # 替换为实际文件路径
    write_file = file_name.split('.')[0]
    palette = np.array([0, 255]).astype(np.uint8)
    with open(file_path, 'r') as file:
        total_bin = ''

        # 使用 for 循环逐行读取文件内容
        for line in file:
            line = line.strip()
            result = [line[i] for i in range(len(line))]
            for i in range(16):
                str_bin = bin(int(result[i], 16))[2:].zfill(4)
                total_bin = total_bin + str_bin

        bin_array = list(total_bin)
        bin_array = np.array(bin_array)
        img = bin_array.reshape((640, 640)).astype(int)
        pred = palette[img]
        cv2.imwrite(write_path+ write_file +'.bmp', pred)
