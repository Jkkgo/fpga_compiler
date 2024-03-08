import os

import cv2
import numpy as np

'''
批量测试生成图片
'''


folder_path = 'D:\WorkStation\BiSeNet/tools/val/img/'  # 替换为实际的文件夹路径
write_path = "./input/"

# 使用 os.listdir() 获取文件夹中所有文件的列表
files_in_folder = os.listdir(folder_path)

# 打印所有文件名
for file_name in files_in_folder:
    img_path = folder_path+file_name
    file = file_name.split('.')[0]
    img = cv2.imread(img_path,0)
    img.tofile(write_path+file+'.bin')

