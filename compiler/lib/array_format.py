import cv2
import numpy as np
import torch

'''
convert_bias:计算量化后新的bias
params:
    z1:输入特征图的zero_point值
    s1:输入特征图的scale值
    s2:权重的scale值
    q2:量化后权重的值
    bias:原始的bias数组
return:
    out_bias:由‘符号位+移位值+定点化的新bias’组成的数组
'''


def convert_bias(z1, s1, s2, q2, bias):
    q2 = q2.astype(np.float32)
    bias_first = bias / (s1 * s2)
    bias_second = q2 * z1
    bias_second = np.sum(bias_second, axis=1)
    bias_second = np.sum(bias_second, axis=1)
    bias_second = np.sum(bias_second, axis=1)

    # new_bias = bias / (s1 * s2) - q2 * z1
    bias = bias_first - bias_second

    # 对bias做定点化
    shift = np.zeros(bias.shape[0], dtype=np.int64)
    for i, num in enumerate(bias):  # i和ii就是从n_bias中取值
        while not ((2 ** 23) <= abs(num) <= (2 ** 24)):
            if shift[i] >= 16:  # fpga里面最多移动16位,这样精度也够了
                break
            else:
                num *= 2
                shift[i] += 1
        bias[i] = round(num)  # n_bias每个数四舍五入

    # 将符号位 移位值 定点化bias拼接在一起
    out_bias = np.zeros(bias.shape[0], dtype=np.int64)
    for index in range(bias.shape[0]):
        # {:024b} 将format后的数字变为24位2二进制，不足补0；
        # & 0xffffff按位与,消去负数的符号位，并将剩下的数字用补码形式保留
        data_integer = format(int(bias[index]) & 0xffffff, '024b')
        n = shift[index]
        symbol = '0'
        if bias[index] < 0:  # 符号位
            symbol = '1'
        elif bias[index] > 0:
            symbol = '0'
        data_decimal = '{:07b}'.format(int(n))  # 位移的次数
        data_total = symbol + str(data_decimal) + str(data_integer)  # 1bit+7bit+24bit
        # n_bias的每个值的符号+n_bias的每个数移位的次数+n_bias的每个值的大小
        out_bias[index] = int(data_total, 2)  # 转成int型 out_bias1为二进制 ；a是十进制

    return out_bias


'''
convert_scale:计算量化后新的scale,并做定点化
params:
    s1:输入特征图的scale值
    s2:权重的scale值
    s3:输出特征图的scale值
return:
    scale:量化后新的scale,并做定点化
    shift - 1:scale定点化所需的移位次数-1(fpga方面做了个+1,因此需要减1,很奇怪)
'''


def convert_scale(s1, s2, s3):
    # scale = (s1 * s2) / s3
    scale = (s1 * s2) / s3
    shift = np.zeros(scale.shape[0], dtype=np.uint32)

    # 对scale做定点化
    for i, num in enumerate(scale):
        while not (0.5 <= num <= 1.0):
            num *= 2
            shift[i] += 1
        scale[i] = np.round(scale[i] * (2 ** (shift[i] + 32)))
    scale = scale.astype(np.uint32)
    return scale, shift - 1


'''
convert_weight:对weight重新进行排序,变为一维数组
params:
    weight:量化后补通道的权重的值
    parallel:fpga的通道并行数
return:
    out_weight:重排后的weight数组
'''


def convert_weight(weight, parallel):
    k = 0
    weight_shape = weight.shape
    weight_size = weight_shape[0] * weight_shape[1] * weight_shape[2] * weight_shape[3]
    out_weight = np.zeros(weight_size, dtype=np.uint32)
    # 出通道读取次数
    out_num = int(weight_shape[0] / parallel)
    # 入通道读取次数
    in_num = int(weight_shape[1] / parallel)

    for row in range(weight_shape[2]):
        for col in range(weight_shape[3]):
            for batch in range(out_num):
                for channel in range(in_num):
                    # i,j代表每次读取parallel个数
                    for i in range(parallel):
                        for j in range(parallel):
                            out_weight[k] = weight[batch * parallel + i][channel * parallel + j][row][col]
                            k += 1
    return out_weight


'''
picture_load:输入图片预处理操作
params:
    shared:共享变量
return: 
    img:处理好的输入图像

'''


def picture_load(shared):
    image_path = shared.img_path
    image_size = shared.img_size
    mean = shared.mean
    std = shared.std
    # 输入图片以灰度图形式读取
    img = cv2.imread(image_path, 0)
    # 输入图片转为规定的尺寸
    if img.shape[0] != image_size:
        img = cv2.resize(img, (image_size, image_size))

    # 归一化处理
    img = (img / 255 - mean) / std

    # 转为tensor格式
    img = torch.from_numpy(img)
    img = img.to(torch.float32)
    img = img.unsqueeze(0)
    img = img.unsqueeze(0)
    return img

#
# s1 = np.array(0.0649)
# s2 = np.array([9.8332e-04, 7.4670e-04, 9.2956e-05, 1.0890e-03, 4.0448e-06, 1.3356e-03,
#                1.3615e-03, 3.3332e-03, 2.0877e-04, 1.8414e-04, 8.9427e-04, 2.9333e-04,
#                1.0383e-03, 1.7247e-06, 2.8482e-04, 6.7435e-04, 1.1350e-03, 2.0811e-02,
#                1.1393e-03, 3.9979e-04, 9.0240e-04, 9.8058e-04, 6.2001e-04, 7.3374e-04,
#                1.2895e-03, 1.1179e-03, 1.2404e-03, 1.2729e-03, 2.3316e-03, 1.1567e-02,
#                3.2449e-03, 9.7334e-04, 5.4277e-03, 1.8092e-03, 2.2526e-03, 1.9541e-03,
#                1.4411e-05, 1.1956e-03, 9.1825e-07, 1.0345e-03, 2.0410e-04, 4.3595e-03,
#                1.3424e-03, 1.2269e-03, 8.8985e-04, 1.6696e-03, 3.5821e-04, 1.3795e-03,
#                1.9346e-05, 1.8360e-03, 7.9314e-04, 1.4554e-03, 8.8995e-04, 8.5802e-04,
#                9.4157e-04, 9.0940e-03, 2.0243e-03, 4.0433e-03, 5.3854e-03, 9.2685e-04,
#                8.4797e-04, 3.6846e-03, 1.9635e-03, 3.0839e-04])
# z1 = np.array(7)
# q2 = np.load("../../para_68/cp.resnet.conv1.weight.int.npy")
# bias = np.load("../../para_68/cp.resnet.conv1.bias.npy")
# convert_bias(z1, s1, s2, q2, bias)

# convert_scale(s1, s2, s3)
