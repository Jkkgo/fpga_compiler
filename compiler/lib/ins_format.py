import numpy as np
import torch
from torch.nn.quantized import functional as qF

'''
conv33para:计算3*3卷积操作中weight、bias、scale、shift的读取次数
params:
    out_channel:权重出通道
    in_channel:权重入通道
    data_size：fpga每次读多少bit数据
'''


def conv33para(out_channel, in_channel, data_size):
    # 一个 3*3的卷积的coe行数
    weight_row_num = int(out_channel * in_channel * 8 * 9 / data_size)
    # 一个 3*3的卷积的scale + shift + bias行数
    bias_row_num = int(out_channel * 32 / data_size)
    # *3是因为有 scale + shift + bias
    total_row_num = weight_row_num + bias_row_num * 3
    # 硬件读取卷积权重总共需要weight_num次（m*c*3*3个权重数字*每个数字8字节/一次读多少字节*9个权重数字）
    weight_num = int((out_channel * in_channel * 3 * 3 * 8) / (data_size * 9))
    # 硬件读三个参数的次数 （m个quant*每个quant32字节/一次读多少个字节）
    quant_num = int(out_channel * 32 / data_size)
    return weight_num, quant_num


'''
conv11para:计算1*1卷积操作中weight、bias、scale、shift的读取次数
params:
    out_channel:权重出通道
    in_channel:权重入通道
    data_size：fpga每次读多少bit数据
'''


def conv11para(out_channel, in_channel, data_size):
    weight_row_num = int(out_channel * in_channel * 8 / data_size)
    bias_row_num = int(out_channel * 32 / data_size)
    total_row_num = weight_row_num + bias_row_num * 3
    # 硬件读取卷积权重总共需要weight_num次（m*c*1*1个权重数字*每个数字8字节/一次读多少字节*1个权重数字）
    weight_num = int((out_channel * in_channel * 8) / data_size)
    # 硬件读三个参数的次数 （m个quant*每个quant32字节/一次读多少个字节）
    quant_num = int(out_channel * 32 / data_size)
    return weight_num, quant_num


'''
leaky_format:计算使用leaky_relu激活函数时所需的修正值
由于torch1.7版本量化不支持leaky_relu算子,所以采用修正的方式实现该算子
1.7版本以后支持leaky_relu算子,理论上不需要再使用此方法
params:
    s3:输出特征图对应的scale
'''


def leaky_format(s3):
    add_data = []
    data1 = torch.ones(16)
    # 将data1中的数据更新为 -index * 10 - 5 范围(-5,-155,stride=-10)
    for index in range(data1.shape[0]):
        data1[index] = -index * 10 - 5

    for index in range(data1.shape[0]):  # 16
        # q_feature = round((data1[index]*s3-0)/s3)  q_feature中所有小于-128的值限制为-128，data1中的其他值保持原样
        # r =s(q-z) q = r/s+z
        q_feature = torch.quantize_per_tensor(data1[index] * s3, scale=float(s3),
                                              zero_point=int(0), dtype=torch.qint8)

        a = q_feature.int_repr()
        out_leak = qF.leaky_relu(input=q_feature, negative_slope=0.1, inplace=True)
        # out为量化之后的值进行leaky relu 运算的结果
        out = out_leak.int_repr()

        out2 = data1[index] * 0.1
        # 将量化值*0.1 四舍五入后的值与 原本值*0.1 四舍五入后的值进行对比
        # 对此方法来说超过 int8 负数范围的值 add_data 值为1，超过正数范围的值 add_data 为-1 范围内对应的add_data为0
        if np.round(out) > np.round(out2):
            add_data.append(1)
        elif np.round(out) < np.round(out2):
            add_data.append(-1)
        else:
            add_data.append(0)

    # print(add_data)
    out_str = ""
    for index in range(data1.shape[0]):
        if add_data[index] == 0:
            out_str += '00'
        elif add_data[index] == 1:
            out_str += '01'
        elif add_data[index] == -1:
            out_str += '10'

    return out_str
