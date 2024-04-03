import numpy as np
import torch
from torch import nn

from compiler.lib.ins_format import leaky_format


def sim_leaky_relu(quant_result, local_zp, local_scale, leakyRatio):
    quant_result = quant_result.astype(np.int64)
    relu_result = quant_result - local_zp
    out_str = leaky_format(local_scale)
    # 使用向量化操作将函数应用到数组的每个元素
    # 仿真硬件的amend修正
    relu_result = np.vectorize(sim_amend)(relu_result, out_str, leakyRatio)
    relu_result += local_zp
    relu_result[relu_result < 0] = 0
    relu_result[relu_result > 255] = 255
    relu_result = relu_result.astype(np.uint8)
    return relu_result


def sim_amend(num, out_str, leakyRatio):
    map_table = [-5, -15, -25, -35, -45, -55, -65, -75, -85, -95, -105, -115, -125, -135, -145, -155]
    if num >= 0:
        pass
    else:
        map_temp = num
        num = num * int(leakyRatio * 2 ** 17)

        num_complement = format(num & 0xffffffff, '032b')
        num_complement = num_complement[::-1]
        mantissa = num_complement[13:17][::-1]
        mantissa = int(mantissa, 2) & 0b1111
        if mantissa == 0:
            num = num >> 17
        else:
            if int(num_complement[17], 2):
                if int(num_complement[16], 2):
                    num = (num >> 17) + 1
                else:
                    num = num >> 17
            else:
                if mantissa > 0b1000:
                    num = (num >> 17) + 1
                else:
                    num = num >> 17

        if map_temp not in map_table:
            pass
        else:
            index = map_table.index(map_temp)
            if out_str[2 * index:2 * index + 2] == '10':
                num = num - 1
            elif out_str[2 * index:2 * index + 2] == '01':
                num = num + 1
            else:
                pass

    return num


def sim_round(num):
    # 判断最后一位是否为1
    if num & 1:
        # 如果最后一位是1，则将该数与二进制的1进行按位或运算
        num = num >> 1
        num = num + 1
    else:
        num = num >> 1

    return num


def sim_quant(conv_result, bias, bias_shift, scale, scale_shift, local_zp):
    quant_result = conv_result.numpy().astype(np.int64)
    for ch in range(conv_result.shape[1]):
        quant_result[0][ch] <<= 16
        quant_result[0][ch] += bias[ch] << (16 - bias_shift[ch])
        quant_result[0][ch] *= scale[ch]
        quant_result[0][ch] >>= (scale_shift[ch] + 32 + 16)
    # 使用向量化操作将函数应用到数组的每个元素
    quant_result = np.vectorize(sim_round)(quant_result)
    quant_result += local_zp
    quant_result[quant_result < 0] = 0
    quant_result[quant_result > 255] = 255
    quant_result = quant_result.astype(np.uint8)
    return quant_result


def sim_conv(feature, weight, option, pre_zp):
    c_in = feature.shape[1]
    c_out = weight.shape[0]
    stride = option[1]
    padding = option[2]
    kernel_size = 0
    if "33" in option[0]:
        kernel_size = 3
    elif "11" in option[0]:
        kernel_size = 1
    feature = np.pad(feature,
                     pad_width=((0, 0), (0, 0), (padding, padding), (padding, padding)),
                     mode='constant',
                     constant_values=pre_zp)

    # 定义卷积操作模块
    conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, bias=False)
    weight = torch.tensor(weight, dtype=torch.float32)
    feature = torch.tensor(feature, dtype=torch.float32)
    conv.weight = nn.Parameter(weight)
    conv_result = conv(feature)

    return conv_result
