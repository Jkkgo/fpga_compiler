import numpy as np
import torch
from torch import nn

from compiler.lib.ins_format import leaky_format

'''
sim_leaky_relu:仿真FPGA的leaky_relu操作
params:
    quant_result: 移位运算仿真结果
    local_zp: 本层zp值(z3)
    local_scale: 本层scale值(s3)
    leakyRatio: leaky_relu的斜率(0.1或者0.01)
return:
    relu_result:leaky_relu计算结果

具体计算过程: leaky_relu(quant_result-local_zp)+local_zp
'''


def sim_leaky_relu(quant_result, local_zp, local_scale, leakyRatio):
    quant_result = quant_result.astype(np.int64)
    relu_result = quant_result - local_zp
    # 计算修正字符串
    out_str = leaky_format(local_scale)

    # 使用向量化操作将函数应用到数组的每个元素
    # 仿真硬件的amend修正
    relu_result = np.vectorize(sim_amend)(relu_result, out_str, leakyRatio)
    relu_result += local_zp

    # 控制边界值
    relu_result[relu_result < 0] = 0
    relu_result[relu_result > 255] = 255
    relu_result = relu_result.astype(np.uint8)
    return relu_result


'''
sim_amend:仿真FPGA leaky_relu中x负半轴的修正操作
params:
    num: 传入的数字
    out_str: 修正字符串
    leakyRatio: leaky_relu的斜率(0.1或者0.01)
return:
    num: 修正后的数字
具体计算过程: 较为复杂,可以宏观的理解为四舍五入操作
'''


def sim_amend(num, out_str, leakyRatio):
    # 特殊值表,对一些没有规律的特殊值做查表工作
    map_table = [-5, -15, -25, -35, -45, -55, -65, -75, -85, -95, -105, -115, -125, -135, -145, -155]

    if num >= 0:
        pass
    else:
        map_temp = num
        num = num * int(leakyRatio * 2 ** 17)

        # 按位与操作,将负十进制数转换为有符号二进制数,即高位为1,其他位是补码
        num_complement = format(num & 0xffffffff, '032b')
        # 翻转二进制数方便取位
        num_complement = num_complement[::-1]
        # 取13到17位,用于复杂的四舍五入
        mantissa = num_complement[13:17][::-1]
        # 将补码二进制数转换为正常形式
        mantissa = int(mantissa, 2) & 0b1111

        # 四位小数的四舍五入操作
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

        # 特殊值查表
        if map_temp not in map_table:
            pass
        # 如果有特殊值，则根据out_str做修正
        else:
            index = map_table.index(map_temp)
            if out_str[2 * index:2 * index + 2] == '10':
                num = num - 1
            elif out_str[2 * index:2 * index + 2] == '01':
                num = num + 1
            else:
                pass

    return num


'''
sim_round:仿真FPGA的四舍五入操作
params:
    num: 传入的数字
return:
    num: 四舍五入后的数字
具体计算过程: (num>>1)+1 or num>>1
'''


def sim_round(num):
    # 如果最后一位是1，则将该数与二进制的1进行按位或运算
    if num & 1:
        # 移位加1实现五入
        num = num >> 1
        num = num + 1
    else:
        # 移位实现四舍
        num = num >> 1

    return num


'''
sim_conv:仿真FPGA conv运算之后的移位运算
params:
    conv_result: conv仿真结果
    bias: 定点化、计算融合后的偏置
    bias_shift: bias的移位信息
    scale: 定点化、计算融合后的scale
    scale_shift: scale的移位信息
    local_zp: z3值
return:
    quant_result:移位运算仿真结果
具体计算过程: (((quant_result<<16)+(bias<<(16-bias_shift)))*scale)>>(scale_shift+1+32+16)
'''


def sim_conv_quant(conv_result, bias, bias_shift, scale, scale_shift, local_zp):
    np.seterr(divide="ignore")
    quant_result = conv_result.detach().numpy().astype(np.float64)
    for ch in range(conv_result.shape[1]):
        quant_result[0][ch] *= 2 ** 16
        quant_result[0][ch] += bias[ch] << (16 - bias_shift[ch])
        quant_result[0][ch] *= scale[ch]
        # 少移1位是为了对最后一位做四舍五入
        quant_result[0][ch] /= 2 ** (scale_shift[ch] + 32 + 16)
    quant_result = np.floor(quant_result).astype(np.int64)
    # 使用向量化操作将函数应用到数组的每个元素
    # 对低位做四舍五入
    quant_result = np.vectorize(sim_round)(quant_result)
    quant_result += local_zp
    # 控制边界值
    quant_result[quant_result < 0] = 0
    quant_result[quant_result > 255] = 255
    quant_result = quant_result.astype(np.uint8)
    return quant_result


'''
sim_conv:仿真FPGA的conv运算
params:
    feature:输入特征图
    weight:本层权重
    option:本层操作信息,包含了stride、padding等信息
    pre_zp:做padding时需要补的值,因为是量化网络，所以补的是z1的值
return:
    conv_result:conv仿真结果
'''


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
    conv = nn.Conv2d(c_in, c_out, kernel_size=weight.shape[3], stride=stride, bias=False)
    weight = torch.tensor(weight, dtype=torch.float32)
    feature = torch.tensor(feature, dtype=torch.float32)
    conv.weight = nn.Parameter(weight)
    conv_result = conv(feature)

    return conv_result


'''
sim_concat_quant:仿真FPGA的concat运算
params:
    feature:输入的特征图
    zp:定点化、计算融合后的zp值
    scale:定点化、计算融合后的scale值
return:
    concat_result:仿真结果
    
具体计算过程: (((feature<<16)+zp)*scale)>>32
'''


def sim_concat_quant(feature, zp, scale):
    concat_result = feature.astype(np.int64)
    concat_result <<= 16
    concat_result += zp
    concat_result *= scale
    # 只移31位是为了对最后一位做四舍五入
    concat_result >>= (15 + 16)
    # 对低位做四舍五入
    concat_result = np.vectorize(sim_round)(concat_result)
    # 控制边界值
    concat_result[concat_result < 0] = 0
    concat_result[concat_result > 255] = 255
    concat_result = concat_result.astype(np.uint8)

    return concat_result


'''
max_pooling:实现最大池化操作
params:
    feature:输入的特征图
    pool_size:池化窗口大小,默认为 (2,2)
    strides:池化窗口的步长,默认为 (2,2)
return:
    result:池化后的结果
'''


def max_pooling(feature, pool_size=(2, 2), strides=(2, 2)):
    # 获取输入数组的形状
    batch, channel, row, col = feature.shape

    # 获取池化窗口的大小和步长
    pool_row, pool_col = pool_size
    stride_row, stride_col = strides

    # 计算输出数组的形状
    output_row = (row - pool_row) // stride_row + 1
    output_col = (col - pool_col) // stride_col + 1

    # 初始化输出数组
    result = np.zeros((batch, channel, output_row, output_col), dtype=np.uint8)

    # 对输入数组进行池化
    for ch in range(channel):
        for r in range(output_row):
            for c in range(output_col):
                # 计算池化窗口在输入数组上的索引范围
                c_start = c * stride_col
                c_end = c_start + pool_col
                r_start = r * stride_row
                r_end = r_start + pool_row

                # 从输入数组中获取池化窗口，并找到窗口中的最大值
                window = feature[0, ch, r_start:r_end, c_start:c_end]
                result[0, ch, r, c] = np.max(window)

    return result


'''
up_sample:最近邻上采样操作
params:
feature:输入的特征图
scale_factor:上采样比例,默认为2
return:
output:上采样后的结果
'''


def up_sample(feature, scale_factor=2):
    # 获取输入数组的形状
    batch, channel, row, col = feature.shape

    # 计算输出数组的形状
    output_row = row * scale_factor
    output_col = col * scale_factor

    # 初始化输出数组
    output = np.zeros((batch, channel, output_row, output_col), dtype=np.uint8)

    # 最近邻上采样：将每个输入像素的值复制到输出数组的相应位置
    for ch in range(channel):
        for r in range(output_row):
            for c in range(output_col):
                x = int(r / scale_factor)
                y = int(c / scale_factor)
                output[0, ch, r, c] = feature[0, ch, x, y]

    return output
