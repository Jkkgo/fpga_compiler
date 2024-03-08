import os

import numpy as np
import torch

from compiler.transit import Transit

from compiler.transit import shared

'''
block:卷积分块操作,将卷积操作按照通道n等分之后再分别计算
params:传入参数即为主函数的所有参数
    para1:输入层的npy文件前缀名 
    para2:本层的npy文件前缀名
    feature:[输入特征图, 输出特征图]
    option:['卷积操作名',stride,padding,是否使用激活函数,分块数量]
'''


def block(para1='', para2='', feature=None, option=None):
    block_num = option[4]

    # 将输出特征图进行分块
    output_block = block_feature(feature[1], block_num)
    # 将权重进行分块
    para_name = block_weight(para2, block_num)

    # 首先对分块后的输入特征图做卷积操作
    for i in range(block_num):
        Transit(para1=para1, para2=para_name[i],
                feature=[feature[0], output_block[i]],
                option=option)

    # 再对卷积操作后的输出特征图进行concat操作
    for i in range(block_num - 1):
        if i != block_num - 2:
            feature_data = torch.cat([output_block[2 * i], output_block[2 * i + 1]], dim=1)
            output_block.append(feature_data)
            Transit(para1=para1, para2=para1, para3=para1,
                    feature=[output_block[2 * i], feature_data, output_block[2 * i + 1]],
                    option=["Concat"])
        else:
            Transit(para1=para1, para2=para1, para3=para1,
                    feature=[output_block[2 * i], feature[1], output_block[2 * i + 1]],
                    option=["Concat"])


'''
block_weight:权重分块操作
params:
    local_para:本层的npy文件前缀名
    block_num:分块数量
return: 
    para_name:分好块的npy文件前缀名

'''


def block_weight(local_para, block_num):
    para_path = shared.para_path + local_para

    local_weight_scale = np.load(para_path + ".weight.scale.npy")
    local_weight_zp = np.load(para_path + ".weight.zero_point.npy")
    local_weight = np.load(para_path + ".weight.npy")
    local_weight_int = np.load(para_path + ".weight.int.npy")
    local_scale = np.load(para_path + ".scale.npy")
    local_zp = np.load(para_path + ".zero_point.npy")
    local_bias = np.load(para_path + ".bias.npy")

    block_shape_out = local_weight_int.shape[0] // block_num

    para_name = []
    for i in range(block_num):
        block_name = para_path + '_' + str(i)
        start_channel_out = i * block_shape_out
        end_channel_out = (i + 1) * block_shape_out

        np.save(block_name + ".weight.scale.npy", local_weight_scale[start_channel_out:end_channel_out])
        np.save(block_name + ".weight.zero_point.npy", local_weight_zp[start_channel_out:end_channel_out])
        np.save(block_name + ".weight.npy",
                local_weight[start_channel_out:end_channel_out, :, :, :])
        np.save(block_name + ".weight.int.npy",
                local_weight_int[start_channel_out:end_channel_out, :, :, :])
        np.save(block_name + ".scale.npy", local_scale)
        np.save(block_name + ".zero_point.npy", local_zp)
        np.save(block_name + ".bias.npy", local_bias[start_channel_out:end_channel_out])
        para_name.append(local_para + '_' + str(i))
    return para_name


'''
block_feature:对特征图按照通道进行n等分
params:
    feature:特征图
    block_num:分块数量
return:
    split_data:分好块的特征图数组
'''


def block_feature(feature, block_num):
    block_shape = feature.shape[1] // block_num

    # 切片操作来分割通道
    split_data = []
    for i in range(block_num):
        start_channel = i * block_shape
        end_channel = (i + 1) * block_shape
        split_data.append(feature[:, start_channel:end_channel, :, :])
    return split_data
