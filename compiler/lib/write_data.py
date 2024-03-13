import os

import numpy as np
import torch
from compiler.lib.add_channel import add_weight, add_array
from compiler.lib.array_format import convert_scale, convert_bias, convert_weight


# 指令集字典
ins_address = {'TJPU_Conv_State': '00', 'TJPU_Conv_Control': '04',
               'TJPU_Shape_State': '08', 'TJPU_Shape_Control': '0C',
               'TJPU_Conv_Reg0': '10', 'TJPU_Conv_Reg1': '14', 'TJPU_Conv_Reg2': '18',
               'TJPU_Conv_Reg3': '1C', 'TJPU_Conv_Reg4': '20', 'TJPU_Shape_Reg0': '24',
               'TJPU_Shape_Reg1': '28', 'TJPU_Shape_Reg2': '2C',
               'TJPU_Shape_Reg3': '30', 'TJPU_Shape_Reg4': '34', 'TJPU_Shape_Reg5': '38',
               'TJPU_Conv_DMA_Write_Addr': '3C', 'TJPU_Conv_DMA_Write_Num': '40',
               'TJPU_Conv_DMA_Read_Addr': '44', 'TJPU_Conv_DMA_Read_Num': '48',
               'TJPU_Shape_DMA_Write_Addr': '4C', 'TJPU_Shape_DMA_Write_Num': '50',
               'TJPU_Shape_DMA_Read_Addr': '54', 'TJPU_Shape_DMA_Read_Num': '58',
               'TJPU_Shape1_DMA_Read_Addr': '5C', 'TJPU_Shape1_DMA_Read_Num': '60'}

'''
write_conv:写入卷积操作相关的指令文件
params:
    file_name:写入文件名
    data_package:计算结果字典
'''


def write_conv(file_name, data_package):
    with open(file_name, 'a+') as f:
        # ------------------写入权重的指令----------------------
        f.write('100000' + ins_address['TJPU_Conv_Reg0'])
        f.write('%08X' % int(data_package['conv_reg0']))
        f.write('\n')
        # 卷积reg1
        f.write('100000' + ins_address['TJPU_Conv_Reg1'])
        f.write('%08X' % int(data_package['conv_reg1']))
        f.write('\n')
        # 卷积reg2
        f.write('100000' + ins_address['TJPU_Conv_Reg2'])
        f.write('%08X' % int(data_package['conv_reg2']))
        f.write('\n')
        # 卷积reg3
        f.write('100000' + ins_address['TJPU_Conv_Reg3'])
        f.write('%08X' % int(data_package['conv_reg3']))
        f.write('\n')
        # 卷积reg4
        f.write('100000' + ins_address['TJPU_Conv_Reg4'])
        f.write('%08X' % int(data_package['conv_reg4']))
        f.write('\n')
        # DMA读权重地址
        f.write('100000' + ins_address['TJPU_Conv_DMA_Read_Addr'])
        f.write('%08X' % int(data_package['weight_address']))
        f.write('\n')
        # DMA读权重长度
        f.write('100000' + ins_address['TJPU_Conv_DMA_Read_Num'])
        f.write('%08X' % int(data_package['weight_size']))
        f.write('\n')
        # 卷积控制加载权重
        f.write('100000' + ins_address['TJPU_Conv_Control'])
        f.write('%08X' % 1)
        f.write('\n')
        # 读F检查状态
        f.write('110000' + ins_address['TJPU_Conv_State'])
        f.write('%08X' % 15)
        f.write('\n')
        # ------------------写入特征图的指令----------------------
        f.write('100000' + ins_address['TJPU_Conv_Reg0'])
        f.write('%08X' % int(data_package['conv_reg0']))
        f.write('\n')
        # 卷积reg1
        f.write('100000' + ins_address['TJPU_Conv_Reg1'])
        f.write('%08X' % int(data_package['conv_reg1']))
        f.write('\n')
        # 卷积reg2
        f.write('100000' + ins_address['TJPU_Conv_Reg2'])
        f.write('%08X' % int(data_package['conv_reg2']))
        f.write('\n')
        # 卷积reg3
        f.write('100000' + ins_address['TJPU_Conv_Reg3'])
        f.write('%08X' % int(data_package['conv_reg3']))
        f.write('\n')
        # 卷积reg4
        f.write('100000' + ins_address['TJPU_Conv_Reg4'])
        f.write('%08X' % int(data_package['conv_reg4']))
        f.write('\n')
        # DMA读图片地址
        f.write('100000' + ins_address['TJPU_Conv_DMA_Read_Addr'])
        f.write('%08X' % int(data_package['feature_address']))
        f.write('\n')
        # DMA读图片长度
        f.write('100000' + ins_address['TJPU_Conv_DMA_Read_Num'])
        f.write('%08X' % int(data_package['feature_size']))
        f.write('\n')
        # DMA写回结果地址
        f.write('100000' + ins_address['TJPU_Conv_DMA_Write_Addr'])
        f.write('%08X' % int(data_package['write_address']))
        f.write('\n')
        # DMA写回结果长度
        f.write('100000' + ins_address['TJPU_Conv_DMA_Write_Num'])
        f.write('%08X' % int(data_package['write_size']))
        f.write('\n')
        # 卷积控制 开始计算
        f.write('100000' + ins_address['TJPU_Conv_Control'])
        f.write('%08X' % 2)
        f.write('\n')
        # 读F检查状态
        f.write('110000' + ins_address['TJPU_Conv_State'])
        f.write('%08X' % 15)
        f.write('\n')


'''
write_shape:写入shape操作相关的指令文件
params:
    file_name:写入文件名
    data_package:计算结果字典
'''


def write_shape(file_name, data_package):
    with open(file_name, 'a+') as fp:
        fp.write('100000' + ins_address['TJPU_Shape_Reg0'])
        fp.write('%08X' % int(data_package['shape_reg0']))
        fp.write('\n')
        fp.write('100000' + ins_address['TJPU_Shape_Reg1'])
        fp.write('%08X' % int(data_package['shape_reg1']))
        fp.write('\n')
        fp.write('100000' + ins_address['TJPU_Shape_Reg2'])
        fp.write('%08X' % int(data_package['shape_reg2']))
        fp.write('\n')
        fp.write('100000' + ins_address['TJPU_Shape_Reg3'])
        fp.write('%08X' % int(data_package['shape_reg3']))
        fp.write('\n')
        fp.write('100000' + ins_address['TJPU_Shape_Reg4'])
        fp.write('%08X' % int(data_package['shape_reg4']))
        fp.write('\n')
        fp.write('100000' + ins_address['TJPU_Shape_Reg5'])
        fp.write('%08X' % int(data_package['shape_reg5']))
        fp.write('\n')

        fp.write('100000' + ins_address['TJPU_Shape_DMA_Read_Addr'])
        fp.write('%08X' % int(data_package['l_feature_address']))
        fp.write('\n')

        fp.write('100000' + ins_address['TJPU_Shape_DMA_Read_Num'])
        fp.write('%08X' % int(data_package['l_feature_size']))
        fp.write('\n')

        fp.write('100000' + ins_address['TJPU_Shape1_DMA_Read_Addr'])
        fp.write('%08X' % int(data_package['r_feature_address']))
        fp.write('\n')

        fp.write('100000' + ins_address['TJPU_Shape1_DMA_Read_Num'])
        fp.write('%08X' % int(data_package['r_feature_size']))
        fp.write('\n')

        fp.write('100000' + ins_address['TJPU_Shape_DMA_Write_Addr'])
        fp.write('%08X' % int(data_package['write_address']))
        fp.write('\n')

        fp.write('100000' + ins_address['TJPU_Shape_DMA_Write_Num'])
        fp.write('%08X' % int(data_package['write_size']))
        fp.write('\n')

        # TJPU_Shape_Control
        fp.write('100000' + ins_address['TJPU_Shape_Control'])
        fp.write('%08X' % int(data_package['shape_control_reg']))
        fp.write('\n')
        # 读F检查状态
        fp.write('110000' + ins_address['TJPU_Shape_State'])
        fp.write('%08X' % 15)
        fp.write('\n')


'''
gen_coe:写入中间结果文件
params:
    coe_name: 写入的coe文件名称
    result: 想要写入的结果,api结果需要调用int_repr()方法
    parallel:fpga的通道并行数
'''


def gen_coe(coe_name, result, parallel):
    shape = result.shape
    out = []
    print('start gen coe file:{}'.format(coe_name))
    with open(coe_name, "w+") as fp:
        for r in range(shape[2]):  # hang
            for c in range(shape[3]):  # lie
                for ch in range(shape[1]):  # channel
                    for n in range(shape[0]):  # image_num
                        out.append(result[n][ch][r][c])
                        if len(out) == parallel:  # 8（一行16g），1（一行两位）
                            out.reverse()
                            for m in out:
                                m = m.item()
                                fp.write('%02x' % m)
                            fp.write('\n')
                            out = []


'''
gen_coe_add:写入补好通道的中间结果文件
params:
    coe_name: 写入的coe文件名称
    result: 想要写入的结果,api结果需要调用int_repr()方法
    add_num:补通道时需要的数字,一般为z3的值
    add_channel:补完通道后的通道数
    parallel:fpga的通道并行数
'''


def gen_coe_add(coe_name, result, add_num, add_channel, parallel):
    shape = result.shape
    out = []
    print('start gen coe-bin-dat-npy-convert file:{}'.format(coe_name))
    with open(coe_name, "w+") as fp:
        for r in range(shape[2]):  # hang
            for c in range(shape[3]):  # lie
                for ch in range(add_channel):  # channel
                    for n in range(shape[0]):  # image_num
                        if ch < shape[1]:
                            out.append(result[n][ch][r][c])
                        else:
                            out.append(torch.tensor(add_num))
                        if len(out) == parallel:  # 16(一行32位数字),8(一行16位数字),1(一行2位数字)
                            out.reverse()
                            for m in out:
                                m = m.item()
                                fp.write('%02x' % m)
                            fp.write('\n')
                            out = []


'''
gen_coe:写入权重文件
params:
    weight_package:权重结果字典
'''


def write_weight(weight_package):
    parallel = weight_package["parallel"]
    out = []
    with open(weight_package["file_name"], "a+") as fp:  # 写入weight
        for r in weight_package["weight"]:
            out.append(r)
            if len(out) == parallel:
                out.reverse()
                for m in out:
                    m = m.item()
                    fp.write('%02x' % m)
                fp.write('\n')
                out = []
        for r in weight_package["bias"]:  # 写入bias
            out.append(r)
            if len(out) == parallel / 4:
                out.reverse()
                for m in out:
                    fp.write('%08x' % int(m))
                fp.write('\n')
                out = []
        for r in weight_package["scale"]:  # 写入SCALE
            out.append(r)
            if len(out) == parallel / 4:
                out.reverse()
                for m in out:
                    m = m.item()
                    fp.write('%08x' % m)
                fp.write('\n')
                out = []
        for r in weight_package["shift"]:  # 写入shift
            out.append(r)
            if len(out) == parallel / 4:
                out.reverse()
                for m in out:
                    m = m.item()
                    fp.write('%08x' % m)
                fp.write('\n')
                out = []


'''
coe2bin:通过coe文件写入bin文件
params:
    coe_path:读取coe文件路径
    bin_path:写入bin文件路径
'''


def coe2bin(coe_path, bin_path):
    torch.set_printoptions(profile="full")
    out_api = open(coe_path)
    out_api = out_api.read().splitlines()

    data_size = int(len(out_api) * len(out_api[0]) / 2)
    p = 0
    # data_size   B的个数
    bin_feature = np.zeros(data_size, dtype=np.uint8, order='C')
    for index in range(len(out_api)):
        data = out_api[index].rsplit(',')
        data = data[0]
        # data[0] = ['str'] ,list 变为str需要取其[0]
        tmp = ''
        out = []
        for index2 in range(len(data)):
            tmp += data[index2]
            if len(tmp) % 2 == 0:
                out.append(tmp)
                tmp = ''
            out.reverse()
        for index3 in range(int(len(data) / 2)):
            bin_feature[p] = int(out[index3], 16)
            p += 1
    write_path1 = bin_path
    fp1 = open(write_path1, "ab+")  # 打开fp1 ab+追加写入
    bin_feature.tofile(fp1)
    fp1.close()


'''
clear_files:删除目标文件夹中的所有文件
params:
    folder_path:文件夹路径
'''


def clear_files(folder_path):
    # 确保路径存在且是一个文件夹
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # 获取文件夹中的所有文件
        files = os.listdir(folder_path)
        # 遍历文件夹中的每个文件
        for file_name in files:
            # 构建文件的完整路径
            file_path = os.path.join(folder_path, file_name)
            # 判断路径是否是文件
            if os.path.isfile(file_path):
                # 删除文件
                os.remove(file_path)

'''
get_feature_count:在特征图字典里查找特征图id对应的层数
params:
    feature_id:特征图id
    layer_table:特征图字典
return:
    feature_count:特征图所对应的层数
'''


def get_feature_count(feature_id, layer_table):
    feature_count = 1
    # 在特征图字典里查找特征图id对应的层数
    for key, value in layer_table.items():
        if feature_id == key:
            feature_count = value
    return feature_count



'''
get_weight:计算scale、bias、shift、weight并写入文件
params:
    para:npy文件路径字典
    parallel:通道并行数
    file_name:写入文件名
'''
def get_weight(para, parallel, file_name):
    pre_scale = np.load(para['pre_scale'])
    pre_zp = np.load(para['pre_zp'])
    local_weight = np.load(para['local_weight_int'])
    local_weight_scale = np.load(para['local_weight_scale'])
    local_scale = np.load(para['local_scale'])
    local_bias = np.load(para['local_bias'])

    # 计算移位之后的scale和移了多少位  scale = (s1 * s2) / s3
    scale, shift = convert_scale(pre_scale, local_weight_scale, local_scale)
    # 计算新的bias bias = symbol(符号位) + data_decimal(移位值) + data_integer(移位之后的bias)
    bias = convert_bias(pre_zp, pre_scale, local_weight_scale, local_weight, local_bias)

    parallel = parallel
    local_weight = add_weight(local_weight, parallel)

    weight = convert_weight(local_weight, parallel)
    scale = add_array(scale, parallel)
    shift = add_array(shift, parallel)
    bias = add_array(bias, parallel)
    weight_package = {
        "file_name": file_name,
        "weight": weight,
        "bias": bias,
        "scale": scale,
        "shift": shift,
        "parallel": parallel
    }
    write_weight(weight_package)
