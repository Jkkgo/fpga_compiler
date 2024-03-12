import os
import shutil

import numpy as np

from compiler.lib.write_data import clear_files, write_conv, gen_coe_add, coe2bin, get_weight, write_shape


class BaseWrite:
    """
    base操作的父类
    规定了conv和shape中一些通用的方法
    """

    '''
    __init__:初始化方法
    params:
        para: npy数据存放路径
        feature: 特征图数据
        option: [卷积类型,步长,padding,激活函数]
        shared: 共享变量集合
    '''

    def __init__(self, para, feature, option, shared):
        self.para = para
        self.feature = feature
        self.option = option
        self.shared = shared

    '''
    write_ins_file:写指令文件
    params:
        data_package: 计算结果字典
    '''

    def write_ins_file(self, data_package):
        layer_count = str(self.shared.layer_count)
        dat_name = 'auto_ins' + layer_count + '.dat'
        file_path = self.shared.file_path + 'ins'
        os.makedirs(file_path, exist_ok=True)
        file_name = "{}/{}".format(file_path, dat_name)

        # 如果是首层，则清空ins文件夹
        if layer_count == '1':
            clear_files(file_path)

        # 如果是联测,则将前一层指令文件复制到本层,再去追加写入本层指令
        if self.shared.generate_mode[0] == 1 and self.shared.layer_count != 1:
            pre_file = file_path + '/auto_ins' + str(self.shared.layer_count - 1) + '.dat'
            shutil.copyfile(pre_file, file_name)

        if 'Conv' in self.option[0]:
            # 写入指令
            write_conv(file_name, data_package)
        else:
            # 写入指令
            write_shape(file_name, data_package)

    '''
    write_result_file:写中间结果文件
    '''

    def write_result_file(self):
        mid_result = self.feature[1]
        result_shape = mid_result.shape

        local_zp = self.para['local_zp']
        local_zp = np.load(local_zp).astype(int).item()

        parallel = self.shared.parallel
        # 判断是否需要补通道,补完通道的形状是多少 np.ceil向上取整
        add_channel = int(parallel * np.ceil(result_shape[1] / parallel))

        layer_count = str(self.shared.layer_count)
        file_name = 'auto_result' + layer_count + '.coe'
        file_path = self.shared.file_path + 'mid_result'
        os.makedirs(file_path, exist_ok=True)
        file_name = "{}/{}".format(file_path, file_name)

        # 如果是联测，则直接生成
        if self.shared.generate_mode[0] == 1:
            gen_coe_add(file_name, mid_result.int_repr(), local_zp, add_channel, parallel)
        # 如果是单测，则只生成一层的中间结果
        elif self.shared.layer_count == self.shared.generate_mode[1]:
            gen_coe_add(file_name, mid_result.int_repr(), local_zp, add_channel, parallel)

        # 如果是首层，则还生成输入bin
        if self.shared.layer_count == 1:
            input_feature = self.feature[0].int_repr()
            channel = self.feature[0].shape[1]
            input_path = file_path + "/auto_input.coe"
            bin_path = file_path + "/auto_input.bin"
            gen_coe_add(input_path, input_feature, 1, channel, channel)
            coe2bin(input_path, bin_path)

    '''
    write_weight_file:写权重文件
    '''

    def write_weight_file(self):

        layer_count = str(self.shared.layer_count)
        coe_name = 'auto_weight' + layer_count + '.coe'
        bin_name = '/auto_weight' + layer_count + '.bin'
        file_path = self.shared.file_path + 'weight'
        os.makedirs(file_path, exist_ok=True)
        file_name = "{}/{}".format(file_path, coe_name)

        # 如果是首层，则清空weight文件夹
        if layer_count == '1':
            clear_files(file_path)

        # 如果是联测,则将前一层权重文件复制到本层,再去追加写入本层权重
        if self.shared.generate_mode[0] == 1 and self.shared.layer_count != 1:
            pre_file = file_path + '/auto_weight' + str(self.shared.layer_count - 1) + '.coe'
            shutil.copyfile(pre_file, file_name)

        if 'Conv' in self.option[0]:
            get_weight(self.para, self.shared.parallel, file_name)
        else:
            with open(file_name, 'a'):
                pass
        # 如果是指定层数，则将该层coe格式权重转为bin
        if self.shared.layer_count == self.shared.generate_mode[1]:
            coe2bin(file_name, file_path + bin_name)
