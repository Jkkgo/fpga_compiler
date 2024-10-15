import os
import shutil
from abc import abstractmethod

import numpy as np
import torch

from compiler.lib.add_channel import add_feature, add_feature_shape
from compiler.lib.write_data import clear_files, write_conv, gen_coe_add, coe2bin, get_weight, write_shape, \
    get_feature_count, coe2np, write_weight


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
        __call__:实例化调用方法
    '''

    def __call__(self, *args, **kwargs):

        data_package = self.packing_data()

        if self.shared.layer_count <= self.shared.generate_mode[1]:
            if self.shared.gen_ins:
                self.write_ins_file(data_package)
            if self.shared.gen_weight:
                self.write_weight_file()
            if self.shared.gen_result:
                self.write_result_file()
            if self.shared.gen_simulate:
                self.write_simulate_file()

        self.update_shared(data_package)

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
            weight_package = get_weight(self.para, self.shared.parallel)
            write_weight(weight_package, file_name)

        else:
            with open(file_name, 'a'):
                pass
        # 如果是指定层数，则将该层coe格式权重转为bin
        if self.shared.layer_count == self.shared.generate_mode[1]:
            coe2bin(file_name, file_path + bin_name)

    '''
    write_simulate_file:写仿真文件
    '''

    def write_simulate_file(self):
        file_path = self.shared.file_path + 'simulate_result'
        layer_count = self.shared.layer_count
        coe_name = 'auto_simulate' + str(layer_count) + '.coe'
        file_name = "{}/{}".format(file_path, coe_name)
        file_path = self.shared.file_path + 'simulate_result'
        os.makedirs(file_path, exist_ok=True)
        parallel = self.shared.parallel
        feature = self.feature
        feature_id_l = id(feature[0])

        # 如果是联测，则直接生成
        if self.shared.generate_mode[0] == 1:
            if self.shared.layer_count == 1:
                feature_l = add_feature(feature[0], parallel)
            else:
                # 通过特征图对应层数来查找地址表中的地址
                feature_count = get_feature_count(feature_id_l, self.shared.layer_table)
                input_name = 'auto_simulate' + str(feature_count) + '.coe'
                input_path = "{}/{}".format(file_path, input_name)
                if os.path.exists(input_path):
                    feature_shape_l = add_feature_shape(feature[0], parallel)
                    feature_l = coe2np(input_path, feature_shape_l)
                # 如果输入coe不存在,则以feature为输入
                else:
                    feature_l = add_feature(feature[0], parallel)

            if "Conv" not in self.option[0] and feature[2] is not None:
                feature_id_r = id(feature[2])
                feature_shape_r = add_feature_shape(feature[2], parallel)
                feature_count_r = get_feature_count(feature_id_r, self.shared.layer_table)
                input_name_r = 'auto_simulate' + str(feature_count_r) + '.coe'
                input_path_r = "{}/{}".format(file_path, input_name_r)
                if os.path.exists(input_path_r):
                    feature_r = coe2np(input_path_r, feature_shape_r)
                # 如果输入coe不存在,则以feature为输入
                else:
                    feature_r = add_feature(feature[2], parallel)
                # 存入新的输入特征图
                feature[2] = feature_r
            # 存入新的输入特征图
            feature[0] = feature_l

            simulate_result = self.simulate(feature)
            if simulate_result is not None:
                gen_coe_add(file_name, simulate_result, 0, simulate_result.shape[1], parallel)

        # 如果是单测，则只生成一层的中间结果
        elif self.shared.layer_count == self.shared.generate_mode[1]:
            feature[0] = add_feature(feature[0], parallel)
            if "Conv" not in self.option[0] and feature[2] is not None:
                feature[2] = add_feature(feature[2], parallel)
            simulate_result = self.simulate(feature)
            if simulate_result is not None:
                gen_coe_add(file_name, simulate_result, 0, simulate_result.shape[1], parallel)

    '''
       packing_data:对计算结果进行打包,该方法为抽象方法,需要子类重写
    '''

    @abstractmethod
    def packing_data(self):
        pass

    '''
    update_shared:更新一些相关的共享变量,该方法为抽象方法,需要子类重写
    '''

    @abstractmethod
    def update_shared(self, data_package):
        pass

    '''
       simulate:模拟FPGA定点运算方式,该方法为抽象方法,需要子类重写
    '''

    @abstractmethod
    def simulate(self, feature):
        pass
