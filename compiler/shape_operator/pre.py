import os

import numpy as np
import torch

from compiler.lib.write_data import gen_coe_add, coe2bin, clear_files, get_feature_count
from compiler.shape_operator.base_shape import BaseShape


class Pre(BaseShape):
    """
    Pre操作
    继承BaseShape类

    Pre:预处理操作,对原始灰度图补通道,并做一次量化，大部分也会做一次归一化
    例:1,1,640,640 => 1,parallel,640,640

    对于归一化：r1 = [(q1/255)-mean]/std
    对于量化: q2 = r1/s1+z1
    可得出 q2 = q1/(255*std*s1) + (std*s1*z1-mean)/(std*s1)

    """
    def __init__(self, para, feature, option, shared):
        super().__init__(para, feature, option, shared)

    def get_shape_reg2(self):
        std = self.shared.std
        scale = self.para["local_scale"]
        scale = np.load(scale)

        shift = 2 ** 17
        scale_new = 1 / (255 * std * scale)
        scale_new = np.round(scale_new * shift).astype(np.uint32)
        scale_new = scale_new.item()

        scale_new = format(scale_new, "032b")

        conv_reg2 = scale_new
        return conv_reg2

    def get_shape_reg3(self):
        mean = self.shared.mean
        std = self.shared.std
        scale = self.para["local_scale"]
        scale = np.load(scale)
        zp = self.para["local_zp"]
        zp = np.load(zp)

        shift = 2 ** 17
        zp_new = (zp * std * scale - mean) / (std * scale)
        zp_new = np.round((zp_new + 0.5) * shift).astype(np.uint32)
        zp_new = zp_new.item()

        zp_new = format(zp_new, "032b")
        conv_reg3 = zp_new
        return conv_reg3

    def get_dma_read(self):
        feature = self.feature[0]
        feature_shape = feature.shape
        feature_id = id(feature)

        if self.shared.layer_count == 1:
            feature_address = self.shared.picture_address
        else:
            # 通过特征图对应层数来查找地址表中的地址
            feature_count = get_feature_count(feature_id, self.shared.layer_table)
            feature_address = self.shared.address_table[feature_count - 1]
        feature_size = feature_shape[0] * feature_shape[1] * feature_shape[2] * feature_shape[3]

        l_feature_address = format(feature_address, '032b')
        l_feature_size = format(feature_size, '032b')
        r_feature_address = format(0, '032b')
        r_feature_size = format(0, '032b')

        return l_feature_address, l_feature_size, r_feature_address, r_feature_size

    def get_dma_write(self):
        feature_shape = self.l_feature_shape

        write_address = self.shared.write_address
        write_size = feature_shape[0] * feature_shape[1] * feature_shape[2] * feature_shape[3]

        write_address = format(write_address, "032b")
        write_size = format(write_size, "032b")

        return write_address, write_size

    def write_result_file(self):
        mid_result = self.feature[1]
        result_shape = mid_result.shape

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
            gen_coe_add(file_name, mid_result.int_repr(), 0, add_channel, parallel)
        # 如果是单测，则只生成一层的中间结果
        elif self.shared.layer_count == self.shared.generate_mode[1]:
            gen_coe_add(file_name, mid_result.int_repr(), 0, add_channel, parallel)

        # 如果是首层，则还生成输入bin
        if self.shared.layer_count == 1:
            input_feature = self.feature[0]
            if type(self.feature[0]) == torch.Tensor:
                input_feature = self.feature[0].int_repr()
            channel = self.feature[0].shape[1]
            input_path = file_path + "/auto_input.coe"
            bin_path = file_path + "/auto_input.bin"
            gen_coe_add(input_path, input_feature, 1, channel, channel)
            coe2bin(input_path, bin_path)

