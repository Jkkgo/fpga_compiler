import os

import numpy as np
import torch

from compiler.lib.add_channel import add_feature_shape
from compiler.lib.write_data import gen_coe_add
from compiler.shape_operator.base_shape import BaseShape


class ArgMax(BaseShape):

    """
    ArgMax操作
    继承BaseShape类

    ArgMax:按通道查找矩阵中的最大值所在位置
    例:1,8,640,640=>1,1,640,640

    """
    def __init__(self, para, feature, option, shared):
        super().__init__(para, feature, option, shared)
        self.l_feature_shape = add_feature_shape(feature[0], 8)
        self.shape_control = shared.shape_control["ArgMax"]

    def get_dma_write(self):
        feature_shape = self.l_feature_shape

        write_address = self.shared.write_address
        write_size = feature_shape[0] * feature_shape[1] * feature_shape[2] * feature_shape[3]
        write_size = int(write_size / 64)

        write_address = format(write_address, "032b")
        write_size = format(write_size, "032b")

        return write_address, write_size

    def get_shape_control(self):
        shape_control = self.shape_control
        shape_control = format(shape_control, '04b')

        shape_control_reg = shape_control.zfill(32)
        return shape_control_reg

    def write_result_file(self):
        mid_result = self.feature[1]
        mid_result = mid_result.to(torch.int)

        layer_count = str(self.shared.layer_count)
        file_name = 'auto_result' + layer_count + '.coe'
        file_path = self.shared.file_path + 'mid_result'
        os.makedirs(file_path, exist_ok=True)
        file_name = "{}/{}".format(file_path, file_name)
        if self.shared.generate_mode[0] == 1:
            gen_coe_add(file_name, mid_result, 1, 1, 1)
        elif self.shared.layer_count == self.shared.generate_mode[1]:
            gen_coe_add(file_name, mid_result, 1, 1, 1)

