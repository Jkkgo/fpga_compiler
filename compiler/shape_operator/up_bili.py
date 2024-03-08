import os

import numpy as np

from lib.add_channel import add_feature_shape, add_feature
from lib.write_data import gen_coe, gen_coe_add
from shape_operator.base_shape import BaseShape


class UpBili(BaseShape):
    """
    UpBili操作
    继承BaseShape类

    UpBili:双线性插值上采样,相比于普通上采样,数据的边缘排布更加平滑
    该操作的fpga具体实现比较复杂,由雨神一手操办,雨神永远是神

    例:1,8,80,80 => 1,8,640,640

    """
    def __init__(self, para, feature, option, shared):
        super().__init__(para, feature, option, shared)
        self.l_feature_shape = add_feature_shape(feature[0], 8)
        self.shape_control = shared.shape_control["UpBili"]

    def get_dma_write(self):
        feature_shape = self.l_feature_shape

        write_address = self.shared.write_address
        write_size = feature_shape[0] * feature_shape[1] * feature_shape[2] * feature_shape[3]
        write_size = write_size * 64

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
        result_shape = mid_result.shape

        local_zp = self.para['local_zp']
        local_zp = np.load(local_zp).astype(int).item()

        parallel = 8
        add_channel = int(parallel * np.ceil(result_shape[1] / parallel))

        layer_count = str(self.shared.layer_count)
        file_name = 'auto_result' + layer_count + '.coe'
        file_path = self.shared.file_path + 'mid_result'
        os.makedirs(file_path, exist_ok=True)

        file_name = "{}/{}".format(file_path, file_name)
        if self.shared.generate_mode[0] == 1:
            gen_coe_add(file_name, mid_result.int_repr(), local_zp, add_channel, parallel)
        elif self.shared.layer_count == self.shared.generate_mode[1]:
            gen_coe_add(file_name, mid_result.int_repr(), local_zp, add_channel, parallel)

