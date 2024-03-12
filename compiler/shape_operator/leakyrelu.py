import numpy as np

from compiler.lib.ins_format import leaky_format
from compiler.shape_operator.base_shape import BaseShape


class LeakyRelu(BaseShape):
    """
    LeakRelu操作
    继承BaseShape类

    LeakRelu:对输入特征图做leaky_relu激活函数,由于是量化后进行操作，因此零值为l_zp

    """
    def __init__(self, para, feature, option, shared):
        super().__init__(para, feature, option, shared)

    def get_shape_reg2(self):
        l_zp = self.para["l_zp"]
        l_zp = np.load(l_zp)

        l_zp = format(l_zp, "032b")

        conv_reg2 = l_zp
        return conv_reg2

    def get_shape_reg3(self):
        l_scale = self.para["l_scale"]
        l_scale = np.load(l_scale)
        l_scale = leaky_format(l_scale)

        conv_reg3 = l_scale
        return conv_reg3

    def get_dma_write(self):
        feature_shape = self.l_feature_shape

        write_address = self.shared.write_address
        write_size = feature_shape[0] * feature_shape[1] * feature_shape[2] * feature_shape[3]

        write_address = format(write_address, "032b")
        write_size = format(write_size, "032b")

        return write_address, write_size