import numpy as np

from shape_operator.base_shape import BaseShape


class MeanPool(BaseShape):
    """
    MeanPool操作
    继承BaseShape类

    MeanPool:全局均值池化,按通道计算输入特征图的平均值
    例:1,256,40,40=>1,256,1,1

    fpga实现方式:特征图按通道求和的值*[1/(feature_shape[2] * feature_shape[3])]

    """
    def __init__(self, para, feature, option, shared):
        super().__init__(para, feature, option, shared)
        self.shape_control = shared.shape_control["MeanPool"]

    def get_shape_reg2(self):
        feature_shape = self.l_feature_shape
        quant_mean = np.round((1 / (feature_shape[2] * feature_shape[3])) * (2 ** 16))
        quant_mean = quant_mean.astype(np.uint32).item()
        quant_mean = format(quant_mean, '032b')

        conv_reg2 = quant_mean
        return conv_reg2

    def get_dma_write(self):
        feature_shape = self.l_feature_shape

        write_address = self.shared.write_address
        write_size = feature_shape[0] * feature_shape[1]

        write_address = format(write_address, "032b")
        write_size = format(write_size, "032b")

        return write_address, write_size

    def get_shape_control(self):
        shape_control = self.shape_control
        shape_control = format(shape_control, '04b')

        shape_control_reg = shape_control.zfill(32)
        return shape_control_reg
