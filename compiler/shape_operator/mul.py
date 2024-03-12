import numpy as np

from compiler.shape_operator.base_shape import BaseShape


class Mul(BaseShape):
    """
    Mul操作
    继承BaseShape类

    Mul:点乘,左输入特征图*右输入特征图,且其中一个特征图通道维度为1
    例:1,256,40,40 * 1,256,1,1 => 1,256,40,40

    对于左输入特征图：r1 = s1(q1-z1)
    对于右输入特征图：r2 = s2(q2-z2)
    对于输出特征图：r3 = r1*r2 ; q3 = r3/s3+z3
    可得出 q3 = (s1*s2/s3)*(q1*q2) + (s1*s2/s3){(z1*z2)+[(s3/s1*s2)*z3]-(q1*z2)-(q2*z1)}

    """
    def __init__(self, para, feature, option, shared):
        super().__init__(para, feature, option, shared)

    def get_shape_reg2(self):

        l_scale = self.para["l_scale"]
        l_scale = np.load(l_scale)
        r_scale = self.para["r_scale"]
        r_scale = np.load(r_scale)
        local_scale = self.para["local_scale"]
        local_scale = np.load(local_scale)

        scale = np.round((l_scale * r_scale / local_scale) * (2 ** 16))
        scale = scale.astype(np.uint32).item()

        scale = format(scale, "032b")
        conv_reg2 = scale

        return conv_reg2

    def get_shape_reg3(self):
        l_scale = self.para["l_scale"]
        l_scale = np.load(l_scale)
        l_zp = self.para["l_zp"]
        l_zp = np.load(l_zp)
        r_scale = self.para["r_scale"]
        r_scale = np.load(r_scale)
        r_zp = self.para["r_zp"]
        r_zp = np.load(r_zp)
        local_scale = self.para["local_scale"]
        local_scale = np.load(local_scale)
        local_zp = self.para["local_zp"]
        local_zp = np.load(local_zp)

        # 计算 z1*z2+s3*z3/(s2*s1)
        zero_point = l_zp * r_zp + (local_scale * local_zp) / (l_scale * r_scale)
        zero_point = np.round(zero_point * (2 ** 16))
        zero_point = zero_point.astype(np.uint32).item()

        zero_point = format(zero_point, "032b")

        conv_reg3 = zero_point
        return conv_reg3

    def get_shape_reg4(self):
        l_zp = self.para["l_zp"]
        l_zp = np.load(l_zp)

        l_zp = format(l_zp, "032b")
        conv_reg4 = l_zp

        return conv_reg4

    def get_shape_reg5(self):
        r_zp = self.para["r_zp"]
        r_zp = np.load(r_zp)

        r_zp = format(r_zp, "032b")
        conv_reg5 = r_zp
        return conv_reg5

    def get_dma_write(self):
        l_feature_shape = self.l_feature_shape

        r_feature_shape = self.r_feature_shape
        write_l_size = l_feature_shape[0] * l_feature_shape[1] * l_feature_shape[2] * l_feature_shape[3]
        write_r_size = r_feature_shape[0] * r_feature_shape[1] * r_feature_shape[2] * r_feature_shape[3]

        write_address = self.shared.write_address
        if write_l_size > write_r_size:
            write_size = write_l_size
        else:
            write_size = write_r_size

        write_address = format(write_address, "032b")
        write_size = format(write_size, "032b")

        return write_address, write_size