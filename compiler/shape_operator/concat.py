import numpy as np

from compiler.shape_operator.base_shape import BaseShape


class Concat(BaseShape):
    """
    Concat操作
    继承BaseShape类

    Concat:左输入特征图与右输入特征图批次、长、宽一致,两张图在通道维度堆叠得到输出特征图。
    例:1,256,40,40 堆叠 1,256,40,40 => 1,512,40,40
    对于左输入特征图：r1 = s1(q1-z1)
    对于右输入特征图：r2 = s2(q2-z2)
    对于输出特征图：r3 = r1堆叠r2 ; q3 = r3/s3+z3
    可得出 q3 = (s1/s3)*q1+(s1/s3)*[(s3/s1)z3-z1] 堆叠 (s2/s3)*q2+(s2/s3)*[(s3/s2)z3-z2]

    """

    def __init__(self, para, feature, option, shared):
        super().__init__(para, feature, option, shared)
        self.shape_control = shared.shape_control["Concat"]

    def get_shape_reg2(self):
        l_scale = self.para["l_scale"]
        l_scale = np.load(l_scale)
        local_scale = self.para["local_scale"]
        local_scale = np.load(local_scale)
        l_scale = np.round((l_scale / local_scale) * (2 ** 16))
        l_scale = l_scale.astype(np.uint32).item()

        l_scale = format(l_scale, "032b")

        conv_reg2 = l_scale
        return conv_reg2

    def get_shape_reg3(self):
        r_scale = self.para["r_scale"]
        r_scale = np.load(r_scale)
        local_scale = self.para["local_scale"]
        local_scale = np.load(local_scale)
        r_scale = np.round((r_scale / local_scale) * (2 ** 16))
        r_scale = r_scale.astype(np.uint32).item()

        r_scale = format(r_scale, "032b")

        conv_reg3 = r_scale
        return conv_reg3

    def get_shape_reg4(self):
        l_scale = self.para["l_scale"]
        l_scale = np.load(l_scale)
        l_zp = self.para["l_zp"]
        l_zp = np.load(l_zp)
        local_scale = self.para["local_scale"]
        local_scale = np.load(local_scale)
        local_zp = self.para["local_zp"]
        local_zp = np.load(local_zp)
        l_zp = (local_scale / l_scale) * local_zp - l_zp
        l_zp = np.round(l_zp * (2 ** 16))
        l_zp = l_zp.astype(np.uint32).item()

        l_zp = format(l_zp, "032b")

        conv_reg4 = l_zp
        return conv_reg4

    def get_shape_reg5(self):
        r_scale = self.para["r_scale"]
        r_scale = np.load(r_scale)
        r_zp = self.para["r_zp"]
        r_zp = np.load(r_zp)
        local_scale = self.para["local_scale"]
        local_scale = np.load(local_scale)
        local_zp = self.para["local_zp"]
        local_zp = np.load(local_zp)
        r_zp = (local_scale / r_scale) * local_zp - r_zp
        r_zp = np.round(r_zp * (2 ** 16))
        r_zp = r_zp.astype(np.uint32).item()
        r_zp = format(r_zp, "032b")

        conv_reg5 = r_zp
        return conv_reg5

    def get_dma_write(self):
        l_feature_shape = self.l_feature_shape
        r_feature_shape = self.r_feature_shape

        write_address = self.shared.write_address
        write_l_size = l_feature_shape[0] * l_feature_shape[1] * l_feature_shape[2] * l_feature_shape[3]
        write_r_size = r_feature_shape[0] * r_feature_shape[1] * r_feature_shape[2] * r_feature_shape[3]
        write_size = write_l_size + write_r_size

        write_address = format(write_address, "032b")
        write_size = format(write_size, "032b")

        return write_address, write_size

    def get_shape_control(self):
        shape_control = self.shape_control
        shape_control = format(shape_control, '04b')

        shape_control_reg = shape_control.zfill(32)
        return shape_control_reg
