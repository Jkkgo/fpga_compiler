from shape_operator.base_shape import BaseShape


class UpSample(BaseShape):
    """
    UpSample操作
    继承BaseShape类

    UpSample:普通上采样,将输入特征图的一个像素点增加为四个像素点

    例:1,8,80,80 => 1,8,160,160

    """
    def __init__(self, para, feature, option, shared):
        super().__init__(para, feature, option, shared)
        self.shape_control = shared.shape_control["UpSample"]

    def get_dma_write(self):
        feature_shape = self.l_feature_shape

        write_address = self.shared.write_address
        write_size = feature_shape[0] * feature_shape[1] * feature_shape[2] * feature_shape[3]
        write_size = write_size * 4

        write_address = format(write_address, "032b")
        write_size = format(write_size, "032b")

        return write_address, write_size

    def get_shape_control(self):
        shape_control = self.shape_control
        shape_control = format(shape_control, '04b')

        shape_control_reg = shape_control.zfill(32)
        return shape_control_reg
