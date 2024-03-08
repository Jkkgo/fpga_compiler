from shape_operator.base_shape import BaseShape


class MaxPool(BaseShape):
    """
    MaxPool操作
    继承BaseShape类

    MaxPool:最大池化,对输入特征图的四个像素点作比较，选出一个最大值
    例: 1,256,40,40=>1,256,20,20

    """
    def __init__(self, para, feature, option, shared):
        super().__init__(para, feature, option, shared)
        self.shape_control = shared.shape_control["MaxPool"]

    def get_dma_write(self):
        feature_shape = self.l_feature_shape

        write_address = self.shared.write_address
        write_size = feature_shape[0] * feature_shape[1] * feature_shape[2] * feature_shape[3]
        write_size = int(write_size / 4)

        write_address = format(write_address, "032b")
        write_size = format(write_size, "032b")

        return write_address, write_size

    def get_shape_control(self):
        shape_control = self.shape_control
        shape_control = format(shape_control, '04b')

        shape_control_reg = shape_control.zfill(32)
        return shape_control_reg
