from compiler.shape_operator.base_shape import BaseShape


class YoloSig(BaseShape):
    """
    YoloSig操作
    继承BaseShape类

    YoloSig:对yolo算法中的置信度进行sigmoid解码,并取前n个高于置信度阈值的数据
    fpga实现yolo算法中的sigmoid函数如下：

    def near_sigmod_fpga(num):
        if num >= 0:
            elif abs(num)>=5<<17   : num =  1<<17
            elif 19<<14 <=abs(num)>=  5<<17  : num = 1 <<12 + 27<<12
            elif 1<<17 <=abs(num)>=  19<<14  : num = 1 <<14 + 5<<14
            elif 0 <=abs(num)>=  1<<17  : num = 1 <<15 + 1<<16
        else:
            #  num = 1<<17 - near_sigmod_fpga(abs(num))
        return num


    """

    def __init__(self, para, feature, option, shared):
        super().__init__(para, feature, option, shared)
        self.shape_control = shared.shape_control["YoloSig"]

    def get_dma_write(self):
        write_address = self.shared.write_address
        write_size = 0

        write_address = format(write_address, "032b")
        write_size = format(write_size, "032b")

        return write_address, write_size

    def get_shape_control(self):
        shape_control = self.shape_control
        shape_control = format(shape_control, '04b')

        shape_control_reg = shape_control.zfill(32)
        return shape_control_reg
