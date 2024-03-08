import numpy as np

from compiler.conv_operator.base_conv import BaseConv
from compiler.lib.add_channel import add_weight
from compiler.lib.ins_format import conv33para


class Conv33(BaseConv):
    """
    3*3卷积操作
    继承BaseConv类
    """

    '''
    __init__:初始化方法
    params:
        para: npy数据存放路径
        feature: 特征图数据
        option: [卷积类型,步长,padding,激活函数]
        shared: 共享变量集合
    '''
    def __init__(self, para, feature, option, shared):
        super().__init__(para, feature, option, shared)

    '''
    get_conv_reg2:计算2寄存器中的数据
    return:
        conv_reg2: 2寄存器数据
    '''
    def get_conv_reg2(self):
        # conv_type = 0代表做3*3卷积
        conv_type = 0
        if (self.shared.layer_count == 1
                and self.shared.start_op == 1):
            first_layer = 1
        else:
            first_layer = 0

        conv_type = format(conv_type, '02b')
        first_layer = format(first_layer, '01b')

        conv_reg2 = first_layer + conv_type
        conv_reg2 = conv_reg2.zfill(32)

        return conv_reg2

    '''
    get_conv_reg3:计算3寄存器中的数据
    return:
        conv_reg3: 3寄存器数据
    '''
    def get_conv_reg3(self):
        data_size = None
        # 8入8出则单次数据量为64bit
        if self.shared.parallel == 8:
            data_size = 64
        # 16入16出则单次数据量为128bit
        elif self.shared.parallel == 16:
            data_size = 128

        weight_shape = self.weight_shape

        # weight_num是fpga读取权重的次数
        # quan_num是fpga读取bias、scale、shift的次数
        # 根据卷积类型,weight_num和quan_num的计算方式也有所不同
        weight_num, quan_num = conv33para(weight_shape[0], weight_shape[1], data_size)

        weight_num = format(weight_num, '016b')
        quan_num = format(quan_num, '016b')

        conv_reg3 = quan_num + weight_num

        return conv_reg3
