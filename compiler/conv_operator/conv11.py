
from compiler.conv_operator.base_conv import BaseConv
from compiler.lib.ins_format import conv11para


class Conv11(BaseConv):
    """
    1*1卷积操作
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
        # 初始化父类
        super().__init__(para, feature, option, shared)

    '''
    get_conv_reg2:计算2寄存器中的数据
    return:
        conv_reg2: 2寄存器数据
        ps：conv_type是硬件规定的
    '''
    def get_conv_reg2(self):
        parallel = self.shared.parallel
        weight_shape = self.weight_shape
        feature_shape = self.feature_shape

        # 如果入通道数小于 8*parallel 或者入通道无法被 8*parallel 整除或者特征图尺寸为1，则采用1*1的卷积方式(硬件方面的bug)
        if (weight_shape[1] < 8 * parallel
                or weight_shape[1] % (8 * parallel) != 0
                or feature_shape[2] == 1):
            conv_type = 2
        # 否则采用1*1*8卷积方式
        else:
            conv_type = 1

        if (self.shared.layer_count == 1
                and self.shared.start_op == 1):
            first_layer = 1
        else:
            first_layer = 0

        conv_type = format(conv_type, '02b')
        first_layer = format(first_layer, '01b')

        conv_reg2 = first_layer + conv_type
        # zfill(32)将字符串conv_reg2填充到总长度为32位，不够左侧补0
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
        weight_num, quan_num = conv11para(weight_shape[0], weight_shape[1], data_size)

        weight_num = format(weight_num, '016b')
        quan_num = format(quan_num, '016b')

        conv_reg3 = quan_num + weight_num

        return conv_reg3
