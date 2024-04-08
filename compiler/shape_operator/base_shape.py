from abc import abstractmethod

from compiler.lib.add_channel import add_feature_shape
from compiler.lib.base_write import BaseWrite
from compiler.lib.write_data import get_feature_count


class BaseShape(BaseWrite):
    """
    shape操作的父类
    规定了shape操作中一些通用的方法
    """
    '''
    __init__:初始化方法
    params:
        para: npy数据存放路径
        feature: 特征图数据
        option: [shape类型]
        shared: 共享变量集合
    '''

    def __init__(self, para, feature, option, shared):
        # 初始化父类
        super().__init__(para, feature, option, shared)
        self.para = para
        self.feature = feature
        self.option = option
        self.shared = shared
        self.l_feature_shape = add_feature_shape(feature[0], self.shared.parallel)
        if feature[2] is not None:
            self.r_feature_shape = add_feature_shape(feature[2], self.shared.parallel)
        else:
            self.r_feature_shape = None

    '''
    packing_data: 对计算结果进行打包
    return:
        data_package: 计算结果字典
    '''

    def packing_data(self):
        shape_reg0 = self.get_shape_reg0()
        shape_reg1 = self.get_shape_reg1()
        shape_reg2 = self.get_shape_reg2()
        shape_reg3 = self.get_shape_reg3()
        shape_reg4 = self.get_shape_reg4()
        shape_reg5 = self.get_shape_reg5()
        (l_feature_address, l_feature_size,
         r_feature_address, r_feature_size) = self.get_dma_read()
        write_address, write_size = self.get_dma_write()
        shape_control_reg = self.get_shape_control()

        data_package = {
            "shape_reg0": int(shape_reg0, 2),
            "shape_reg1": int(shape_reg1, 2),
            "shape_reg2": int(shape_reg2, 2),
            "shape_reg3": int(shape_reg3, 2),
            "shape_reg4": int(shape_reg4, 2),
            "shape_reg5": int(shape_reg5, 2),
            "l_feature_address": int(l_feature_address, 2),
            "l_feature_size": int(l_feature_size, 2),
            "r_feature_address": int(r_feature_address, 2),
            "r_feature_size": int(r_feature_size, 2),
            "write_address": int(write_address, 2),
            "write_size": int(write_size, 2),
            "shape_control_reg": int(shape_control_reg, 2)
        }
        return data_package

    '''
    get_shape_reg0: 计算0寄存器中的数据
    return:
        shape_reg0: 0寄存器数据
    '''

    def get_shape_reg0(self):
        feature_shape = self.l_feature_shape
        row_in = feature_shape[2]
        col_in = feature_shape[3]
        row_in = format(row_in, '011b')
        col_in = format(col_in, '011b')
        # 该寄存器存储左入通道数、行数、列数
        shape_reg0 = col_in + row_in
        shape_reg0 = shape_reg0.zfill(32)
        return shape_reg0

    '''
    get_shape_reg1: 计算1寄存器中的数据
    return:
        shape_reg1: 1寄存器数据
    '''

    def get_shape_reg1(self):
        r_channel_in = 0
        input_para_r = self.para['input_para_r']
        if input_para_r:
            r_feature_shape = self.r_feature_shape
            r_channel_in = r_feature_shape[1]

        # 该寄存器存储左右入通道数
        feature_shape = self.l_feature_shape
        l_channel_in = feature_shape[1]
        l_channel_in = format(l_channel_in, '011b')
        r_channel_in = format(r_channel_in, '011b')
        shape_reg1 = l_channel_in + r_channel_in
        shape_reg1 = shape_reg1.zfill(32)
        return shape_reg1

    '''
    get_shape_reg2: 计算2寄存器中的数据
    return:
        shape_reg2: 2寄存器数据
    '''

    def get_shape_reg2(self):
        l_scale = 0
        l_scale = format(l_scale, "032b")

        shape_reg2 = l_scale
        return shape_reg2

    '''
    get_shape_reg3: 计算3寄存器中的数据
    return:
        shape_reg3: 3寄存器数据
    '''

    def get_shape_reg3(self):
        r_scale = 0

        r_scale = format(r_scale, "032b")

        shape_reg3 = r_scale
        return shape_reg3

    '''
    get_shape_reg4: 计算4寄存器中的数据
    return:
        shape_reg4: 4寄存器数据
    '''

    def get_shape_reg4(self):
        l_zp = 0

        l_zp = format(l_zp, "032b")

        shape_reg4 = l_zp
        return shape_reg4

    '''
    get_shape_reg5: 计算5寄存器中的数据
    return:
        shape_reg5: 5寄存器数据
    '''

    def get_shape_reg5(self):
        r_zp = 0

        r_zp = format(r_zp, "032b")

        shape_reg5 = r_zp
        return shape_reg5

    '''
    get_dma_read:获取左输入特征图、右输入特征图的读取地址以及长度
    return:
        l_feature_address: 左输入特征图读取地址
        l_feature_size: 左输入特征图读取长度
        r_feature_address: 右输入特征图读取地址
        r_feature_size: 右输入特征图读取长度
    '''

    def get_dma_read(self):

        l_feature_address, l_feature_size = self.get_one_read(self.feature[0], self.l_feature_shape)
        r_feature_address, r_feature_size = self.get_one_read(self.feature[2], self.r_feature_shape)

        l_feature_address = format(l_feature_address, '032b')
        l_feature_size = format(l_feature_size, '032b')
        r_feature_address = format(r_feature_address, '032b')
        r_feature_size = format(r_feature_size, '032b')

        return l_feature_address, l_feature_size, r_feature_address, r_feature_size

    '''
    get_one_read:获取输入特征图的读取地址以及长度
    params:
        feature:输入特征图
        feature_shape:输入特征图形状
    return:
        feature_address: 输入特征图读取地址
        feature_size: 输入特征图读取长度
    '''

    def get_one_read(self, feature, feature_shape):
        # 通过特征图id来查找地址表中的地址
        feature_id = id(feature)
        if feature is None:
            return 0, 0
        else:

            # 通过特征图对应层数来查找地址表中的地址
            feature_count = get_feature_count(feature_id, self.shared.layer_table)
            feature_address = self.shared.address_table[feature_count - 1]

            feature_size = feature_shape[0] * feature_shape[1] * feature_shape[2] * feature_shape[3]
            return feature_address, feature_size

    '''
    get_dma_write:计算输出特征图的写回地址以及写回长度,该方法为抽象方法,需要子类重写
    return:
        write_address: 输出特征图写回地址
        write_size: 输入特征图写回长度
    '''

    @abstractmethod
    def get_dma_write(self):
        write_address = 0
        write_size = 0
        write_address = format(write_address, '032b')
        write_size = format(write_size, '032b')
        return write_address, write_size

    '''
    get_shape_control:计算shape类型,该方法为抽象方法,需要子类重写
    return:
        shape_control_reg: shape类型寄存器数据
    '''

    @abstractmethod
    def get_shape_control(self):
        shape_name = self.option[0]
        shape_control = self.shared.shape_control[shape_name]
        shape_control = format(shape_control, '04b')

        shape_control_reg = shape_control.zfill(32)
        return shape_control_reg

    '''
    update_shared:更新一些相关的共享变量
    更新特征图地址
    更新特征图地址表
    更新层数
    '''

    def update_shared(self, data_package):
        feature_id = id(self.feature[1])
        write_address = data_package["write_address"]
        write_size = data_package["write_size"]

        self.shared.write_address += write_size

        self.shared.layer_table[feature_id] = self.shared.layer_count
        self.shared.address_table.append(write_address)

        self.shared.layer_count += 1

    '''
       simulate:模拟FPGA定点运算方式,该方法为抽象方法,需要子类重写
    '''
    def simulate(self, feature):
        pass
