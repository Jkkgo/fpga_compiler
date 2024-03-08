import os
import shutil
from abc import abstractmethod

from compiler.lib.array_format import convert_scale, convert_bias, convert_weight
from compiler.lib.write_data import write_conv, gen_coe_add, write_weight, clear_files, coe2bin
from compiler.lib.add_channel import *
from compiler.lib.ins_format import leaky_format


class BaseConv:
    """
    卷积操作的父类
    规定了卷积中一些通用的方法
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
        self.para = para
        self.feature = feature
        self.option = option
        self.shared = shared
        if shared.layer_count != 1:
            self.feature_shape = add_feature_shape(feature[0], self.shared.parallel)
        else:
            self.feature_shape = feature[0].shape

        weight = np.load(self.para['local_weight_int'])
        weight = add_weight(weight, self.shared.parallel)
        self.weight_shape = weight.shape

    '''
    __call__:实例化调用方法
    '''

    def __call__(self, *args, **kwargs):

        data_package = self.packing_data()

        if self.shared.layer_count <= self.shared.generate_mode[1]:
            if self.shared.gen_ins:
                self.write_ins_file(data_package)
            if self.shared.gen_weight:
                self.write_weight_file()
            if self.shared.gen_result:
                self.write_result_file()

        self.update_shared(data_package)

    '''
    packing_data:对计算结果进行打包
    return:
        data_package: 计算结果字典
    '''

    def packing_data(self):
        conv_reg0 = self.get_conv_reg0()
        conv_reg1 = self.get_conv_reg1()
        conv_reg2 = self.get_conv_reg2()
        conv_reg3 = self.get_conv_reg3()
        conv_reg4 = self.get_conv_reg4()
        weight_address, weight_size, feature_address, feature_size = self.get_dma_read()
        write_address, write_size = self.get_dma_write()

        data_package = {
            "conv_reg0": int(conv_reg0, 2),
            "conv_reg1": int(conv_reg1, 2),
            "conv_reg2": int(conv_reg2, 2),
            "conv_reg3": int(conv_reg3, 2),
            "conv_reg4": int(conv_reg4, 2),
            "weight_address": int(weight_address, 2),
            "weight_size": int(weight_size, 2),
            "feature_address": int(feature_address, 2),
            "feature_size": int(feature_size, 2),
            "write_address": int(write_address, 2),
            "write_size": int(write_size, 2),
        }
        return data_package

    '''
    get_conv_reg0:计算0寄存器中的数据
    return:
        conv_reg0: 0寄存器数据
    '''

    def get_conv_reg0(self):
        feature_shape = self.feature_shape
        row_in = feature_shape[2]
        col_in = feature_shape[3]
        channel_in = feature_shape[1]
        row_in = format(row_in, '011b')
        col_in = format(col_in, '010b')
        channel_in = format(channel_in, '011b')
        # 该寄存器存储入通道数、行数、列数
        conv_reg0 = channel_in + col_in + row_in
        return conv_reg0

    '''
    get_conv_reg1:计算1寄存器中的数据
    return:
        conv_reg1: 1寄存器数据
    '''

    def get_conv_reg1(self):
        weight_shape = self.weight_shape
        channel_out = weight_shape[0]
        # en_padding指明了fpga是否需要做padding操作
        en_padding = 1 if self.option[2] > 0 else 0
        # en_padding指明了fpga是否需要做激活函数操作
        en_activation = self.option[3]
        z1 = np.load(self.para['pre_zp']).item()
        # z1_num指明了fpga做padding时需要补多少圈
        z1_num = self.option[2] if self.option[2] > 0 else 0
        # z3指明了fpga做padding时需要补的数据,由于fpga采用量化后的定点数进行运算,所以不能单纯的去补零,而是补z3的值
        z3 = np.load(self.para['local_zp'])
        # en_stride=0代表步长为1,en_stride=1代表步长为2
        en_stride = 1 if self.option[1] == 2 else 0

        channel_out = format(channel_out, '010b')
        en_padding = format(en_padding, '01b')
        en_activation = format(en_activation, '01b')
        z1 = format(z1, '08b')
        z1_num = format(z1_num, '03b')
        z3 = format(z3, '08b')
        en_stride = format(en_stride, '01b')
        conv_reg1 = (en_stride + z3 + z1_num + z1 +
                     en_activation + en_padding + channel_out)
        return conv_reg1

    '''
    get_conv_reg2:计算2寄存器中的数据,该方法为抽象方法,需要子类重写
    return:
        conv_reg2: 2寄存器数据
    '''

    @abstractmethod
    def get_conv_reg2(self):
        # conv_type需要根据conv类型来具体实现
        conv_type = 00
        '''
        此段if else 代码是为了适配卷积作为首层时，用first_layer这个参数来指示在卷积之前做一些特殊操作
        这种操作职能可以用shape操作中的pre代替
        '''
        if (self.shared.layer_count + 1 == 1
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
    get_conv_reg3:计算3寄存器中的数据,该方法为抽象方法,需要子类重写
    return:
        conv_reg3: 3寄存器数据
    '''

    @abstractmethod
    def get_conv_reg3(self):
        # weight_num,quan_num 需要根据conv类型来具体实现
        weight_num = 00
        quan_num = 00

        weight_num = format(weight_num, '016b')
        quan_num = format(quan_num, '016b')

        conv_reg3 = quan_num + weight_num

        return conv_reg3

    '''
    get_conv_reg4:计算4寄存器中的数据
    return:
        conv_reg4: 4寄存器数据
    '''

    def get_conv_reg4(self):
        en_activation = self.option[3]
        if en_activation == 1:
            s3 = np.load(self.para['local_scale'])
            # amend用于修正leakrelu
            # 当采用relu做激活函数时，amend值无意义。
            # 当采用leakrelu做激活函数时,fpga需要用amend值做修正
            amend = leaky_format(s3)
            conv_reg4 = amend
        else:
            conv_reg4 = format(0, '032b')

        return conv_reg4

    '''
    get_dma_read:获取权重、特征图的读取地址以及长度
    return:
        weight_address: 权重读取地址
        weight_size: 权重读取长度
        feature_address: 特征图读取地址
        feature_size: 特征图读取长度
    '''

    def get_dma_read(self):

        weight_address, weight_size = self.get_weight_address()
        feature_address, feature_size = self.get_feature_address()

        weight_address = format(weight_address, '032b')
        weight_size = format(weight_size, '032b')
        feature_address = format(feature_address, '032b')
        feature_size = format(feature_size, '032b')

        return weight_address, weight_size, feature_address, feature_size

    '''
    get_feature_address:查找特征图的读取地址以及计算特征图长度
    return:
        feature_address: 特征图读取地址
        feature_size: 特征图读取长度
    '''

    def get_feature_address(self):
        feature_shape = self.feature_shape
        # 获取特征图的唯一id
        feature_id = id(self.feature[0])

        feature_address = 0
        # 通过特征图id来查找地址表中的地址
        for key, value in self.shared.address_table.items():
            if feature_id == key:
                feature_address = value
        # 计算特征图长度
        feature_size = feature_shape[0] * feature_shape[1] * feature_shape[2] * feature_shape[3]

        return feature_address, feature_size

    '''
    get_weight_address:查找权重的读取地址以及计算权重长度
    return:
        feature_address: 权重读取地址
        feature_size: 权重读取长度
    '''

    def get_weight_address(self):
        weight_shape = self.weight_shape

        # 权重采用堆叠形式存储,直接调用地址变量即可
        weight_address = self.shared.weight_address
        # 计算权重长度
        weight_size = weight_shape[0] * weight_shape[1] * weight_shape[2] * weight_shape[3]
        other_size = 3 * weight_shape[0] * 4
        weight_size += other_size

        return weight_address, weight_size

    '''
    get_dma_write:计算特征图的写回地址以及写回长度
    return:
        write_address: 特征图写回地址
        write_size: 特征图写回长度
    '''

    def get_dma_write(self):
        feature_shape = self.feature_shape
        weight_shape = self.weight_shape
        padding = self.option[2]
        stride = self.option[1]
        # 卷积输出特征图长宽的计算方式
        out_size = int((feature_shape[2] - weight_shape[2] + 2 * padding) / stride) + 1

        write_address = self.shared.write_address
        write_size = feature_shape[0] * weight_shape[0] * out_size * out_size

        write_address = format(write_address, '032b')
        write_size = format(write_size, '032b')

        return write_address, write_size

    '''
    update_shared:更新一些相关的共享变量
    更新权重地址
    更新特征图地址
    更新特征图地址表
    更新层数
    '''

    def update_shared(self, data_package):
        feature_id = id(self.feature[1])
        weight_address = data_package["weight_address"]
        weight_size = data_package["weight_size"]
        write_address = data_package["write_address"]
        write_size = data_package["write_size"]

        self.shared.weight_address += weight_size
        self.shared.write_address += write_size
        self.shared.address_table[feature_id] = write_address

        self.shared.layer_count += 1

    '''
    write_ins_file:写指令文件
    params:
        data_package: 计算结果字典
    '''

    def write_ins_file(self, data_package):
        layer_count = str(self.shared.layer_count)
        dat_name = 'auto_ins' + layer_count + '.dat'
        file_path = self.shared.file_path + 'ins'
        os.makedirs(file_path, exist_ok=True)
        file_name = "{}/{}".format(file_path, dat_name)

        # 如果是首层，则清空ins文件夹
        if layer_count == '1':
            clear_files(file_path)

        # 如果是联测,则将前一层指令文件复制到本层,再去追加写入本层指令
        if self.shared.generate_mode[0] == 1 and self.shared.layer_count != 1:
            pre_file = file_path + '/auto_ins' + str(self.shared.layer_count - 1) + '.dat'
            shutil.copyfile(pre_file, file_name)
        # 写入指令
        write_conv(file_name, data_package)

    '''
    write_result_file:写中间结果文件
    '''

    def write_result_file(self):
        mid_result = self.feature[1]
        result_shape = mid_result.shape

        local_zp = self.para['local_zp']
        local_zp = np.load(local_zp).astype(int).item()

        parallel = self.shared.parallel
        # 判断是否需要补通道,补完通道的形状是多少 np.ceil向上取整
        add_channel = int(parallel * np.ceil(result_shape[1] / parallel))

        layer_count = str(self.shared.layer_count)
        file_name = 'auto_result' + layer_count + '.coe'
        file_path = self.shared.file_path + 'mid_result'
        os.makedirs(file_path, exist_ok=True)
        file_name = "{}/{}".format(file_path, file_name)

        # 如果是联测，则直接生成
        if self.shared.generate_mode[0] == 1:
            gen_coe_add(file_name, mid_result.int_repr(), local_zp, add_channel, parallel)
        # 如果是单测，则只生成一层的中间结果
        elif self.shared.layer_count == self.shared.generate_mode[1]:
            gen_coe_add(file_name, mid_result.int_repr(), local_zp, add_channel, parallel)

        # 如果是首层，则还生成输入bin
        if self.shared.layer_count == 1:
            input_feature = self.feature[0].int_repr()
            channel = self.feature[0].shape[1]
            input_path = file_path + "/auto_input.coe"
            bin_path = file_path + "/auto_input.bin"
            gen_coe_add(input_path, input_feature, 1, channel, channel)
            coe2bin(input_path, bin_path)

    '''
    write_weight_file:写权重文件
    '''

    def write_weight_file(self):
        pre_scale = np.load(self.para['pre_scale'])
        pre_zp = np.load(self.para['pre_zp'])
        local_weight = np.load(self.para['local_weight_int'])
        local_weight_scale = np.load(self.para['local_weight_scale'])
        local_scale = np.load(self.para['local_scale'])
        local_bias = np.load(self.para['local_bias'])

        # 计算移位之后的scale和移了多少位  scale = (s1 * s2) / s3
        scale, shift = convert_scale(pre_scale, local_weight_scale, local_scale)
        # 计算新的bias bias = symbol(符号位) + data_decimal(移位值) + data_integer(移位之后的bias)
        bias = convert_bias(pre_zp, pre_scale, local_weight_scale, local_weight, local_bias)

        parallel = self.shared.parallel
        local_weight = add_weight(local_weight, parallel)
        weight = convert_weight(local_weight, parallel)
        scale = add_array(scale, parallel)
        shift = add_array(shift, parallel)
        bias = add_array(bias, parallel)

        layer_count = str(self.shared.layer_count)
        coe_name = 'auto_weight' + layer_count + '.coe'
        bin_name = '/auto_weight' + layer_count + '.bin'
        file_path = self.shared.file_path + 'weight'
        os.makedirs(file_path, exist_ok=True)
        file_name = "{}/{}".format(file_path, coe_name)

        # 如果是首层，则清空weight文件夹
        if layer_count == '1':
            clear_files(file_path)

        # 如果是联测,则将前一层权重文件复制到本层,再去追加写入本层权重
        if self.shared.generate_mode[0] == 1 and self.shared.layer_count != 1:
            pre_file = file_path + '/auto_weight' + str(self.shared.layer_count - 1) + '.coe'
            shutil.copyfile(pre_file, file_name)

        weight_package = {
            "file_name": file_name,
            "weight": weight,
            "bias": bias,
            "scale": scale,
            "shift": shift,
            "parallel": parallel
        }
        write_weight(weight_package)
        # 如果是指定层数，则将该层coe格式权重转为bin
        if self.shared.layer_count == self.shared.generate_mode[1]:
            coe2bin(file_name, file_path + bin_name)
