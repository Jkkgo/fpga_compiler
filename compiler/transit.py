import os

from conv_operator.conv11 import Conv11
from conv_operator.conv33 import Conv33
from shape_operator.add import Add
from shape_operator.argmax import ArgMax
from shape_operator.concat import Concat
from shape_operator.leakyrelu import LeakyRelu
from shape_operator.max_pool import MaxPool
from shape_operator.mean_pool import MeanPool
from shape_operator.mul import Mul
from shape_operator.pre import Pre
from shape_operator.up_bili import UpBili
from shape_operator.upsample import UpSample
from shape_operator.yolo_sig import YoloSig
from shape_operator.split import Split
from shared_variable import SharedVariableContainer

# 算子字典,添加新算子后记得把新算子的类名注册到此算子字典
module_factory = {
    'Conv33': Conv33,
    'Conv11': Conv11,
    'Concat': Concat,
    'Add': Add,
    'Mul': Mul,
    'MeanPool': MeanPool,
    'MaxPool': MaxPool,
    'UpSample': UpSample,
    'UpBili': UpBili,
    'LeakyRelu': LeakyRelu,
    'ArgMax': ArgMax,
    'Split': Split,
    'Pre': Pre,
    'YoloSig': YoloSig
}

# 所有的分发器共用一个共享变量
shared = SharedVariableContainer()


class Transit:
    """
    Transit分发器
    根据操作类型调用不同的算子

    """
    '''
    __init__:初始化方法
    params:
        若为conv操作:
        para1:'输入层的npy文件前缀名'
        para2:'本层的npy文件前缀名'
        feature:[输入特征图, 输出特征图]
        option:['卷积操作名',stride,padding,是否使用激活函数]
        若为shape操作:
        para1:'左输入层的npy文件前缀名'
        para2:'本层的npy文件前缀名'
        para3:'右输入层的npy文件前缀名'(可省略)
        feature:[左输入特征图,输出特征图,右输入特征图(可省略)]
        option:['shape操作名']
    '''
    def __init__(self, para1="", para2="", para3="", feature=None, option=None):

        # 所有的分发器共用一个共享变量
        self.shared = shared
        para_path = shared.para_path
        module_name = option[0]
        # 若为conv操作
        if 'Conv' in option[0]:
            conv_para = self.get_conv(para_path, para1, para2)
            para = conv_para
        else:
            if len(feature) == 2:
                feature = [feature[0], feature[1], None]
            shape_para = self.get_shape(para_path, para1, para2, para3)
            para = shape_para
        # 工厂模式,根据操作类型调用不同的算子
        module = module_factory[module_name](para, feature, option, shared)
        # 调用算子的__call__函数
        module()

    '''
    get_conv:列出conv操作所需要的npy文件路径
    params:
        para_path:npy文件包的路径
        pre_para:输入层的npy文件前缀名
        local_para:本层的npy文件前缀名
    return:
        npy文件路径字典
    '''
    def get_conv(self, para_path, pre_para, local_para):
        para = {"pre_scale": para_path + pre_para + ".scale.npy",
                "pre_zp": para_path + pre_para + ".zero_point.npy",
                "local_weight_scale": para_path + local_para + ".weight.scale.npy",
                "local_weight_zp": para_path + local_para + ".weight.zero_point.npy",
                "local_weight": para_path + local_para + ".weight.npy",
                "local_weight_int": para_path + local_para + ".weight.int.npy",
                "local_scale": para_path + local_para + ".scale.npy",
                "local_zp": para_path + local_para + ".zero_point.npy",
                "local_bias": para_path + local_para + ".bias.npy",
                "pre_para": pre_para,
                "local_para": local_para
                }

        return para

    '''
    get_shape:列出shape操作所需要的npy文件路径
    params:
        para_path:npy文件包的路径
        input_para_l:左输入层的npy文件前缀名
        local_para:本层的npy文件前缀名
        input_para_r:右输入层的npy文件前缀名
    return:
        npy文件路径字典
    '''
    def get_shape(self, para_path, input_para_l, local_para, input_para_r):
        para = {"l_scale": para_path + input_para_l + ".scale.npy",
                "l_zp": para_path + input_para_l + ".zero_point.npy",
                "r_scale": para_path + input_para_r + ".scale.npy",
                "r_zp": para_path + input_para_r + ".zero_point.npy",
                "local_scale": para_path + local_para + ".scale.npy",
                "local_zp": para_path + local_para + ".zero_point.npy",
                "input_para_l": input_para_l,
                "input_para_r": input_para_r,
                "local_para": local_para
                }

        return para
