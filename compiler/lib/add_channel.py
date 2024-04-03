import numpy as np

'''
add_weight:根据fpga的通道并行数来对权重进行补通道
params:
    weight:未补通道的权重
    parallel:fpga的通道并行数
'''


def add_weight(weight, parallel):
    weight_shape = weight.shape

    if (weight_shape[1] % parallel != 0):  # 权重输入通道变成parallel的倍数,不足补0
        channel_in_num = weight_shape[1] + parallel - weight_shape[1] % parallel
    else:
        channel_in_num = weight_shape[1]

    if (weight_shape[0] % parallel != 0):  # 权重输出通道变成parallel的倍数,不足补0
        channel_out_num = weight_shape[0] + parallel - weight_shape[0] % parallel
    else:
        channel_out_num = weight_shape[0]

    weight_add = np.zeros((channel_out_num, channel_in_num, weight_shape[2], weight_shape[3]), dtype=np.int32)
    weight_add[:weight_shape[0], :weight_shape[1], :, :] = weight

    return weight_add


'''
add_weight:根据fpga的通道并行数来对权重进行补通道
params:
    weight:未补通道的权重
    parallel:fpga的通道并行数
'''


def add_feature(feature, parallel):
    feature = feature.int_repr().numpy()
    feature_shape = feature.shape

    if feature_shape[1] % parallel != 0:  # 特征图输入通道变成parallel的倍数,不足补0
        channel_in_num = feature_shape[1] + parallel - feature_shape[1] % parallel
        feature_add = np.zeros((feature_shape[0], channel_in_num, feature_shape[2], feature_shape[3]))
        feature_add[:, :feature_shape[1], :, :] = feature
        feature_add = feature_add.astype(np.uint8)
    else:
        feature_add = feature.astype(np.uint8)

    return feature_add


'''
add_array:根据fpga的通道并行数来对一维数组补通道
用于对scale、shift、bias数组补通道
params:
    array:未补通道的一维数组
    parallel:fpga的通道并行数
'''


def add_array(array, parallel):
    array_shape = array.shape

    if array_shape[0] % parallel != 0:  # 特征图输入通道变成parallel的倍数,不足补0
        channel_in_num = array_shape[0] + parallel - array_shape[0] % parallel
    else:
        channel_in_num = array_shape[0]

    array_add = np.zeros(channel_in_num, dtype=np.uint32)
    array_add[:array_shape[0]] = array

    return array_add


'''
add_feature_shape:根据fpga的通道并行数来计算出特征图补通道后的形状
只计算形状,节约资源
params:
    feature:未补通道的特征图
    parallel:fpga的通道并行数
'''


def add_feature_shape(feature, parallel):
    feature_shape = feature.shape

    if feature_shape[1] % parallel != 0:  # 特征图输入通道变成parallel的倍数,不足补0
        channel_in_num = feature_shape[1] + parallel - feature_shape[1] % parallel
    else:
        channel_in_num = feature_shape[1]

    feature_shape = [feature_shape[0], channel_in_num, feature_shape[2], feature_shape[3]]

    return feature_shape
