import numpy as np


def coe_comparison(coe_file1_path, coe_file2_path):
    # if need_dat2coe:
    #     # 遍历dat文件夹下的所有dat文件，进行dat到coe的转换，不过此时需要保证该文件夹下的dat格式一致
    #     dat_file_list = [x for x in os.listdir(dat_dir_path) if '.dat' in x]
    #     for dat_file in dat_file_list:
    #         dat2coe(dat_file, is_weight, no_head)

    coe_file1 = open(coe_file1_path)
    coe_file1_lines = coe_file1.read().splitlines()
    coe_file2 = open(coe_file2_path)
    coe_file2_lines = coe_file2.read().splitlines()
    coe_file1_out1 = []
    coe_file1_out2 = []
    # 将该coe_file1文件下的每两个16进制值转化为对应的10进制 保存在 coe_file1_out2中
    for index1 in range(len(coe_file2_lines)):
        data1 = coe_file1_lines[index1].rsplit(',')[0]
        tmp = ''
        for data1_index, data1_item in enumerate(data1):
            tmp += data1_item
            if data1_index % 2 != 0:
                tmp = int(tmp, 16)
                coe_file1_out2.append(tmp)
                tmp = ''
        coe_file1_out1.append(data1)

    coe_file2_out1 = []
    coe_file2_out2 = []
    # 将该coe_file2文件下的每两个16进制值转化为对应的10进制 保存在 coe_file2_out2中
    for index2 in range(len(coe_file2_lines)):
        data2 = coe_file2_lines[index2].rsplit(',')[0]
        tmp = ''
        for data2_index, data2_item in enumerate(data2):
            tmp += data2_item
            if data2_index % 2 != 0:
                tmp = int(tmp, 16)
                coe_file2_out2.append(tmp)
                tmp = ''
        coe_file2_out1.append(data2)
    assert len(coe_file1_out1) == len(coe_file2_out1)
    # 将coe_file1_out2中的10进制值转换为np数组保存在final_out1中
    final_out1 = np.array(coe_file1_out2[:len(coe_file2_out2)])
    # 将coe_file2_out2中的10进制值转换为np数组保存在final_out2中
    # 按理来说进行对比的两个文件内的数据量是一致的，不需要使用第二个文件的数据量作为切分值
    final_out2 = np.array(coe_file2_out2)
    dif = final_out2 - final_out1
    result_dict = {}
    list_array = list(dif)
    set_array = set(list_array)
    for item in set_array:
        result_dict.update({item: list_array.count(item)})
    for key in result_dict.keys():
        val = result_dict[key]
        val = val / len(coe_file2_out2) * 100
        result_dict[key] = val

    print(coe_file1_path[-10:-4], ': result_dict:', result_dict)
