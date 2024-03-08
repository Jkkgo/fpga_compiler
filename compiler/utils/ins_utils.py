def instruction_parsing(ins_path):
    conv_op_type = {'2': 'conv'}
    shape_op_type = {'1': 'maxpool',  '3': 'upsample', '7': 'mean',
                     '4': 'cat', '5': 'add', '8': 'mul', '6': 'leakyrelu'}

    layer = 1
    instructions = open(ins_path).read().splitlines()
    print('------------------------------------------------------------'
          '------------------------------------------------------------')
    for i, everyins in enumerate(instructions):

        ins_address_0 = everyins[0] + everyins[1]
        ins_address_1 = everyins[6] + everyins[7]

        if(ins_address_0 == '11'):
            print('第{}层，第{}行指令({})(卷积状态):{}'.format(layer, i+1, ins_address_1, everyins[-1]))
            if (instructions[i-1][-1] != '1' or instructions[i-1][6]+instructions[i-1][7] == '0C'):
                layer += 1
                print('------------------------------------------------------------'
                      '------------------------------------------------------------')
        else:
            if(ins_address_1 == '04'):
                print('第{}层，第{}行指令({})(卷积控制):{}\t\t\t\t\t\t\t\t\t\t\t\t{}'.format(layer, i + 1, ins_address_1, everyins[-1],
                            conv_op_type[everyins[-1]] if everyins[-1] in conv_op_type.keys() else ''))
            if(ins_address_1 == '08'):
                print('第{}层，第{}行指令({})(shape状态):{}'.format(layer, i + 1, ins_address_1, everyins[-1]))
            if(ins_address_1 == '0C'):
                print('第{}层，第{}行指令({})(shape控制):{}\t\t\t\t\t\t\t\t\t\t\t\t{}'.format(layer, i + 1, ins_address_1, everyins[-1],
                            shape_op_type[everyins[-1]] if everyins[-1] in shape_op_type.keys() else ''))
            if(ins_address_1 == '10'):
                everyins = bin(int(everyins[8:],16))[2:].zfill(8*4) #拆成32位
                channelin = int(everyins[0:10],2)
                col_num_in = int(everyins[10:21],2)
                row_num_in = int(everyins[21:32],2)
                print('第{}层，第{}行指令({})(ConvReg0):channelin={},col_num_in={},row_num_in={}'.format(layer, i+1, ins_address_1, channelin, col_num_in, row_num_in))

            if (ins_address_1 == '14'):
                everyins = bin(int(everyins[8:], 16))[2:].zfill(8 * 4)  # 拆成32位
                EN_STRIDE = int(everyins[0],2)
                z3 = int(everyins[1:9],2)
                z1_NUM= int(everyins[9:12],2)
                z1= int(everyins[12:20],2)
                EN_ACTIVATION= int(everyins[20],2)
                EN_PADDING= int(everyins[21],2)
                CHANNEL_OUT= int(everyins[22:32],2)
                print('第{}层，第{}行指令({})(ConvReg1):EN_STRIDE={}, z3={}, z1_NUM={}, z1={}, EN_ACTIVATION={}, EN_PADDING={}, CHANNEL_OUT={}'.format
                      (layer, i + 1, ins_address_1, EN_STRIDE, z3, z1_NUM, z1, EN_ACTIVATION, EN_PADDING, CHANNEL_OUT))

            if(ins_address_1 == '18'):
                everyins = bin(int(everyins[8:], 16))[2:].zfill(8 * 4)  # 拆成32位
                FIRST_LAYER = int(everyins[-3],2)
                CONV_TYPE = int(everyins[-2:],2)
                print('第{}层，第{}行指令({})(ConvReg2):FIRST_LAYER={}, CONV_TYPE={}'.format(layer, i + 1, ins_address_1, FIRST_LAYER, CONV_TYPE))

            if(ins_address_1 == '1C'):
                everyins = bin(int(everyins[8:], 16))[2:].zfill(8 * 4)  # 拆成32位
                QUAN_NUM = int(everyins[0:16], 2)
                WEIGHT_NUM = int(everyins[16:32], 2)
                print('第{}层，第{}行指令({})(ConvReg3):QUAN_NUM={}, WEIGHT_NUM={}'.format(layer, i + 1, ins_address_1, QUAN_NUM, WEIGHT_NUM))

            if(ins_address_1 == '20'):
                everyins = bin(int(everyins[8:], 16))[2:].zfill(8 * 4)  # 拆成32位
                AMEND = int(everyins,2)
                print('第{}层，第{}行指令({})(ConvReg4):AMEND修正={}'.format(layer, i + 1, ins_address_1, AMEND))

            if (ins_address_1 == '24'):
                everyins = bin(int(everyins[8:], 16))[2:].zfill(8 * 4)  # 拆成32位
                channelin = int(everyins[0:10], 2)
                col_num_in = int(everyins[10:21], 2)
                row_num_in = int(everyins[21:32], 2)
                print(
                    '第{}层，第{}行指令({})(ShapeReg0):channelin={},col_num_in={},row_num_in={}'.format
                    (layer, i + 1, ins_address_1, channelin, col_num_in, row_num_in))

            if (ins_address_1 == '28'):
                everyins = bin(int(everyins[8:], 16))[2:].zfill(8 * 4)  # 拆成32位
                CHANNEL_IN1 = int(everyins[-10:], 2)
                print('第{}层，第{}行指令({})(ShapeReg1):CHANNEL_IN1={}'.format(layer, i + 1, ins_address_1, CHANNEL_IN1))

            if (ins_address_1 == '2C'):
                everyins = bin(int(everyins[8:], 16))[2:].zfill(8 * 4)  # 拆成32位
                SCALE = int(everyins, 2)
                print('第{}层，第{}行指令({})(ShapeReg2):SCALE={}'.format(layer, i + 1, ins_address_1, SCALE))

            if (ins_address_1 == '30'):
                everyins = bin(int(everyins[8:], 16))[2:].zfill(8 * 4)  # 拆成32位
                SCALE1 = int(everyins, 2)
                print('第{}层，第{}行指令({})(ShapeReg3):SCALE1={}'.format(layer, i + 1, ins_address_1, SCALE1))

            if (ins_address_1 == '34'):
                everyins = bin(int(everyins[8:], 16))[2:].zfill(8 * 4)  # 拆成32位
                ZERO = int(everyins, 2)
                print('第{}层，第{}行指令({})(ShapeReg4):ZERO={}'.format(layer, i + 1, ins_address_1, ZERO))

            if (ins_address_1 == '38'):
                everyins = bin(int(everyins[8:], 16))[2:].zfill(8 * 4)  # 拆成32位
                ZERO1 = int(everyins, 2)
                print('第{}层，第{}行指令({})(ShapeReg5):ZERO1={}'.format(layer, i + 1, ins_address_1, ZERO1))

            if (ins_address_1 == '3C' or ins_address_1 == '4C'):
                everyins = bin(int(everyins[8:], 16))[2:].zfill(8 * 4)  # 拆成32位
                WRITEDDRREG = int(everyins, 2)
                print('第{}层，第{}行指令({})(DMA写地址):WRITEDDRREG={}({})'.format(layer, i + 1, ins_address_1, WRITEDDRREG, hex(int(everyins, 2))[2:].zfill(8)))

            if ins_address_1 == '40' or ins_address_1 == '50':
                everyins = bin(int(everyins[8:], 16))[2:].zfill(8 * 4)  # 拆成32位
                WRITELENREG = int(everyins, 2)
                print('第{}层，第{}行指令({})(DMA写长度):WRITELENREG={}({})'.format(layer, i + 1, ins_address_1,  WRITELENREG, hex(int(everyins, 2))[2:].zfill(8)))

            if (ins_address_1 == '44' or ins_address_1 == '54' or ins_address_1 == '5C'):
                everyins = bin(int(everyins[8:], 16))[2:].zfill(8 * 4)  # 拆成32位
                READDDRREG = int(everyins, 2)
                print('第{}层，第{}行指令({})(DMA读地址):READDDRREG={}({})'.format(layer, i + 1, ins_address_1, READDDRREG, hex(int(everyins, 2))[2:].zfill(8)))

            if (ins_address_1 == '48' or ins_address_1 == '58' or ins_address_1 == '60'):
                everyins = bin(int(everyins[8:], 16))[2:].zfill(8 * 4)  # 拆成32位
                READLENREG = int(everyins, 2)
                print('第{}层，第{}行指令({})(DMA读长度):READLENREG={}({})'.format(layer, i + 1, ins_address_1, READLENREG, hex(int(everyins, 2))[2:].zfill(8)))

