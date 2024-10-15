import torch
from torch import nn

from compiler.lib.array_format import picture_load
from compiler.lib.write_data import gen_coe_add

from compiler.transit import Transit
from compiler.transit import shared
from lib.block_format import block

'''
   create_files:主函数
   1.将网络模型的每一层拆分出来
   2.将拆分出来的网络输入输出、提取出来的量化参数、本层操作传入transit分发器
'''


def create_files():
    model = torch.jit.load(shared.model_path)
    model.eval()
    img = picture_load(shared)
    img = img.unsqueeze(0)
    with torch.no_grad():
        # ------------------------网络Init-------------------------- #
        stem_0 = list(model.backbone.stem.children())[0]  # Conv
        stem_1 = list(model.backbone.stem.children())[1]  # Conv
        stem_2 = list(model.backbone.stem.children())[2]  # Conv
        dark2_0 = list(model.backbone.dark2.children())[0]  # Conv
        dark2_1 = list(model.backbone.dark2.children())[1]  # Multi_Concat_Block
        dark3_0 = list(model.backbone.dark3.children())[0]  # Transition_Block
        dark3_1 = list(model.backbone.dark3.children())[1]  # Multi_Concat_Block
        dark4_0 = list(model.backbone.dark4.children())[0]  # Transition_Block
        dark4_1 = list(model.backbone.dark4.children())[1]  # Multi_Concat_Block
        dark5_0 = list(model.backbone.dark5.children())[0]  # Transition_Block
        dark5_1 = list(model.backbone.dark5.children())[1]  # Multi_Concat_Block

        feature_f = model.quant(img)
        # 图片补通道
        feature_q = feature_f.int_repr()
        feature_addchannel = torch.zeros([1, 16, feature_q.shape[2], feature_q.shape[3]], dtype=torch.int16)
        feature_addchannel[:, :feature_q.shape[1], :, :] = feature_q
        # gen_coe_add("../sim_data/input_add.coe",feature_addchannel,0,8,8)

        '''
                主函数具体写法:
                    若为正常conv操作:
                        Transit(
                        para1='输入层的npy文件前缀名', 
                        para2='本层的npy文件前缀名', 
                        feature=[输入特征图, 输出特征图], 
                        option=['卷积操作名',stride,padding,是否使用激活函数]
                        )
                    若为shape操作:
                        Transit(
                        para1='左输入层的npy文件前缀名', 
                        para2='本层的npy文件前缀名', 
                        para3='右输入层的npy文件前缀名', 
                        feature=[左输入特征图,输出特征图,右输入特征图], 
                        option=['shape操作名']
                        )
                    若为分块conv操作:
                        Block(
                        para1='输入层的npy文件前缀名', 
                        para2='本层的npy文件前缀名', 
                        feature=[输入特征图, 输出特征图], 
                        option=['卷积操作名',stride,padding,是否使用激活函数,分块数量]
                        )
                    特殊规则：
                        1.第0层的para1为空
                        2.若本层的para2没有可用的npy文件,则用本层的para1参数代替
                        3.若shape操作中只有一个输入,则该操作的para3和feature[2]可省略
                        4.分块conv操作,根据分块数量生成 2*分块数量-1 层指令(满二叉树)
        '''

        # 斜框没有预处理层,第零层不用管
        # 0
        # Transit(para1='', para2='quant', feature=[img, quant_feature_f], option=['Pre'])

        # ------------------------网络推理-------------------------- #
        # ------------- stem --------------- #
        # 1
        stem_0_r = stem_0.conv(feature_f)
        stem_0_r_int = stem_0_r.int_repr()
        # gen_coe_add("./1.coe", stem_0_r_int, 0, 16, 8)
        stem_0_act_r = stem_0.act(stem_0_r)
        Transit(para1='quant', para2='backbone.stem.0.conv',
                feature=[feature_f, stem_0_act_r],
                option=['Conv33', 1, 1, 1])
        # exit()

        stem_1_r = stem_1.conv(stem_0_act_r)
        stem_1_r_int = stem_1_r.int_repr()
        # gen_coe_add("./2.coe", stem_1_r_int, 0, 32, 16)
        stem_1_act_r = stem_1.act(stem_1_r)
        # 2
        Transit(para1='backbone.stem.0.conv', para2='backbone.stem.1.conv',
                feature=[stem_0_act_r, stem_1_act_r],
                option=['Conv33', 2, 1, 1])
        # 3
        stem_2_r = stem_2.conv(stem_1_act_r)
        stem_2_r_int = stem_2_r.int_repr()
        # gen_coe_add("./3.coe", stem_2_r_int, 0, 32, 16)
        stem_2_act_r = stem_2.act(stem_2_r)
        Transit(para1='backbone.stem.1.conv', para2='backbone.stem.2.conv',
                feature=[stem_1_act_r, stem_2_act_r],
                option=['Conv33', 1, 1, 1])
        # ------------ dark2 -------------- #
        # 4
        dark2_0_r = dark2_0.conv(stem_2_act_r)
        dark2_0_act_r = dark2_0.act(dark2_0_r)
        Transit(para1='backbone.stem.2.conv', para2='backbone.dark2.0.conv',
                feature=[stem_2_act_r, dark2_0_act_r],
                option=['Conv33', 2, 1, 1])

        # --dark2.1--Multi_Concat_Block ---#
        # 5
        dark2_1_cv1_r = dark2_1.cv1.conv(dark2_0_act_r)
        dark2_1_cv1_act_r = dark2_1.cv1.act(dark2_1_cv1_r)  # r1
        Transit(para1='backbone.dark2.0.conv', para2='backbone.dark2.1.cv1.conv',
                feature=[dark2_0_act_r, dark2_1_cv1_act_r],
                option=['Conv11', 1, 0, 1])
        # 6
        dark2_1_cv2_r = dark2_1.cv2.conv(dark2_0_act_r)
        dark2_1_cv2_act_r = dark2_1.cv2.act(dark2_1_cv2_r)  # r2
        Transit(para1='backbone.dark2.0.conv', para2='backbone.dark2.1.cv2.conv',
                feature=[dark2_0_act_r, dark2_1_cv2_act_r],
                option=['Conv11', 1, 0, 1])
        # 7
        dark2_1_cv3_1_r = dark2_1.cv3_1.conv(dark2_1_cv2_act_r)
        dark2_1_cv3_1_act_r = dark2_1.cv3_1.act(dark2_1_cv3_1_r)
        Transit(para1='backbone.dark2.1.cv2.conv', para2='backbone.dark2.1.cv3_1.conv',
                feature=[dark2_1_cv2_act_r, dark2_1_cv3_1_act_r],
                option=['Conv33', 1, 1, 1])
        # 8
        dark2_1_cv3_2_r = dark2_1.cv3_2.conv(dark2_1_cv3_1_act_r)
        dark2_1_cv3_2_act_r = dark2_1.cv3_2.act(dark2_1_cv3_2_r)  # r3
        Transit(para1='backbone.dark2.1.cv3_1.conv', para2='backbone.dark2.1.cv3_2.conv',
                feature=[dark2_1_cv3_1_act_r, dark2_1_cv3_2_act_r],
                option=['Conv33', 1, 1, 1])
        # 9
        dark2_1_cv4_1_r = dark2_1.cv4_1.conv(dark2_1_cv3_2_act_r)
        dark2_1_cv4_1_act_r = dark2_1.cv4_1.act(dark2_1_cv4_1_r)
        Transit(para1='backbone.dark2.1.cv3_2.conv', para2='backbone.dark2.1.cv4_1.conv',
                feature=[dark2_1_cv3_2_act_r, dark2_1_cv4_1_act_r],
                option=['Conv33', 1, 1, 1])
        # 10
        dark2_1_cv4_2_r = dark2_1.cv4_2.conv(dark2_1_cv4_1_act_r)
        dark2_1_cv4_2_act_r = dark2_1.cv4_2.act(dark2_1_cv4_2_r)  # r4
        Transit(para1='backbone.dark2.1.cv4_1.conv', para2='backbone.dark2.1.cv4_2.conv',
                feature=[dark2_1_cv4_1_act_r, dark2_1_cv4_2_act_r],
                option=['Conv33', 1, 1, 1])

        # ----cat r4,r3,r2,r1----#
        # 11
        dark2_qf00_cat_r = dark2_1.qf00.cat([dark2_1_cv4_2_act_r, dark2_1_cv3_2_act_r], dim=1)
        Transit(para1='backbone.dark2.1.cv4_2.conv', para2='backbone.dark2.1.qf00', para3='backbone.dark2.1.cv3_2.conv',
                feature=[dark2_1_cv4_2_act_r, dark2_qf00_cat_r, dark2_1_cv3_2_act_r],
                option=['Concat'])
        # 12
        dark2_qf01_cat_r = dark2_1.qf01.cat([dark2_1_cv2_act_r, dark2_1_cv1_act_r], dim=1)
        Transit(para1='backbone.dark2.1.cv2.conv', para2='backbone.dark2.1.qf01', para3='backbone.dark2.1.cv1.conv',
                feature=[dark2_1_cv2_act_r, dark2_qf01_cat_r, dark2_1_cv1_act_r],
                option=['Concat'])
        # 13
        dark2_cat_r = dark2_1.qf0.cat([dark2_qf00_cat_r, dark2_qf01_cat_r], dim=1)
        Transit(para1='backbone.dark2.1.qf00', para2='backbone.dark2.1.qf0', para3='backbone.dark2.1.qf01',
                feature=[dark2_qf00_cat_r, dark2_cat_r, dark2_qf01_cat_r],
                option=['Concat'])

        # ---------cv5----------#
        # 14
        dark2_1_cv5_r = dark2_1.cv5.conv(dark2_cat_r)
        dark2_1_cv5_act_r = dark2_1.cv5.act(dark2_1_cv5_r)
        Transit(para1='backbone.dark2.1.qf0', para2='backbone.dark2.1.cv5.conv',
                feature=[dark2_cat_r, dark2_1_cv5_act_r],
                option=['Conv11', 1, 0, 1])

        # ------------ dark3 -------------- #
        # --------Transition_Block------left==mp+cv1,right=cv2+cv3--cat(right,left)--#
        # 15
        dark3_0_cv1_mp_r = dark3_0.mp(dark2_1_cv5_act_r)
        Transit(para1='backbone.dark2.1.cv5.conv', para2='backbone.dark2.1.cv5.conv',
                feature=[dark2_1_cv5_act_r, dark3_0_cv1_mp_r],
                option=['MaxPool'])
        # 16
        dark3_0_cv1_r = dark3_0.cv1.conv(dark3_0_cv1_mp_r)
        dark3_0_cv1_act_r = dark3_0.cv1.act(dark3_0_cv1_r)  # left
        Transit(para1='backbone.dark2.1.cv5.conv', para2='backbone.dark3.0.cv1.conv',
                feature=[dark3_0_cv1_mp_r, dark3_0_cv1_act_r],
                option=['Conv11', 1, 0, 1])
        # 17
        dark3_0_cv2_r = dark3_0.cv2.conv(dark2_1_cv5_act_r)
        dark3_0_cv2_act_r = dark3_0.cv2.act(dark3_0_cv2_r)
        Transit(para1='backbone.dark2.1.cv5.conv', para2='backbone.dark3.0.cv2.conv',
                feature=[dark2_1_cv5_act_r, dark3_0_cv2_act_r],
                option=['Conv11', 1, 0, 1])
        # 18
        dark3_0_cv3_r = dark3_0.cv3.conv(dark3_0_cv2_act_r)
        dark3_0_cv3_act_r = dark3_0.cv3.act(dark3_0_cv3_r)  # right
        Transit(para1='backbone.dark3.0.cv2.conv', para2='backbone.dark3.0.cv3.conv',
                feature=[dark3_0_cv2_act_r, dark3_0_cv3_act_r],
                option=['Conv33', 2, 1, 1])
        # -----Transition_Block--- cat --------#
        # 19
        dark3_0_cat_r = dark3_0.qf1.cat([dark3_0_cv3_act_r, dark3_0_cv1_act_r], dim=1)
        Transit(para1='backbone.dark3.0.cv3.conv', para2='backbone.dark3.0.qf1', para3='backbone.dark3.0.cv1.conv',
                feature=[dark3_0_cv3_act_r, dark3_0_cat_r, dark3_0_cv1_act_r],
                option=['Concat'])

        # -------- dark3.1--Multi_Concat_Block ---------#
        # 20
        dark3_1_cv1_r = dark3_1.cv1.conv(dark3_0_cat_r)
        dark3_1_cv1_act_r = dark3_1.cv1.act(dark3_1_cv1_r)  # r1
        Transit(para1='backbone.dark3.0.qf1', para2='backbone.dark3.1.cv1.conv',
                feature=[dark3_0_cat_r, dark3_1_cv1_act_r],
                option=['Conv11', 1, 0, 1])
        # 21
        dark3_1_cv2_r = dark3_1.cv2.conv(dark3_0_cat_r)
        dark3_1_cv2_act_r = dark3_1.cv2.act(dark3_1_cv2_r)  # r2
        Transit(para1='backbone.dark3.0.qf1', para2='backbone.dark3.1.cv2.conv',
                feature=[dark3_0_cat_r, dark3_1_cv2_act_r],
                option=['Conv11', 1, 0, 1])

        # 22
        dark3_1_cv3_1_r = dark3_1.cv3_1.conv(dark3_1_cv2_act_r)
        dark3_1_cv3_1_act_r = dark3_1.cv3_1.act(dark3_1_cv3_1_r)
        Transit(para1='backbone.dark3.1.cv2.conv', para2='backbone.dark3.1.cv3_1.conv',
                feature=[dark3_1_cv2_act_r, dark3_1_cv3_1_act_r],
                option=['Conv33', 1, 1, 1])
        # 23
        dark3_1_cv3_2_r = dark3_1.cv3_2.conv(dark3_1_cv3_1_act_r)
        dark3_1_cv3_2_act_r = dark3_1.cv3_2.act(dark3_1_cv3_2_r)  # r3
        Transit(para1='backbone.dark3.1.cv3_1.conv', para2='backbone.dark3.1.cv3_2.conv',
                feature=[dark3_1_cv3_1_act_r, dark3_1_cv3_2_act_r],
                option=['Conv33', 1, 1, 1])
        # 24
        dark3_1_cv4_1_r = dark3_1.cv4_1.conv(dark3_1_cv3_2_act_r)
        dark3_1_cv4_1_act_r = dark3_1.cv4_1.act(dark3_1_cv4_1_r)
        Transit(para1='backbone.dark3.1.cv3_2.conv', para2='backbone.dark3.1.cv4_1.conv',
                feature=[dark3_1_cv3_2_act_r, dark3_1_cv4_1_act_r],
                option=['Conv33', 1, 1, 1])
        # 25
        dark3_1_cv4_2_r = dark3_1.cv4_2.conv(dark3_1_cv4_1_act_r)
        dark3_1_cv4_2_act_r = dark3_1.cv4_2.act(dark3_1_cv4_2_r)  # r4
        Transit(para1='backbone.dark3.1.cv4_1.conv', para2='backbone.dark3.1.cv4_2.conv',
                feature=[dark3_1_cv4_1_act_r, dark3_1_cv4_2_act_r],
                option=['Conv33', 1, 1, 1])
        # ----cat r4,r3,r2,r1----#
        # 26
        dark3_qf00_cat_r = dark3_1.qf00.cat([dark3_1_cv4_2_act_r, dark3_1_cv3_2_act_r], dim=1)
        Transit(para1='backbone.dark3.1.cv4_2.conv', para2='backbone.dark3.1.qf00', para3='backbone.dark3.1.cv3_2.conv',
                feature=[dark3_1_cv4_2_act_r, dark3_qf00_cat_r, dark3_1_cv3_2_act_r],
                option=['Concat'])
        # 27
        dark3_qf01_cat_r = dark3_1.qf01.cat([dark3_1_cv2_act_r, dark3_1_cv1_act_r], dim=1)
        Transit(para1='backbone.dark3.1.cv2.conv', para2='backbone.dark3.1.qf01', para3='backbone.dark3.1.cv1.conv',
                feature=[dark3_1_cv2_act_r, dark3_qf01_cat_r, dark3_1_cv1_act_r],
                option=['Concat'])
        # 28
        dark3_cat_r = dark3_1.qf0.cat([dark3_qf00_cat_r, dark3_qf01_cat_r], dim=1)
        Transit(para1='backbone.dark3.1.qf00', para2='backbone.dark3.1.qf0', para3='backbone.dark3.1.qf01',
                feature=[dark3_qf00_cat_r, dark3_cat_r, dark3_qf01_cat_r],
                option=['Concat'])
        # 29 30 31
        dark3_1_cv5_r = dark3_1.cv5.conv(dark3_cat_r)
        dark3_1_cv5_act_r = dark3_1.cv5.act(dark3_1_cv5_r)  # feat1
        block(para1='backbone.dark3.1.qf0', para2='backbone.dark3.1.cv5.conv',
              feature=[dark3_cat_r, dark3_1_cv5_act_r],
              option=['Conv11', 1, 0, 1, 2])

        # ------------ dark4 -------------- #
        # --------Transition_Block------left==mp+cv1,right=cv2+cv3--cat(right,left)--#
        # 32
        dark4_0_cv1_mp_r = dark4_0.mp(dark3_1_cv5_act_r)
        Transit(para1='backbone.dark3.1.cv5.conv', para2='backbone.dark3.1.cv5.conv',
                feature=[dark3_1_cv5_act_r, dark4_0_cv1_mp_r],
                option=['MaxPool'])
        # 33
        dark4_0_cv1_r = dark4_0.cv1.conv(dark4_0_cv1_mp_r)
        dark4_0_cv1_act_r = dark4_0.cv1.act(dark4_0_cv1_r)  # left
        Transit(para1='backbone.dark3.1.cv5.conv', para2='backbone.dark4.0.cv1.conv',
                feature=[dark4_0_cv1_mp_r, dark4_0_cv1_act_r],
                option=['Conv11', 1, 0, 1])
        # 34
        dark4_0_cv2_r = dark4_0.cv2.conv(dark3_1_cv5_act_r)
        dark4_0_cv2_act_r = dark4_0.cv2.act(dark4_0_cv2_r)
        Transit(para1='backbone.dark3.1.cv5.conv', para2='backbone.dark4.0.cv2.conv',
                feature=[dark3_1_cv5_act_r, dark4_0_cv2_act_r],
                option=['Conv11', 1, 0, 1])
        # 35
        dark4_0_cv3_r = dark4_0.cv3.conv(dark4_0_cv2_act_r)
        dark4_0_cv3_act_r = dark4_0.cv3.act(dark4_0_cv3_r)  # right
        Transit(para1='backbone.dark4.0.cv2.conv', para2='backbone.dark4.0.cv3.conv',
                feature=[dark4_0_cv2_act_r, dark4_0_cv3_act_r],
                option=['Conv33', 2, 1, 1])

        # -----Transition_Block--- cat --------#
        # 36
        dark4_0_cat_r = dark4_0.qf1.cat([dark4_0_cv3_act_r, dark4_0_cv1_act_r], dim=1)
        Transit(para1='backbone.dark4.0.cv3.conv', para2='backbone.dark4.0.qf1', para3='backbone.dark4.0.cv1.conv',
                feature=[dark4_0_cv3_act_r, dark4_0_cat_r, dark4_0_cv1_act_r],
                option=['Concat'])
        # -------- dark4.1--Multi_Concat_Block ---------#
        # 37
        dark4_1_cv1_r = dark4_1.cv1.conv(dark4_0_cat_r)
        dark4_1_cv1_act_r = dark4_1.cv1.act(dark4_1_cv1_r)  # r1
        Transit(para1='backbone.dark4.0.qf1', para2='backbone.dark4.1.cv1.conv',
                feature=[dark4_0_cat_r, dark4_1_cv1_act_r],
                option=['Conv11', 1, 0, 1])
        # 38
        dark4_1_cv2_r = dark4_1.cv2.conv(dark4_0_cat_r)
        dark4_1_cv2_act_r = dark4_1.cv2.act(dark4_1_cv2_r)  # r2
        Transit(para1='backbone.dark4.0.qf1', para2='backbone.dark4.1.cv2.conv',
                feature=[dark4_0_cat_r, dark4_1_cv2_act_r],
                option=['Conv11', 1, 0, 1])
        # 39
        dark4_1_cv3_1_r = dark4_1.cv3_1.conv(dark4_1_cv2_act_r)
        dark4_1_cv3_1_act_r = dark4_1.cv3_1.act(dark4_1_cv3_1_r)
        Transit(para1='backbone.dark4.1.cv2.conv', para2='backbone.dark4.1.cv3_1.conv',
                feature=[dark4_1_cv2_act_r, dark4_1_cv3_1_act_r],
                option=['Conv33', 1, 1, 1])
        # 40
        dark4_1_cv3_2_r = dark4_1.cv3_2.conv(dark4_1_cv3_1_act_r)
        dark4_1_cv3_2_act_r = dark4_1.cv3_2.act(dark4_1_cv3_2_r)  # r3
        Transit(para1='backbone.dark4.1.cv3_1.conv', para2='backbone.dark4.1.cv3_2.conv',
                feature=[dark4_1_cv3_1_act_r, dark4_1_cv3_2_act_r],
                option=['Conv33', 1, 1, 1])
        # 41
        dark4_1_cv4_1_r = dark4_1.cv4_1.conv(dark4_1_cv3_2_act_r)
        dark4_1_cv4_1_act_r = dark4_1.cv4_1.act(dark4_1_cv4_1_r)
        Transit(para1='backbone.dark4.1.cv3_2.conv', para2='backbone.dark4.1.cv4_1.conv',
                feature=[dark4_1_cv3_2_act_r, dark4_1_cv4_1_act_r],
                option=['Conv33', 1, 1, 1])
        # 42
        dark4_1_cv4_2_r = dark4_1.cv4_2.conv(dark4_1_cv4_1_act_r)
        dark4_1_cv4_2_act_r = dark4_1.cv4_2.act(dark4_1_cv4_2_r)  # r4
        Transit(para1='backbone.dark4.1.cv4_1.conv', para2='backbone.dark4.1.cv4_2.conv',
                feature=[dark4_1_cv4_1_act_r, dark4_1_cv4_2_act_r],
                option=['Conv33', 1, 1, 1])

        # ----cat r4,r3,r2,r1----#
        # 43
        dark4_qf00_cat_r = dark4_1.qf00.cat([dark4_1_cv4_2_act_r, dark4_1_cv3_2_act_r], dim=1)
        Transit(para1='backbone.dark4.1.cv4_2.conv', para2='backbone.dark4.1.qf00', para3='backbone.dark4.1.cv3_2.conv',
                feature=[dark4_1_cv4_2_act_r, dark4_qf00_cat_r, dark4_1_cv3_2_act_r],
                option=['Concat'])
        # 44
        dark4_qf01_cat_r = dark4_1.qf01.cat([dark4_1_cv2_act_r, dark4_1_cv1_act_r], dim=1)
        Transit(para1='backbone.dark4.1.cv2.conv', para2='backbone.dark4.1.qf01', para3='backbone.dark4.1.cv1.conv',
                feature=[dark4_1_cv2_act_r, dark4_qf01_cat_r, dark4_1_cv1_act_r],
                option=['Concat'])
        # 45
        dark4_cat_r = dark4_1.qf0.cat([dark4_qf00_cat_r, dark4_qf01_cat_r], dim=1)
        Transit(para1='backbone.dark4.1.qf00', para2='backbone.dark4.1.qf0', para3='backbone.dark4.1.qf01',
                feature=[dark4_qf00_cat_r, dark4_cat_r, dark4_qf01_cat_r],
                option=['Concat'])
        # 46 47 48 49 50 51 52
        dark4_1_cv5_r = dark4_1.cv5.conv(dark4_cat_r)
        dark4_1_cv5_act_r = dark4_1.cv5.act(dark4_1_cv5_r)  # feat2
        block(para1='backbone.dark4.1.qf0', para2='backbone.dark4.1.cv5.conv',
              feature=[dark4_cat_r, dark4_1_cv5_act_r],
              option=['Conv11', 1, 0, 1, 4])

        # ------------ dark5 -------------- #
        # --------Transition_Block------left==mp+cv1,right=cv2+cv3--cat(right,left)--#
        # 53
        dark5_0_cv1_mp_r = dark5_0.mp(dark4_1_cv5_act_r)
        Transit(para1='backbone.dark4.1.cv5.conv', para2='backbone.dark4.1.cv5.conv',
                feature=[dark4_1_cv5_act_r, dark5_0_cv1_mp_r],
                option=['MaxPool'])
        # 54 55 56
        dark5_0_cv1_r = dark5_0.cv1.conv(dark5_0_cv1_mp_r)
        dark5_0_cv1_act_r = dark5_0.cv1.act(dark5_0_cv1_r)  # left
        block(para1='backbone.dark4.1.cv5.conv', para2='backbone.dark5.0.cv1.conv',
              feature=[dark5_0_cv1_mp_r, dark5_0_cv1_act_r],
              option=['Conv11', 1, 0, 1, 2])

        # 57 58 59
        dark5_0_cv2_r = dark5_0.cv2.conv(dark4_1_cv5_act_r)
        dark5_0_cv2_act_r = dark5_0.cv2.act(dark5_0_cv2_r)
        block(para1='backbone.dark4.1.cv5.conv', para2='backbone.dark5.0.cv2.conv',
              feature=[dark4_1_cv5_act_r, dark5_0_cv2_act_r],
              option=['Conv11', 1, 0, 1, 2])
        # 60 61 62
        dark5_0_cv3_r = dark5_0.cv3.conv(dark5_0_cv2_act_r)
        dark5_0_cv3_act_r = dark5_0.cv3.act(dark5_0_cv3_r)  # right
        block(para1='backbone.dark5.0.cv2.conv', para2='backbone.dark5.0.cv3.conv',
              feature=[dark5_0_cv2_act_r, dark5_0_cv3_act_r],
              option=['Conv33', 2, 1, 1, 2])

        # -----Transition_Block--- cat --------#1
        # 63
        dark5_0_cat_r = dark5_0.qf1.cat([dark5_0_cv3_act_r, dark5_0_cv1_act_r], dim=1)
        Transit(para1='backbone.dark5.0.cv3.conv', para2='backbone.dark5.0.qf1', para3='backbone.dark5.0.cv1.conv',
                feature=[dark5_0_cv3_act_r, dark5_0_cat_r, dark5_0_cv1_act_r],
                option=['Concat'])

        # -------- dark5.1--Multi_Concat_Block ---------#
        # 64
        dark5_1_cv1_r = dark5_1.cv1.conv(dark5_0_cat_r)
        dark5_1_cv1_act_r = dark5_1.cv1.act(dark5_1_cv1_r)  # r1
        Transit(para1='backbone.dark5.0.qf1', para2='backbone.dark5.1.cv1.conv',
                feature=[dark5_0_cat_r, dark5_1_cv1_act_r],
                option=['Conv11', 1, 0, 1])
        # 65
        dark5_1_cv2_r = dark5_1.cv2.conv(dark5_0_cat_r)
        dark5_1_cv2_act_r = dark5_1.cv2.act(dark5_1_cv2_r)  # r2
        Transit(para1='backbone.dark5.0.qf1', para2='backbone.dark5.1.cv2.conv',
                feature=[dark5_0_cat_r, dark5_1_cv2_act_r],
                option=['Conv11', 1, 0, 1])
        # 66
        dark5_1_cv3_1_r = dark5_1.cv3_1.conv(dark5_1_cv2_act_r)
        dark5_1_cv3_1_act_r = dark5_1.cv3_1.act(dark5_1_cv3_1_r)
        Transit(para1='backbone.dark5.1.cv2.conv', para2='backbone.dark5.1.cv3_1.conv',
                feature=[dark5_1_cv2_act_r, dark5_1_cv3_1_act_r],
                option=['Conv33', 1, 1, 1])
        # 67
        dark5_1_cv3_2_r = dark5_1.cv3_2.conv(dark5_1_cv3_1_act_r)
        dark5_1_cv3_2_act_r = dark5_1.cv3_2.act(dark5_1_cv3_2_r)  # r3
        Transit(para1='backbone.dark5.1.cv3_1.conv', para2='backbone.dark5.1.cv3_2.conv',
                feature=[dark5_1_cv3_1_act_r, dark5_1_cv3_2_act_r],
                option=['Conv33', 1, 1, 1])

        # 68
        dark5_1_cv4_1_r = dark5_1.cv4_1.conv(dark5_1_cv3_2_act_r)
        dark5_1_cv4_1_act_r = dark5_1.cv4_1.act(dark5_1_cv4_1_r)
        Transit(para1='backbone.dark5.1.cv3_2.conv', para2='backbone.dark5.1.cv4_1.conv',
                feature=[dark5_1_cv3_2_act_r, dark5_1_cv4_1_act_r],
                option=['Conv33', 1, 1, 1])
        # 69
        dark5_1_cv4_2_r = dark5_1.cv4_2.conv(dark5_1_cv4_1_act_r)
        dark5_1_cv4_2_act_r = dark5_1.cv4_2.act(dark5_1_cv4_2_r)  # r4
        Transit(para1='backbone.dark5.1.cv4_1.conv', para2='backbone.dark5.1.cv4_2.conv',
                feature=[dark5_1_cv4_1_act_r, dark5_1_cv4_2_act_r],
                option=['Conv33', 1, 1, 1])

        # ----cat r4,r3,r2,r1----#
        # 70
        dark5_qf00_cat_r = dark5_1.qf00.cat([dark5_1_cv4_2_act_r, dark5_1_cv3_2_act_r], dim=1)
        Transit(para1='backbone.dark5.1.cv4_2.conv', para2='backbone.dark5.1.qf00', para3='backbone.dark5.1.cv3_2.conv',
                feature=[dark5_1_cv4_2_act_r, dark5_qf00_cat_r, dark5_1_cv3_2_act_r],
                option=['Concat'])
        # 71
        dark5_qf01_cat_r = dark5_1.qf01.cat([dark5_1_cv2_act_r, dark5_1_cv1_act_r], dim=1)
        Transit(para1='backbone.dark5.1.cv2.conv', para2='backbone.dark5.1.qf01', para3='backbone.dark5.1.cv1.conv',
                feature=[dark5_1_cv2_act_r, dark5_qf01_cat_r, dark5_1_cv1_act_r],
                option=['Concat'])
        # 72
        dark5_cat_r = dark5_1.qf0.cat([dark5_qf00_cat_r, dark5_qf01_cat_r], dim=1)
        Transit(para1='backbone.dark5.1.qf00', para2='backbone.dark5.1.qf0', para3='backbone.dark5.1.qf01',
                feature=[dark5_qf00_cat_r, dark5_cat_r, dark5_qf01_cat_r],
                option=['Concat'])
        # 73 74 75 76 77 78 79
        dark5_1_cv5_r = dark5_1.cv5.conv(dark5_cat_r)
        dark5_1_cv5_act_r = dark5_1.cv5.act(dark5_1_cv5_r)
        block(para1='backbone.dark5.1.qf0', para2='backbone.dark5.1.cv5.conv',
              feature=[dark5_cat_r, dark5_1_cv5_act_r],
              option=['Conv11', 1, 0, 1, 4])

        # ------------------------ SPP------------------------------ #
        # -----Conv--#
        # 80
        spp_r = model.sppcspc.conv(dark5_1_cv5_act_r)
        spp_act_r = model.sppcspc.act(spp_r)  # feat3
        Transit(para1='backbone.dark5.1.cv5.conv', para2='sppcspc.conv',
                feature=[dark5_1_cv5_act_r, spp_act_r],
                option=['Conv11', 1, 0, 1])
        # ------------------------ FPN ------------------------------ #

        feat1 = dark3_1_cv5_act_r
        feat2 = dark4_1_cv5_act_r
        feat3 = spp_act_r  # 用于向下融合
        # 81
        for_P5_r = model.conv_for_P5.conv(feat3)
        for_P5_act_r = model.conv_for_P5.act(for_P5_r)
        Transit(para1='sppcspc.conv', para2='conv_for_P5.conv',
                feature=[feat3, for_P5_act_r],
                option=['Conv11', 1, 0, 1])
        # 82
        for_P5_upsample_r = model.upsample(for_P5_act_r)

        Transit(para1='conv_for_P5.conv', para2='conv_for_P5.conv',
                feature=[for_P5_act_r, for_P5_upsample_r],
                option=['UpSample'])
        # 83
        for_P4_r = model.conv_for_feat2.conv(feat2)
        for_P4_act_r = model.conv_for_feat2.act(for_P4_r)
        Transit(para1='backbone.dark4.1.cv5.conv', para2='conv_for_feat2.conv',
                feature=[feat2, for_P4_act_r],
                option=['Conv11', 1, 0, 1])
        # 84
        p54_cat_r = model.qf4.cat([for_P4_act_r, for_P5_upsample_r], dim=1)
        Transit(para1='conv_for_feat2.conv', para2='qf4', para3='conv_for_P5.conv',
                feature=[for_P4_act_r, p54_cat_r, for_P5_upsample_r],
                option=['Concat'])

        # -----向上融合的第一个----Multi_Concat_Block---- #
        # 85
        for_P54_Mul_cv1_r = model.conv3_for_upsample1.cv1.conv(p54_cat_r)
        for_P54_Mul_cv1_act_r = model.conv3_for_upsample1.cv1.act(for_P54_Mul_cv1_r)  # r1
        Transit(para1='qf4', para2='conv3_for_upsample1.cv1.conv',
                feature=[p54_cat_r, for_P54_Mul_cv1_act_r],
                option=['Conv11', 1, 0, 1])
        # 86
        for_P54_Mul_cv2_r = model.conv3_for_upsample1.cv2.conv(p54_cat_r)
        for_P54_Mul_cv2_act_r = model.conv3_for_upsample1.cv2.act(for_P54_Mul_cv2_r)  # r2
        Transit(para1='qf4', para2='conv3_for_upsample1.cv2.conv',
                feature=[p54_cat_r, for_P54_Mul_cv2_act_r],
                option=['Conv11', 1, 0, 1])
        # 87
        for_P54_Mul_cv3_1_r = model.conv3_for_upsample1.cv3_1.conv(for_P54_Mul_cv2_act_r)
        for_P54_Mul_cv3_1_act_r = model.conv3_for_upsample1.cv3_1.act(for_P54_Mul_cv3_1_r)
        Transit(para1='conv3_for_upsample1.cv2.conv', para2='conv3_for_upsample1.cv3_1.conv',
                feature=[for_P54_Mul_cv2_act_r, for_P54_Mul_cv3_1_act_r],
                option=['Conv33', 1, 1, 1])
        # 88
        for_P54_Mul_cv3_2_r = model.conv3_for_upsample1.cv3_2.conv(for_P54_Mul_cv3_1_act_r)
        for_P54_Mul_cv3_2_act_r = model.conv3_for_upsample1.cv3_2.act(for_P54_Mul_cv3_2_r)  # r3
        Transit(para1='conv3_for_upsample1.cv3_1.conv', para2='conv3_for_upsample1.cv3_2.conv',
                feature=[for_P54_Mul_cv3_1_act_r, for_P54_Mul_cv3_2_act_r],
                option=['Conv33', 1, 1, 1])
        # 89
        for_P54_Mul_cv4_1_r = model.conv3_for_upsample1.cv4_1.conv(for_P54_Mul_cv3_2_act_r)
        for_P54_Mul_cv4_1_act_r = model.conv3_for_upsample1.cv4_1.act(for_P54_Mul_cv4_1_r)
        Transit(para1='conv3_for_upsample1.cv3_2.conv', para2='conv3_for_upsample1.cv4_1.conv',
                feature=[for_P54_Mul_cv3_2_act_r, for_P54_Mul_cv4_1_act_r],
                option=['Conv33', 1, 1, 1])
        # 90
        for_P54_Mul_cv4_2_r = model.conv3_for_upsample1.cv4_2.conv(for_P54_Mul_cv4_1_act_r)
        for_P54_Mul_cv4_2_act_r = model.conv3_for_upsample1.cv4_2.act(for_P54_Mul_cv4_2_r)  # r4
        Transit(para1='conv3_for_upsample1.cv4_1.conv', para2='conv3_for_upsample1.cv4_2.conv',
                feature=[for_P54_Mul_cv4_1_act_r, for_P54_Mul_cv4_2_act_r],
                option=['Conv33', 1, 1, 1])
        # ----cat r4,r3,r2,r1----#
        # 91
        for_P54_Mul_qf00_cat_r = model.conv3_for_upsample1.qf00.cat(
            [for_P54_Mul_cv4_2_act_r, for_P54_Mul_cv3_2_act_r], dim=1)
        Transit(para1='conv3_for_upsample1.cv4_2.conv', para2='conv3_for_upsample1.qf00',
                para3='conv3_for_upsample1.cv3_2.conv',
                feature=[for_P54_Mul_cv4_2_act_r, for_P54_Mul_qf00_cat_r, for_P54_Mul_cv3_2_act_r],
                option=['Concat'])
        # 92
        for_P54_Mul_qf01_cat_r = model.conv3_for_upsample1.qf01.cat([for_P54_Mul_cv2_act_r, for_P54_Mul_cv1_act_r],
                                                                    dim=1)
        Transit(para1='conv3_for_upsample1.cv2.conv', para2='conv3_for_upsample1.qf01',
                para3='conv3_for_upsample1.cv1.conv',
                feature=[for_P54_Mul_cv2_act_r, for_P54_Mul_qf01_cat_r, for_P54_Mul_cv1_act_r],
                option=['Concat'])
        # 93
        for_P54_Mul_cat_r = model.conv3_for_upsample1.qf0.cat([for_P54_Mul_qf00_cat_r, for_P54_Mul_qf01_cat_r],
                                                              dim=1)
        Transit(para1='conv3_for_upsample1.qf00', para2='conv3_for_upsample1.qf0', para3='conv3_for_upsample1.qf01',
                feature=[for_P54_Mul_qf00_cat_r, for_P54_Mul_cat_r, for_P54_Mul_qf01_cat_r],
                option=['Concat'])
        # 94
        for_P54_Mul_cv5_r = model.conv3_for_upsample1.cv5.conv(for_P54_Mul_cat_r)
        for_P54_Mul_cv5_act_r = model.conv3_for_upsample1.cv5.act(for_P54_Mul_cv5_r)  # 用于向下融合
        Transit(para1='conv3_for_upsample1.qf0', para2='conv3_for_upsample1.cv5.conv',
                feature=[for_P54_Mul_cat_r, for_P54_Mul_cv5_act_r],
                option=['Conv11', 1, 0, 1])

        # 95
        for_P54_r = model.conv_for_P4.conv(for_P54_Mul_cv5_act_r)
        for_P54_act_r = model.conv_for_P4.act(for_P54_r)
        Transit(para1='conv3_for_upsample1.cv5.conv', para2='conv_for_P4.conv',
                feature=[for_P54_Mul_cv5_act_r, for_P54_act_r],
                option=['Conv11', 1, 0, 1])
        # 96
        for_P54_upsample_r = model.upsample(for_P54_act_r)
        Transit(para1='conv_for_P4.conv', para2='conv_for_P4.conv',
                feature=[for_P54_act_r, for_P54_upsample_r],
                option=['UpSample'])

        # 97
        for_P3_r = model.conv_for_feat1.conv(feat1)
        for_P3_act_r = model.conv_for_feat1.act(for_P3_r)
        Transit(para1='backbone.dark3.1.cv5.conv', para2='conv_for_feat1.conv',
                feature=[feat1, for_P3_act_r],
                option=['Conv11', 1, 0, 1])
        # 98
        P43_cat_r = model.qf5.cat([for_P3_act_r, for_P54_upsample_r], dim=1)
        Transit(para1='conv_for_feat1.conv', para2='qf5', para3='conv_for_P4.conv',
                feature=[for_P3_act_r, P43_cat_r, for_P54_upsample_r],
                option=['Concat'])

        # -----向上融合的第二个----Multi_Concat_Block---- #
        # 99
        for_P43_Mul_cv1_r = model.conv3_for_upsample2.cv1.conv(P43_cat_r)
        for_P43_Mul_cv1_act_r = model.conv3_for_upsample2.cv1.act(for_P43_Mul_cv1_r)  # r1
        Transit(para1='qf5', para2='conv3_for_upsample2.cv1.conv',
                feature=[P43_cat_r, for_P43_Mul_cv1_act_r],
                option=['Conv11', 1, 0, 1])
        # 100
        for_P43_Mul_cv2_r = model.conv3_for_upsample2.cv2.conv(P43_cat_r)
        for_P43_Mul_cv2_act_r = model.conv3_for_upsample2.cv2.act(for_P43_Mul_cv2_r)  # r2
        Transit(para1='qf5', para2='conv3_for_upsample2.cv2.conv',
                feature=[P43_cat_r, for_P43_Mul_cv2_act_r],
                option=['Conv11', 1, 0, 1])
        # 101
        for_P43_Mul_cv3_1_r = model.conv3_for_upsample2.cv3_1.conv(for_P43_Mul_cv2_act_r)
        for_P43_Mul_cv3_1_act_r = model.conv3_for_upsample2.cv3_1.act(for_P43_Mul_cv3_1_r)
        Transit(para1='conv3_for_upsample2.cv2.conv', para2='conv3_for_upsample2.cv3_1.conv',
                feature=[for_P43_Mul_cv2_act_r, for_P43_Mul_cv3_1_act_r],
                option=['Conv33', 1, 1, 1])
        # 102
        for_P43_Mul_cv3_2_r = model.conv3_for_upsample2.cv3_2.conv(for_P43_Mul_cv3_1_act_r)
        for_P43_Mul_cv3_2_act_r = model.conv3_for_upsample2.cv3_2.act(for_P43_Mul_cv3_2_r)  # r3
        Transit(para1='conv3_for_upsample2.cv3_1.conv', para2='conv3_for_upsample2.cv3_2.conv',
                feature=[for_P43_Mul_cv3_1_act_r, for_P43_Mul_cv3_2_act_r],
                option=['Conv33', 1, 1, 1])
        # 103
        for_P43_Mul_cv4_1_r = model.conv3_for_upsample2.cv4_1.conv(for_P43_Mul_cv3_2_act_r)
        for_P43_Mul_cv4_1_act_r = model.conv3_for_upsample2.cv4_1.act(for_P43_Mul_cv4_1_r)
        Transit(para1='conv3_for_upsample2.cv3_2.conv', para2='conv3_for_upsample2.cv4_1.conv',
                feature=[for_P43_Mul_cv3_2_act_r, for_P43_Mul_cv4_1_r],
                option=['Conv33', 1, 1, 1])
        # 104
        for_P43_Mul_cv4_2_r = model.conv3_for_upsample2.cv4_2.conv(for_P43_Mul_cv4_1_act_r)
        for_P43_Mul_cv4_2_act_r = model.conv3_for_upsample2.cv4_2.act(for_P43_Mul_cv4_2_r)  # r4
        Transit(para1='conv3_for_upsample2.cv4_1.conv', para2='conv3_for_upsample2.cv4_2.conv',
                feature=[for_P43_Mul_cv4_1_act_r, for_P43_Mul_cv4_2_act_r],
                option=['Conv33', 1, 1, 1])
        # ----cat r4,r3,r2,r1----#
        # 105
        for_P43_Mul_qf00_cat_r = model.conv3_for_upsample2.qf00.cat(
            [for_P43_Mul_cv4_2_act_r, for_P43_Mul_cv3_2_act_r], dim=1)
        Transit(para1='conv3_for_upsample2.cv4_2.conv', para2='conv3_for_upsample2.qf00',
                para3='conv3_for_upsample2.cv3_2.conv',
                feature=[for_P43_Mul_cv4_2_act_r, for_P43_Mul_qf00_cat_r, for_P43_Mul_cv3_2_act_r],
                option=['Concat'])
        # 106
        for_P43_Mul_qf01_cat_r = model.conv3_for_upsample2.qf01.cat([for_P43_Mul_cv2_act_r, for_P43_Mul_cv1_act_r],
                                                                    dim=1)
        Transit(para1='conv3_for_upsample2.cv2.conv', para2='conv3_for_upsample2.qf01',
                para3='conv3_for_upsample2.cv1.conv',
                feature=[for_P43_Mul_cv2_act_r, for_P43_Mul_qf01_cat_r, for_P43_Mul_cv1_act_r],
                option=['Concat'])
        # 107
        for_P43_Mul_cat_r = model.conv3_for_upsample2.qf0.cat([for_P43_Mul_qf00_cat_r, for_P43_Mul_qf01_cat_r],
                                                              dim=1)
        Transit(para1='conv3_for_upsample2.qf00', para2='conv3_for_upsample2.qf0', para3='conv3_for_upsample2.qf01',
                feature=[for_P43_Mul_qf00_cat_r, for_P43_Mul_cat_r, for_P43_Mul_qf01_cat_r],
                option=['Concat'])
        # 108
        for_P43_Mul_cv5_r = model.conv3_for_upsample2.cv5.conv(for_P43_Mul_cat_r)
        for_P43_Mul_cv5_act_r = model.conv3_for_upsample2.cv5.act(for_P43_Mul_cv5_r)  # 用于向下融合,且是 P3
        Transit(para1='conv3_for_upsample2.qf0', para2='conv3_for_upsample2.cv5.conv',
                feature=[for_P43_Mul_cat_r, for_P43_Mul_cv5_act_r],
                option=['Conv11', 1, 0, 1])

        # -----向下融合的第一个----Transition_Block---- #
        # 109
        for_P34_mp_r = model.down_sample1.mp(for_P43_Mul_cv5_act_r)
        Transit(para1='conv3_for_upsample2.cv5.conv', para2='conv3_for_upsample2.cv5.conv',
                feature=[for_P43_Mul_cv5_act_r, for_P34_mp_r],
                option=['MaxPool'])
        # 110
        for_P34_cv1_r = model.down_sample1.cv1.conv(for_P34_mp_r)
        for_P34_cv1_act_r = model.down_sample1.cv1.act(for_P34_cv1_r)  # left
        Transit(para1='conv3_for_upsample2.cv5.conv', para2='down_sample1.cv1.conv',
                feature=[for_P34_mp_r, for_P34_cv1_act_r],
                option=['Conv11', 1, 0, 1])
        # 111
        for_P34_cv2_r = model.down_sample1.cv2.conv(for_P43_Mul_cv5_act_r)
        for_P34_cv2_act_r = model.down_sample1.cv2.act(for_P34_cv2_r)
        Transit(para1='conv3_for_upsample2.cv5.conv', para2='down_sample1.cv2.conv',
                feature=[for_P43_Mul_cv5_act_r, for_P34_cv2_act_r],
                option=['Conv11', 1, 0, 1])
        # 112
        for_P34_cv3_r = model.down_sample1.cv3.conv(for_P34_cv2_act_r)
        for_P34_cv3_act_r = model.down_sample1.cv3.act(for_P34_cv3_r)  # right
        Transit(para1='down_sample1.cv2.conv', para2='down_sample1.cv3.conv',
                feature=[for_P34_cv2_act_r, for_P34_cv3_act_r],
                option=['Conv33', 2, 1, 1])

        # 113
        for_P34_cat_r = model.down_sample1.qf1.cat([for_P34_cv3_act_r, for_P34_cv1_act_r], dim=1)
        Transit(para1='down_sample1.cv3.conv', para2='down_sample1.qf1', para3='down_sample1.cv1.conv',
                feature=[for_P34_cv3_act_r, for_P34_cat_r, for_P34_cv1_act_r],
                option=['Concat'])

        # 114
        P34_cat_r = model.qf6.cat([for_P34_cat_r, for_P54_Mul_cv5_act_r], dim=1)
        Transit(para1='down_sample1.qf1', para2='qf6', para3='conv3_for_upsample1.cv5.conv',
                feature=[for_P34_cat_r, P34_cat_r, for_P54_Mul_cv5_act_r],
                option=['Concat'])

        # -----向下融合的第一个----Multi_Concat_Block---- #
        # 115
        for_P34_Mul_cv1_r = model.conv3_for_downsample1.cv1.conv(P34_cat_r)
        for_P34_Mul_cv1_act_r = model.conv3_for_downsample1.cv1.act(for_P34_Mul_cv1_r)  # r1
        Transit(para1='qf6', para2='conv3_for_downsample1.cv1.conv',
                feature=[P34_cat_r, for_P34_Mul_cv1_act_r],
                option=['Conv11', 1, 0, 1])
        # 116
        for_P34_Mul_cv2_r = model.conv3_for_downsample1.cv2.conv(P34_cat_r)
        for_P34_Mul_cv2_act_r = model.conv3_for_downsample1.cv2.act(for_P34_Mul_cv2_r)  # r2
        Transit(para1='qf6', para2='conv3_for_downsample1.cv2.conv',
                feature=[P34_cat_r, for_P34_Mul_cv2_act_r],
                option=['Conv11', 1, 0, 1])
        # 117
        for_P34_Mul_cv3_1_r = model.conv3_for_downsample1.cv3_1.conv(for_P34_Mul_cv2_act_r)
        for_P34_Mul_cv3_1_act_r = model.conv3_for_downsample1.cv3_1.act(for_P34_Mul_cv3_1_r)
        Transit(para1='conv3_for_downsample1.cv2.conv', para2='conv3_for_downsample1.cv3_1.conv',
                feature=[for_P34_Mul_cv2_act_r, for_P34_Mul_cv3_1_act_r],
                option=['Conv33', 1, 1, 1])
        # 118
        for_P34_Mul_cv3_2_r = model.conv3_for_downsample1.cv3_2.conv(for_P34_Mul_cv3_1_act_r)
        for_P34_Mul_cv3_2_act_r = model.conv3_for_downsample1.cv3_2.act(for_P34_Mul_cv3_2_r)  # r3
        Transit(para1='conv3_for_downsample1.cv3_1.conv', para2='conv3_for_downsample1.cv3_2.conv',
                feature=[for_P34_Mul_cv3_1_act_r, for_P34_Mul_cv3_2_act_r],
                option=['Conv33', 1, 1, 1])
        # 119
        for_P34_Mul_cv4_1_r = model.conv3_for_downsample1.cv4_1.conv(for_P34_Mul_cv3_2_act_r)
        for_P34_Mul_cv4_1_act_r = model.conv3_for_downsample1.cv4_1.act(for_P34_Mul_cv4_1_r)
        Transit(para1='conv3_for_downsample1.cv3_2.conv', para2='conv3_for_downsample1.cv4_1.conv',
                feature=[for_P34_Mul_cv3_2_act_r, for_P34_Mul_cv4_1_act_r],
                option=['Conv33', 1, 1, 1])
        # 120
        for_P34_Mul_cv4_2_r = model.conv3_for_downsample1.cv4_2.conv(for_P34_Mul_cv4_1_act_r)
        for_P34_Mul_cv4_2_act_r = model.conv3_for_downsample1.cv4_2.act(for_P34_Mul_cv4_2_r)  # r4
        Transit(para1='conv3_for_downsample1.cv4_1.conv', para2='conv3_for_downsample1.cv4_2.conv',
                feature=[for_P34_Mul_cv4_1_act_r, for_P34_Mul_cv4_2_act_r],
                option=['Conv33', 1, 1, 1])
        # ----cat r4,r3,r2,r1----#
        # 121
        for_P34_Mul_qf00_cat_r = model.conv3_for_downsample1.qf00.cat(
            [for_P34_Mul_cv4_2_act_r, for_P34_Mul_cv3_2_act_r],
            dim=1)
        Transit(para1='conv3_for_downsample1.cv4_2.conv', para2='conv3_for_downsample1.qf00',
                para3='conv3_for_downsample1.cv3_2.conv',
                feature=[for_P34_Mul_cv4_2_act_r, for_P34_Mul_qf00_cat_r, for_P34_Mul_cv3_2_act_r],
                option=['Concat'])
        # 122
        for_P34_Mul_qf01_cat_r = model.conv3_for_downsample1.qf01.cat(
            [for_P34_Mul_cv2_act_r, for_P34_Mul_cv1_act_r], dim=1)
        Transit(para1='conv3_for_downsample1.cv2.conv', para2='conv3_for_downsample1.qf01',
                para3='conv3_for_downsample1.cv1.conv',
                feature=[for_P34_Mul_cv2_act_r, for_P34_Mul_qf01_cat_r, for_P34_Mul_cv1_act_r],
                option=['Concat'])
        # 123
        for_P34_Mul_cat_r = model.conv3_for_downsample1.qf0.cat([for_P34_Mul_qf00_cat_r, for_P34_Mul_qf01_cat_r],
                                                                dim=1)
        Transit(para1='conv3_for_downsample1.qf00', para2='conv3_for_downsample1.qf0',
                para3='conv3_for_downsample1.qf01',
                feature=[for_P34_Mul_qf00_cat_r, for_P34_Mul_cat_r, for_P34_Mul_qf01_cat_r],
                option=['Concat'])
        # 124
        for_P34_Mul_cv5_r = model.conv3_for_downsample1.cv5.conv(for_P34_Mul_cat_r)
        for_P34_Mul_cv5_act_r = model.conv3_for_downsample1.cv5.act(for_P34_Mul_cv5_r)  # P4
        Transit(para1='conv3_for_downsample1.qf0', para2='conv3_for_downsample1.cv5.conv',
                feature=[for_P34_Mul_cat_r, for_P34_Mul_cv5_act_r],
                option=['Conv11', 1, 0, 1])

        # -----向下融合的第二个----Transition_Block---- #
        # 125
        for_P45_mp_r = model.down_sample2.mp(for_P34_Mul_cv5_act_r)
        Transit(para1='conv3_for_downsample1.cv5.conv', para2='conv3_for_downsample1.cv5.conv',
                feature=[for_P34_Mul_cv5_act_r, for_P45_mp_r],
                option=['MaxPool'])
        # 126
        for_P45_cv1_r = model.down_sample2.cv1.conv(for_P45_mp_r)
        for_P45_cv1_act_r = model.down_sample2.cv1.act(for_P45_cv1_r)  # left
        Transit(para1='conv3_for_downsample1.cv5.conv', para2='down_sample2.cv1.conv',
                feature=[for_P45_mp_r, for_P45_cv1_act_r],
                option=['Conv11', 1, 0, 1])
        # 127
        for_P45_cv2_r = model.down_sample2.cv2.conv(for_P34_Mul_cv5_act_r)
        for_P45_cv2_act_r = model.down_sample2.cv2.act(for_P45_cv2_r)
        Transit(para1='conv3_for_downsample1.cv5.conv', para2='down_sample2.cv2.conv',
                feature=[for_P34_Mul_cv5_act_r, for_P45_cv2_act_r],
                option=['Conv11', 1, 0, 1])
        # 128
        for_P45_cv3_r = model.down_sample2.cv3.conv(for_P45_cv2_act_r)
        for_P45_cv3_act_r = model.down_sample2.cv3.act(for_P45_cv3_r)  # right
        Transit(para1='down_sample2.cv2.conv', para2='down_sample2.cv3.conv',
                feature=[for_P45_cv2_act_r, for_P45_cv3_act_r],
                option=['Conv33', 2, 1, 1])
        # 129
        for_P45_cat_r = model.down_sample2.qf1.cat([for_P45_cv3_act_r, for_P45_cv1_act_r], dim=1)
        Transit(para1='down_sample2.cv3.conv', para2='down_sample2.qf1', para3='down_sample2.cv1.conv',
                feature=[for_P45_cv3_act_r, for_P45_cat_r, for_P45_cv1_act_r],
                option=['Concat'])
        # 130
        P45_cat_r = model.qf7.cat([for_P45_cat_r, feat3], dim=1)
        Transit(para1='down_sample2.qf1', para2='qf7', para3='sppcspc.conv',
                feature=[for_P45_cat_r, P45_cat_r, feat3],
                option=['Concat'])

        # -----向下融合的第二个----Multi_Concat_Block---- #
        # 131
        for_P45_Mul_cv1_r = model.conv3_for_downsample2.cv1.conv(P45_cat_r)
        for_P45_Mul_cv1_act_r = model.conv3_for_downsample2.cv1.act(for_P45_Mul_cv1_r)  # r1
        Transit(para1='qf7', para2='conv3_for_downsample2.cv1.conv',
                feature=[P45_cat_r, for_P45_Mul_cv1_act_r],
                option=['Conv11', 1, 0, 1])
        # 132
        for_P45_Mul_cv2_r = model.conv3_for_downsample2.cv2.conv(P45_cat_r)
        for_P45_Mul_cv2_act_r = model.conv3_for_downsample2.cv2.act(for_P45_Mul_cv2_r)  # r2
        Transit(para1='qf7', para2='conv3_for_downsample2.cv2.conv',
                feature=[P45_cat_r, for_P45_Mul_cv2_act_r],
                option=['Conv11', 1, 0, 1])
        # 133
        for_P45_Mul_cv3_1_r = model.conv3_for_downsample2.cv3_1.conv(for_P45_Mul_cv2_act_r)
        for_P45_Mul_cv3_1_act_r = model.conv3_for_downsample2.cv3_1.act(for_P45_Mul_cv3_1_r)
        Transit(para1='conv3_for_downsample2.cv2.conv', para2='conv3_for_downsample2.cv3_1.conv',
                feature=[for_P45_Mul_cv2_act_r, for_P45_Mul_cv3_1_act_r],
                option=['Conv33', 1, 1, 1])
        # 134
        for_P45_Mul_cv3_2_r = model.conv3_for_downsample2.cv3_2.conv(for_P45_Mul_cv3_1_act_r)
        for_P45_Mul_cv3_2_act_r = model.conv3_for_downsample2.cv3_2.act(for_P45_Mul_cv3_2_r)  # r3
        Transit(para1='conv3_for_downsample2.cv3_1.conv', para2='conv3_for_downsample2.cv3_2.conv',
                feature=[for_P45_Mul_cv3_1_act_r, for_P45_Mul_cv3_2_act_r],
                option=['Conv33', 1, 1, 1])
        # 135
        for_P45_Mul_cv4_1_r = model.conv3_for_downsample2.cv4_1.conv(for_P45_Mul_cv3_2_act_r)
        for_P45_Mul_cv4_1_act_r = model.conv3_for_downsample2.cv4_1.act(for_P45_Mul_cv4_1_r)
        Transit(para1='conv3_for_downsample2.cv3_2.conv', para2='conv3_for_downsample2.cv4_1.conv',
                feature=[for_P45_Mul_cv3_2_act_r, for_P45_Mul_cv4_1_act_r],
                option=['Conv33', 1, 1, 1])
        # 136
        for_P45_Mul_cv4_2_r = model.conv3_for_downsample2.cv4_2.conv(for_P45_Mul_cv4_1_act_r)
        for_P45_Mul_cv4_2_act_r = model.conv3_for_downsample2.cv4_2.act(for_P45_Mul_cv4_2_r)  # r4
        Transit(para1='conv3_for_downsample2.cv4_1.conv', para2='conv3_for_downsample2.cv4_2.conv',
                feature=[for_P45_Mul_cv4_1_act_r, for_P45_Mul_cv4_2_act_r],
                option=['Conv33', 1, 1, 1])
        # ----cat r4,r3,r2,r1----#
        # 137
        for_P45_Mul_qf00_cat_r = model.conv3_for_downsample2.qf00.cat(
            [for_P45_Mul_cv4_2_act_r, for_P45_Mul_cv3_2_act_r],
            dim=1)
        Transit(para1='conv3_for_downsample2.cv4_2.conv', para2='conv3_for_downsample2.qf00',
                para3='conv3_for_downsample2.cv3_2.conv',
                feature=[for_P45_Mul_cv4_2_act_r, for_P45_Mul_qf00_cat_r, for_P45_Mul_cv3_2_act_r],
                option=['Concat'])
        # 138
        for_P45_Mul_qf01_cat_r = model.conv3_for_downsample2.qf01.cat(
            [for_P45_Mul_cv2_act_r, for_P45_Mul_cv1_act_r], dim=1)
        Transit(para1='conv3_for_downsample2.cv2.conv', para2='conv3_for_downsample2.qf01',
                para3='conv3_for_downsample2.cv1.conv',
                feature=[for_P45_Mul_cv2_act_r, for_P45_Mul_qf01_cat_r, for_P45_Mul_cv1_act_r],
                option=['Concat'])
        # 139
        for_P45_Mul_cat_r = model.conv3_for_downsample2.qf0.cat([for_P45_Mul_qf00_cat_r, for_P45_Mul_qf01_cat_r],
                                                                dim=1)
        Transit(para1='conv3_for_downsample2.qf00', para2='conv3_for_downsample2.qf0',
                para3='conv3_for_downsample2.qf01',
                feature=[for_P45_Mul_qf00_cat_r, for_P45_Mul_cat_r, for_P45_Mul_qf01_cat_r],
                option=['Concat'])
        # 140
        for_P45_Mul_cv5_r = model.conv3_for_downsample2.cv5.conv(for_P45_Mul_cat_r)
        for_P45_Mul_cv5_act_r = model.conv3_for_downsample2.cv5.act(for_P45_Mul_cv5_r)  # P5
        Transit(para1='conv3_for_downsample2.qf0', para2='conv3_for_downsample2.cv5.conv',
                feature=[for_P45_Mul_cat_r, for_P45_Mul_cv5_act_r],
                option=['Conv11', 1, 0, 1])

        # ------------------------Head------------------------------ #
        P3 = for_P43_Mul_cv5_act_r
        P4 = for_P34_Mul_cv5_act_r
        P5 = for_P45_Mul_cv5_act_r
        # 141
        rep_conv_1_r = model.rep_conv_1.conv(P3)
        rep_conv_1_act_r = model.rep_conv_1.act(rep_conv_1_r)
        Transit(para1='conv3_for_upsample2.cv5.conv', para2='rep_conv_1.conv',
                feature=[P3, rep_conv_1_act_r],
                option=['Conv33', 1, 1, 1])
        # 142
        rep_conv_2_r = model.rep_conv_2.conv(P4)
        rep_conv_2_act_r = model.rep_conv_2.act(rep_conv_2_r)
        Transit(para1='conv3_for_downsample1.cv5.conv', para2='rep_conv_2.conv',
                feature=[P4, rep_conv_2_act_r],
                option=['Conv33', 1, 1, 1])
        # 143
        rep_conv_3_r = model.rep_conv_3.conv(P5)
        rep_conv_3_act_r = model.rep_conv_3.act(rep_conv_3_r)
        Transit(para1='conv3_for_downsample2.cv5.conv', para2='rep_conv_3.conv',
                feature=[P5, rep_conv_3_act_r],
                option=['Conv33', 1, 1, 1])

        out0 = model.yolo_head_P3(rep_conv_1_act_r)
        out1 = model.yolo_head_P4(rep_conv_2_act_r)
        out2 = model.yolo_head_P5(rep_conv_3_act_r)
        out0 = model.dequant(out0)
        out1 = model.dequant(out1)
        out2 = model.dequant(out2)
        out = [out2, out1, out0]
        print('#================前向推理完毕==================#\n')
        result = model(img)
        if result[0].equal(out[0]) & result[1].equal(out[1]) & result[2].equal(out[2]):
            print('ok')
        else:
            print('error')


if __name__ == '__main__':
    create_files()
