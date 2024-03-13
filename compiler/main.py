import cv2
import numpy as np
import torch
from torch import nn

from compiler.lib.array_format import picture_load

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

    image_path = shared.img_path
    image_size = shared.img_size
    # 输入图片以灰度图形式读取
    input_img = cv2.imread(image_path, 0)
    # 输入图片转为规定的尺寸
    if input_img.shape[0] != image_size:
        input_img = cv2.resize(input_img, (image_size, image_size))
    input_img = np.expand_dims(input_img,axis = 0)
    input_img = np.expand_dims(input_img,axis = 0)


    # picture_load为模型训练时的预处理方法，一般包含了将图片尺寸统一、转为灰度图、归一化{[(image/255)-mean]/std}
    img = picture_load(shared)
    with torch.no_grad():
        quant_feature_f = model.quant(img)
        # ------------------------------------api网络重命名--------------------------------------------------------
        quant = model.quant
        dequant = model.dequant
        # ------------------------ContextPath------------------------------
        ContextPath = model.cp
        # ------------------------cp.resnet18------------------------------
        cp_Resnet18 = ContextPath.resnet
        cp_Resnet18_conv1 = cp_Resnet18.conv1
        cp_Resnet18_bn1 = cp_Resnet18.bn1
        cp_Resnet18_relu = cp_Resnet18.relu
        cp_Resnet18_maxpool = cp_Resnet18.maxpool

        # ------------------------cp.resnet18.layer1------------------------------
        cp_Resnet18_layer1 = cp_Resnet18.layer1
        cp_Resnet18_layer1_list = list(cp_Resnet18_layer1.children())
        cp_Resnet18_layer1_b0 = cp_Resnet18_layer1_list[0]
        cp_Resnet18_layer1_b0_conv1 = cp_Resnet18_layer1_b0.conv1
        cp_Resnet18_layer1_b0_bn1 = cp_Resnet18_layer1_b0.bn1
        cp_Resnet18_layer1_b0_conv2 = cp_Resnet18_layer1_b0.conv2
        cp_Resnet18_layer1_b0_bn2 = cp_Resnet18_layer1_b0.bn2
        cp_Resnet18_layer1_b0_relu1 = cp_Resnet18_layer1_b0.relu1
        cp_Resnet18_layer1_b0_relu2 = cp_Resnet18_layer1_b0.relu2
        cp_Resnet18_layer1_b0_qf = cp_Resnet18_layer1_b0.qf

        cp_Resnet18_layer1_b1 = cp_Resnet18_layer1_list[1]
        cp_Resnet18_layer1_b1_conv1 = cp_Resnet18_layer1_b1.conv1
        cp_Resnet18_layer1_b1_bn1 = cp_Resnet18_layer1_b1.bn1
        cp_Resnet18_layer1_b1_conv2 = cp_Resnet18_layer1_b1.conv2
        cp_Resnet18_layer1_b1_bn2 = cp_Resnet18_layer1_b1.bn2
        cp_Resnet18_layer1_b1_relu1 = cp_Resnet18_layer1_b1.relu1
        cp_Resnet18_layer1_b1_relu2 = cp_Resnet18_layer1_b1.relu2
        cp_Resnet18_layer1_b1_qf = cp_Resnet18_layer1_b1.qf

        # ------------------------cp.resnet18.layer2------------------------------
        cp_Resnet18_layer2 = cp_Resnet18.layer2
        cp_Resnet18_layer2_list = list(cp_Resnet18_layer2.children())
        cp_Resnet18_layer2_b0 = cp_Resnet18_layer2_list[0]
        cp_Resnet18_layer2_b0_conv1 = cp_Resnet18_layer2_b0.conv1
        cp_Resnet18_layer2_b0_bn1 = cp_Resnet18_layer2_b0.bn1
        cp_Resnet18_layer2_b0_conv2 = cp_Resnet18_layer2_b0.conv2
        cp_Resnet18_layer2_b0_bn2 = cp_Resnet18_layer2_b0.bn2
        cp_Resnet18_layer2_b0_relu1 = cp_Resnet18_layer2_b0.relu1
        cp_Resnet18_layer2_b0_relu2 = cp_Resnet18_layer2_b0.relu2
        cp_Resnet18_layer2_b0_downsample = cp_Resnet18_layer2_b0.downsample
        cp_Resnet18_layer2_b0_qf = cp_Resnet18_layer2_b0.qf

        cp_Resnet18_layer2_b1 = cp_Resnet18_layer2_list[1]
        cp_Resnet18_layer2_b1_conv1 = cp_Resnet18_layer2_b1.conv1
        cp_Resnet18_layer2_b1_bn1 = cp_Resnet18_layer2_b1.bn1
        cp_Resnet18_layer2_b1_conv2 = cp_Resnet18_layer2_b1.conv2
        cp_Resnet18_layer2_b1_bn2 = cp_Resnet18_layer2_b1.bn2
        cp_Resnet18_layer2_b1_relu1 = cp_Resnet18_layer2_b1.relu1
        cp_Resnet18_layer2_b1_relu2 = cp_Resnet18_layer2_b1.relu2
        cp_Resnet18_layer2_b1_qf = cp_Resnet18_layer2_b1.qf

        # ------------------------cp.resnet18.layer3------------------------------
        cp_Resnet18_layer3 = cp_Resnet18.layer3
        cp_Resnet18_layer3_list = list(cp_Resnet18_layer3.children())
        cp_Resnet18_layer3_b0 = cp_Resnet18_layer3_list[0]
        cp_Resnet18_layer3_b0_conv1 = cp_Resnet18_layer3_b0.conv1
        cp_Resnet18_layer3_b0_bn1 = cp_Resnet18_layer3_b0.bn1
        cp_Resnet18_layer3_b0_conv2 = cp_Resnet18_layer3_b0.conv2
        cp_Resnet18_layer3_b0_bn2 = cp_Resnet18_layer3_b0.bn2
        cp_Resnet18_layer3_b0_relu1 = cp_Resnet18_layer3_b0.relu1
        cp_Resnet18_layer3_b0_relu2 = cp_Resnet18_layer3_b0.relu2
        cp_Resnet18_layer3_b0_downsample = cp_Resnet18_layer3_b0.downsample
        cp_Resnet18_layer3_b0_qf = cp_Resnet18_layer3_b0.qf

        cp_Resnet18_layer3_b1 = cp_Resnet18_layer3_list[1]
        cp_Resnet18_layer3_b1_conv1 = cp_Resnet18_layer3_b1.conv1
        cp_Resnet18_layer3_b1_bn1 = cp_Resnet18_layer3_b1.bn1
        cp_Resnet18_layer3_b1_conv2 = cp_Resnet18_layer3_b1.conv2
        cp_Resnet18_layer3_b1_bn2 = cp_Resnet18_layer3_b1.bn2
        cp_Resnet18_layer3_b1_relu1 = cp_Resnet18_layer3_b1.relu1
        cp_Resnet18_layer3_b1_relu2 = cp_Resnet18_layer3_b1.relu2
        cp_Resnet18_layer3_b1_qf = cp_Resnet18_layer3_b1.qf

        # ------------------------cp.resnet18.layer4------------------------------
        cp_Resnet18_layer4 = cp_Resnet18.layer4
        cp_Resnet18_layer4_list = list(cp_Resnet18_layer4.children())
        cp_Resnet18_layer4_b0 = cp_Resnet18_layer4_list[0]
        cp_Resnet18_layer4_b0_conv1 = cp_Resnet18_layer4_b0.conv1
        cp_Resnet18_layer4_b0_bn1 = cp_Resnet18_layer4_b0.bn1
        cp_Resnet18_layer4_b0_conv2 = cp_Resnet18_layer4_b0.conv2
        cp_Resnet18_layer4_b0_bn2 = cp_Resnet18_layer4_b0.bn2
        cp_Resnet18_layer4_b0_relu1 = cp_Resnet18_layer4_b0.relu1
        cp_Resnet18_layer4_b0_relu2 = cp_Resnet18_layer4_b0.relu2
        cp_Resnet18_layer4_b0_downsample = cp_Resnet18_layer4_b0.downsample
        cp_Resnet18_layer4_b0_qf = cp_Resnet18_layer4_b0.qf

        cp_Resnet18_layer4_b1 = cp_Resnet18_layer4_list[1]
        cp_Resnet18_layer4_b1_conv1 = cp_Resnet18_layer4_b1.conv1
        cp_Resnet18_layer4_b1_bn1 = cp_Resnet18_layer4_b1.bn1
        cp_Resnet18_layer4_b1_conv2 = cp_Resnet18_layer4_b1.conv2
        cp_Resnet18_layer4_b1_bn2 = cp_Resnet18_layer4_b1.bn2
        cp_Resnet18_layer4_b1_relu1 = cp_Resnet18_layer4_b1.relu1
        cp_Resnet18_layer4_b1_relu2 = cp_Resnet18_layer4_b1.relu2
        cp_Resnet18_layer4_b1_qf = cp_Resnet18_layer4_b1.qf

        # ------------------------cp.arm16------------------------------
        cp_arm16 = ContextPath.arm16
        cp_arm16_conv = ContextPath.arm16.conv
        cp_arm16_conv_atten = ContextPath.arm16.conv_atten
        cp_arm16_bn_atten = ContextPath.arm16.bn_atten
        cp_arm16_relu = ContextPath.arm16.relu
        cp_arm16_qf = ContextPath.arm16.qf

        # ------------------------cp.arm32------------------------------
        cp_arm32 = ContextPath.arm32
        cp_arm32_conv = ContextPath.arm32.conv
        cp_arm32_conv_atten = ContextPath.arm32.conv_atten
        cp_arm32_bn_atten = ContextPath.arm32.bn_atten
        cp_arm32_relu = ContextPath.arm32.relu
        cp_arm32_qf = ContextPath.arm32.qf

        # ------------------------cp.conv_head32------------------------------
        cp_conv_head32 = ContextPath.conv_head32
        # ------------------------cp.conv_head16------------------------------
        cp_conv_head16 = ContextPath.conv_head16
        # ------------------------cp.conv_avg------------------------------
        cp_conv_avg = ContextPath.conv_avg
        # ------------------------cp.up32------------------------------
        cp_up32 = ContextPath.up32
        # ------------------------cp.up16------------------------------
        cp_up16 = ContextPath.up16
        # ------------------------cp.qf------------------------------
        cp_qf = ContextPath.qf

        # ------------------------SpatialPath---------------------------------
        SpatialPath = model.sp
        SpatialPath_conv1 = SpatialPath.conv1
        SpatialPath_conv2 = SpatialPath.conv2
        SpatialPath_conv3 = SpatialPath.conv3
        SpatialPath_conv_out = SpatialPath.conv_out

        # ------------------------FeatureFusionModule---------------------------------
        FeatureFusionModule = model.ffm
        FeatureFusionModule_convblk = FeatureFusionModule.convblk
        FeatureFusionModule_conv = FeatureFusionModule.conv
        FeatureFusionModule_bn = FeatureFusionModule.bn
        FeatureFusionModule_relu = FeatureFusionModule.relu
        FeatureFusionModule_qf0 = FeatureFusionModule.qf0
        FeatureFusionModule_qf1 = FeatureFusionModule.qf1
        FeatureFusionModule_qf2 = FeatureFusionModule.qf2

        # ------------------------BiSeNetOutput---------------------------------
        BiSeNetOutput = model.conv_out
        BiSeNetOutput_conv = BiSeNetOutput.conv
        BiSeNetOutput_conv_out = BiSeNetOutput.conv_out
        BiSeNetOutput_up = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)

        # ------------------------------------api网络推理--------------------------------------------------------
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

        # layer ↓ 1
        Transit(para1='', para2='quant', feature=[input_img, quant_feature_f], option=['Pre'])
        # layer ↓ 2
        cp_Resnet18_conv1_feature = cp_Resnet18_conv1(quant_feature_f)
        cp_Resnet18_bn1_feature = cp_Resnet18_bn1(cp_Resnet18_conv1_feature)
        cp_Resnet18_relu_feature = cp_Resnet18_relu(cp_Resnet18_bn1_feature)
        # 不分块
        Transit(para1='quant', para2='cp.resnet.conv1', feature=[quant_feature_f, cp_Resnet18_relu_feature],
                option=['Conv33', 2, 1, 1])
        # # 分块操作
        # block(para1='quant', para2='cp.resnet.conv1', feature=[quant_feature_f, cp_Resnet18_relu_feature],
        #       option=['Conv33', 2, 1, 1, 2])

        # layer ↓ 3
        cp_Resnet18_maxpool_feature = cp_Resnet18_maxpool(cp_Resnet18_relu_feature)
        Transit(para1='cp.resnet.conv1', para2='cp.resnet.conv1',
                feature=[cp_Resnet18_relu_feature, cp_Resnet18_maxpool_feature], option=['MaxPool'])

        # layer ↓ 4
        cp_Resnet18_layer1_b0_conv1_feature = cp_Resnet18_layer1_b0_conv1(cp_Resnet18_maxpool_feature)
        cp_Resnet18_layer1_b0_bn1_feature = cp_Resnet18_layer1_b0_bn1(cp_Resnet18_layer1_b0_conv1_feature)
        cp_Resnet18_layer1_b0_relu1_feature = cp_Resnet18_layer1_b0_relu1(cp_Resnet18_layer1_b0_bn1_feature)

        # 不分块
        Transit(para1='cp.resnet.conv1', para2='cp.resnet.layer1.0.conv1',
                feature=[cp_Resnet18_maxpool_feature, cp_Resnet18_layer1_b0_relu1_feature], option=['Conv33', 1, 1, 1])

        # layer ↓ 5
        cp_Resnet18_layer1_b0_conv2_feature = cp_Resnet18_layer1_b0_conv2(cp_Resnet18_layer1_b0_relu1_feature)
        cp_Resnet18_layer1_b0_bn2_feature = cp_Resnet18_layer1_b0_bn2(cp_Resnet18_layer1_b0_conv2_feature)
        Transit(para1='cp.resnet.layer1.0.conv1', para2='cp.resnet.layer1.0.conv2',
                feature=[cp_Resnet18_layer1_b0_relu1_feature, cp_Resnet18_layer1_b0_bn2_feature],
                option=['Conv33', 1, 1, 0])

        # layer ↓ 6
        cp_Resnet18_layer1_b0_out_feature = cp_Resnet18_layer1_b0_qf.add(cp_Resnet18_layer1_b0_bn2_feature,
                                                                         cp_Resnet18_maxpool_feature)
        Transit(para1='cp.resnet.layer1.0.conv2', para2='cp.resnet.layer1.0.qf', para3='cp.resnet.conv1',
                feature=[cp_Resnet18_layer1_b0_bn2_feature, cp_Resnet18_layer1_b0_out_feature,
                         cp_Resnet18_maxpool_feature],
                option=['Add'])

        # layer ↓ 7
        cp_Resnet18_layer1_b0_relu_feature = cp_Resnet18_layer1_b0_relu2(cp_Resnet18_layer1_b0_out_feature)
        Transit(para1='cp.resnet.layer1.0.qf', para2='cp.resnet.layer1.0.qf',
                feature=[cp_Resnet18_layer1_b0_out_feature, cp_Resnet18_layer1_b0_relu_feature], option=['LeakyRelu'])

        # layer ↓ 8
        cp_Resnet18_layer1_b1_conv1_feature = cp_Resnet18_layer1_b1_conv1(cp_Resnet18_layer1_b0_relu_feature)
        cp_Resnet18_layer1_b1_bn1_feature = cp_Resnet18_layer1_b1_bn1(cp_Resnet18_layer1_b1_conv1_feature)
        cp_Resnet18_layer1_b1_relu1_feature = cp_Resnet18_layer1_b1_relu1(cp_Resnet18_layer1_b1_bn1_feature)

        Transit(para1='cp.resnet.layer1.0.qf', para2='cp.resnet.layer1.1.conv1',
                feature=[cp_Resnet18_layer1_b0_relu_feature, cp_Resnet18_layer1_b1_relu1_feature],
                option=['Conv33', 1, 1, 1])

        # layer ↓ 9
        cp_Resnet18_layer1_b1_conv2_feature = cp_Resnet18_layer1_b1_conv2(cp_Resnet18_layer1_b1_relu1_feature)
        cp_Resnet18_layer1_b1_bn2_feature = cp_Resnet18_layer1_b1_bn2(cp_Resnet18_layer1_b1_conv2_feature)
        Transit(para1='cp.resnet.layer1.1.conv1', para2='cp.resnet.layer1.1.conv2',
                feature=[cp_Resnet18_layer1_b1_relu1_feature, cp_Resnet18_layer1_b1_bn2_feature],
                option=['Conv33', 1, 1, 0])

        # layer ↓ 10
        cp_Resnet18_layer1_b1_out_feature = cp_Resnet18_layer1_b1_qf.add(cp_Resnet18_layer1_b1_bn2_feature,
                                                                         cp_Resnet18_layer1_b0_relu_feature)
        Transit(para1='cp.resnet.layer1.1.conv2', para2='cp.resnet.layer1.1.qf', para3='cp.resnet.layer1.0.qf',
                feature=[cp_Resnet18_layer1_b1_bn2_feature, cp_Resnet18_layer1_b1_out_feature,
                         cp_Resnet18_layer1_b0_relu_feature],
                option=['Add'])
        # layer ↓ 11
        cp_Resnet18_layer1_b1_relu_feature = cp_Resnet18_layer1_b1_relu2(cp_Resnet18_layer1_b1_out_feature)
        Transit(para1='cp.resnet.layer1.1.qf', para2='cp.resnet.layer1.1.qf',
                feature=[cp_Resnet18_layer1_b1_out_feature, cp_Resnet18_layer1_b1_relu_feature], option=['LeakyRelu'])

        # layer ↓ 12
        cp_Resnet18_layer2_b0_conv1_feature = cp_Resnet18_layer2_b0_conv1(cp_Resnet18_layer1_b1_relu_feature)
        cp_Resnet18_layer2_b0_bn1_feature = cp_Resnet18_layer2_b0_bn1(cp_Resnet18_layer2_b0_conv1_feature)
        cp_Resnet18_layer2_b0_relu1_feature = cp_Resnet18_layer2_b0_relu1(cp_Resnet18_layer2_b0_bn1_feature)
        Transit(para1='cp.resnet.layer1.1.qf', para2='cp.resnet.layer2.0.conv1',
                feature=[cp_Resnet18_layer1_b1_relu_feature, cp_Resnet18_layer2_b0_relu1_feature],
                option=['Conv33', 2, 1, 1])

        # layer ↓ 13
        cp_Resnet18_layer2_b0_conv2_feature = cp_Resnet18_layer2_b0_conv2(cp_Resnet18_layer2_b0_relu1_feature)
        cp_Resnet18_layer2_b0_bn2_feature = cp_Resnet18_layer2_b0_bn2(cp_Resnet18_layer2_b0_conv2_feature)
        Transit(para1='cp.resnet.layer2.0.conv1', para2='cp.resnet.layer2.0.conv2',
                feature=[cp_Resnet18_layer2_b0_relu1_feature, cp_Resnet18_layer2_b0_bn2_feature],
                option=['Conv33', 1, 1, 0])

        # layer ↓ 14
        cp_Resnet18_layer2_b0_downsample_feature = cp_Resnet18_layer2_b0_downsample(
            cp_Resnet18_layer1_b1_relu_feature)
        Transit(para1='cp.resnet.layer1.1.qf', para2='cp.resnet.layer2.0.downsample.0',
                feature=[cp_Resnet18_layer1_b1_relu_feature, cp_Resnet18_layer2_b0_downsample_feature],
                option=['Conv11', 2, 0, 0])

        # layer ↓ 15
        cp_Resnet18_layer2_b0_out_feature = cp_Resnet18_layer2_b0_qf.add(cp_Resnet18_layer2_b0_downsample_feature,
                                                                         cp_Resnet18_layer2_b0_bn2_feature)
        Transit(para1='cp.resnet.layer2.0.downsample.0', para2='cp.resnet.layer2.0.qf',
                para3='cp.resnet.layer2.0.conv2',
                feature=[cp_Resnet18_layer2_b0_downsample_feature, cp_Resnet18_layer2_b0_out_feature,
                         cp_Resnet18_layer2_b0_bn2_feature],
                option=['Add'])

        # layer ↓ 16
        cp_Resnet18_layer2_b0_relu_feature = cp_Resnet18_layer2_b0_relu2(cp_Resnet18_layer2_b0_out_feature)
        Transit(para1='cp.resnet.layer2.0.qf', para2='cp.resnet.layer2.0.qf',
                feature=[cp_Resnet18_layer2_b0_out_feature, cp_Resnet18_layer2_b0_relu_feature], option=['LeakyRelu'])

        # layer ↓ 17
        cp_Resnet18_layer2_b1_conv1_feature = cp_Resnet18_layer2_b1_conv1(cp_Resnet18_layer2_b0_relu_feature)
        cp_Resnet18_layer2_b1_bn1_feature = cp_Resnet18_layer2_b1_bn1(cp_Resnet18_layer2_b1_conv1_feature)
        cp_Resnet18_layer2_b1_relu1_feature = cp_Resnet18_layer2_b1_relu1(cp_Resnet18_layer2_b1_bn1_feature)
        Transit(para1='cp.resnet.layer2.0.qf', para2='cp.resnet.layer2.1.conv1',
                feature=[cp_Resnet18_layer2_b0_relu_feature, cp_Resnet18_layer2_b1_relu1_feature],
                option=['Conv33', 1, 1, 1])

        # layer ↓ 18
        cp_Resnet18_layer2_b1_conv2_feature = cp_Resnet18_layer2_b1_conv2(cp_Resnet18_layer2_b1_relu1_feature)
        cp_Resnet18_layer2_b1_bn2_feature = cp_Resnet18_layer2_b1_bn2(cp_Resnet18_layer2_b1_conv2_feature)
        Transit(para1='cp.resnet.layer2.1.conv1', para2='cp.resnet.layer2.1.conv2',
                feature=[cp_Resnet18_layer2_b1_relu1_feature, cp_Resnet18_layer2_b1_bn2_feature],
                option=['Conv33', 1, 1, 0])

        # layer ↓ 19
        cp_Resnet18_layer2_b1_out_feature = cp_Resnet18_layer2_b1_qf.add(cp_Resnet18_layer2_b1_bn2_feature,
                                                                         cp_Resnet18_layer2_b0_relu_feature)
        Transit(para1='cp.resnet.layer2.1.conv2', para2='cp.resnet.layer2.1.qf', para3='cp.resnet.layer2.0.qf',
                feature=[cp_Resnet18_layer2_b1_bn2_feature, cp_Resnet18_layer2_b1_out_feature,
                         cp_Resnet18_layer2_b0_relu_feature],
                option=['Add'])

        # gen_coe('input&output/out18.coe', cp_Resnet18_layer2_b1_out_feature.int_repr())

        # layer ↓ 20
        cp_Resnet18_layer2_b1_relu_feature = cp_Resnet18_layer2_b1_relu2(cp_Resnet18_layer2_b1_out_feature)
        Transit(para1='cp.resnet.layer2.1.qf', para2='cp.resnet.layer2.1.qf',
                feature=[cp_Resnet18_layer2_b1_out_feature, cp_Resnet18_layer2_b1_relu_feature],
                option=['LeakyRelu'])

        # layer ↓ 21
        cp_Resnet18_layer3_b0_conv1_feature = cp_Resnet18_layer3_b0_conv1(cp_Resnet18_layer2_b1_relu_feature)
        cp_Resnet18_layer3_b0_bn1_feature = cp_Resnet18_layer3_b0_bn1(cp_Resnet18_layer3_b0_conv1_feature)
        cp_Resnet18_layer3_b0_relu1_feature = cp_Resnet18_layer3_b0_relu1(cp_Resnet18_layer3_b0_bn1_feature)
        Transit(para1='cp.resnet.layer2.1.qf', para2='cp.resnet.layer3.0.conv1',
                feature=[cp_Resnet18_layer2_b1_relu_feature, cp_Resnet18_layer3_b0_relu1_feature],
                option=['Conv33', 2, 1, 1])

        # layer ↓ 22
        cp_Resnet18_layer3_b0_conv2_feature = cp_Resnet18_layer3_b0_conv2(cp_Resnet18_layer3_b0_relu1_feature)
        cp_Resnet18_layer3_b0_bn2_feature = cp_Resnet18_layer3_b0_bn2(cp_Resnet18_layer3_b0_conv2_feature)
        Transit(para1='cp.resnet.layer3.0.conv1', para2='cp.resnet.layer3.0.conv2',
                feature=[cp_Resnet18_layer3_b0_relu1_feature, cp_Resnet18_layer3_b0_bn2_feature],
                option=['Conv33', 1, 1, 0])

        # layer ↓ 23
        cp_Resnet18_layer3_b0_downsample_feature = cp_Resnet18_layer3_b0_downsample(
            cp_Resnet18_layer2_b1_relu_feature)
        Transit(para1='cp.resnet.layer2.1.qf', para2='cp.resnet.layer3.0.downsample.0',
                feature=[cp_Resnet18_layer2_b1_relu_feature, cp_Resnet18_layer3_b0_downsample_feature],
                option=['Conv11', 2, 0, 0])

        # layer ↓ 24
        cp_Resnet18_layer3_b0_out_feature = cp_Resnet18_layer3_b0_qf.add(cp_Resnet18_layer3_b0_downsample_feature,
                                                                         cp_Resnet18_layer3_b0_bn2_feature)
        Transit(para1='cp.resnet.layer3.0.downsample.0', para2='cp.resnet.layer3.0.qf',
                para3='cp.resnet.layer3.0.conv2',
                feature=[cp_Resnet18_layer3_b0_downsample_feature, cp_Resnet18_layer3_b0_out_feature,
                         cp_Resnet18_layer3_b0_bn2_feature],
                option=['Add'])

        # layer ↓ 25
        cp_Resnet18_layer3_b0_relu_feature = cp_Resnet18_layer3_b0_relu2(cp_Resnet18_layer3_b0_out_feature)
        Transit(para1='cp.resnet.layer3.0.qf', para2='cp.resnet.layer3.0.qf',
                feature=[cp_Resnet18_layer3_b0_out_feature, cp_Resnet18_layer3_b0_relu_feature],
                option=['LeakyRelu'])

        # layer ↓ 26
        cp_Resnet18_layer3_b1_conv1_feature = cp_Resnet18_layer3_b1_conv1(cp_Resnet18_layer3_b0_relu_feature)
        cp_Resnet18_layer3_b1_bn1_feature = cp_Resnet18_layer3_b1_bn1(cp_Resnet18_layer3_b1_conv1_feature)
        cp_Resnet18_layer3_b1_relu1_feature = cp_Resnet18_layer3_b1_relu1(cp_Resnet18_layer3_b1_bn1_feature)
        Transit(para1='cp.resnet.layer3.0.qf', para2='cp.resnet.layer3.1.conv1',
                feature=[cp_Resnet18_layer3_b0_relu_feature, cp_Resnet18_layer3_b1_relu1_feature],
                option=['Conv33', 1, 1, 1])

        # layer ↓ 27
        cp_Resnet18_layer3_b1_conv2_feature = cp_Resnet18_layer3_b1_conv2(cp_Resnet18_layer3_b1_relu1_feature)
        cp_Resnet18_layer3_b1_bn2_feature = cp_Resnet18_layer3_b1_bn2(cp_Resnet18_layer3_b1_conv2_feature)
        Transit(para1='cp.resnet.layer3.1.conv1', para2='cp.resnet.layer3.1.conv2',
                feature=[cp_Resnet18_layer3_b1_relu1_feature, cp_Resnet18_layer3_b1_bn2_feature],
                option=['Conv33', 1, 1, 0])

        # layer ↓ 28
        cp_Resnet18_layer3_b1_out_feature = cp_Resnet18_layer3_b1_qf.add(cp_Resnet18_layer3_b1_bn2_feature,
                                                                         cp_Resnet18_layer3_b0_relu_feature)
        Transit(para1='cp.resnet.layer3.1.conv2', para2='cp.resnet.layer3.1.qf', para3='cp.resnet.layer3.0.qf',
                feature=[cp_Resnet18_layer3_b1_bn2_feature, cp_Resnet18_layer3_b1_out_feature,
                         cp_Resnet18_layer3_b0_relu_feature],
                option=['Add'])

        # layer ↓ 29
        cp_Resnet18_layer3_b1_relu_feature = cp_Resnet18_layer3_b1_relu2(cp_Resnet18_layer3_b1_out_feature)
        Transit(para1='cp.resnet.layer3.1.qf', para2='cp.resnet.layer3.1.qf',
                feature=[cp_Resnet18_layer3_b1_out_feature, cp_Resnet18_layer3_b1_relu_feature],
                option=['LeakyRelu'])

        # layer ↓ 30
        cp_Resnet18_layer4_b0_conv1_feature = cp_Resnet18_layer4_b0_conv1(cp_Resnet18_layer3_b1_relu_feature)
        cp_Resnet18_layer4_b0_bn1_feature = cp_Resnet18_layer4_b0_bn1(cp_Resnet18_layer4_b0_conv1_feature)
        cp_Resnet18_layer4_b0_relu1_feature = cp_Resnet18_layer4_b0_relu1(cp_Resnet18_layer4_b0_bn1_feature)
        Transit(para1='cp.resnet.layer3.1.qf', para2='cp.resnet.layer4.0.conv1',
                feature=[cp_Resnet18_layer3_b1_relu_feature, cp_Resnet18_layer4_b0_relu1_feature],
                option=['Conv33', 2, 1, 1])

        # layer ↓ 31
        cp_Resnet18_layer4_b0_conv2_feature = cp_Resnet18_layer4_b0_conv2(cp_Resnet18_layer4_b0_relu1_feature)
        cp_Resnet18_layer4_b0_bn2_feature = cp_Resnet18_layer4_b0_bn2(cp_Resnet18_layer4_b0_conv2_feature)
        Transit(para1='cp.resnet.layer4.0.conv1', para2='cp.resnet.layer4.0.conv2',
                feature=[cp_Resnet18_layer4_b0_relu1_feature, cp_Resnet18_layer4_b0_bn2_feature],
                option=['Conv33', 1, 1, 0])

        # layer ↓ 32
        cp_Resnet18_layer4_b0_downsample_feature = cp_Resnet18_layer4_b0_downsample(
            cp_Resnet18_layer3_b1_relu_feature)
        Transit(para1='cp.resnet.layer3.1.qf', para2='cp.resnet.layer4.0.downsample.0',
                feature=[cp_Resnet18_layer3_b1_relu_feature, cp_Resnet18_layer4_b0_downsample_feature],
                option=['Conv11', 2, 0, 0])

        # layer ↓ 33
        cp_Resnet18_layer4_b0_out_feature = cp_Resnet18_layer4_b0_qf.add(cp_Resnet18_layer4_b0_downsample_feature,
                                                                         cp_Resnet18_layer4_b0_bn2_feature)
        Transit(para1='cp.resnet.layer4.0.downsample.0', para2='cp.resnet.layer4.0.qf',
                para3='cp.resnet.layer4.0.conv2',
                feature=[cp_Resnet18_layer4_b0_downsample_feature, cp_Resnet18_layer4_b0_out_feature,
                         cp_Resnet18_layer4_b0_bn2_feature],
                option=['Add'])

        # layer ↓ 34
        cp_Resnet18_layer4_b0_relu_feature = cp_Resnet18_layer4_b0_relu2(cp_Resnet18_layer4_b0_out_feature)
        Transit(para1='cp.resnet.layer4.0.qf', para2='cp.resnet.layer4.0.qf',
                feature=[cp_Resnet18_layer4_b0_out_feature, cp_Resnet18_layer4_b0_relu_feature],
                option=['LeakyRelu'])

        # layer ↓ 35
        cp_Resnet18_layer4_b1_conv1_feature = cp_Resnet18_layer4_b1_conv1(cp_Resnet18_layer4_b0_relu_feature)
        cp_Resnet18_layer4_b1_bn1_feature = cp_Resnet18_layer4_b1_bn1(cp_Resnet18_layer4_b1_conv1_feature)
        cp_Resnet18_layer4_b1_relu1_feature = cp_Resnet18_layer4_b1_relu1(cp_Resnet18_layer4_b1_bn1_feature)
        Transit(para1='cp.resnet.layer4.0.qf', para2='cp.resnet.layer4.1.conv1',
                feature=[cp_Resnet18_layer4_b0_relu_feature, cp_Resnet18_layer4_b1_relu1_feature],
                option=['Conv33', 1, 1, 1])

        # layer ↓ 36
        cp_Resnet18_layer4_b1_conv2_feature = cp_Resnet18_layer4_b1_conv2(cp_Resnet18_layer4_b1_relu1_feature)
        cp_Resnet18_layer4_b1_bn2_feature = cp_Resnet18_layer4_b1_bn2(cp_Resnet18_layer4_b1_conv2_feature)
        Transit(para1='cp.resnet.layer4.1.conv1', para2='cp.resnet.layer4.1.conv2',
                feature=[cp_Resnet18_layer4_b1_relu1_feature, cp_Resnet18_layer4_b1_bn2_feature],
                option=['Conv33', 1, 1, 0])

        # layer ↓ 37
        cp_Resnet18_layer4_b1_out_feature = cp_Resnet18_layer4_b1_qf.add(cp_Resnet18_layer4_b1_bn2_feature,
                                                                         cp_Resnet18_layer4_b0_relu_feature)
        Transit(para1='cp.resnet.layer4.1.conv2', para2='cp.resnet.layer4.1.qf', para3='cp.resnet.layer4.0.qf',
                feature=[cp_Resnet18_layer4_b1_bn2_feature, cp_Resnet18_layer4_b1_out_feature,
                         cp_Resnet18_layer4_b0_relu_feature],
                option=['Add'])

        # layer ↓ 38
        cp_Resnet18_layer4_b1_relu_feature = cp_Resnet18_layer4_b1_relu2(cp_Resnet18_layer4_b1_out_feature)
        Transit(para1='cp.resnet.layer4.1.qf', para2='cp.resnet.layer4.1.qf',
                feature=[cp_Resnet18_layer4_b1_out_feature, cp_Resnet18_layer4_b1_relu_feature],
                option=['LeakyRelu'])

        # layer ↓ 39
        cp_arm32_conv_feature = cp_arm32_conv(cp_Resnet18_layer4_b1_relu_feature)
        Transit(para1='cp.resnet.layer4.1.qf', para2='cp.arm32.conv.conv',
                feature=[cp_Resnet18_layer4_b1_relu_feature, cp_arm32_conv_feature],
                option=['Conv33', 1, 1, 1])

        # layer ↓ 40
        cp_arm32_mean_feature = torch.mean(cp_arm32_conv_feature, dim=(2, 3), keepdim=True)
        Transit(para1='cp.arm32.conv.conv', para2='cp.arm32.conv.conv',
                feature=[cp_arm32_conv_feature, cp_arm32_mean_feature],
                option=['MeanPool'])

        # layer ↓ 41
        cp_arm32_conv_atten_feature = cp_arm32_conv_atten(cp_arm32_mean_feature)
        cp_arm32_bn_atten_feature = cp_arm32_bn_atten(cp_arm32_conv_atten_feature)
        cp_arm32_relu_feature = cp_arm32_relu(cp_arm32_bn_atten_feature)
        Transit(para1='cp.arm32.conv.conv', para2='cp.arm32.conv_atten',
                feature=[cp_arm32_mean_feature, cp_arm32_relu_feature],
                option=['Conv11', 1, 0, 1])

        # layer ↓ 42
        cp_arm32_qf_feature = cp_arm32_qf.mul(cp_arm32_conv_feature, cp_arm32_relu_feature)
        Transit(para1='cp.arm32.conv.conv', para2='cp.arm32.qf', para3='cp.arm32.conv_atten',
                feature=[cp_arm32_conv_feature, cp_arm32_qf_feature, cp_arm32_relu_feature],
                option=['Mul'])

        # layer ↓ 43
        cp_up32_feature = cp_up32(cp_arm32_qf_feature)
        Transit(para1='cp.arm32.qf', para2='cp.arm32.qf',
                feature=[cp_arm32_qf_feature, cp_up32_feature],
                option=['UpSample'])

        # layer ↓ 44
        cp_conv_head32_feature = cp_conv_head32(cp_up32_feature)
        Transit(para1='cp.arm32.qf', para2='cp.conv_head32.conv',
                feature=[cp_up32_feature, cp_conv_head32_feature],
                option=['Conv33', 1, 1, 1])

        # layer ↓ 45
        cp_arm16_conv_feature = cp_arm16_conv(cp_Resnet18_layer3_b1_relu_feature)
        Transit(para1='cp.resnet.layer3.1.qf', para2='cp.arm16.conv.conv',
                feature=[cp_Resnet18_layer3_b1_relu_feature, cp_arm16_conv_feature],
                option=['Conv33', 1, 1, 1])

        # layer ↓ 46
        cp_arm16_mean_feature = torch.mean(cp_arm16_conv_feature, dim=(2, 3), keepdim=True)
        Transit(para1='cp.arm16.conv.conv', para2='cp.arm16.conv.conv',
                feature=[cp_arm16_conv_feature, cp_arm16_mean_feature],
                option=['MeanPool'])

        # layer ↓ 47
        cp_arm16_conv_atten_feature = cp_arm16_conv_atten(cp_arm16_mean_feature)
        cp_arm16_bn_atten_feature = cp_arm16_bn_atten(cp_arm16_conv_atten_feature)
        cp_arm16_relu_feature = cp_arm16_relu(cp_arm16_bn_atten_feature)
        Transit(para1='cp.arm16.conv.conv', para2='cp.arm16.conv_atten',
                feature=[cp_arm16_mean_feature, cp_arm16_relu_feature],
                option=['Conv11', 1, 0, 1])

        # layer ↓ 48
        cp_arm16_qf_feature = cp_arm16_qf.mul(cp_arm16_conv_feature, cp_arm16_relu_feature)
        Transit(para1='cp.arm16.conv.conv', para2='cp.arm16.qf', para3='cp.arm16.conv_atten',
                feature=[cp_arm16_conv_feature, cp_arm16_qf_feature, cp_arm16_relu_feature],
                option=['Mul'])

        # layer ↓ 49
        cp_qf_feature = cp_qf.add(cp_arm16_qf_feature, cp_conv_head32_feature)
        Transit(para1='cp.arm16.qf', para2='cp.qf', para3='cp.conv_head32.conv',
                feature=[cp_arm16_qf_feature, cp_qf_feature, cp_conv_head32_feature],
                option=['Add'])

        # layer ↓ 50
        cp_up16_feature = cp_up16(cp_qf_feature)
        Transit(para1='cp.qf', para2='cp.qf',
                feature=[cp_qf_feature, cp_up16_feature],
                option=['UpSample'])

        # layer ↓ 51
        cp_conv_head16_feature = cp_conv_head16(cp_up16_feature)
        Transit(para1='cp.qf', para2='cp.conv_head16.conv',
                feature=[cp_up16_feature, cp_conv_head16_feature],
                option=['Conv33', 1, 1, 1])

        # layer ↓ 52
        SpatialPath_conv1_feature = SpatialPath_conv1(quant_feature_f)
        Transit(para1='quant', para2='sp.conv1.conv',
                feature=[quant_feature_f, SpatialPath_conv1_feature],
                option=['Conv33', 2, 1, 1])

        # layer ↓ 53
        SpatialPath_conv2_feature = SpatialPath_conv2(SpatialPath_conv1_feature)
        Transit(para1='sp.conv1.conv', para2='sp.conv2.conv',
                feature=[SpatialPath_conv1_feature, SpatialPath_conv2_feature],
                option=['Conv33', 2, 1, 1])

        # layer ↓ 54
        SpatialPath_conv3_feature = SpatialPath_conv3(SpatialPath_conv2_feature)
        Transit(para1='sp.conv2.conv', para2='sp.conv3.conv',
                feature=[SpatialPath_conv2_feature, SpatialPath_conv3_feature],
                option=['Conv33', 2, 1, 1])

        # layer ↓ 55
        SpatialPath_conv_out_feature = SpatialPath_conv_out(SpatialPath_conv3_feature)
        Transit(para1='sp.conv3.conv', para2='sp.conv_out.conv',
                feature=[SpatialPath_conv3_feature, SpatialPath_conv_out_feature],
                option=['Conv11', 1, 0, 1])

        # exit()
        # layer ↓ 56
        FeatureFusionModule_qf0_feature = FeatureFusionModule_qf0.cat(
            [SpatialPath_conv_out_feature, cp_conv_head16_feature], dim=1)
        Transit(para1='sp.conv_out.conv', para2='ffm.qf0', para3='cp.conv_head16.conv',
                feature=[SpatialPath_conv_out_feature, FeatureFusionModule_qf0_feature, cp_conv_head16_feature],
                option=['Concat'])

        # layer ↓ 57
        FeatureFusionModule_convblk_feature = FeatureFusionModule_convblk(FeatureFusionModule_qf0_feature)
        Transit(para1='ffm.qf0', para2='ffm.convblk.conv',
                feature=[FeatureFusionModule_qf0_feature, FeatureFusionModule_convblk_feature],
                option=['Conv11', 1, 0, 1])

        # layer ↓ 58
        FeatureFusionModule_convblk_mean_feature = torch.mean(FeatureFusionModule_convblk_feature, dim=(2, 3),
                                                              keepdim=True)
        Transit(para1='ffm.convblk.conv', para2='ffm.convblk.conv',
                feature=[FeatureFusionModule_convblk_feature, FeatureFusionModule_convblk_mean_feature],
                option=['MeanPool'])

        # layer ↓ 59
        FeatureFusionModule_conv_feature = FeatureFusionModule_conv(FeatureFusionModule_convblk_mean_feature)
        FeatureFusionModule_bn_feature = FeatureFusionModule_bn(FeatureFusionModule_conv_feature)
        FeatureFusionModule_relu_feature = FeatureFusionModule_relu(FeatureFusionModule_bn_feature)
        Transit(para1='ffm.convblk.conv', para2='ffm.conv',
                feature=[FeatureFusionModule_convblk_mean_feature, FeatureFusionModule_relu_feature],
                option=['Conv11', 1, 0, 1])

        # layer ↓ 60
        FeatureFusionModule_qf1_feature = FeatureFusionModule_qf1.mul(FeatureFusionModule_convblk_feature,
                                                                      FeatureFusionModule_relu_feature)
        Transit(para1='ffm.convblk.conv', para2='ffm.qf1', para3='ffm.conv',
                feature=[FeatureFusionModule_convblk_feature, FeatureFusionModule_qf1_feature,
                         FeatureFusionModule_relu_feature], option=['Mul'])

        # layer ↓ 61
        FeatureFusionModule_qf2_feature = FeatureFusionModule_qf2.add(FeatureFusionModule_qf1_feature,
                                                                      FeatureFusionModule_convblk_feature)
        Transit(para1='ffm.qf1', para2='ffm.qf2', para3='ffm.convblk.conv',
                feature=[FeatureFusionModule_qf1_feature, FeatureFusionModule_qf2_feature,
                         FeatureFusionModule_convblk_feature], option=['Add'])

        # layer ↓ 62
        BiSeNetOutput_conv_feature = BiSeNetOutput_conv(FeatureFusionModule_qf2_feature)
        Transit(para1='ffm.qf2', para2='conv_out.conv.conv',
                feature=[FeatureFusionModule_qf2_feature, BiSeNetOutput_conv_feature],
                option=['Conv33', 1, 1, 1])

        # layer ↓ 63
        BiSeNetOutput_conv_out_feature = BiSeNetOutput_conv_out(BiSeNetOutput_conv_feature)
        Transit(para1='conv_out.conv.conv', para2='conv_out.conv_out',
                feature=[BiSeNetOutput_conv_feature, BiSeNetOutput_conv_out_feature],
                option=['Conv11', 1, 0, 0])

        # layer ↓ 64

        BiSeNetOutput_split_feature = torch.zeros((1, 8,
                                                   BiSeNetOutput_conv_out_feature.shape[2],
                                                   BiSeNetOutput_conv_out_feature.shape[3]))
        Transit(para1='conv_out.conv_out', para2='conv_out.conv_out',
                feature=[BiSeNetOutput_conv_out_feature, BiSeNetOutput_split_feature],
                option=['Split'])

        # layer ↓ 65
        BiSeNetOutput_up_feature = BiSeNetOutput_up(BiSeNetOutput_conv_out_feature)
        Transit(para1='conv_out.conv_out', para2='conv_out.conv_out',
                feature=[BiSeNetOutput_split_feature, BiSeNetOutput_up_feature],
                option=['UpBili'])

        # layer ↓ 66
        BiSeNetOutput_up_feature = dequant(BiSeNetOutput_up_feature)
        argmax = torch.argmax(BiSeNetOutput_up_feature, dim=1)
        argmax = torch.unsqueeze(argmax, dim=0)
        Transit(para1='conv_out.conv_out', para2='conv_out.conv_out',
                feature=[BiSeNetOutput_up_feature, argmax],
                option=['ArgMax'])


if __name__ == '__main__':
    create_files()
