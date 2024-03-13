import cv2
import torch
from compiler.lib.array_format import picture_load
from compiler.transit import Transit
from compiler.transit import shared

'''
   create_files:主函数
   1.将网络模型的每一层拆分出来
   2.将拆分出来的网络输入输出、提取出来的量化参数、本层操作传入transit分发器
'''


def create_files():
    model = torch.jit.load(shared.model_path)
    model.eval()

    with torch.no_grad():
        # ------------------------网络Init-------------------------- #

        torch.set_printoptions(profile='full')

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

        # 1(pre预处理层)：详情请看shape_operator里面的pre.py的注释，里面写了第一层干了啥，让硬件那边知道
        # 现在把预处理当作第一层，就会比原本多出14条指令
        # img没有进行归一化，保存图片最原始的状态！不要用toTensor(),因为里面自带归一化
        img = cv2.imread(shared.img_path, 0)
        # 输入图片转为规定的尺寸
        if img.shape[0] != shared.img_size:
            img = cv2.resize(img, (shared.img_size, shared.img_size))

        # 转为tensor格式
        img = torch.from_numpy(img)
        img = img.to(torch.float32)
        img = img.unsqueeze(0)
        img = img.unsqueeze(0)
        img = img.to(torch.int)
        image = picture_load(shared)
        feature_f = model.quant(image)
        Transit(para1='', para2='quant', feature=[img, feature_f], option=['Pre'])

        # ------------------------网络推理-------------------------- #
        # ------------- stem --------------- #
        # 2
        bconv1_r = model.bconv1.conv(feature_f)
        bconv1_act_r = model.bconv1.relu(bconv1_r)
        Transit(para1='quant', para2='bconv1.conv',
                feature=[feature_f, bconv1_act_r],
                option=['Conv33', 2, 1, 1])  # stride=2, padding=1, 激活操作=True
        # gen_coe('../output1.coe', bconv1_act_r.int_repr())
        # 3
        bconv2_r = model.bconv2.conv(bconv1_act_r)
        bconv2_act_r = model.bconv2.relu(bconv2_r)
        Transit(para1='bconv1.conv', para2='bconv2.conv',
                feature=[bconv1_act_r, bconv2_act_r],
                option=['Conv33', 1, 1, 1])
        # gen_coe('../output2.coe', bconv2_act_r.int_repr())
        # 4
        bconv3_r = model.bconv3.conv(bconv2_act_r)
        bconv3_act_r = model.bconv3.relu(bconv3_r)
        Transit(para1='bconv2.conv', para2='bconv3.conv',
                feature=[bconv2_act_r, bconv3_act_r],
                option=['Conv33', 1, 1, 1])
        # gen_coe('../output3.coe', bconv3_act_r.int_repr())
        # 5
        qf1_r = model.qf1.cat([bconv1_act_r, bconv3_act_r], 1)
        Transit(para1='bconv1.conv', para2='qf1', para3='bconv3.conv',
                feature=[bconv1_act_r, qf1_r, bconv3_act_r],
                option=['Concat'])
        # gen_coe('../output4.coe', qf1_r.int_repr())
        # 6
        bconv4_r = model.bconv4.conv(qf1_r)
        bconv4_act_r = model.bconv4.relu(bconv4_r)
        Transit(para1='qf1', para2='bconv4.conv',
                feature=[qf1_r, bconv4_act_r],
                option=['Conv33', 2, 1, 1])
        # print(bconv4_act_r.int_repr())
        # gen_coe('../1_sim_data/output_0000.coe', bconv4_act_r.int_repr())
        # coe2bin('../1_sim_data/output_0000.coe', '../output_0000.bin')
        print('#================前向推理完毕==================#\n')
        # result = model(img)
        # if result[0].equal(out[0]) & result[1].equal(out[1]) & result[2].equal(out[2]):
        #     print('ok')
        # else:
        #     print('error')


if __name__ == '__main__':
    create_files()
