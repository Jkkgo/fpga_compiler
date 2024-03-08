import cv2
import numpy as np

'''
根据均值方差融合出新的s跟z
'''

mean = 0.0558
std = 0.1208
scale = np.load("../../para_68/quant.scale.npy")
zp = np.load("../../para_68/quant.zero_point.npy")
# image = cv2.imread('./234.jpg',0)
# gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
image = np.zeros(256)

for i in range(256):
    image[i] = i



image_quant = np.round((image-255*mean+zp*255*std*scale)/(255*std*scale))




shift = 2**17
scale_new = 1/(255*std*scale)
zp_new = (zp*std*scale-mean)/(std*scale)
scale_new = np.round(scale_new*shift)
zp_new = np.round((zp_new+0.5)*shift)
image_new = np.floor((image*scale_new+zp_new)/shift)

print(np.floor((0*scale_new+zp_new)/shift))

# with open("0_255_result.coe", 'w') as output_file:
#     for i in range(256):
#         img = str(hex(int(image_new[i])))[2:]
#         output_file.write(img+"\n")
#
# exit()


map = image_quant-image_new
unique_values, counts = np.unique(map,return_counts=True)


# image_quant = np.expand_dims(image_quant, axis=0)
# image_quant = np.expand_dims(image_quant, axis=1).astype(int)
# gen_coe('./Bisenet_16in/1/input1.coe',quant_feature.int_repr())

print("Unique values:", unique_values)
print("Counts:", counts)
print("scale_new:", scale_new)
print("zp_new:", zp_new)






