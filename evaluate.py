import torch
import PIL.Image as Image
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import matplotlib
matplotlib.use('tkagg')
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# import cv2
name = ["background","road","sidewalk","building","wall","fence","pole","traffic light",
"traffic sign","vegetation","terrain","sky","person","rider","car","truck","bus","train",
             "motorcycle","bicycle"]

color = [[0,0,0],[128,64,128],[244, 35,232],[70, 70, 70],[102, 102, 156],[190, 153, 153],
            [153, 153, 153],[250, 170, 30],[220, 220, 0],[107, 142, 35],[152, 251, 152],[70, 130, 180],
            [220, 20, 60],[255, 0, 0],[0, 0, 142],[0, 0, 70],[0, 60, 100],[0, 80,100],[0, 0,230],[119, 11, 32]]

def label_to_RGB(tensor ,lable_clors_RGB):
    label_colors = lable_clors_RGB
    label_img = np.array(tensor,dtype=np.uint8)
    h, w = label_img.shape
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
    # 根据映射表进行转换
    for row in range(h):
        for col in range(w):
            label = label_img[row, col]
            rgb_img[row, col] = label_colors[label]
    rgb_img = rgb_img.swapaxes(0, 2)
    return rgb_img
import torchvision.transforms as transforms
trans = transforms.Compose([
    transforms.ToTensor()
])
modle = torch.load(r"D:\PythonProject\TestRGB\best.pth", map_location=torch.device("cpu"), weights_only=False)
test_image = r"D:\PythonProject\TestRGB\2.png"
test_image = trans(Image.open(test_image).convert('RGB')).unsqueeze(0)
with torch.no_grad():
    modle.eval()
    output = modle(test_image)
    image = output.argmax(dim=1).cpu().squeeze(0)
    print(image.shape)
    new_image = label_to_RGB(image,color)
    new_image = new_image.swapaxes(0,2)
    print(new_image.shape)
    plt.imshow(new_image)
    plt.show()
