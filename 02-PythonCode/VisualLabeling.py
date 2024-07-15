##图像分析与可视化标注
import pandas as pd
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import os


#ctrl / 组合键进行快速注释与取消注释
#1.4读取训练集图像数据，检查训练集所有图像的尺寸是否一致，检查每张图像是否可以正常打开，检查训练集中图像的尺寸和数目
train_size_dict = defaultdict(int)
train_path = Path("C:/Users/yangb/Desktop/test/yuandata/train_images/")
for img_name in train_path.iterdir():
    img = Image.open(img_name)
    train_size_dict[img.size] += 1
print(train_size_dict)

#1.5读取测试集图像数据
test_size_dict = defaultdict(int)
test_path = Path("C:/Users/yangb/Desktop/test/yuandata/test_images/")
for img_name in test_path.iterdir():
    img = Image.open(img_name)
    test_size_dict[img.size] += 1
print(test_size_dict)


#2、可视化数据集，
#2.1读取csv文本数据
train_df = pd.read_csv("C:/Users/yangb/Desktop/test/yuandata/train.csv")

#2.2为不同的缺陷类别设置颜色显示
palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249,50,12)]
fig, ax = plt.subplots(1, 4, figsize=(15, 5))
for i in range(4):
    ax[i].axis('off')
    ax[i].imshow(np.ones((50, 50, 3), dtype=np.uint8) * palet[i])
    ax[i].set_title("class color: {}".format(i+1))

fig.suptitle("each class colors")
plt.show()

#2.3将不同的缺陷标识归类：
idx_no_defect = []
idx_class_1 = []
idx_class_2 = []
idx_class_3 = []
idx_class_4 = []
idx_class_multi = []
idx_class_triple = []

# 遍历训练数据集的每一行，从col开始，每次取4行
for col in range(0, len(train_df), 4):
    img_names = [str(i).split("_")[0] for i in train_df.iloc[col:col+4, 0].values]
    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):
        raise ValueError

    labels = train_df.iloc[col:col+4, 1]
    if labels.isna().all():
        idx_no_defect.append(col)
    elif (labels.isna() == [False, True, True, True]).all():
        idx_class_1.append(col)
    elif (labels.isna() == [True, False, True, True]).all():
        idx_class_2.append(col)
    elif (labels.isna() == [True, True, False, True]).all():
        idx_class_3.append(col)
    elif (labels.isna() == [True, True, True, False]).all():
        idx_class_4.append(col)
    elif labels.isna().sum() == 1:
        idx_class_triple.append(col)
    else:
        idx_class_multi.append(col)
train_path = Path("C:/Users/yangb/Desktop/test/yuandata/train_images/")

#2.4创建可视化标注函数：
def name_and_mask(start_idx):
    col = start_idx
    img_names = [str(i).split("_")[0] for i in train_df.iloc[col:col+4, 0].values]
    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):
        raise ValueError

    labels = train_df.iloc[col:col+4, 1]
    mask = np.zeros((256, 1600, 4), dtype=np.uint8)

    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            mask_label = np.zeros(1600*256, dtype=np.uint8)
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            for pos, le in zip(positions, length):
                mask_label[pos-1:pos+le-1] = 1
            mask[:, :, idx] = mask_label.reshape(256, 1600, order='F')  #按列取值reshape

    return img_names[0], mask

#def类缺陷图片保存
def save_image(image, path):
  # 检查路径是否存在，如果不存在则创建路径
  if not os.path.exists(os.path.dirname(path)):
      os.makedirs(os.path.dirname(path))
  cv2.imwrite(path, image)   # 保存图片

   
def show_mask_image(col):
    name, mask = name_and_mask(col)
    img = cv2.imread(str(train_path / name))
    fig, ax = plt.subplots(figsize=(15, 15))
    ##以下注释为标注函数，若不需将csv中的缺陷轮廓坐标标注可注释一下for循环中的代码
    for ch in range(4):
        contours, _ = cv2.findContours(mask[:, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i in range(0, len(contours)):
            cv2.polylines(img, contours[i], True, palet[ch], 2)

    ax.set_title(name)
    ax.imshow(img)
    #plt.show()  #展示图片 看需求
    save_image(img, f'2/{name}') #注意：保存不同的缺陷类型时修改输出文件路径      
        
        
        
##分类展示并保存标注好的各种缺陷图片。在输出不同缺陷图片时要修改save_image(img, f'output_triple/{name}')中utput_triple为指定文件名手动取消注释  
  
#展示并保存3132张无缺陷图
# for idx in idx_no_defect[5:10]:
#    show_mask_image(idx)
   
#第一类缺陷图769张
for idx in idx_class_1[:]:
     show_mask_image(idx)

#第2种195张
# for idx in idx_class_2[:]:
#    show_mask_image(idx)
    
#第三种
# for idx in idx_class_3[:]:
#    show_mask_image(idx)

#第4种516张
# for idx in idx_class_4[:]:
#    show_mask_image(idx)

#多种
# for idx in idx_class_triple:
#    show_mask_image(idx)