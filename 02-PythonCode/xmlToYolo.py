##使用labelme标注直接导出txt文件则不需要转换
#此代码仅实现xml->>txt（同时实现坐标归一化（yolov8训练过程中需要归一化坐标）），能自动生成标签文件夹
#需要根据需求更改对于classes=[]及postfix = 'jpg'、imgpath = '图片路径'、xmlpath = 'xml路径'、txtpath = '转换的标签路径'进行修改
#注意：如果在yolo网络中，对于数据集来源的配置文件指定了name:[](也就是配置文件中种类的声明)，在这里需要使classes与之对应，如果classes置空，则会依据字母升序顺序编号，如果两者编号不同，会产生P、R及mAP为0的问题
import xml.etree.ElementTree as ET
import os, cv2
import numpy as np
from os import listdir
from os.path import join
 
classes = ['defect0', 'defect1', 'defect2', 'defect3', 'defect4']#修改1：根据需要进行修改，须保证与yolov8网络配置的name一致
#归一化
def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)
 
 
def convert_annotation(xmlpath, xmlname):
    with open(xmlpath, "r", encoding='utf-8') as in_file:
        txtname = xmlname[:-4] + '.txt'
        txtfile = os.path.join(txtpath, txtname)
        tree = ET.parse(in_file)
        root = tree.getroot()
        filename = root.find('filename')
        img = cv2.imdecode(np.fromfile('{}/{}.{}'.format(imgpath, xmlname[:-4], postfix), np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        res = []
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes:
                classes.append(cls)
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            res.append(str(cls_id) + " " + " ".join([str(a) for a in bb]))
        if len(res) != 0:
            with open(txtfile, 'w+') as f:
                f.write('\n'.join(res))
 
 
if __name__ == "__main__":
    #以下4个参数和路径根据情况进行修改
    postfix = 'jpg'
    imgpath = 'xmltest/images'
    xmlpath = 'xmltest/1'
    txtpath = 'xmltest/xmloutput'
 
    if not os.path.exists(txtpath):
        os.makedirs(txtpath, exist_ok=True)
 
    list = os.listdir(xmlpath)
    error_file_list = []
    for i in range(0, len(list)):
        try:
            path = os.path.join(xmlpath, list[i])
            if ('.xml' in path) or ('.XML' in path):
                convert_annotation(path, list[i])
                print(f'file {list[i]} convert success.')
            else:
                print(f'file {list[i]} is not xml format.')
        except Exception as e:
            print(f'file {list[i]} convert error.')
            print(f'error message:\n{e}')
            error_file_list.append(list[i])
    print(f'this file convert failure\n{error_file_list}')
    print(f'Dataset Classes:{classes}')