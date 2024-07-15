# 基于YOLOv8的钢铁材料表面缺陷检测

## Part 1 项目说明

从环境搭建到训练YOLO的项目说明，包含了除最后部署的大部分内容，以下是项目文档目录：

```

```

## Part 2 环境配置

### 一、YOLO项目仓库

国内gitcode镜像：[YOLOv8镜像项目地址](https://gitcode.com/mirrors/ultralytics/ultralytics/overview)、github官方仓库地址：[YOLOv8官方仓库地址](https://github.com/ultralytics/ultralytics)、[yolo官方说明文档](https://docs.ultralytics.com/zh/)

直接下载项目压缩包文件并解压至项目目录，或通过clone下载（但这首先需要[安装git](https://git-scm.com/)）：复制如下代码

![Figure_1](./pic/01.png)

在C盘外的其他盘新建一个工程文件夹，在此文件夹中右键选择“Git Bash Here”（发现右键没有或消失了？[点这里](https://yangbing70.gitee.io/2023/04/07/z7z8/gitbashrepair/)，百度吧，网站挂了┭┮﹏┭┮），输入“git clone 复制的代码”即可将项目复制到本地

```git
git clone https://yangbing70.gitee.io/2023/04/07/z7z8/gitbashrepair/
```

### 二、环境的配置（pytorch）

#### 1.前言

本部分是深度学习的重要内容，也比较繁琐和复杂。要进行深度学习首先要配置好虚拟环境，这包括安装Anaconda、安装Visual Studio、安装配置Pycharm、创建虚拟环境、安装CUDA、安装cuDNN、安装pytorch。对于YOLO项目的环境要求可参考说明文档requirements.txt，确保环境符合要求，官方文档说明如下：

```cmd
# Ultralytics requirements
# Example: pip install -r requirements.txt

# Base --------------------------------r--------
matplotlib>=3.3.0
numpy>=1.22.2 # pinned by Snyk to avoid a vulnerability
opencv-python>=4.6.0
pillow>=7.1.2
pyyaml>=5.3.1
requests>=2.23.0
scipy>=1.4.1
torch>=1.8.0
torchvision>=0.9.0
tqdm>=4.64.0

# Logging -------------------------------------
# tensorboard>=2.13.0
# dvclive>=2.12.0
# clearml
# comet

# Plotting ------------------------------------
pandas>=1.1.4
seaborn>=0.11.0

# Export --------------------------------------
# coremltools>=7.0  # CoreML export
# onnx>=1.12.0  # ONNX export
# onnxsim>=0.4.1  # ONNX simplifier
# nvidia-pyindex  # TensorRT export
# nvidia-tensorrt  # TensorRT export
# scikit-learn==0.19.2  # CoreML quantization
# tensorflow>=2.4.1,<=2.13.1  # TF exports (-cpu, -aarch64, -macos)
# tflite-support
# tensorflowjs>=3.9.0  # TF.js export
# openvino-dev>=2023.0  # OpenVINO export

# Extras --------------------------------------
psutil  # system utilization
py-cpuinfo  # display CPU info
thop>=0.1.1  # FLOPs computation
# ipython  # interactive notebook
# albumentations>=1.0.3  # training augmentations
# pycocotools>=2.0.6  # COCO mAP
# roboflow
```

#### 2.软件

对应软件安装教程：[Anaconda安装](https://blog.csdn.net/m0_61607990/article/details/129531686)、[Visual Studio](https://blog.csdn.net/Javachichi/article/details/131358012)、[Pycharm](https://blog.csdn.net/wangyuxiang946/article/details/130634049)、[安装CUDA、cuDNN、pytorch](https://blog.csdn.net/weixin_58283091/article/details/127841182?spm=1001.2014.3001.5502)。

注意：①在安装VS时根据开发所用语言选择安装即可，本项目只需python环境。②安装VS是由于如果不安装的话后面安装CUDA时会有问题。③在配置pytorch环境若由于安装的是cuda11.2版本，去官网发现没有与之对应的torch版本的话，考虑到版本向下兼容，选择低一个版本的便可，本文选择cuda 11.1版本，在官网选择cuda11.1 复制如下安装代码进行安装。实际安装过程中可能还会有各种奇奇怪怪的问题，自行百度解决。

```pip
# CUDA 11.1
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

安装完成后进行测试：

```python
import torch
print(torch.cuda.is_available()) # cuda是否可用，可用则返回TRUE
```

#### 3.虚拟环境

##### 1. 常用命令

```
//conda基本常用命令，Anconda PowerShell Prompt中运行
conda env list					//查看conda中的虚拟环境
conda activate 环境名字			//激活或进入环境，若虚拟环境没有名字，按conda env list 显示的某一虚拟环境路径激活
pip list					//查看该环境的包
python						// 可以查看该环境python的版本
conda deactivate 			//退出环境

//删除环境
conda deactivate			//退出环境
conda env list				//查看虚拟环境列表，此时出现列表的同时还会显示其所在路径
conda env remove -p xxx		//删除名为xxx的环境
conda env remove -p /home/kuucoss/anaconda3/envs/tfpy36 //路径删除环境的例子
//或
conda deactivate		//退出环境
conda remove -n xxx --all	//删除名为xxx的环境

//重命名环境：由于conda没有重命名命令，通过clone一个新环境，删除原环境实现重命名
conda create -n NewName --clone OldName		//把环境 OldName 重命名成 NewName，注意此时环境的路径
conda remove -n OldName --all 			//删除旧环境

//查看安装的pytorch版本
//方法一：进入安装pytorch的虚拟环境，依次输入如下命令
import torch
print(torch.__version__)
//或cmd
python -c "import torch; print(torch.__version__)"
```

##### 2.进入环境

法一：在命令提示符中输入`activate`

```cmd
> activate	//进入base环境
> conda env list	//查看环境名称，再进入相应环境
```

法二：在开始界面找到Anaconda3文件夹下的Anaconda Powershell Prompt，点击可直接进入base环境，随后步骤同方法一。

##### 3.创建虚拟环境

若要创建python解释器版本为3.8（本机已有的Python版本）、环境名为xxx的虚拟环境，在打开的命令提示符内输入如下代码：

```conda
conda create -n xxx python==3.8
```

创建过程中显示Procced ([y]/n) ？，输入 `y` ，回车继续安装，*此时虚拟环境默认被安装在C盘。*


若想在其他盘符创建python虚拟环境，例如盘符路径E:\IDE\Venvlist，python版本号3.8，则可输入：

```conda
conda create --prefix=E:\IDE\Venvlist\xxx python==3.8
```

此时将自动在\Venvlist目录下创建xxx文件环境。

上述方法每次装环境还得显式地指定路径，可以更改conda config 一劳永逸：

```conda
conda config --append envs_dirs /scratch/reza/anaconda3/envs
```

激活环境也就不用每次都写路径了：

```conda
conda create -n xxx python==3.8.0 # 默认就装在了/scratch/reza/anaconda3/envs/

conda activate xxx		#环境名称可以随意起名，python版本的选择也可以自行确定
```

##### 4.激活虚拟环境

在命令提示符内输入如下代码激活名为xxx的虚拟环境

```conda
conda activate xxx
```

 注意，激活环境首先要设置环境名字（[参考博客7](https://blog.csdn.net/m0_46306264/article/details/131858440)），若不设置名字而直接激活时将会报错，当然也可以通过路径激活，如`0.conda常用命令`中的按路径激活。

##### 5.⭐安装依赖⭐

首先创建一个虚拟环境和项目文件，二者可以在不同的目录，激活虚拟环境后，进入到项目所在文件夹（步骤1-3），在此项目目录下安装依赖（步骤4），如yolo安装依赖输入如下代码：

```pip
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

![Figure_2](./pic/02.png)

##### **添加国内镜像源**

使用conda默认的源来安装包速度非常慢，甚至失败，因此可以添加国内源来加快安装包的网速

(1)首先，找到用户目录下的.condarc文件（C:\Users\username）。
(2)打开.condarc文件之后，添加或修改.condarc 中的 env_dirs 设置环境路径，按顺序第⼀个路径作为默认存储路径，搜索环境按先后顺序在各⽬录中查找。直接在.condarc添加：

```txt
# 中科大源
channels:
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
  - https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
ssl_verify: true

# 清华源
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
ssl_verify: true
```

镜像pip加速

```pip
pip install 模块名字 -i https://pypi.douban.com/simple/
```

### 三、使用YOLO

#### 1.数据集的制作

首先要制作属于自己的数据集，数据集分为多种如coco格式数据集，参考如下[文章](https://huaweicloud.csdn.net/637f7ce4dacf622b8df86169.html?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-2-125757449-blog-88353970.235%5Ev38%5Epc_relevant_sort_base2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-2-125757449-blog-88353970.235%5Ev38%5Epc_relevant_sort_base2)，或者使用labelimg（labelme同理）进行标注创建 txt格式数据集，过程如下：

新建labelimg虚拟环境-->激活环境 -->安装python包，对应代码如下

```conda
conda create -n labelimg python=3.8	##可以按路径创建
 
activate labelimg
 
pip install PyQt5
pip install pyqt5-tools
pip install lxml
pip install labelimg
```

最后，在该虚拟环境下输入labelimg（或labelme）启动软件

进入软件后要进行修改为voc格式的话需要将输出的xml转换为yolo 的txt格式，可以选择yolo直接输出txt格式。在view中勾选自动保存/展示标签等选项；**w快捷键进行开始标注，a，d，进行前后图片的切换,若只能打正方形框，可按ctrl shift R组合键进行恢复**

标注好后，按照如下的数据文件目录进行数据整理

```
 ├── datasets
     └── test 
     |		└──images
     |		|	└──num=260	#测试集图像及数量
     |		└──labels
     |			└──num=260	#标注后测试集yolo txt格式文件数量
     └── train 
     |		└──images
     |		|	└──num=909
     |		└──labels
     |	    	└──num=909
     └─── val 			
  	        └──images
      		|	└──num=131
      		└──labels
      	    	└──num=131
```

#### 2.检查yolo配置环境

##### 1. 下载预训练权重

在YOLO[官方GitHub](https://github.com/ultralytics/ultralytics)README文档中选择自己的训练模型对应的权重并下载，将其放在*ultralytics根目录*下面

##### 2.  检验环境是否配置成功

在项目终端运行如下代码

```
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
```

其中`yolov8n.pt`是刚刚放在根目录下的权重对应的模型；`'https://ultralytics.com/images/bus.jpg'`是下载GitHub上的测试图片，该图片可以在项目文档`ultralytics/ultralytics/assets/bus.jpg`路径下找到，复制其来自内容根的路径，由于本文选择的是segmentation训练模型，则测试代码为：

```
yolo predict model=yolov8n-seg.pt source= ultralytics/ultralytics/assets/bus.jpg
```

终端输出如此则项目环境便没问题

![Figure_3](./pic/03.png)

##### 3.训练自己的数据集

在项目根目录下新建datasets文件夹用来存放在本大节第一小节标注好的数据集，其文件目录格式如下

```
├── ultralytics
 └── datasets
     └── test 
     |		└──images	
     |		└──labels		
     └── train 
     |		└──images
     |		└──labels 	
     └─── val 			
  	        └──images	
      		└──labels
```

##### 4. coco128.yaml文件

```yaml
path: ../datasets/coco128  # 数据集coco128文件夹所在路径，即刚刚新建的datasets中的182路径，最好使用绝对路径给
train: images/train2017  # 从上面的coco128文件路径下找到里面的images文件夹，再找到训练数据文件
val: images/train2017  # 同上，注：val可以用和train一样的数据，即 images/train
test:  # test images (optional)

# Classes，有几个写几个
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
```

### 问题解决 

1、解决训练函数YOLO OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.报错问题

在对应的训练函数（yolo/segment/runs/train）中导入如下代码并重新运行

```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
```

2、如下报错的可能原因是：选择的训练模型为segment，但打标注为labelimg的矩形框标注（矩形框标注适合 检测模型），导致数据集与分割模型不匹配，若想训练分割模型，则应选择labelme多点标注，将其导出为voc格式，再将其进行xml转yolo格式（或直接导出为yolo的txt格式，labelimg导出的 txt文件数据是归一化好的）

![Figure_4](./pic/04.png)

分割数据集可能格式不正确，或者不是 YOLOv8 模型的分割数据集。请确保您的数据集格式正确，符合 YOLov8 文档中提供的训练分割模型的指导原则。关于分割数据集中坐标的归一化，应根据图像大小对坐标进行归一化。x 和 y 的值范围应为 [0，1]。因此，如果在大小为 (640, 320) 的图像上有一个坐标为 (639, 31) 的点，则应将坐标归一化如下： x = 639 / 640 = 0.998，y = 31 / 320 = 0.097。但是，如果发现任何性能问题，可以尝试分别用（图像宽度 - 1）和（图像高度 - 1）来分割点坐标，而不是用（图像宽度）和（图像高度）来分割点坐标。

3、出现如下错误：***\*OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program.\****解决方法

```python
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
```

### 参考文章

1. [YOLOv8手把手教程--超级无敌详细系列](https://franpper.blog.csdn.net/article/details/132645365?spm=1001.2101.3001.6650.13&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-13-132645365-blog-130472736.235%5Ev38%5Epc_relevant_sort_base2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-13-132645365-blog-130472736.235%5Ev38%5Epc_relevant_sort_base2&utm_relevant_index=14)
2. [深度学习环境配置(pytorch版本)----超级无敌详细版（有手就行）](https://blog.csdn.net/weixin_58283091/article/details/127841182?spm=1001.2014.3001.5502)
3. [Anaconda安装（超详细版）](https://blog.csdn.net/m0_61607990/article/details/129531686)
4. [Visual Studio下载安装教程（非常详细）从零基础入门到精通，看完这一篇就够了](https://blog.csdn.net/Javachichi/article/details/131358012)
5. [PyCharm安装教程，图文教程（超详细）](https://blog.csdn.net/wangyuxiang946/article/details/130634049)
6. [cuda11.2版本的对应安装的pytorch版本](https://blog.csdn.net/wangmengmeng99/article/details/128318248)
7. [conda在D盘创建虚拟环境](https://blog.csdn.net/m0_46306264/article/details/131858440)
8. [解决OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.报错问题](https://zhuanlan.zhihu.com/p/599835290)
9. [OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program.](https://blog.csdn.net/qq_46684028/article/details/130676647)

## Part 2 训练与分析

### 一、数据集处理与分析

数据集来源 ：[Kaggle](https://www.kaggle.com/c/severstal-steel-defect-detection/overview/)

Python版本：Python3.11+Python3.8

YOLOv8 训练硬件信息：torch-2.1.0+cpu CPU（Intel Core（TM）i5-9300H 2.40GHz）

#### 1.1 数据集分析

**前言：**下载后的数据集文档目录中包含train.csv&train_images文件夹，train_images中存放训练用的 12,568 张图像，本项目随机选取1300张图像进行训练（图像选取见1.2节），train.csv文件包含了图像信息，如下所示：![Figure_5](.\pic\1.png)

表格中，`ImageId_ClassId`为图像的ID，与train_images中存放的图像ID一一对应，大多数图像分为4类，如红蓝框所示，代表不同种类的缺陷；`EncodedPixels`为图像标签，以“列位置-长度”方式从像素级别框选缺陷，以减小数据量，若某一行有数值，代表该图像有第几类缺陷，如红框第一行jpg_1表示此图像有第一类缺陷，29102像素位置开始的12长度均为非背景，相当于在图像29102像素位置上画一条长度为12像素的竖线；

##### 1.1.1 统计缺陷类别数

![Figure_6](.\pic\2.png)

输出结果表明，数据集中没有缺陷的钢板数量为5902，有缺陷的钢板数量为6666，可以看出**大多数图像没有缺陷或仅含一种缺陷，有无缺陷的图像数量大致相当，极少的图像同时包含3种缺陷，并且在训练集中没有一张图像会同时包含4中缺陷**。

##### 1.1.2 有缺陷的钢板分类

![Figure_7](.\pic\3.png)

表明钢材的表面缺陷类别的数量是不均衡的，以第三类为主，第一类和第四类数量大致相当

##### 1.1.3 图像检查

检查图像的格式，是否能正常打开，以及检查训练集所有图像的尺寸是否一致，Code终端输出如下

![Figure_8](.\pic\4.png)

表明训练集图像数量为12568张，测试集图像数量为1801张，所有的图像大小都为1600*256，且图像均能正常打开

##### 1.1.4  缺陷的可视化标注

注：此部分可视化标注与后续Labelimg标注不同，此节可视化标注是为了画出前言中的`EncodedPixels`像素位置，并将不同种类的可视化标注好的图像进行 分类，便于区分不同缺陷的表面特征，为后续手动Labelimg标注提供参考，Labelimg标注时并未使用可视化标注后的图片，而是使用train_images中的源图像标注，原因见1.2节。

可视化标注后的典型缺陷特征如下所示：

![Figure_9](.\pic\5.png)

可见，第一种缺陷主要为黑色凹坑及麻点，在局部区域呈现集中分布，单个此类缺陷在整个图像范围内像素点较小；第二种缺陷呈现黑色长条状，从第二种缺陷图像总结其特征可知，此种缺陷大多分布在钢带的边缘位置且不易分辨，并没有贯穿钢带的整个表面；第三种缺陷与的特征第二种缺陷类似，不同的是，其缺陷区域较大，往往由数条贯穿表面的线型缺陷组成缺陷区域，且该类缺陷易于识别；第四类缺陷为大面积的黑色损毁区域，容易识别。

#### 1.2 YOLO标准数据集制作

从12,568张图像数据中随机选取1300张图像用于训练、验证与测试，其中各类型的缺陷图像数量如下表所示，各类缺陷图像的数量具有随机性。

| 缺陷种类代码 | 图像数量 | 缺陷特征                     |
| ------------ | -------- | ---------------------------- |
| defect0      | 276      | 无缺陷                       |
| defect1      | 284      | 黑色坑状麻点                 |
| defect2      | 195      | 条状黑色划痕                 |
| defect3      | 274      | 贯穿钢板的条状与片状区域缺陷 |
| defect4      | 271      | 大面积黑色损毁区域缺陷       |

使用labelimg软件将上述图像数据集进行标注并转换为适用于YOLO训练的txt格式数据集（标注后的输出数据集格式有json、xml、提醒他格式，本次选用txt格式，其余2种需要进行格式转换），示例过程如下：（此过程是漫长的重复过程）

![Figure_10](.\pic\6.png)

在这里，标注图像并没有在1.1.4节中可视化 标注后的图片基础上进行标注，使用的为按缺陷分类好的源图像，这是因为可视化标注时，每种缺陷 都选用了一种固定的 颜色进行标注，若以可视化后的图像进行标注，Labelimg标注框中会有可视化的颜色框，进而在后续模型训练中会把框的颜色作为可能的缺陷特征之一。

数据集制作好后对其按照如下格式进行如下分类，代码见第四节：

```
└── data
     └── test 
     |	   └──images
     |	   |	└──num=260	#测试集jpg图像及数量
     |	   └──labels
     |		└──num=260	#标注后测试集txt格式文件数量
     └── train 
     |	   └──images
     |	   |	└──num=909
     |	   └──labels
     |	    	└──num=909
     └─── val 			
           └──images
      	   |	└──num=131
      	   └──labels
      	    	└──num=131
```

### 二、YOLOv8训练与结果分析

#### 2.1 训练

训练时，所使用的软件环境与硬件信息为：Ultralytics YOLOv8.0.215 Python-3.11.5 torch-2.1.0+cpu CPU（Intel Core（TM）i5-9300H 2.40GHz），训练时长为12.8小时，如下所示

![Figure_11](.\pic\7.png)

#### 2.2 结果分析

训练输出结果如下所示，重点分析以下内容：

![Figure_12](.\pic\8.png)

##### 2.2.1 F1_curve.png

F1曲线图，被称为查准率和召回率的调和平均数，最大值为1，表明模型性能最好，0代表模型性能最差。一般而言，置信度阈值（该样本被判定为某一类的概率阈值）较低的时候，很多置信度低的样本被认为是真，召回率高，精确率低；置信度阈值较高的时候，置信度高的样本才能被认为是真，类别检测的越准确，即精准率较大（只有confidence很大，才被判断是某一类别），所以前后两头的F1分数比较少。下图为训练本次训练得到的F1_curve图像，说明在置信度为0.2-0.4区间内得到比较好的F1分数。

![Figure_13](.\pic\9.png)

##### 2.2.2 P_curve.png

单一类准确率，是准确率和置信度的关系图。当判定概率超过置信度阈值时，各个类别识别的准确率，置信度越大，类别检测越准确，但是这样就有可能漏掉一些判定概率较低的真实样本。

<img src=".\pic\10.png" alt="P_curve width=" style="zoom:60%;" />

可以看到，除defect2类型缺陷外，当置信度增大的时候，类别检测的准确率越高，而defect2类型在置信度大于0.4后准确率骤然下降，在达到0.6是时接近为0，随后准确率反弹至最大值1，这也导致所有类型的准确率曲线在置信度为0.5~0.7之间有较大的波动。这可能是由于训练样本中defect2类型缺陷的缺陷特征不明显所引起的当置信度在0.5~0.7区间时，defect2的区分度并不明显，有将其误检测为defect3的可能性，这与所选择的目标检测模型有关。

##### 2.2.3 PR_curve.png

由于精度越高时召回率越低，因此模型要在精确率很高的前提下尽可能的检测到全部的类别，即曲线越接近(1,1)，模型的性能越好。曲线围成的面积AP接近于1，所有类别曲线的AP的平均值称为mAP，即平均精确度。由于defect4曲线更接近（1，1）坐标，且波动情况较其他类型的缺陷更小，表明对不同的缺陷进行检测时，第四类缺陷的检测性能优于其他类型，但使用yolov8n的预权重训练得到的对于其他类型的缺陷检测并不理想，仍有很大的优化空间。 

![Figure_15](.\pic\11.png)

##### 2.2.4 results.png

损失函数是用来衡量模型预测值和真实值不一样的程度，极大程度上决定了模型的性能。其中，box_loss为定位损失，即预测框与标定框之间的误差，该值越小表明定位的越准确； cls_loss为分类损失，计算标定框与对应的标定分类是否正确，越小检测的越准。dfl_loss用以优化bbox；模型的训练与验证结果如下图所示。

![Figure_16](.\pic\12.png)

图中，训练的数据集box_loss与cls_loss均随着训练次数的增多而逐渐减小，呈收敛形式，表明随着训练次数增加，模型对缺陷的定位精确度和检测准确率提高；而对于验证集，虽然box_loss与cls_loss的起始数值均大于训练集相应的起始数值，但在开始训练后的数值快速减小并逐渐收敛，其收敛速度大于训练集收敛速度。可以看到，无论是训练集还是验证集，均在训练100次时仍有减小的趋势，证明模型可以通过增加训练次数的方式得到优化。图中训练集的精确率与召回、验证集的mAP分析如前所述，可见其整体趋势仍在上涨，没有达到稳定值或围绕某一值小范围波动，这是由于训练次数较少导致的准确度不足。

#### 2.3 误差分析

模型训练次数较少

如2.2.4节所述，训练集的精确度和召回、验证集的mAP曲线均在100次时仍出现增长的态势，并没有在一个具体数值附近小范围波动，证明训练次数较少，这也是模型不够精确的最主要原因，为增加准确率、提高模型性能，可将训练次数从100epochs增加到300~500epochs，直至mAP曲线达到稳定的数值。

检测模型精度不足

目标检测精度不太合适。目标检测模型只是给出目标物和目标物的具体位置，并没有从像素级别上进行目标物的区分，为提高识别率可进行分割模型训练，即钢板图像的每个像素只属于一种缺陷类别（或者没缺陷），由于需要定位出钢板缺陷的精细区域，因此可以把这个任务看作是一个语义分割任务，即按照像素级别精度判断每个像素所属的缺陷类别。

为进行像素级别的缺陷分割，在手动进行标注时需要更改矩形标注框为多边形标注框，这无疑会加大手动标注缺陷区域的工作量，尤其是大面积缺陷的不规则区域（如1.1.4节 图d）。若手动标注时仍为矩形框标注，为提高模型的识别精度而选取YOLOv8-seg分割模型进行训练，这将导致数据集与分割模型不匹配，会出现如下所示的错误信息。因此，在充分考虑时间成本的前提下，可以选择标注时间投入较少的检测模型代替像素级的语义分割模型，相较而言，其准确率会有所下降

![Figure_17](.\pic\13.png)

一次输入的图片数量过多，导致组合的图片的缺陷分辨率下降
