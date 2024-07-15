##数据集分析
import pandas as pd
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
 
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑字体
plt.rcParams['axes.unicode_minus'] = False  # 处理负号显示异常


#读取和分析文本数据
train_df = pd.read_csv("C:/Users/yangb/Desktop/test/yuandata/train.csv")
sample_df = pd.read_csv("C:/Users/yangb/Desktop/test/yuandata/sample_submission.csv")
#查看
print(train_df.head())
print('\r\n')
print(sample_df.head())

#统计有无缺陷及每类缺陷的图像数量：
class_dict = defaultdict(int)
kind_class_dict = defaultdict(int)
no_defects_num = 0
defects_num = 0

for col in range(0, len(train_df), 4):
    img_names = [str(i).split("_")[0] for i in train_df.iloc[col:col+4, 0].values]
    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):
        raise ValueError

    labels = train_df.iloc[col:col+4, 1]
    if labels.isna().all():
        no_defects_num += 1
    else:
        defects_num += 1

    kind_class_dict[sum(labels.isna().values == False)] += 1

    for idx, label in enumerate(labels.isna().values.tolist()):
        if label == False:
            class_dict[idx+1] += 1

print("无缺陷钢板数量: {}".format(no_defects_num))
print("有缺陷钢板数量: {}".format(defects_num))


#对有缺陷的图像进行分类统计
fig, ax = plt.subplots()
sns.barplot(x=list(class_dict.keys()), y=list(class_dict.values()), ax=ax)
#ax.set_title("每个类别的图像数量")
ax.set_xlabel("缺陷类别")
for p in ax.patches:
    height = p.get_height()
    ax.annotate('{:.0f}'.format(height), (p.get_x() + p.get_width() / 2, height), 
        ha='center', va='bottom')
plt.show()
print(class_dict)


#统计一张图像中可能包含的缺陷种类数
fig, ax = plt.subplots()
sns.barplot(x=list(kind_class_dict.keys()), y=list(kind_class_dict.values()), ax=ax)
#ax.set_title("每幅图像中包含的类别数");
ax.set_xlabel("图像中的类别数")
for p in ax.patches:
    height = p.get_height()
    ax.annotate('{:.0f}'.format(height), (p.get_x() + p.get_width() / 2, height), 
        ha='center', va='bottom')
plt.show()
print(kind_class_dict)

