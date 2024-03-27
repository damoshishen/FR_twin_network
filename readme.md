# 《基于改进孪生网络的小样本人脸识别算法与系统设计》主要代码及复现
版本：V0.9

最后更新时间：2024.3.36

作者：物理系的计算机选手

## 一.项目介绍
作为一个有“算法梦”的物理系渣渣，在读研期间水了一篇国内普刊（关于人脸识别算法）也算过足了瘾。之前总是有人问几年前写的那篇粗稿，最近工作之余，将当时的论文主体复现并讲解一下。以下是各文件夹下代码的含义：
+ analysis:
  1. data_analysis.ipynb: 数据分析
+ data_Generated: 
  1. datasetFacesORL.npy：处理好的数据
+ data_process:
  1. data_process:
  2. image_pair_creation:
+ model:
  1. distance_function: 各种距离函数
  2. network: 基干网络及变体距离
  3. tf_test: 测试windows是否可以使用tensorflow--GPU
  4. train: 模型训练文件


## 二.使用方法
1. 配置环境
2. 运行model/train.ipynb 进行模型训练