# IndustrialDefectDetectionInside
# 实验说明
工业缺陷检测课程实验，仅用于模型训练与本地推理练习。
# 环境
CU118  
torch2.00
ultralytics

# DATASET
## 金属厚度预测数据集
https://www.kaggle.com/datasets/rusuanjun/pec-dataset

## 超声流量计（Ultrasonic Flowmeters）运行工况与健康状态
https://www.selectdataset.com/dataset/e3d68a3257fe7803a0043dde68d89f38

## RWDD 焊缝 X 射线照片中的缺陷检测——焊接缺陷检测
Gazprom Neft（俄罗斯天然气工业石油公司）黑客马拉松提供的数据，该数据集包含大量焊接接头的 X 射线检测图像，用于解决焊接缺陷检测与分类任务。
类别名称：

| 编号	 | 缺陷类别（俄语）         | 	中文翻译                                        |
|-----|------------------|----------------------------------------------|
| 0	  | Пора	            | 气孔（孔隙缺陷，Porosity）                            |
| 1	  | Включение        | 	夹杂物（焊缝夹杂 Inclusion）                         |
| 2	  | Подрез           | 	咬边（Under-cut）                               |
| 3   | 	Прожог          | 	烧穿（Burn-through）                            |
| 4   | 	Трещина         | 	裂纹（Crack）                                   |
| 5   | 	Наплыв          | 	焊瘤 / 熔积（Overlap / Excess weld metal）        |
| 6   | 	Эталон 1        | 	标准样本 1（Reference 1）                         |
| 7   | 	Эталон 2        | 	标准样本 2（Reference 2）                         |
| 8   | 	Эталон 3        | 	标准样本 3（Reference 3）                         |
| 9   | 	Пора-скрытая    | 	隐藏气孔（Hidden porosity）                       |
| 10  | 	Утяжина         | 	凹陷 / 收缩缺陷（Shrinkage cavity / Groove defect） |
| 11  | 	Несплавление	   | 未熔合（Lack of fusion）                          |
| 12  | 	Непровар корня	 | 根部未焊透（Incomplete root penetration）           |

https://www.kaggle.com/datasets/viacheslavasadchiy/radiographs-welding-defect-detection?resource=download

## WDD焊缝缺陷检测数据集 射线射线照片图像
### 介绍：
https://github.com/admin1523/Weld-defect-detection-datasets

数据集覆盖了射线底片中常见的焊接缺陷，包括但不限于：
Incomplete penetration	未焊透 / 根部未完全熔透
Porosity	气孔
Slag inclusion	夹渣
Incomplete fusion	未熔合
Cracks	裂纹
Other typical weld anomalies	其他典型焊接异常
### 数据集 1
https://pan.baidu.com/s/1gZBmIyyV1NCvUP2NUUxMgQ?pwd=gyxi
### 数据集 2
相比数据集 1：
该数据集中的缺陷通常表现为：
更低的对比度
更不明显的边界
更强的背景干扰
更加接近真实工业检测环境。
https://pan.baidu.com/s/1mf3gkusHvUvokVvlW_tKuA?pwd=2dut

## 焊缝 X 射线图像数据集
https://zenodo.org/records/10618962
# 期望
目标是低耦合的代码设计，+oop的思想组织项目，训练模型
# 结果
## 针对PCE两个数据集厚度预测 EPOCH=40

| 模型+材质 | MLP铝 | ResNet铝 | 1D-CNN铝 | MLP钢 | ResNet钢 | 1D-CNN钢 |
|-------| ---- | ---- | ---- | ---- | ---- | ---- |
| 查全率   |  84.59% | 77.85% | 78.02% | 95.67%   | 65.81% |   96.46%   |    
| 查准率   |  84.75%   | 78.64% |  78.12% | 96.51% |62.17%  | 97.27%     |  
| MSE   |  0.3369   | 0.4569 |  0.4323 | 0.0542  | 2.1627  | 0.0413 |

注意 需要将ResNet的epoch改为60才有好的效果，40轮还欠拟合
Precision 93.54% 查全率Recall 92.33%  MSE 0.0801

## 针对WDD焊缝缺陷检测数据集 射线射线照片图像 数据集-识别任务
### 使用yoloV11n-seg，EPOCH=50，1600张数据集，batchsize=16 ，耗时40min训练

指标 | 边界框检测指标 (Box) | 缺陷分割指标 (Mask) 
---- | ---- | ----
查准率 |  0.9991 | 0.9455
查全率| 0.9880| 0.9356
F1-Score|  0.9935| 0.9405
mAP50|    0.9947| 0.9542
### 使用DeepLabV3和使用FCN算法，epoch=5，训练集200张
指标\算法 | DeepLabV3 | FCN
---- | ---- | ----
像素查准率 |0.6651| 0.7159
像素查全率 |  0.1075|0.0460
F1-Score | 0.1851| 0.0864
缺陷交并比 (IoU / mAP):| 0.1020| 0.0452


## 针对RWDD焊缝X射线照片中的缺陷检测——焊接缺陷检测数据集
yolov11n-seg，EPOCH=20，1000张数据集，batchsize=16 

指标 | 值
 ---|---
查准| 0.544  
查全 |0.415  
f1 |0.4712
mAP50|  0.355 

## USM采集超声波流量计 采集Ultrasonic Flowmeter Diagnostics Data Set（超声波流量计诊断数据集）
细分为四个数据集 分别是Meter A, B, C, D

流量计编号 | 样本量 | 属性(特征) | 数声道配置覆盖的健康状态（分类目标）
---- | ---| --- | ---
Meter A | 87 | 37 | 8 声道 (8-path)| 2 类：健康、安装效应（管道布局引起的流态异常）
Meter B | 92 | 52 | 4 声道 (4-path) | 3 类：健康、气体注入（气泡干扰）、结蜡（内部结垢）
Meter C | 181 | 44 | 4 声道 (4-path) | 4 类：健康、气体注入、安装效应、结蜡
Meter D | 180 | 44 | 4 声道 (4-path) | 4 类：健康、气体注入、安装效应、结蜡

所有数据集中的特征都是连续的（非离散）
### MLP模型 EPOCH=30

数据集+类别 | B1 | B2 | B3 |A1 | A2 | C1 | C2 | C3 | C4| D1 |D2| D3 |D4
---|----|---|---|---|---|---|---|---|---|----|---|-----|---
查准率 | 1 | 1  | 1 | 1 | 1 | 1  | 1 | 1 | 1 | 1  | 1 | 0.9167 | 1 
查全率| 1  | 1  | 1 | 1 | 1 | 1  | 1 | 1 | 1 | 0.9 | 1 | 1   | 1 
F1-Score| 1 | 1 | 1 | 1 | 1 | 1  | 1 | 1 | 1 | 0.9474 | 1 | 0.9565 | 1 

按照评估采样样本加权平均

数据集 | B |A | C  |D
---|----|--|------|---
查准率 | 1 | 1 | 1 | 0.9745 
查全率| 1  | 1 | 1 | 0.9722 
F1-Score| 1 | 1 | 1 |  0.9721  

### XGBoost
按照测试集各类别样本加权平均

数据集 | B | A       | C  |D
---|---|---------|------|---
查准率 | 1 | 0.8889  | 0.9754| 0.9213
查全率| 1 | 0.8889  | 0.9730 | 0.9167
F1-Score| 1 | 0.8889  |  0.9721| 0.9172 

### 随机森林
按照测试集各类别样本加权平均

数据集 | B | A       | C |D
---|---|---------|---|---
查准率 | 1 | 0.8889  | 1 |  0.9182
查全率| 1 | 0.8889  | 1 |  0.9167
F1-Score| 1 | 0.8889  | 1 | 0.9163
