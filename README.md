# IndustrialDefectDetectionInside
# 实验说明
工业缺陷检测课程实验，仅用于模型训练与本地推理练习。
# 环境
CU118  
torch2.10
# DATASET
https://www.kaggle.com/datasets/rusuanjun/pec-dataset
https://www.selectdataset.com/dataset/e3d68a3257fe7803a0043dde68d89f38
https://www.kaggle.com/datasets/viacheslavasadchiy/radiographs-welding-defect-detection?resource=download
https://pan.baidu.com/s/1gZBmIyyV1NCvUP2NUUxMgQ?pwd=gyxi
https://pan.baidu.com/s/1mf3gkusHvUvokVvlW_tKuA?pwd=2dut
https://zenodo.org/records/10618962
# 期望
目标是低耦合的代码设计+oop的思想组织项目
# 结果
针对PCE两个数据集厚度预测 EPOC=40

| 模型+材质 | MLP铝 | ResNet铝 | 1D-CNN铝 | MLP钢 | ResNet钢 | 1D-CNN钢 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 查全率 |  84.59% | 77.85% | 78.02% | 95.67%   | 65.81% |   96.46%   |    
| 查准率 |  84.75%   | 78.64% |  78.12% | 96.51% |62.17%  | 97.27%     |  
| MSE  |  0.3369   | 0.4569 |  0.4323 | 0.0542  | 2.1627  | 0.0413 |

注意 需要将ResNet的epoch改为60才有好的效果，40轮还欠拟合
Precision 93.54% 查全率Recall 92.33%  MSE 0.0801