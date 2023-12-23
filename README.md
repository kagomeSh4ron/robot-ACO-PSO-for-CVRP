# robot-ACO-PSO-for-CVRP

## 项目介绍

来自专业课——《机器人基础》期末论文

解决小车路径问题（Vehicle Routing Problem,VRP）的目标是通过一些时间管理将产品或货物交付到正确的目的地，为了达到这些要求，需要进行有效的路线规划。因此小车路径问题一直是物流工程、应用数学和计算机科学领域研究得最多的问题之一。启发式算法在求解类似CVRP问题这样的组合优化问题时取得了很好的效果，如蚁群算法和粒子群算法。本文提出了基于蚁群算法和粒子群算法分别求解CVRP的两种模型，寻找CVRP问题的最优解，并通过E-n22-k4和E-n101-k14测试集进行验证。通过对比解的准确度、运行时间、Gap值等方面，结果显示在数据量不同的情况下，PSO相比于ACO算法更能得到接近最优解的结果，随着数据集规模增大，PSO算法的速度是ACO算法的10倍，Gap值相较ACO算法降低了8.97%，求解所得最优路径与现标准结果分别相差0%与8.09%。实验结果表明，使用基于粒子群算法建立的模型对于求解CVRP问题具有可行性。 

## 编译环境

ubuntu20.04

采用vscode软件

## 运行结果
<img width="215" alt="image" src="https://github.com/ckxSh4ron/robot-ACO-PSO-for-CVRP/assets/138695155/dc9f0151-57d3-4d6d-9600-454a63769ede">
PSO算法下E-n22-k4最佳路径图

<img width="217" alt="image" src="https://github.com/ckxSh4ron/robot-ACO-PSO-for-CVRP/assets/138695155/b12df01e-c8d1-4221-8694-e614484d61fc">
PSO算法下E-n22-k4最佳路径图

<img width="231" alt="image" src="https://github.com/ckxSh4ron/robot-ACO-PSO-for-CVRP/assets/138695155/445e89db-3e0d-4172-aeb6-7d89ac8b83a9">
PSO算法下E-n101-k14最佳路径图

<img width="230" alt="image" src="https://github.com/ckxSh4ron/robot-ACO-PSO-for-CVRP/assets/138695155/dc1b3721-814b-4393-b086-0c387bef79d4">
ACO算法下E-n101-k14最佳路径图
