import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import numpy as np
import copy
import time
import matplotx
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文


def calDistance(CityCoordinates):
    '''
        计算城市间距离
        输入：CityCoordinates-地点坐标
        输出：dis_matrix-地点间距离矩阵
    '''
    dis_matrix = pd.DataFrame(data=None, columns=range(len(CityCoordinates)), index=range(len(CityCoordinates)))
    for i in range(len(CityCoordinates)):
        xi, yi = CityCoordinates[i][0], CityCoordinates[i][1]
        for j in range(len(CityCoordinates)):
            xj, yj = CityCoordinates[j][0], CityCoordinates[j][1]
            dis_matrix.iloc[i, j] = round(math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2), 2)
    return dis_matrix


def greedy(CityCoordinates, dis_matrix):
    '''
    贪婪策略构造初始解,初始化时将VRP简化为TSP进行构造
        输入：CityCoordinates-节点坐标,dis_matrix-距离矩阵
        输出：line-初始解
    '''
    # 修改dis_matrix以适应求解需要
    dis_matrix = dis_matrix.astype('float64')
    for i in range(len(CityCoordinates)): dis_matrix.loc[i, i] = math.pow(10, 10)
    dis_matrix.loc[:, 0] = math.pow(10, 10)  # 0不在编码内
    line = []  # 初始化
    now_city = random.randint(1, len(CityCoordinates) - 1)  # 随机生成出发地点
    line.append(now_city)  # 添加当前地点到路径
    dis_matrix.loc[:, now_city] = math.pow(10, 10)  # 更新距离矩阵，已经过地点不再被取出
    for i in range(1, len(CityCoordinates) - 1):
        next_city = dis_matrix.loc[now_city, :].idxmin()  # 距离最近的地点
        line.append(next_city)  # 添加进路径
        dis_matrix.loc[:, next_city] = math.pow(10, 10)  # 更新距离矩阵
        now_city = next_city  # 更新当前地点

    return line


def calFitness(birdPop, Demand, dis_matrix, CAPACITY, DISTABCE, C0, C1):
    '''
    贪婪策略分配车辆（解码），计算路径距离（评价函数）
        输入：birdPop-路径，Demand-客户需求,dis_matrix-城市间距离矩阵，CAPACITY-车辆最大载重,DISTABCE-车辆最大行驶距离,C0-车辆启动成本,C1-车辆单位距离行驶成本；
        输出：birdPop_car-分车后路径,fits-适应度
    '''
    birdPop_car, fits = [], []  # 初始化
    for j in range(len(birdPop)):
        bird = birdPop[j]
        lines = []  # 存储线路分车
        line = [0]  # 每辆车服务客户点
        dis_sum = 0  # 线路距离
        dis, d = 0, 0  # 当前客户距离前一个客户的距离、当前客户需求量
        i = 0  # 指向中心仓库
        while i < len(bird):
            if line == [0]:  # 车辆未分配客户点
                dis += dis_matrix.loc[0, bird[i]]  # 记录距离
                line.append(bird[i])  # 为客户点分车
                d += Demand[bird[i]]  # 记录需求量
                i += 1  # 指向下一个客户点
            else:  # 已分配客户点则需判断车辆载重和行驶距离
                if (dis_matrix.loc[line[-1], bird[i]] + dis_matrix.loc[bird[i], 0] + dis <= DISTABCE) & (
                        d + Demand[bird[i]] <= CAPACITY):
                    dis += dis_matrix.loc[line[-1], bird[i]]
                    line.append(bird[i])
                    d += Demand[bird[i]]
                    i += 1
                else:
                    dis += dis_matrix.loc[line[-1], 0]  # 当前车辆装满
                    line.append(0)
                    dis_sum += dis
                    lines.append(line)
                    # 下一辆车
                    dis, d = 0, 0
                    line = [0]

        # 最后一辆车
        dis += dis_matrix.loc[line[-1], 0]
        line.append(0)
        dis_sum += dis
        lines.append(line)

        birdPop_car.append(lines)
        fits.append(round(C1 * dis_sum + C0 * len(lines), 1))

    return birdPop_car, fits


def crossover(bird, pLine, gLine, w, c1, c2):
    '''
    采用顺序交叉方式；交叉的parent1为粒子本身，分别以w/(w+c1+c2),c1/(w+c1+c2),c2/(w+c1+c2)
    的概率接受粒子本身逆序、当前最优解、全局最优解作为parent2,只选择其中一个作为parent2；
        输入：bird-粒子,pLine-当前最优解,gLine-全局最优解,w-惯性因子,c1-自我认知因子,c2-社会认知因子；
        输出：交叉后的粒子-croBird；
    '''
    croBird = [None] * len(bird)  # 初始化
    parent1 = bird  # 选择 parent1
    # 选择 parent2（轮盘赌操作）
    randNum = random.uniform(0, sum([w, c1, c2]))
    if randNum <= w:
        parent2 = [bird[i] for i in range(len(bird) - 1, -1, -1)]  # bird的逆序
    elif randNum <= w + c1:
        parent2 = pLine
    else:
        parent2 = gLine

    # parent1-> croBird
    start_pos = random.randint(0, len(parent1) - 1)
    end_pos = random.randint(0, len(parent1) - 1)
    if start_pos > end_pos: start_pos, end_pos = end_pos, start_pos
    croBird[start_pos:end_pos + 1] = parent1[start_pos:end_pos + 1].copy()

    # parent2 -> croBird
    list2 = list(range(0, start_pos))
    list1 = list(range(end_pos + 1, len(parent2)))
    list_index = list1 + list2  # croBird从后往前填充
    j = -1
    for i in list_index:
        for j in range(j + 1, len(parent2) + 1):
            if parent2[j] not in croBird:
                croBird[i] = parent2[j]
                break

    return croBird

def draw_path(car_routes, CityCoordinates):
    '''
    画路径图
        输入：line-路径，CityCoordinates-地点坐标；
        输出：路径图
    '''
    len=0
    with plt.style.context(matplotx.styles.pitaya_smoothie['light']):
        for route in car_routes:
            pic = ['o-', '*-', 's-', 'D-', 'x-', '+-', 'h-', 'p-','H-','1-','v-','d-','P-','X-']
            x, y = [], []
            for i in route:
                Coordinate = CityCoordinates[i]
                x.append(Coordinate[0])
                y.append(Coordinate[1])
                plt.annotate(str(i), xy=(Coordinate[0], Coordinate[1]))
            x.append(x[0])
            y.append(y[0])
            plt.plot(x, y, pic[len], alpha=1, linewidth=0.9)
            len = len + 1
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xticks(np.arange(125, 175, 5))
        plt.yticks(np.arange(180, 270, 10))
    #    plt.xticks(np.arange(0, 70, 5))
    #    plt.yticks(np.arange(0, 80, 5))
        plt.title('PSO: "E-n22-k4"测试集的路径图')
    #    plt.title('PSO: "E-n101-k14"测试集的路径图')
        plt.show()


def draw_fitness(iter,bestfit):
    with plt.style.context(matplotx.styles.pitaya_smoothie['light']):
        plt.plot(np.arange(1,iter+1,1),bestfit,alpha=1, linewidth=0.9,label="Fitness")
        plt.xlabel('迭代次数')
        plt.ylabel('适应度')
        plt.legend()
        plt.title('PSO: "E-n22-k4"测试集的适应度曲线')
    #    plt.title('PSO: "E-n101-k14"测试集的适应度曲线')
        plt.show()

if __name__ == '__main__':
    # 车辆参数
    CAPACITY = 6000  # 车辆最大容量 # 112
    DISTABCE = 1000  # 车辆最大行驶距离 # 500
    C0 = 0  # 车辆启动成本
    C1 = 1  # 车辆单位距离行驶成本

    # PSO参数
    birdNum = 22  # 粒子数量  #101
    w = 0.9  # 惯性因子
    c1 = 2  # 自我认知因子
    c2 = 2  # 社会认知因子
    pBest, pLine = 0, []  # 当前最优值、当前最优解，（自我认知部分）
    gBest, gLine = 0, []  # 全局最优值、全局最优解，（社会认知部分）

    # 其他参数
    iterMax = 1000  # 迭代次数  #3000
    iterI = 1  # 当前迭代次数
    bestfit = []  # 记录每代最优值

    """ 
    ""E - n101 - k14""
    Customer = [(35, 35),(41, 49),(35, 17),(55, 45),(55, 20),(15, 30),(25, 30),(20, 50),(10, 43),(55, 60),
                 (30, 60),(20, 65),(50, 35),(30, 25),(15, 10),(30, 5),(10, 20),(5, 30),(20, 40),(15, 60),
                 (45, 65),(45, 20),(45, 10),(55, 5),(65, 35),(65, 20),(45, 30),(35, 40),(41, 37),(64, 42),
                 (40, 60),(31, 52),(35, 69),(53, 52),(65, 55),(63, 65),(2, 60),(20, 20),(5, 5),(60, 12),
                 (40, 25),(42, 7),(24, 12),(23, 3),(11, 14),(6, 38),(2, 48),(8, 56),(13, 52),(6, 68),(47, 47),
                 (49, 58),(27, 43),(37, 31),(57, 29),(63, 23),(53, 12),(32, 12),(36, 26),(21, 24),(17, 34),
                 (12, 24),(24, 58),(27, 69),(15, 77),(62, 77),(49, 73),(67, 5),(56, 39),(37, 47),(37, 56),
                 (57, 68),(47, 16),(44, 17),(46, 13),(49, 11),(49, 42),(53, 43),(61, 52),(57, 48),(56, 37),
                 (55, 54),(15, 47),(14, 37),(11, 31),(16, 22),(4, 18),(28, 18),(26, 52),(26, 35),(31, 67),
                 (15, 19),(22, 22),(18, 24),(26, 27),(25, 24),(22, 27),(25, 21),(19, 21),(20, 26),(18, 18)]
    Demand = [0, 10, 7, 13, 19, 26, 3, 5, 9, 16, 16, 12, 19, 23, 20, 8, 19, 2, 12, 17, 9, 11, 18, 29, 3, 6, 17, 16, 16, 9,
              21, 27, 23, 11, 14, 8, 5, 8, 16, 31, 9, 5, 5, 7, 18, 16, 1, 27, 36, 30, 13, 10, 9, 14, 18, 2, 6, 7, 18, 28, 3,
              13, 19, 10, 9, 20, 25, 25, 36, 6, 5, 15, 25, 9, 8, 18, 13, 14, 3, 23, 6, 26, 16, 11, 7, 41, 35, 26, 9, 15, 3,
              1, 2, 22, 27, 20, 11, 12, 10, 9, 17]
    """

    """E-n22-k4"""
    Customer = [(145, 215),(151, 264),( 159, 261),(130, 254),(128, 252),(163, 247),(146, 246),(161, 242),
                 (142, 239),(163, 236),(148, 232),(128, 231),(156, 217),(129, 214),(146, 208),(164, 208),
                 (141, 206),(147, 193),(164, 193),(129, 189),(155, 185),(139, 182)]
    Demand = [0,1100,700,800,1400,2100,400,800,100,500,600,1200,1300,1300,300,900,2100,1000,900,2500,1800,700]
    start = time.time()
    dis_matrix = calDistance(Customer)  # 计算城市间距离

    birdPop = [greedy(Customer, dis_matrix) for i in range(birdNum)]  # 贪婪算法构造初始解
    # birdPop = [random.sample(range(1,len(Customer)),len(Customer)-1) for i in range(birdNum)] # 客户点编码，随机初始化生成种群

    birdPop_car, fits = calFitness(birdPop, Demand, dis_matrix, CAPACITY, DISTABCE, C0, C1)  # 分配车辆，计算种群适应度

    gBest = pBest = min(fits)  # 全局最优值、当前最优值
    gLine = pLine = birdPop[fits.index(min(fits))]  # 全局最优解、当前最优解
    gLine_car = pLine_car = birdPop_car[fits.index(min(fits))]
    bestfit.append(gBest)

    while iterI <= iterMax:  # 迭代开始
        for i in range(birdNum):
            birdPop[i] = crossover(birdPop[i], pLine, gLine, w, c1, c2)

        birdPop_car, fits = calFitness(birdPop, Demand, dis_matrix, CAPACITY, DISTABCE, C0, C1)  # 分配车辆，计算种群适应度
        pBest, pLine, pLine_car = min(fits), birdPop[fits.index(min(fits))], birdPop_car[fits.index(min(fits))]
        if min(fits) <= gBest:
            gBest, gLine, gLine_car = min(fits), birdPop[fits.index(min(fits))], birdPop_car[fits.index(min(fits))]

        bestfit.append(gBest)
        print(iterI, gBest)  # 打印当前代数和最佳适应度值
        iterI += 1  # 迭代计数加一

    temp = copy.deepcopy(gLine_car)
    for i in range(len(gLine_car)):
        for j in range(len(gLine_car[i])):
            gLine_car[i][j] += 1
    print("Solution: ", gLine_car, ",", gBest)  # 路径顺序
    end = time.time()
    optimalSolution = [[18, 21, 19, 16, 13], [17, 20, 22, 15], [14, 12, 5, 4, 9, 11], [10, 8, 6, 3, 2, 7]], 375
    """
    optimalSolution = [[92,37,100,85,93,99,96], [13,87,97,95,94], [18,8,46,36,49,64,11,62,88], [52,7,19,47,48,82],
                       [53,27],[61,16,86,38,44,91,98],[69,70,30,32,90,63,10,31],[23,67,39,25,55,54],[28,76,77,3,68,12],
                       [80,24,29,78,34,35,71,65,66,20],[50,79,33,81,9,51,1],[26,4,56,75,72,21,40],[58,2,57,42,14,43,15,41,22,74,73],
                       [89,60,83,45,17,84,5,59,6],1067]
    """
    print("Optimal solution:", str(optimalSolution))
    print("TIME: {:.2f}s".format(end - start))
    draw_path(temp, Customer)  # 画路径图
    draw_fitness(iterMax,bestfit[1:])