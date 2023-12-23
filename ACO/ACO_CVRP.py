import ACO_RegExService
from matplotlib.pylab import mpl
import numpy
from functools import reduce
import matplotlib.pyplot as plt
import time
import numpy as np
import matplotx

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文

# 参数设置
alfa = 2
beta = 5
sigm = 3
ro = 0.8
th = 400  #1000
fileName = "E-n22-k4"
#fileName = "E-n101-k14.txt"
iterations = 1000   #3000
ants = 22      #101

def generateGraph():
    '''
    生成数据信息
        输出：vertices-地点顶点集, edges-距离字典集合, capacityLimit-车辆负载限制值,
             demand-客户需求矩阵, feromones-信息素字典集合, optimalValue-最优值
    '''
    capacityLimit, graph, demand, optimalValue = ACO_RegExService.getData(fileName)
    vertices = list(graph.keys())
    vertices.remove(1)
    edges = { (min(a,b),max(a,b)) : numpy.sqrt((graph[a][0]-graph[b][0])**2 + (graph[a][1]-graph[b][1])**2) for a in graph.keys() for b in graph.keys()}
    feromones = { (min(a,b),max(a,b)) : 1 for a in graph.keys() for b in graph.keys() if a!=b }
    
    return vertices, edges, capacityLimit, demand, feromones, optimalValue

def solutionOfOneAnt(vertices, edges, capacityLimit, demand, feromones):
    '''
    生成一辆车行驶覆盖所有地点的行驶路径
        输入：vertices-地点顶点集，edges-距离字典集合，capacityLimit-车辆负载限制值，demand-客户需求矩阵，feromones-信息素字典集合
        输出：行驶路径
    '''
    solution = []   # 车辆行驶路径
    while(len(vertices)!=0):
        path = []  # 一条路径
        # 随机选择第一个地点，加入路径，从顶点删除
        city = numpy.random.choice(vertices)
        capacity = capacityLimit - demand[city]
        path.append(city)
        vertices.remove(city)
        while(len(vertices)!=0):
            # 地点转移概率公式计算
            probabilities = list(map(lambda x: ((feromones[(min(x,city), max(x,city))])**alfa)*((1/edges[(min(x,city), max(x,city))])**beta), vertices))
            # 归一化
            probabilities = probabilities/numpy.sum(probabilities)
            # 随机选择下一个地点
            city = numpy.random.choice(vertices, p=probabilities)
            capacity = capacity - demand[city]
            # 判断车辆负载是否可承担当前地点运输需求
            if(capacity>0):
                path.append(city)
                vertices.remove(city)
            else:
                break
        solution.append(path)

    return solution

def rateSolution(solution, edges):
    '''
    生成行驶路径的总距离
        输入：solution-行驶路径，edges-距离字典集合
        输出：路径总距离
    '''
    s = 0   # 记录距离
    for i in solution:
        a = 1   # 起点，首先是中心仓库
        for j in i:
            b = j  # 终点
            s = s + edges[(min(a,b), max(a,b))]   # 首先加入了中心仓库到第一个地点的距离
            a = b    # 起点移动至路径中下一个地点
        b = 1   # 最后回到中心仓库
        s = s + edges[(min(a,b), max(a,b))]

    return s

def updateFeromone(feromones, solutions, bestSolution):
    '''
    更新信息素字典集合
        输入：feromones-信息素字典集合，solutions-所有车辆的行驶路径，bestSolution-最优路径与最短路径
        输出：路径总距离
    '''
    Lavg = reduce(lambda x,y: x+y, (i[1] for i in solutions))/len(solutions)  # 距离平均值
    feromones = { k : (ro + th/Lavg)*v for (k,v) in feromones.items() }
    solutions.sort(key = lambda x: x[1])  # 按距离大小升序
    if(bestSolution!=None):
        if(solutions[0][1] < bestSolution[1]):
            bestSolution = solutions[0]  #更新最优路径和最短路径
        for path in bestSolution[0]:
            for i in range(len(path)-1):
                feromones[(min(path[i],path[i+1]), max(path[i],path[i+1]))] = sigm/bestSolution[1] + feromones[(min(path[i],path[i+1]), max(path[i],path[i+1]))]
    else:
        bestSolution = solutions[0]
    for l in range(sigm):
        paths = solutions[l][0]
        L = solutions[l][1]
        for path in paths:
            for i in range(len(path)-1):
                feromones[(min(path[i],path[i+1]), max(path[i],path[i+1]))] = (sigm-(l+1)/L**(l+1)) + feromones[(min(path[i],path[i+1]), max(path[i],path[i+1]))]
    return bestSolution

def draw_path(car_routes, CityCoordinates):
    '''
    画路径图
        输入：line-路径，CityCoordinates-地点坐标
        输出：路径图
    '''
    len=0
    with plt.style.context(matplotx.styles.pitaya_smoothie['light']):
        for route in car_routes:
            pic=['o-', '*-', 's-', 'D-', 'x-', '+-', 'h-', 'p-','H-','1-','v-','d-','P-','X-']
            x, y = [], []
            for i in route:
                Coordinate = CityCoordinates[i-1]
                x.append(Coordinate[0])
                y.append(Coordinate[1])
                plt.annotate(str(i), xy=(Coordinate[0], Coordinate[1]))
            x.append(x[0])
            y.append(y[0])
            plt.plot(x, y, pic[len],alpha=1, linewidth=0.9)
            len=len+1

        plt.xlabel('x')
        plt.ylabel('y')
        plt.xticks(np.arange(125, 175, 5))
        plt.yticks(np.arange(180, 270, 10))
    #    plt.xticks(np.arange(0, 70, 5))
    #    plt.yticks(np.arange(0, 80, 5))
        plt.title('ACO: "E-n22-k4"测试集的路径图')
    #    plt.title('ACO: "E-n101-k14"测试集的路径图')
        plt.show()

def draw_fitness(iter,bestfit):
    '''
    画适应度曲线
        输入：iter-迭代次数，bestfit-距离值
        输出：适应度曲线
    '''
    with plt.style.context(matplotx.styles.pitaya_smoothie['light']):
        plt.plot(np.arange(1, iter + 1, 1), bestfit, alpha=1, linewidth=0.9, label="Fitness")
        plt.xlabel('迭代次数')
        plt.ylabel('适应度')
        plt.legend()
        plt.title('ACO: "E-n22-k4"测试集的适应度曲线')
        plt.show()


def main():
    bestSolution = None
    vertices, edges, capacityLimit, demand, feromones, optimalValue = generateGraph()
    fitness=[]
    for i in range(iterations):
        solutions = list()
        for _ in range(ants):
            solution = solutionOfOneAnt(vertices.copy(), edges, capacityLimit, demand, feromones)
            solutions.append((solution, rateSolution(solution, edges)))
        bestSolution = updateFeromone(feromones, solutions, bestSolution)
        print(str(i+1)+" "+str(round(bestSolution[1],1)))
        fitness.append(round(bestSolution[1],1))
    return bestSolution,fitness

if __name__ == "__main__":
    start = time.time()
    solution,fitness= main()
    capacityLimit, graph, demand, optimalValue = ACO_RegExService.getData(fileName)
    line=solution[0]
    for i in range(len(line)):
        line[i].reverse()
        line[i].append(1)
        line[i].reverse()
        line[i].append(1)
    end = time.time()
    print("Solution: ",solution[0],",",round(solution[1],1))

    if(fileName=="E-n22-k4"):   #E-n101-k14
         optimalSolution = [[18, 21, 19, 16, 13], [17, 20, 22, 15], [14, 12, 5, 4, 9, 11], [10, 8, 6, 3, 2, 7]], 375
         """
         optimalSolution = [[92, 37, 100, 85, 93, 99, 96], [13, 87, 97, 95, 94], [18, 8, 46, 36, 49, 64, 11, 62, 88],
                           [52, 7, 19, 47, 48, 82],
                           [53, 27], [61, 16, 86, 38, 44, 91, 98], [69, 70, 30, 32, 90, 63, 10, 31],
                           [23, 67, 39, 25, 55, 54], [28, 76, 77, 3, 68, 12],
                           [80, 24, 29, 78, 34, 35, 71, 65, 66, 20], [50, 79, 33, 81, 9, 51, 1],
                           [26, 4, 56, 75, 72, 21, 40], [58, 2, 57, 42, 14, 43, 15, 41, 22, 74, 73],
                           [89, 60, 83, 45, 17, 84, 5, 59, 6], 1067]
         """
         print("Optimal solution: "+str(optimalSolution))
    print("TIME: {:.2f}s".format(end - start))
    draw_path(line,list(graph.values()))
    draw_fitness(iterations,fitness)