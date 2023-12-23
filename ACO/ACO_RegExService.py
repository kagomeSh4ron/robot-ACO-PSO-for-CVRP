import re
def getData(fileName):
    '''
    读取文件
        输入：fileName-文件名
        输出：capacity-车辆负载限制值, graph-节点坐标, demand-需求, optimalValue-标准最优值
    '''
    f = open("E-n22-k4.txt", 'r')
    #f = open("E-n101-k14.txt",'r')
    content = f.read()
    optimalValue = re.search("Optimal value: (\d+)", content, re.MULTILINE)
    if(optimalValue != None):
        optimalValue = optimalValue.group(1)
    else:
        optimalValue = re.search("Best value: (\d+)", content, re.MULTILINE)
        if(optimalValue != None):
            optimalValue = optimalValue.group(1)
    capacity = re.search("^CAPACITY : (\d+)$", content, re.MULTILINE).group(1)
    graph = re.findall(r"^(\d+) (\d+) (\d+)$", content, re.MULTILINE)
    demand = re.findall(r"^(\d+) (\d+)$", content, re.MULTILINE)
    graph = {int(a):(int(b),int(c)) for a,b,c in graph}
    demand = {int(a):int(b) for a,b in demand}
    capacity = int(capacity)
    optimalValue = int(optimalValue)
    return capacity, graph, demand, optimalValue