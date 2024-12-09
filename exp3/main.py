import math
import random
import numpy as np
import matplotlib.pyplot as plt
'''
每个粒子对应的路径表示一种配送顺序
'''


class PSO:

    # 参数初始化
    def __init__(self):
        # 车辆参数
        # 车辆最大容量
        self.capacity = 120
        # 车辆最大行驶距离
        self.distance = 250
        # 车辆启动成本
        self.C0 = 30
        # 车辆单位距离行驶成本
        self.C1 = 1

        # PSO参数
        # 粒子数量
        self.bird_num = 50
        # 惯性因子
        self.w = 0.2
        # 自我认知因子
        self.c1 = 0.4
        # 社会认知因子
        self.c2 = 0.4
        # 当前最优值
        self.local_cost = 0
        # 当前最优解
        self.local_paths = list()
        # 全局最优值
        self.global_cost = 0
        # 全局最优解
        self.global_paths = list()
        # 行驶路径
        self.paths = []
        # 车辆分配情况
        self.cars = []
        # 粒子适应度
        self.fits = []
        # 全局最优路线
        self.global_lines = []

        # 其他参数
        # 最大迭代次数
        self.epochs = 1000
        # 每次迭代的最优解
        self.history_cost = list()

        # 客户数据
        # 客户点坐标
        self.customers = [(50, 50), (96, 24), (40, 5), (49, 8), (13, 7),
                          (29, 89), (48, 30), (84, 39), (14, 47), (2, 24),
                          (3, 82), (65, 10), (98, 52), (84, 25), (41, 69),
                          (1, 65), (51, 71), (75, 83), (29, 32), (83, 3),
                          (50, 93), (80, 94), (5, 42), (62, 70), (31, 62),
                          (19, 97), (91, 75), (27, 49), (23, 15), (20, 70),
                          (85, 60), (98, 85)]

        # 客户需求
        self.demands = [
            0, 16, 11, 6, 10, 7, 12, 16, 6, 16, 8, 14, 7, 16, 3, 22, 18, 19, 1,
            14, 8, 12, 4, 8, 24, 24, 2, 10, 15, 2, 14, 9
        ]
        # 客户间的距离
        self.distances = None

    # 计算客户之间的距离
    def get_distance(self):
        n = len(self.customers)
        distances = np.full((n, n), np.inf, dtype=float)
        for i in range(n):
            x1, y1 = self.customers[i][0], self.customers[i][1]
            for j in range(i + 1, n):
                x2, y2 = self.customers[j][0], self.customers[j][1]
                distance = math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))
                distances[i, j] = distance
                distances[j, i] = distance
        self.distances = distances

    # 贪心初始化
    def greedy_init(self):
        paths = list()
        for i in range(self.bird_num):
            path = list()
            distances = self.distances.copy()
            source_city = random.randint(1, len(self.customers) - 1)
            path.append(source_city)
            distances[:, source_city] = np.inf
            for j in range(1, len(self.customers) - 1):
                next_city = np.argmin(distances[source_city])
                path.append(next_city)
                distances[:, next_city] = np.inf
                source_city = next_city
            paths.append(path)
        self.paths = paths

    # 计算粒子的适应度
    def get_fitness(self):
        # 车辆分配情况
        cars = []
        # 粒子适应度
        fits = []
        for path in self.paths:
            # 总线路
            lines = []
            # 每辆车的服务客户
            line = [0]
            # 当前满足的客户需求量
            capacity = 0
            # 当前线路的距离
            distance = 0
            # 总线路的距离
            distance_sum = 0
            i = 0
            while i < len(path):
                # 当前车辆未分配客户点
                if line == [0]:
                    capacity += self.demands[path[i]]
                    distance += self.distances[0, path[i]]
                    line.append(path[i])
                    i += 1
                # 当前车辆以分配客户点
                else:
                    # 同时满足重量和距离约束
                    if (distance + self.distances[line[-1], path[i]] +
                            self.distances[path[i], 0] <= self.distance) and (
                                capacity + self.demands[path[i]]
                                <= self.capacity):
                        capacity += self.demands[path[i]]
                        distance += self.distances[line[-1], path[i]]
                        line.append(path[i])
                        i += 1
                    # 不满足重量或距离约束
                    else:
                        # 返回配送中心
                        distance += self.distances[line[-1], 0]
                        distance_sum += distance
                        line.append(0)
                        lines.append(line)

                        # 分配下一辆车
                        capacity = 0
                        distance = 0
                        line = [0]
            # 特殊处理最后一辆车
            distance += self.distances[line[-1], 0]
            distance_sum += distance
            line.append(0)
            lines.append(line)

            cars.append(lines)
            fit = self.C0 * len(lines) + self.C1 * distance_sum
            fit = round(fit, 1)
            fits.append(fit)
        self.cars = cars
        self.fits = fits

    # 顺序交叉
    def crossover(self):
        paths = self.paths.copy()
        for i in range(self.bird_num):
            child = [None] * len(paths[i])
            parent1 = paths[i]
            rand_num = random.uniform(0, sum([self.w, self.c1, self.c2]))
            # 粒子本身逆序
            if rand_num <= self.w:
                parent2 = list(reversed(parent1))
            # 局部最优解
            elif rand_num <= self.w + self.C1:
                parent2 = self.local_paths
            # 全局最优解
            else:
                parent2 = self.global_paths

            start_pos = random.randint(0, len(parent1) - 1)
            end_pos = random.randint(0, len(parent1) - 1)
            if start_pos > end_pos:
                start_pos, end_pos = end_pos, start_pos

            # parent1 -> child
            child[start_pos:end_pos + 1] = parent1[start_pos:end_pos +
                                                   1].copy()

            # parent2 -> child
            list1 = list(range(0, start_pos))
            list2 = list(range(end_pos + 1, len(parent2)))
            list_index = list1 + list2
            j = -1
            for index in list_index:
                while j < len(parent2) - 1:
                    j += 1
                    if parent2[j] not in child:
                        child[index] = parent2[j]
                        break
            self.paths[i] = child

    def init(self):
        # 计算城市间的距离
        self.get_distance()

        # 使用贪心算法构造初始解
        self.greedy_init()

        # 分配车辆，并计算适应度
        self.get_fitness()

        # 记录全局最优值和当前最优值
        self.global_cost = min(self.fits)
        self.local_cost = self.global_cost

        # 记录全局最优路径和当前最优路径
        min_index = self.fits.index(min(self.fits))
        self.global_paths = self.paths[min_index]
        self.local_paths = self.global_paths

        # 记录全局最优的路径
        self.global_lines = self.cars[min_index]

    # 更新粒子
    def evolve(self):
        for epoch in range(1, self.epochs + 1):
            self.crossover()
            self.get_fitness()

            # 更新当前最优值
            self.local_cost = min(self.fits)
            min_index = self.fits.index(min(self.fits))
            self.local_paths = self.paths[min_index]

            # 更新全局最优解
            if self.local_cost <= self.global_cost:
                self.global_cost = self.local_cost
                self.global_paths = self.local_paths
                self.global_lines = self.cars[min_index]

            self.history_cost.append(self.local_cost)

            print(f'epoch:{epoch},local_cost:{self.local_cost}')

    # 绘制路径
    def show_path(self):
        for line in self.global_lines:
            x, y = [], []
            for i in line:
                customer = self.customers[i]
                x.append(customer[0])
                y.append(customer[1])
            plt.plot(x, y, 'o-')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


if __name__ == '__main__':
    pso = PSO()
    pso.init()
    pso.evolve()
    print(f'global cost:{pso.global_cost}')
    print(f'global lines:{pso.global_lines}')
    pso.show_path()
