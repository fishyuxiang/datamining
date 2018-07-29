# coding:UTF-8
'''
Date:20180720
@author: yuxiang
'''

import numpy as np
from random import random
from Kmeans import load_data, kmeans, distance, save_result,showCluser,randCent,kmedoids


FLOAT_MAX = 1e100 # 设置一个较大的值作为初始化的最小的距离

def nearest(point, cluster_centers):
    min_dist = FLOAT_MAX
    m = np.shape(cluster_centers)[0]  # 当前已经初始化的聚类中心的个数
    for i in range(m):
        # 计算point与每个聚类中心之间的距离
        d = distance(point, cluster_centers[i, ])
        # 选择最短距离
        if min_dist > d:
            min_dist = d
    return min_dist

def get_centroids(points, k):#Kmeans++,轮盘法
    m, n = np.shape(points)
    cluster_centers = np.mat(np.zeros((k , n)))
    # 1、随机选择一个样本点为第一个聚类中心
    index = np.random.randint(0, m)
    cluster_centers[0, ] = np.copy(points[index, ])
 #   print("0------",cluster_centers[0, ],index)
    # 2、初始化一个距离的序列
    d = [0.0 for _ in range(m)]
    for i in range(1, k):
        sum_all = 0
        for j in range(m):
            # 3、对每一个样本找到最近的聚类中心点
            d[j] = nearest(points[j, ], cluster_centers[0:i, ])
            # 4、将所有的最短距离相加
            sum_all += d[j]
        # 5、取得sum_all之间的随机值
        sum_all *= random()
        # 6、获得距离较远的样本点（概率较大，也可能取得与已知聚类中心近距离的点）作为聚类中心点
        for j, di in enumerate(d):
            #print(di)
            sum_all -= di
            if sum_all > 0:
                continue
            cluster_centers[i] = np.copy(points[j, ])
            #print(i,"-------",sum_all,"-------",cluster_centers[i],j,"\n")
            break

    return cluster_centers

def get_centroids_b(points, k):#kmeans++改进
    m, n = np.shape(points)
    cluster_centers = np.mat(np.zeros((k , n)))
    # 1、随机选择一个样本点为第一个聚类中心
    index = np.random.randint(0, m)
    cluster_centers[0, ] = np.copy(points[index, ])
    #print("0------",cluster_centers[0, ],index)
    # 2、初始化一个距离的序列
    d = [0.0 for _ in range(m)]
    for i in range(1, k):
        for j in range(m):
            # 3、对每一个样本找到最近的聚类中心点
            d[j] = nearest(points[j, ], cluster_centers[0:i, ])
        maxdi=d[0]
        ind=0
        # 4、获得距离最远的样本点作为聚类中心点
        for j, di in enumerate(d):
            if di>maxdi:
                maxdi=di
                ind=j
            cluster_centers[i] = np.copy(points[ind, ])

    return cluster_centers

if __name__ == "__main__":
    k = 4#聚类中心的个数
    file_path = "kmeans.txt"
    # 1、导入数据
    print ("---------- 1.load data ------------")
    data = load_data(file_path)
    # 2、KMeans++的聚类中心初始化方法
    print ("---------- 2.K-Means++ generate centers ------------")
    centroids = get_centroids_b(data, k)
    # 3、聚类计算
    print ("---------- 3.kmeans ------------")
    subCenter = kmedoids(data, k, centroids)
    # 4、保存所属的类别文件
    print ("---------- 4.save subCenter ------------")
    save_result("sub_pp", subCenter)
    # 5、保存聚类中心
    print ("---------- 5.save centroids ------------")
    save_result("center_pp", centroids)
    # 6、可视化
    print("---------- 6.showCluser ------------")
    showCluser(data, k, centroids, subCenter)
