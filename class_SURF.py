#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 12:57:46 2019

@author: vaeahc
"""

import cv2
import numpy as np

class SURF:
    
    def __init__(self, surf_harr_size0, octaves, s, sum_, img):
        
        self.img = img
        self.surf_harr_size0 = surf_harr_size0
        self.octaves = octaves
        self.s = s
        self.sum = sum_
        self.surf_harr_size_inc = 6
        self.keypoint = {}
        self.keypoint_index = 0
        self.SURF_ORI_sigma = 1.2
        
        self.total_s= (self.s + 2) * self.octaves
        #hessian矩阵行列式
        det = [0 for x in range(self.total_s)]
        #hessian矩阵迹
        trace = [0 for x in range(self.total_s)]
        #每一层haar模版大小
        size = [0 for x in range(self.total_s)]
        #每一层的采样步长
        samplestep = [0 for x in range(self.total_s)]
        #中间层的缩影
        middleindic = [0 for x in range(self.octaves * self.s)]
        step = 1
        index = 0
        middleindex = 0
        
        for i in range(self.octaves):
            for j in range(self.s + 2):                
            
                det[index] = np.zeros(((self.sum.shape[0] - 1) / step, (self.sum.shape[1] - 1) / step))
                trace[index] = np.zeros(((self.sum.shape[0] - 1) / step, (self.sum.shape[1] - 1) / step))
                size[index] = self.surf_harr_size0 + self.surf_harr_size_inc * j
                samplestep[i] = step
            
                if j > 0 and j <= self.s + 1:
                    middleindic[middleindex] = index
                    middleindex += 1
                index += 1
            step *= 2
        
        self.det = np.array(det)
        self.trace = np.array(trace)
        self.size = np.array(size)
        self.samplestep = np.array(samplestep)
        self.middleindic = np.array(middleindic)
        
    def resizehaar(self, point, n, oldsize, newsize, widthstep):
        #转换比例
        #point为5 * n的array,每一列为np.array([x1, y1, x2, y2, w])
        ratio = newsize / oldsize
        #转换到积分图左上角点的拉直距离
        dst = np.array((5, n))
        X = np.round(ratio * point[0:4, :])
        temp = np.array([[1 ,widthstep, 0, 0], [0, widthstep, 1, 0], [1, 0, 0, widthstep], [0, 0, 1, widthstep]])
            
        dst[0:4, :] = np.dot(temp, X)
        for i in range(n):
                
            dst[4, n] = point[4, n] / (point[2][n] - point[0][n]) * (point[3][n] - point[1][n])
            
        return dst
    
    def calchaar(self,origin, n, dst):
        #dst为到积分图左上角的拉直距离
        #result为计算模版的值
        #返回盒子滤波后的值
        result = 0
        for i in range(n):
                
            result += (origin[dst[0][n]] + origin[dst[3][n]] - origin[dst[1][n]] - origin[dst[2][n]]) * dst[4][n]
                
        return result
        
    def calclayerDetAndTrace(self, harr_size, sample_step, det, trace):
        #把模版中的点坐标转换到积分图
        #harr_size为此时模版的尺寸, det为当前层的行列式矩阵
        #sample_step为此时采样步长, trace为当前层的迹矩阵
        self.Nx = 3
        self.Ny = 3
        self.Nxy = 4
        #三个方向盒子滤波模版 9*9
        self.dx = np.array([[0, 3, 6], [2, 2, 2], [3, 6, 9], [7, 7, 7], [1, -2, 1]])
        self.dy = np.array([[2, 2, 2], [0, 3, 6], [7, 7, 7], [3, 6, 9], [1, -2, 1]])
        self.dxy = np.array([[1, 5, 1, 5], [1, 1, 5, 5], [4, 8, 4, 8], [4, 4, 8, 8], [1, -1, -1, 1]])
        
        if harr_size > self.sum.shape[0] - 1 or harr_size > self.sum.shape[1] - 1:
            return 0
        
        Dx = self.resizehaar(self.dx, self.Nx, 9, harr_size, self.sum.shape[1])
        Dy = self.resizehaar(self.dy, self.Ny, 9, harr_size, self.sum.shape[1])
        Dxy = self.resizehaar(self.dxy, self.Nxy, 9, harr_size, self.sum.shape[1])

        #积分图比实际的尺寸大1
        sample_i = int(1 + (self.sum.shape[1] - 1 - harr_size) / sample_step)
        sample_j = int(1 + (self.sum.shape[0] - 1 - harr_size) / sample_step)
        
        margin = int((harr_size / 2) / sample_step)
        
        for i in range(sample_i):
            for j in range(sample_j):
                
                dx = self.calchaar(self.sum[i * self.sum.shape[1] + j:], self.Nx, Dx)
                dy = self.calchaar(self.sum[i * self.sum.shape[1] + j:], self.Ny, Dy)
                dxy = self.calchaar(self.sum[i * self.sum.shape[1] + j:], self.Nxy, Dxy)
                #实际考察点[i + margin, j + margin]处的响应
                det[i + margin][j + margin] = dx * dy - dxy ** 2 * 0.81
                trace[i+ margin][j + margin] = dx + dy
    
    #3D非极大值抑制           
    def findMaximaInlayer(self, octave, layer, hessianThreshold, sample_step):
        #octave为组序号, layer为层数, hessianThreshold为矩阵行列式阈值，检测低能量点（不稳定点）
        #插值得到亚像素   
        def interpolatepoint(area, dx, dy, ds, kpt_index):
            #area为以检测点为中心的26邻域
            #b为三个方向的梯度取反
            b = -np.array([(area[1][5] - area[1][3]) / 2, (area[1][7] - area[1][2]) / 2, (area[2][4] - area[0][4]) / 2])
            #A为hessian矩阵
            A = np.array([[area[1][3] + area[1][5] - 2 * area[1][4], 
                           (area[1][8] + area[1][0] - area[1][2] - area[1][6]) / 4, 
                           (area[2][5] + area[0][3] - area[2][3] - area[0][5]) / 4, 
                           ], 
                          [(area[1][8] + area[1][0] - area[1][2] - area[1][6]) / 4,
                           area[1][1] + area[1][7] - 2 * area[1][4], 
                           (area[2][7] + area[0][1] - area[0][7] - area[2][1]) / 4
                           ],
                          [(area[2][5] + area[0][3] - area[2][3] - area[0][5]) / 4, 
                           (area[2][7] + area[0][1] - area[0][7] - area[2][1]) / 4, 
                           area[0][4] + area[2][4] - area[1][4], 
                           ]])
            #求出偏移量， Ax = b
            x_hat = np.dot(np.linalg.inv(A), b)
            
            flag = x_hat[0] != 0 and x_hat[1] != 0 and x_hat[2] != 0 and abs(x_hat[0]) <= 1 and abs(x_hat[1]) <= 1 and abs(x_hat[2]) <= 1
            
            if flag:
                self.keypoint[kpt_index]['center_j'] += x_hat[0] * dx
                self.keypoint[kpt_index]['center_i'] += x_hat[1] * dy
                self.keypoint[kpt_index]['size'] += x_hat[2] * ds
                
        
        harr_size = self.size[layer]
        dm = np.array([0, 0, 9, 9, 1])
        layer_i = int((self.sum.shape[0] - 1) / sample_step)
        layer_j = int((self.sum.shape[1] - 1) / sample_step)
        #由于上下三层比较，取最大空白
        margin = int(self.size[layer + 1] / sample_step + 1)
        
        if self.sum:
            dst = self.resizehaar(dm, 1, 9, harr_size, self.sum.shape[1])
            
        for i in range(margin, layer_i - margin):
            for j in range(margin, layer_j - margin):

                center = self.det[layer][i][j]
                #除去不稳定点
                if center > hessianThreshold:
                    #计算模版在积分图上起始位置，筛选模版阈值
                    sum_i = int(sample_step * (i - harr_size / 2 / sample_step))
                    sum_j = int(sample_step * (j - harr_size / 2 / sample_step))
                    
                    N9 = np.array([[self.det[layer - 1][i - 1][j - 1], 
                                    self.det[layer - 1][i - 1][j], 
                                    self.det[layer - 1][i - 1][j + 1], 
                                    self.det[layer - 1][i][j - 1], 
                                    self.det[layer - 1][i][j], 
                                    self.det[layer - 1][i][j + 1], 
                                    self.det[layer - 1][i + 1][j - 1], 
                                    self.det[layer - 1][i + 1][j], 
                                    self.det[layer - 1][i + 1][j + 1]
                                    ], 
                                   [self.det[layer][i - 1][j - 1], 
                                    self.det[layer][i - 1][j], 
                                    self.det[layer][i - 1][j + 1], 
                                    self.det[layer][i][j - 1], 
                                    self.det[layer][i][j], 
                                    self.det[layer][i][j + 1], 
                                    self.det[layer][i + 1][j - 1], 
                                    self.det[layer][i + 1][j], 
                                    self.det[layer][i + 1][j + 1]
                                    ], 
                                   [self.det[layer + 1][i - 1][j - 1], 
                                    self.det[layer + 1][i - 1][j], 
                                    self.det[layer + 1][i - 1][j + 1], 
                                    self.det[layer + 1][i][j - 1], 
                                    self.det[layer + 1][i][j], 
                                    self.det[layer + 1][i][j + 1], 
                                    self.det[layer + 1][i + 1][j - 1], 
                                    self.det[layer + 1][i + 1][j], 
                                    self.det[layer + 1][i + 1][j + 1]
                                    ]
                                  ]                                    )
                    if self.sum:
                        res = self.calchaar(self.sum[i * self.sum.shape[1] + j], 1, dst)
                        if res < 0.5:
                            continue
                   
                    #极大值检测
                    if center >= np.max(N9):
                        #原始积分图上模版中心位置
                        center_i = sum_i + (harr_size - 1) * 0.5
                        center_j = sum_j + (harr_size - 1) * 0.5
                        
                        self.keypoint[self.keypoint_index] = {'center_i': center_i, 
                                                              'center_j': center_j, 
                                                              'size': harr_size, 
                                                              'det_value': center, 
                                                              'octaves': octave, 
                                                              }
                        self.keypoint_index += 1
                        
                        ds = harr_size - self.size[layer - 1]
                        #调用插值函数
                        interpolatepoint(N9, sample_step, sample_step, ds, self.keypoint_index - 1)
                        
    def get_descr(self):
        #圆形区域的半径， s = 1
        Ori_radius = 6
        Patch_size = 20
        
        #圆域内坐标点
        apt = []
        #圆域内高斯权重
        aptw = []
        DW = []
        #圆内点数
        nOrisamples = 0
        #圆域内高斯加权系数
        for i in range(-Ori_radius, Ori_radius + 1):
            for j in range(-Ori_radius, Ori_radius + 1):
                
                apt[nOrisamples] = (i, j)
                sigma = 2 * self.SURF_ORI_sigma
                aptw[nOrisamples] = np.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2))
                nOrisamples += 1
        #正方形内高斯加权系数
        #正方形内点数
        nOri_square = 0
        for i in range(Patch_size):
            for j in range(Patch_size):
                
                sigma = 3.3 * self.SURF_ORI_sigma
                DW[nOri_square] = np.exp(-((i - 10) ** 2 + (j - 10) ** 2) / (2 * sigma ** 2))
                nOri_square += 1
                
        #harr小波模版
        Nx = 2
        Ny = 2
        #水平分享模版
        dx_s = np.array([[0, 2], [0, 0], [2, 4], [4, 4], [-1, 1]])
        #竖直方向模版
        dy_s = np.array([[0, 0], [0, 2], [4, 4], [2, 4], [1, -1]])
        nangle = 0
        X = []
        Y = []
        ag = []
        #为每一个关键点建立描述子
        for kpy in self.keypoint:
            #计算关键点的尺度
            s = kpy['size'] / 9 * self.SURF_ORI_sigma
            #harr_size
            harr_grad_width = 2 * np.round(s)
            
            if self.sum.shape[0] < harr_grad_width or self.sum.shape[1] < harr_grad_width:
                #越界处理
                kpy['size'] = -1
                continue
            
            dst_x = self.resizehaar(dx_s, Nx, 4, harr_grad_width, self.sum.shape[1])
            dst_y = self.resizehaar(dy_s, Ny, 4, harr_grad_width, self.sum.shape[1])
            #计算圆域内点的小波响应
            for i in range(nOrisamples):
                #求关键点左上角和右下角模版处点
                x = np.round(kpy['center_j'] + apt[i][0] * s - (harr_grad_width - 1) / 2)
                y = np.round(kpy['center_i'] + apt[i][1] * s - (harr_grad_width - 1) / 2)
                
                if x < 0 or x >= self.sum.shape[1] or y < 0 or y >= self.sum.shape[0]:
                    continue
                
            
                vx = self.calchaar(self.sum[x * self.sum.shape[1] + y: ], 2, dst_x)
                vy = self.calchaar(self.sum[x * self.sum.shape[1] + y: ], 2, dst_y)
                #不同方向的梯度
                X[nangle] = vx * aptw[i]
                Y[nangle] = vy * aptw[i]
                ag[nangle] = np.arctan(y / x)
                nangle += 1
            #越界处理  
            if nangle == 0:
                kpy['size'] = -1
                continue
            
            #60度滑动窗口
            angle = 0
            max_mod = 0
            while angle <= 360:
                #记录扇形区域内点的harr小波响应和
                sumx = 0
                sumy = 0
                
                for j in range(nangle):
                    #点在扇形区域内
                    if ag[j] + np.pi > angle * np.pi / 180 and ag[j] + np.pi <= (angle + 60) * np.pi / 180:
                        sumx += X[j]
                        sumy += Y[j]
                        
                temp = sumx ** 2 + sumy ** 2
                #寻找最大响应方向
                if temp > max_mod:
                    max_mod = temp
                    bestx = sumx
                    besty = sumy
                    
                angle += 36 / np.pi
                
            desc_dir = np.arctan(besty / bestx) + np.pi    
            #使用20s * 20s构建描述子
            win_size = 20 * s
            win_offset = -(win_size - 1) / 2
            #旋转矩阵
            rot_mat = np.array([[np.cos(desc_dir), -np.sin(desc_dir)], 
                                [np.sin(desc_dir), np.cos(desc_dir)]])

            #win中存储旋转后像素值
            win = np.zeros((win_size, win_size))
            #求旋转后坐标
            for i in range(win_offset, -win_offset + 1):
                for j in range(win_offset, -win_offset + 1):
                    
                    X_Y = np.array([kpy['center_j'], kpy['center_i']]) + np.dot(rot_mat, np.array([j, i]))
                    #边界处理
                    x = min(max(X_Y[0], 0), self.sum.shape[1] - 1)
                    y = min(max(X_Y[1], 0), self.sum.shape[0] - 1)
                    win[i][j] = self.img[y][x]
            #将尺寸缩小为20 * 20        
            win = cv2.resize(win, (np.shape[0] / s, np.shape[1] / s))
            #用2s *2s模版计算harr响应
            DX = np.zeros((20, 20))
            DY = np.zeros((20, 20))
            for i in range(20):
                for j in range(20):
                    
                    dw = DW[i * 20 + j]
                    vx = (win[i][j + 1] + win[i + 1][j + 1] - win[i][j] - win[i + 1][j]) * dw
                    vy = (win[i + 1][j] + win[i + 1][j + 1] - win[i][j] - win[i][j + 1]) * dw
                    DX[i][j] = vx
                    DY[i][j] = vy
                    
            #建立64维描述子
            vec = np.zeros((64, ))
            for i in range(4):
                for j in range(4):                  
                    for y in range(5 * i, 5 * i + 5):
                        for x in range(5 * j, 5 * j + 5):
                            
                            tx = DX[y][x]
                            ty = DY[y][x]
                            
                            vec[i * 4 + j * 4] += tx
                            vec[i * 4 + j * 4 + 1] += abs(tx)
                            vec[i * 4 + j * 4 + 2] += ty
                            vec[i * 4 + j * 4 + 3] += abs(ty)
            #归一化后的特征描述子            
            vec /= np.linalg.norm(vec)
            
            
                    
                    
            
            
                    
                   
                
            
                    
            
            
                    
                    
                    
                
                
                
                
                
                
                
            
            
            
        
                
        
                
            
                        
                        
                        
            
        
                
            
        
        
            
        
                
                
        
        
        
        
        
        
                
            
            
            

                
            
            
        

            
            
            
        
        
    
