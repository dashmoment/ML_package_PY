import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import random

class gaussion_2D:
#2D Gaussian f(x,mean,var) = 1/(var*sqrt(2*pi))*exp(-(x-mean)^2/2*var^2) 
    def __init__(self, mean, var = 5, l_bound = -100, h_bound = 100):
        self.mean = mean
        self.var = var
        self.l_bound = l_bound
        self.h_bound = h_bound
        print "Initial 2D_Gaussian..."

    def g2d_build(self):
        if self.h_bound <= self.l_bound:
            raise ValueError('h_bound should larger than l_bound')

        size = self.h_bound - self.l_bound + 1
        x_size = range(size)
        x_pt = x_size + np.ones(size)*self.l_bound

        g_num = 1./(self.var*np.sqrt(2*np.pi))   
        array_mean = np.ones(size)*self.mean


        A = pow(x_pt - array_mean, 2)
        B = (-1./(2*pow(self.var, 2)))*np.ones(size)
        C = [aa*bb for aa,bb in zip(A,B)]
   
        res = g_num*np.exp(C)

        return res

    def g2d_plot(self, g_func, plotname):

        size = self.h_bound - self.l_bound + 1
        x_size = range(size)
        x_pt = x_size + np.ones(size)*self.l_bound

        #print len(x_pt), len(g_func)

        
        fig1 = plt.figure()
        title = plt.title(plotname)
        plot = plt.plot(x_pt, g_func)
        plt.show()
        

    def g2d_writefile(self, g_func):
        ##        'r' （默认值）表示从文件读取数据。
        ##        'w' 表示要向文件写入数据，并截断以前的内容。
        ##        'a' 表示要向文件写入数据，但是添加到当前内容尾部。
        ##        'r+' 表示对文件进行读写操作（删除以前的所有数据）。
        ##        'r+a' 表示对文件进行读写操作（添加到当前内容尾部）。
        ##        'b' 表示要读写二进制数据。
        
    # -*- coding: utf-8 -*-
        filename = input('Write filename:')
        f = open(filename, 'w')
        
        for i in g_func:
            f.write("%s\n" %i)
            print i
            
        f.close()

    def g2d_readfile(self):
        res = []
        
        filename = input('Read filename:')
        f = open(filename, 'r')
        #f = open('test.txt', 'r')

        while(1):
            line = f.readline();
            if len(line) == 0:
                break;
            res += [line.strip()]         
        f.close()
        
        res = map(float, res)
        return res
    
    def g2dpoint(self, xx):
        g_num = 1./(self.var*np.sqrt(2*np.pi))   

        A = pow(xx - self.mean, 2)
        B = -1./(2*pow(self.var, 2))

        result = g_num*np.exp(A*B)
        print result


class gaussion_3D:
    size = 0
    r_size = 0
    xy_pt = 0
    
    def __init__(self, mean_x, mean_y, var_x, var_y, co_r,l_bound, h_bound):
        
        
        self.mean_x = mean_x
        self.mean_y = mean_y
        self.var_x = var_x     #standard devieation
        self.var_y = var_y
        self.co_r = co_r      #corelation coefficient(<1)

        if(co_r > 1):
            raise ValueError('corelation coefficient should less than 1') 
        
        self.l_bound = l_bound  
        self.h_bound = h_bound

        print "Initial 3D Gaussian..."
        
    def g3d_build(self):
        gaussion_3D.size = self.h_bound - self.l_bound + 1
        gaussion_3D.r_size = range(gaussion_3D.size)
        gaussion_3D.xy_pt = gaussion_3D.r_size + np.ones(gaussion_3D.size)*self.l_bound


        pt = [[i,j] for i in gaussion_3D.xy_pt for j in gaussion_3D.xy_pt]

        #x_mean = np.ones(size*size)*self.mean_x
        #y_mean = np.ones(size*size)*self.mean_y

        minus_cr = 1-pow(self.co_r,2)
        g_num = 1./(2*np.pi*self.var_x*self.var_y*np.sqrt(minus_cr))

        res = []
        for i in range(len(pt)):
            a_x = pt[i][0] - self.mean_x 
            a_y = pt[i][1] - self.mean_y
            A_x = pow(a_x, 2)
            A_y = pow(a_y, 2)
            AA = -(2*self.co_r*a_x*a_y)/(self.var_x*self.var_y)
            B_x = (1./pow(self.var_x, 2))
            B_y = (1./pow(self.var_y, 2))
            C = -(1./(2*minus_cr))

            D = C*(A_x*B_x+A_y*B_y+AA)

            res.append(g_num*np.exp(D))

        ##Sum of result which should be 1##
##        sum_g = 0
##
##        for k in range(len(res)):
##            sum_g += res[k]
##     
##        print sum_g
        return res

    def g3d_plot(self, g_func, plotname):
        
        pt = [[i,j] for i in gaussion_3D.xy_pt for j in gaussion_3D.xy_pt]
        
        X =[]
        Y =[]
        for i in range(len(pt)):
            X.append(pt[i][0])
            Y.append(pt[i][1])
        #print len(X)


        fig = plt.figure()
        title = plt.title(plotname)
        ax = fig.add_subplot(111, projection='3d')
        plot = ax.scatter(X,Y, g_func)
        plt.show()

class rnd_line:
    def __init__(self, al, bl, l_bound = 0, h_bound = 100, p_num = 100):
        self.al = al
        self.bl = bl
        self.l_bound = l_bound
        self.h_bound = h_bound
        self.p_num = p_num

        if(l_bound >= h_bound):
            raise ValueError('l_bound should be larger than h_bound')
        if(p_num <= 0):
            raise ValueError('number of point > 0')

    def build(self, rnd_range = 2):

        xx = []
        yy = []
        yrnd = []


        for i in range(self.p_num):
            xx.append(random.randint(self.l_bound, self.h_bound))

        yy = [self.al*xp+self.bl for xp in xx]

        #print xx
        #print '\n'
        #print yy

        [yrnd.append(random.randint(yp-rnd_range, yp+rnd_range)) for yp in yy] 

        
        
        return xx,yrnd

    def build_qua(self, rnd_range = 20):

        xx = []
        yy = []
        yrnd = []


        for i in range(self.p_num):
            xx.append(random.randint(self.l_bound, self.h_bound))

        yy = [self.al*xp+self.bl for xp in xx]

        #print xx
        #print '\n'
        #print yy

        [yrnd.append(random.randint(yp-rnd_range, yp+rnd_range)) for yp in yy] 

        
        
        return xx,yrnd
        
        
        

        
#gausssian = gaussion_2D(0,10,-100,100)
#res = gausssian.g2d_build()
#gausssian.g2d_plot(res, "Gaussian distribution")
#gausssian.g2d_writefile(res)
#result = gausssian.g2d_readfile()
#print result

##g_3d = gaussion_3D(0,0,20,20,0.5,-50,50)
##res = g_3d.g3d_build()
##g_3d.g3d_plot(res,'g3d')

##line = rnd_line(1,0,5)
##line.build(20)



