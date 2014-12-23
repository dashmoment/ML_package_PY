import numpy as np
import matplotlib.pyplot as plt

class gaussion_2D:
#2D Gaussian f(x,mean,var) = 1/(var*sqrt(2*pi))*exp(-(x-mean)^2/2*var^2) 
    def __init__(self, mean, var, l_bound, h_bound):
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


class gaussion_3D:
    def __init__(self, mean_x, mean_y, var_x, var_y, l_bound, h_bound):
        self.mean_x = mean_x
        self.mean_y = mean_y
        self.var_x = var_x
        self.var_y = var_y
        self.l_bound = l_bound
        self.h_bound = h_bound

        print "Initial 3D Gaussian..."
        
    def g3d_build(self):
        size = self.h_bound - self.l_bound + 1
        r_size = range(size)
        xy_pt = r_size + np.ones(size)*self.l_bound

        x_mean = np.ones(size)*self.mean_x
        y_mean = np.ones(size)*self.mean_y

        A_x = pow(xy_pt - x_mean, 2)
        A_y = pow(xy_pt - y_mean, 2)

        B_x = -(0.5/pow(self.var_x, 2))*np.ones(size)
        B_y = -(0.5/pow(self.var_y, 2))*np.ones(size)

        C = [ax*bx+ay*by for ax,bx,ay,by in zip(A_x,B_x,A_y,B_y) ]

        g_num = 1./(2*np.pi*self.var_x*self.var_y)
        res = g_num*np.exp(C)

        print res


#gausssian = gaussion_2D(0,10,-100,100)
#res = gausssian.g2d_build()
#gausssian.g2d_plot(res, "Gaussian distribution")
#gausssian.g2d_writefile(res)
#result = gausssian.g2d_readfile()
#print result


g_3d = gaussion_3D(0,0,10,10,-50,50)
g_3d.g3d_build()



