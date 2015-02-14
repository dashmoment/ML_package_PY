import numpy as np
import matplotlib.pyplot as plt
import randomiser as rnd
from scipy import misc as sc
from cvxopt import matrix, solvers
import sort

class find_pla:
    def __init__(self, wt , x_pt , y_pt):
        self.wt = wt
        self.x_pt = x_pt
        self.y_pt = y_pt
        print "Initialized\n"

    def update(self):
        j = 1
        print self.wt
        #while j != len (x_pt):
        for i in range(len (self.x_pt)):

            #sep_line = [-1/self.wt[0],1]
            #result = np.dot(sep_line,self.x_pt[i])
        
            result = np.dot(self.wt,self.x_pt[i])
            print result, self.wt, self.x_pt[i]


            sign = 1 if result >= -0.1 else -1
        
            if sign != self.y_pt[i]:
                wt1 = self.wt + self.y_pt[i]*self.x_pt[i]
                self.wt = wt1
                #print "Revised\n", i,self.wt                
                break
            elif j == len (self.x_pt):
                print "All Correct"
                break
            else:
                print "Correct"
                j+=1
                print j
                

    def plot(self):
        fig1 = plt.figure(1)
        title = plt.title("PLA Result")
        plot = plt.scatter(self.x_pt[:,0],self.x_pt[:,1],c=self.y_pt)

        xx = np.linspace(0,10)
        yy = -self.wt[0]/self.wt[1]*xx
        line = plt.plot(xx,yy,color="red")
        plt.show();

class linear_reg:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def build(self):
        xx = [self.X*np.ones(len(self.X)), np.ones(len(self.X))]
        yy = self.Y*np.ones(len(self.Y))
        x_pinv = np.linalg.pinv(xx)

        ww= np.zeros(len(xx)) 
    
        for i in range(len(x_pinv)):
            for j in range(len(ww)):
                ww[j] += x_pinv[i][j]*yy[i]
            
        res = [ww[0],ww[1]]
        
        return res
          

class qua_2dreg:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def build(self):

        xx = [self.X*np.ones(len(self.X)), np.ones(len(self.X))]
        yy = self.Y*np.ones(len(self.Y))

        X12 = [xx[0][i] for i in range(len(xx[0]))]
        X1sqr = [xx[0][i]*xx[0][i] for i in range(len(xx[0]))]
        X2sqr = xx[1]
        X_trans = [ X2sqr, X12, X1sqr]

        x_pinv = np.linalg.pinv(X_trans)  
         
        ww = np.zeros(len(X_trans))
        
        for i in range(len(x_pinv)):
            for j in range(len(X_trans)):
               ww[j] += x_pinv[i][j]*yy[i]

        return ww 

class qua_ndreg:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def build(self):
        x_trans = np.zeros((len(self.X)+sc.comb(len(self.X),2))*len(self.X[0]))
        x_trans.shape = (len(self.X)+sc.comb(len(self.X),2)), len(self.X[0])
        

        for i in range(len(self.X)):
            x_trans[i] = [aa*bb for aa,bb in zip(self.X[i],self.X[i])]


        j = len(self.X)

        h = 0
        n = 1
        for j in range(len(x_trans)-len(self.X)):
            

            if h+n < len(self.X):
                #print h,h+n,'\n'
                for k in range(len(self.X[0])):
                     x_trans[j+len(self.X)][k] = self.X[h][k]*self.X[h+n][k]
                n+=1   
            else:
                h+=1
                n = 1
                #print h,h+n,'\n'
                for k in range(len(self.X[0])):
                     x_trans[j+len(self.X)][k] = self.X[h][k]*self.X[h+n][k]
                n+=1
        #print x_trans

        x_pinv = np.linalg.pinv(x_trans)
        ww = np.zeros(len(x_trans))

        for i in range(len(x_pinv)):
            for j in range(len(x_trans)):
               ww[j] += x_pinv[i][j]*self.Y[i]
        #print ww

        return ww, x_trans
       
       
        
class hardm_SVM:
    def __init__(self, yin, xin, kernal = 2):
        if len(yin) > 0 and len(yin) == len(xin[0]) :
            self.y = yin
            self.x = xin
            self.kernal = 2
        else:
            raise ValueError('len(yin) = len(xin[0]) && len > 0')
        
    def build(self):
        datasize = len(self.x[0])
        dim = len(self.x)

        q = np.zeros((datasize,datasize))

        if self.kernal == 2:
            for i in range(datasize):
                for j in range(datasize):
                    sum_k = sum((self.x[d][i]*self.x[d][j] for d in range(dim)))
                    q[i][j] = self.y[i]*self.y[j]*(1+sum_k + pow(sum_k,2))

                    
            #k.shape = (datasize,datasize)

            P = matrix(q)
            #Q = matrix(-1,(datasize,1))
            q = np.ones((datasize,1))
            Q = matrix(q)
            #a = np.transpose(self.y)
            A = matrix(self.y)
            A = A.trans()
            #B = matrix(0,(datasize,1),'d')
            B = matrix(0.)
            G = matrix(0.0, (datasize,datasize))
            G[::datasize+1] = -1.0
            #G = matrix(1,(1,datasize))
            h = np.zeros((datasize,1))
            H = matrix(h)
            #H = matrix(0,(datasize,1))
            #print B 

            sol = solvers.qp(P, Q, G, H, A, B)
            returns = [x for x in sol]
            print  sol['x']



class decision_stump:
   
    def thresh(self,sort_dat,axis):
        thresh = []
        for i in range(len(sort_dat) - 1):
            thresh.append((sort_dat[i+1][axis]+sort_dat[i][axis])/2)
        return thresh
            

    def b_thresh(self, b_data):
        
        dat = b_data
        if len(dat[0]) == 0:
            raise ValueError('dimension of data should not be zero')
        
        else:
            dim = len(b_data[0])-1
            d_num = len(dat)
            sort_res = [[]]
            sortmp = []
            thresh_a = [[]]

            
            for d in range(dim): #sort each axis in data
                res = sort.sort(dat)
                sortmp = res.quicksort(0,d_num -1,d)
                sort_res[len(sort_res) - 1] = sortmp
                sort_res.append([])
               
            sort_res.pop()

            for i in range(len(sort_res)): #derive thesh of all axis
                thresh_a[len(thresh_a)-1] = self.thresh(sort_res[i], i)
                thresh_a.append([])
            thresh_a.pop()
            
            return thresh_a   #thesh_a[d][n] d: axis, n: number of threshold

    def tree_gini(self, tdata, k): #B_tree branch rule by gini index
        ncls = len(k)
        num = [len(tdata[0]),len(tdata[1])]       
        gini = [0,0]

        
        for j in range(len(gini)):
            for n in range(ncls):
                tmp = 0
                for i in range(num[j]):
                    if tdata[j][i]*k[n] > 0:
                        tmp += 1
                    else:
                        tmp = tmp
                gini[j] += pow(tmp,2)

                
                
            if num[j] != 0:
                gini[j] = 1. - (float(gini[j])/pow(num[j],2))
            else:
                gini[j] = 0
            
        branch = sum(num[m]*gini[m] for m in range(len(num)))

       

        return branch

    def tds_branch(self, td_data):
        tds_thresh = self.b_thresh(td_data)

        if len(td_data) == 0:
            raise ValueError('data size should be larger than zero')
        else:
            num = len(td_data)
            dim = len(td_data[0]) - 1
            ds_s = [+1. , -1.]
            branch = num
            result = []
            s = 0
            axis = 0
            thresh = 0

            #print len(tds_thresh),len(tds_thresh[0])

            for i in range(len(ds_s)):
                for d in range(dim):
                    for k in range(len(tds_thresh[d])):
                        b_data = [[],[]]
                        b_label = [[],[]]
                        for n in range(num):
                            tmp = ds_s[i]*(td_data[n][d] - tds_thresh[d][k])

                            if tmp > 0:
                                b_data[0].append(td_data[n])                                            
                                b_label[0].append(td_data[n][dim])
                                
                            if tmp <= 0:
                                b_data[1].append(td_data[n])                                            
                                b_label[1].append(td_data[n][dim])
                        
                        
                        gini = self.tree_gini(b_label, ds_s)

                        if gini < branch:
                            s = ds_s[i]
                            axis = d
                            thresh = tds_thresh[d][k]
                            branch = gini
                            result = b_data
                            gidx = 0
                            
                        elif gini == 0:
                            s = ds_s[i]
                            axis = d
                            thresh = tds_thresh[d][k]
                            result = b_data
                            gidx = -1
                            break

              
            #print result,s,axis,thresh
            return result,s,axis,thresh,gidx #result[0] for left branch, and result[1] for right
                
                
                
            
class error_in:
    def __init__(self, y_real, wml, xin):

        if len(y_real) > 0 and len(wml)>0 and len(xin)>0:
            self.y_real = y_real
            self.wml = wml
            self.xin = xin
        else:
            raise ValueError ('Size of input should larger than 0')


    def build(self):

        yml = []
        for i in range(len(self.xin[0])):
            yml.append(sum(self.wml[j]*self.xin[j][i] for j in range(len(self.wml))))

        err = sum(pow(aa-bb,2) for aa,bb in zip(yml,self.y_real))
        print 'E_in = ',err
        return yml


##traindat  = np.loadtxt('test.dat')
##res = decision_stump()
#res.b_thresh()

##res.tds_branch(traindat)



##al = 3
##bl = 100
##line = rnd.rnd_line(al,bl)
##xpt,yrnd = line.build(50)
##
##reg = linear_reg(xpt,yrnd)
##ww = reg.build()
##
##yy = [al*xp+bl for xp in xpt]
##y_pre = [ww[0]*xp+ww[1] for xp in xpt]
##
##print 'Desired result [w0,w1] = [',al,',',bl,']'
##print 'Linear regression result = [',ww[0],',',ww[1],']'
##
##fig = plt.figure()
##title = plt.title('Linear regression')
##plot = plt.scatter(xpt,yrnd)
##goal = plt.plot(xpt,yy,'r',label = 'Goal')
##lr = plt.plot(xpt,y_pre,'g--',label = 'LR')
##plt.show()

##xx = [[2,3,4],[1,1,1]]
##yy = [9,25,49]
##qua = qua_2dreg(xx,yy)
##qua.build()
##
##al = 3
##bl = 200
##line = rnd.rnd_line(al,bl)
##xpt,yrnd = line.build_qua(rnd_range = 5)
###print xpt
##reg_2d = qua_2dreg(xpt,yrnd)
##w_sqr = reg_2d.build()
##
##yy = [al*xp+bl for xp in xpt]
##y_pre = [w_sqr[2]*xp*xp+ w_sqr[1]*xp + w_sqr[0] for xp in range(100)]
##
##
##
##xx = [np.ones(len(xpt)), xpt*np.ones(len(xpt))]
##reg_nd = qua_ndreg(xx, yrnd)
##ww_2d, x_2d = reg_nd.build()
##err = error_in(yrnd, ww_2d, x_2d)
##yml = err.build()
##
##
##print 'Desired result [w0,w1] = [',al,',',bl,']'
##print 'Nonlinear regression = ',w_sqr
##print 'nd Nonlinear regression = ',ww_2d
##fig = plt.figure()
##title = plt.title('squre_nonLinear regression')
##plot = plt.scatter(xpt,yrnd)
##goal = plt.plot(xpt,yy,'r',label = 'Goal')
##lr = plt.plot(range(100),y_pre,'g--',label = 'LR')
##lr2 = plt.scatter(xpt,yml,36,'r')
##plt.show()

##ww_real = [4,3,2,1]
##x_nd = [[1,1,1,1,1],[2,3,4,1,2],[4,1,2,4,3],[3,2,1,2,1]]
##y_nd = np.zeros(len(x_nd[0]))
##
##for i in range(len(x_nd[0])):
##    for j in range(len(x_nd)):
##        y_nd[i]+= x_nd[j][i]*ww_real[j]
##print y_nd
##
##reg_nd = qua_ndreg(x_nd, y_nd)
##ww_nd, x_trans = reg_nd.build()
###print len(ww_nd), x_trans
##
##
##y_ml = []
##
##for j in range(len(x_trans[0])):
##    y_ml.append(sum(ww_nd[i]*x_trans[i][j] for i in range(len(ww_nd))))
##
##print y_ml,'\n'
##
##err = error_in(y_nd, ww_real,x_nd)
##err.build()




