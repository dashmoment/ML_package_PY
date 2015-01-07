import numpy as np
import matplotlib.pyplot as plt
import randomiser as rnd

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

class qua_classifier:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def build(self):
        X12 = [self.X[0][i]*self.X[1][i] for i in range(len(self.X[0]))]
        X1sqr = [self.X[0][i]*self.X[0][i] for i in range(len(self.X[0]))]
        X2sqr =[self.X[1][i]*self.X[1][i] for i in range(len(self.X[0]))]
        X_trans = [self.X[0], self.X[1], X12, X1sqr, X2sqr]

        yy = self.Y*np.ones(len(self.Y))
        x_pinv = np.linalg.pinv(X_trans)

        print len(X_trans), len(yy)
         
        ww = np.zeros(len(X_trans))
        
        for i in range(len(x_pinv)):
            for j in range(len(X_trans)):
               ww[j] += x_pinv[i][j]*yy[i]
        print ww
          

       
        
class logistic_reg:
    def __init__(self, X, Y, step):
        self.X = X
        self.Y = Y
        self.step = step
        
    
        
        

al = 3
bl = 100
line = rnd.rnd_line(al,bl)
xpt,yrnd = line.build(50)

reg = linear_reg(xpt,yrnd)
ww = reg.build()

yy = [al*xp+bl for xp in xpt]
y_pre = [ww[0]*xp+ww[1] for xp in xpt]

print 'Desired result [w0,w1] = [',al,',',bl,']'
print 'Linear regression result = [',ww[0],',',ww[1],']'

fig = plt.figure()
title = plt.title('Linear regression')
plot = plt.scatter(xpt,yrnd)
goal = plt.plot(xpt,yy,'r',label = 'Goal')
lr = plt.plot(xpt,y_pre,'g--',label = 'LR')
plt.show()

xx = [[1,2,3],[2,3,4]]
yy = [1,1,-1]
qua = qua_classifier(xx,yy)
qua.build()


