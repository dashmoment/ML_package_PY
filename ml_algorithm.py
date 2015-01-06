import numpy as np
import matplotlib.pyplot as plt
import randomiser as rnd

class linear_reg:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def build(self):
        xx = [self.X*np.ones(len(self.X)),np.ones(len(self.X))]
        yy = self.Y*np.ones(len(self.Y))
        x_pinv = np.linalg.pinv(xx)

        w0 = 0
        w1 = 0
        for i in range(len(x_pinv)):
            w0 += x_pinv[i][0]*yy[i]
            w1 += x_pinv[i][1]*yy[i]
        res = [w0,w1]
        
        return res


class logistic_reg:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def build(self):
        
        

        

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




