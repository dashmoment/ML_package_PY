import numpy as np
import matplotlib.pyplot as plt


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



                
                
               
        
