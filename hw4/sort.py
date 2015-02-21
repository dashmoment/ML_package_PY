import numpy as np

class sort:
    def __init__(self, data):
        self.data = data

    def swap(self,i,j):

        temp = np.copy(self.data[i])
        self.data[i] = self.data[j]
        self.data[j] = temp


    def quicksort(self, left, right,axis):
        
        pivot = self.data[left][axis]
        i = left + 1
        j = right

        if left < right:
            while i < j:
                #print self.data
                while self.data[i][axis] < pivot and i+1 < len(self.data):
                    i = i+1
                while self.data[j][axis] > pivot and j-1 >=0:   
                    j = j-1 
                    
                if i < j:
                    self.swap(i,j)
                    i = i+1
                    j = j-1

            if(self.data[left][axis] > self.data[j][axis]):
                self.swap(left,j)
            
            self.quicksort(left,j-1,axis)
            
            if j+1 < len(self.data):
                self.quicksort(j+1,right,axis)
       

        return np.copy(self.data)

##traindat  = np.loadtxt('hw2_adaboost_train.dat')
##testdat = np.loadtxt('hw2_adaboost_test.dat')
##sortdata = sort(traindat)
##dat = sortdata.quicksort(0,len(traindat)-1,0)
##
##print dat


        
