import numpy as np
traindat  = np.loadtxt('hw2_adaboost_train.dat')
testdat = np.loadtxt('hw2_adaboost_test.dat')


a = [7,8,11,6,17,2,3,18,3,10,16,1,8]

def swap(i,j,a):

    temp = np.copy(a[i])
    a[i] = a[j]
    a[j] = temp


def quicksort(a,left,right,axis):
    pivot = a[left][axis]
    i = left + 1
    j = right

    if left < right:
        while i < j:
            
            while a[i][axis] < pivot and i+1 < len(a):
                i = i+1
            while a[j][axis] > pivot and j-1 >=0:   
                j = j-1 
                
            if i < j:
                swap(i,j,a)
                i = i+1
                j = j-1
        if(a[left][axis] > a[j][axis]):
            swap(left,j,a)
        
        quicksort(a,left,j-1,axis)
        
        if j+1 < len(a):
            quicksort(a,j+1,right,axis)
   

    return np.copy(a)
    

quicksort(traindat,0,len(traindat)-1,0)
train_axis0 = np.copy(traindat)
quicksort(traindat,0,len(traindat)-1,1)
train_axis1 = np.copy(traindat)

print train_axis1,train_axis0
np.savetxt('adB_train0.dat', train_axis0, delimiter=" ", fmt="%s")
np.savetxt('adB_train1.dat', train_axis1, delimiter=" ", fmt="%s")
        
