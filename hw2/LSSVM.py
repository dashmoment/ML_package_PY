import numpy as np
from numpy.linalg import inv


def LSSVM(traindat = 'hw2_lssvm_all.dat',validdata = []):
    data = np.loadtxt('hw2_lssvm_all.dat',float)
    num  = 400        
    dim = len(data[0])
    
    y = []
    [y.append(data[n][dim-1]) for n in range(num)]

    kernel = np.zeros((num,num))
    kernel_v = np.zeros((num,len(validdata)))
    
    gamma_f ={1:32.,2:2., 3:0.125}
    gamma = gamma_f[3]
    lamda_f = {1:0.001, 2:1., 3:1000}
    lamda_a = lamda_f[3]*np.identity(num)

    print 'Gamma = ', gamma
    print 'lambda = ', lamda_a

    
    for n in range(num): #kernel for training
        for m in range(num):
            if n == m:
                kernel[n][m] = 1
            else:
                div = [pow(xn-xm,2) for xn,xm in zip(data[n][:dim-1],data[m][:dim-1])]
                ksum = -gamma*sum(div[n] for n in range(len(div)))
                kernel[n][m] = np.exp(ksum)
  

    for n in range(num): #kernel for validation
        for m in range(len(validdata)):
            div = [pow(xn-xm,2) for xn,xm in zip(data[n][:dim-1],validdata[m][:dim-1])]
            ksum = -gamma*sum(div[j] for j in range(len(div)))
            kernel_v[n][m] = np.exp(ksum)
        
    inv_k = inv(lamda_a + kernel)
    beta = np.dot(inv_k,y) #Derive beta
    print 'beta=',beta
    print 'kernel_v=',len(kernel_v)

    #Validation
    err = 0 
    pre = np.dot(beta,kernel_v)
    for num_p in range(len(pre)):
        if pre[num_p]*validdata[num_p][dim-1] < 0:
            err +=1
    print len(pre)
    print err/float(len(validdata))

    
    

data = np.loadtxt('hw2_lssvm_all.dat',float)
validdata = data[400:500]
LSSVM(validdata = validdata)





