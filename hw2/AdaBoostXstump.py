import numpy as np

train_0 = np.loadtxt('adB_train0.dat',float)

u=[]
[u.append(1./len(train_0)) for i in range(len(train_0))]
err_idx = []


dim = len(train_0[0])-1
hset = []#hypothesise set
alpha = np.zeros(dim+2)
dir_temp = [1.,-1.]

def ada_stump(train_0 = train_0, time = 5):

    atemp = []
    for n in range(len(train_0)):
            if n+1 < len(train_0):
                atemp.append((train_0[n][:dim] + train_0[n+1][:dim])/2)
    
    for t in range(time):
        e_min = len(u)
        e_sum = sum(u[i] for i in range(len(u)))
        err = e_sum

##        if t == time - 1:
##            print e_sum
        
        for m in range(len(atemp)):              # atemp: set of all possible alpha
            for d in range(len(atemp[0])):       # hypothesis in different dim 
                for q in range(len(dir_temp)):   # different direction : dir_temp = [1.,-1.]

                    #reset ein and err for each iteration
                    ein = 0
                    err_num = 0
                    errtemp = np.zeros(len(train_0))
                          
                    for n in range(len(train_0)):
                        sgn = dir_temp[q]*(train_0[n][d] - atemp[m][d])         

                        if train_0[n][dim]*sgn < 0:
                            ein += u[n]
                            errtemp[n] = -1
                            err_num +=1
               
                    if ein < err:
                        e_min = err_num
                        err = ein
                        err_idx = errtemp
                        
                        for k in range(dim):
                            if k == d:
                                alpha[k] = atemp[m][d]
                            else:
                                alpha[k] = 0.
                                
                        alpha[dim] = dir_temp[q]
                        
        if err != 0:
           
            err = err/e_sum
            print err
            p_factor = np.sqrt((1.-err)/err)
            
            
            if p_factor!= 0.:
                for i in range(len(err_idx)):
                    if err_idx[i] != 0:
                        u[i] = u[i]*p_factor
                    else:
                        u[i] = u[i]/p_factor
            #print u
            alpha[dim+1] = np.log(p_factor)
            
        elif err == 0 :
            alpha[dim+1] = 1

        if len(alpha) != 0:
            hset.append(np.copy(alpha))
            #print alpha[dim+1]
            
        np.savetxt('predictor2.txt', hset, delimiter=" ", fmt="%s")

def prediction(hypeset = 'predictor2.txt', testset = ''):
    hset = np.loadtxt(hypeset,float)
    valid = np.loadtxt(testset,float)
    err = 0
    err_g1 = 0
    
    for i in range(len(valid)):
        prediction = 0
        for n in range(len(hset)):
            for d in range(dim):
                if hset[n][d] != 0.:
                    gt = hset[n][dim]*(valid[i][d]-hset[n][d])
                    prediction += hset[n][dim+1]*(gt)

                    #print gt,valid[i][d], hset[n][d],hset[n][dim+1]                  
        if prediction*valid[i][dim] < 0:
            err+=1.
    for i in range(len(valid)):
        prediction = 0
       
        for d in range(dim):
            if hset[n][d] != 0.:
                gt = hset[0][dim]*(valid[i][d]-hset[0][d])
                prediction = gt
                    #print gt,valid[i][d], hset[n][d],hset[n][dim+1]                  
        if prediction*valid[i][dim] < 0:
            err_g1+=1.
            
            
    print 'err_g1 = ',err_g1/len(valid)
    print 'err = ',err/len(valid)

ada_stump(time = 300)
prediction(testset = 'hw2_adaboost_test.dat')
prediction(testset = 'adB_train0.dat')



