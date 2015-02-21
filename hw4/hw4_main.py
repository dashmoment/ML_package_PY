import numpy as np
import sort
import random

def cartesian(mat1, mat2):
    if len(mat1) != len(mat2):
        raise ValueError('dim of two matric should be the same')
    else:
        dist = 0
        for i in range(len(mat1)-1):
            res = mat1[i] - mat2[i]
            res = pow(res,2)
            dist += res

    return dist

def knearest(test, data, k):
    if len(data) <= 0:
        raise ValueError('N of data should not be zero')
    elif len(test) != len(data[0]):
        raise ValueError('dim of two test and data should be the same')
    else:
        count = 0
        tmp = []
        for i in range(len(data)):
            tmp.append([cartesian(test,data[i]),i])
            
        s= sort.sort(tmp)
        sort_res = s.quicksort(0, len(tmp)-1,0)
        #print sort_res

        k_res = []
        for j in range(k):
            k_res.append(sort_res[j])

        #print k_res[0]
        return k_res

def kvalid(test, data, k):
    err = 0
    for j in range(len(test)):
        knear = knearest(test[j],data,k)
        pre = 0
        for i in range(len(knear)):
            pre_tmp = data[knear[i][1]][len(data[0])-1]
            pre += pre_tmp
            
        if pre*test[j][len(test[j])-1] < 0:
            err+=1
    err = float(err)/len(test)
    return err


def sortdat(data, axis):
    s = sort.sort(data)
    sort_res = s.quicksort(0, len(data)-1,axis)
    return sort_res

def kmeans_init(data, k):
    seed = []
    clustered = np.copy(data)
    classes = []
    for i in range(k):
        rnd = random.randint(0,len(data) - 1)
        seed.append([data[rnd],i])

    #print seed
    for i in range(len(data)):
        for j in range(len(seed)):
            tmp = cartesian(data[i],seed[j][0])
            if j == 0:
                nearestseed = tmp
                cluster = seed[j][1]
            elif tmp < nearestseed:
                nearestseed = tmp
                cluster = seed[j][1]
            else:
                nearestseed = nearestseed
                cluster = cluster
        classes.append(cluster)
        
    clustered = np.column_stack((clustered,classes))
    #print clustered
    return clustered, seed
   
def kmeans(data, k):
    clustered, seed = kmeans_init(data,k)
    

    for t in range(20):
        sum_cls = np.zeros((k,len(clustered[0])-1))
        num_cls = np.zeros(k)

        #print clustered[0]
        print seed
        
        
        for i in range(len(clustered)):
            for j in range(len(seed)):
                if clustered[i][len(clustered[0])-1] == seed[j][1]:
                    
                    sum_cls[j] = [a+b for a,b in zip(sum_cls[j], clustered[i])]
                    num_cls[j] += 1
                    #print 'sum', num_cls
       
        #print sum_cls
        #print num_cls
        
        seed = []              
        for i in range(len(sum_cls)):
            sum_cls[i] = sum_cls[i]/ num_cls[i]
            seed.append([sum_cls[i], float(i)])
        
        for m in range(len(data)):
            c_num = 0
            min_dist = 0
            for c in range(len(seed)):
                tmp = seed[c][0]       
                c_tmp = cartesian(data[m], tmp)
                if c == 0 or c_tmp < min_dist:
                    min_dist = c_tmp
                    c_num = seed[c][1]
                else:
                    min_dist = min_dist
                    c_num = c_num

            clustered[m][len(clustered[m])-1] = c_num
        #print clustered
    return seed

def min_dest(valid_dat, seed):
    c_num = 0
    min_dist = 0

    for j in range(len(seed)):
        tmp = seed[j][0]       
        c_tmp = cartesian(valid_dat, tmp)

        if j == 0 or c_tmp < min_dist:
            min_dist = c_tmp
            c_num = seed[j][1]
        else:
            min_dist = min_dist
            c_num = c_num

    
    return min_dist
    
def kmeans_valid(data,valid_dat, itr_t):

    total_err = 0

    for t in range(itr_t):
        seed = kmeans(data,10)
        num = len(valid_dat)
        err = 0

        for i in range(num):
             err += min_dest(valid_dat[i], seed)

        total_err += float(err)/num
    total_err = total_err/itr_t
    print total_err
    
             

train_dat = np.loadtxt('hw4_knn_train.dat',float)
valid_dat = np.loadtxt('hw4_knn_test.dat', float)
k_train = np.loadtxt('hw4_kmeans_train.dat', float)

#ein = kvalid(train_dat,train_dat,5)
eout = kvalid(valid_dat,train_dat,5)
print eout
#print ein,'\n',eout


#kmeans(k_train, 5)
#kmeans_valid(k_train,k_train,500)


