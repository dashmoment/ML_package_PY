import numpy as np
import ml_algorithm as mla
import random as rnd


def node_c(c = 2):
    node = [[]]
    for i in range(c):
        node.append([])
    return node


def tree(data,idx):
    global tree_g
    global itr

    if idx == 0:
        tree_g = [[]]

    leaf = len(tree_g)
    
       
    if leaf == 1:
        
        n_node = len(data)      
        pow_n = 0 
        while n_node > pow(2,pow_n):
            pow_n +=1         
        itr = sum(pow(2,i) for i in range(pow_n+1))

            
    if leaf < itr + 1:

        if leaf == 1:
            tree_g[0] = node_c()
            tree_g[0][1] = []
            tree_g.append(node_c())
        else:    
            for n in range(leaf):
                if tree_g[n][0] == []:
                    tree_g.append(node_c())
                    tree_g[leaf][1] = []
                    tree_g[n][0] = leaf-1                   
                    break
                if tree_g[n][2] == []:
                    tree_g.append(node_c())
                    tree_g[leaf][1] = []
                    tree_g[n][2] = leaf-1          
                    break
        tree(data,1)
    
    return tree_g
     
def tree_r(tree):

    for n in range(len(tree)):
        data = tree[n][1]
        
        if len(data) > 1:
            rule = len(data)/2
            for i in range(len(data)):
                idxl = tree[n][0]
                idxr = tree[n][2]
                
                if i < rule and idxl != []:
                   tree[idxl][1].append(data[i])
                elif i >= rule and idxr != []:              
                   tree[idxr][1].append(data[i])
            
    tree.pop()
        

     

def tree_ds(data):
    
    tmp_dat = np.copy(data)
    fg_tree = tree(data,0)
    ds = mla.decision_stump()

   

    for n in range(len(fg_tree)):
        if n != 0:
            tmp_dat = fg_tree[n][1]
            
        if len(tmp_dat) > 1:
            
            d_res,s,axis,thresh, gidx  = ds.tds_branch(tmp_dat)
            fg_tree[n][1] = [d_res,s,axis,thresh]

          
            if fg_tree[n][0] != [] and fg_tree[n][0] != -1:
                if d_res[0] != []:
                    fg_tree[fg_tree[n][0]][1] = d_res[0]
                    if gidx != 0:
                        fg_tree[fg_tree[n][0]][0] = -1
                        fg_tree[fg_tree[n][0]][2] = -1
                        fg_tree[fg_tree[n][0]][1] = d_res[0]
                elif d_res[1] != []:
                    fg_tree[fg_tree[n][0]][1] = -np.copy(d_res[1])
                    fg_tree[fg_tree[n][0]][0] = -1
                    fg_tree[fg_tree[n][0]][2] = -1
                  
              
                    
            if fg_tree[n][2] != [] and fg_tree[n][2] != -1:

                if len(d_res[1]) != 0:
                    fg_tree[fg_tree[n][2]][1] = d_res[1]
                    if gidx != 0:
                        fg_tree[fg_tree[n][2]][0] = -1
                        fg_tree[fg_tree[n][2]][2] = -1
                        fg_tree[fg_tree[n][2]][1] = d_res[1]
                elif len(d_res[0]) != 0:
                    fg_tree[fg_tree[n][2]][1] = -np.copy(d_res[0])
                    fg_tree[fg_tree[n][2]][0] = -1
                    fg_tree[fg_tree[n][2]][2] = -1
                    
##                if  fg_tree[fg_tree[n][2]][1] == []:
##                    print '2=', fg_tree[fg_tree[n][2]][1]

            
                    
                
                
        elif len(tmp_dat) == 1 and tmp_dat != [] and fg_tree[n][0] != -1:
            fg_tree[n][1] = tmp_dat
            fg_tree[n][0] = -1
            fg_tree[n][2] = -1
        
        
    return fg_tree


def d_valid(res_tree, val_data):
    num = len(val_data)
    asgn = []
    err = 0

    if num > 0:      
        dim = len(val_data[0])

        for n in range(num):
           if len(asgn) != n:
               print res_tree[i]
               print val_data[n]
               asgn.append([])
           
           i = 0
           while len(res_tree[i][1])> 0:

               if len(res_tree[i][1]) == 1:
                   #print res_tree[i]
                   sgn = res_tree[i][1][0][2]
                   if val_data[n][dim-1]*sgn < 0:
                       err+=1
                       asgn.append(sgn)
                       break
                   else:
                       asgn.append(sgn)
                       break
               else:
                   
                   s = res_tree[i][1][1]
                   d = res_tree[i][1][2]
                   th = res_tree[i][1][3]


                   sgn = s*(val_data[n][d] - th)
                   vsgn = 0
                   
                   if sgn > 0:

                       if res_tree[i][0] == -1 or res_tree[i][0] == []:
                          
                           for sd in range(len(res_tree[i][1][0])):
                                for sn in range(len(res_tree[i][1][0][sd])):
                                    vsgn +=  res_tree[i][1][0][sd][sn][dim-1]
                           asgn.append(vsgn)
                           
                           if val_data[n][dim-1]*vsgn< 0:
                               err +=1
                               #print len(res_tree[i][1][0][0]),'\n'
                           break
                       else:
                           i = res_tree[i][0]
                          
                        
                           
                   if sgn <= 0:
                       if res_tree[i][2] == -1 or res_tree[i][2] == []:
                           for sd in range(len(res_tree[i][1][0])):
                                for sn in range(len(res_tree[i][1][0][sd])):
                                    vsgn +=  res_tree[i][1][0][sd][sn][dim-1]
                           asgn.append(vsgn)
                           
                           if val_data[n][dim-1]*vsgn < 0:
                               err +=1
                               #print len(res_tree[i][1][0][0]),'\n'
                           break
                       else:
                           i = res_tree[i][2]
                              
                
        #print 'Error = ', err
        #print len(asgn)
        return err,asgn
    
    else:
        raise ValueError('len of data should not be zero')



def bootstrap(data, itr_T):
    num = len(data)
    bp_res = []
    for t in range(itr_T):
        rnd_tmp = []
        for i in range(num):
            rnd_tmp.append(data[rnd.randint(0,num-1)])
        bp_res.append(np.copy(rnd_tmp))
            
            
    return bp_res


def rndforest(data, dtrees):
    dnum = len(data)
    tnum = len(dtrees)
    sgn = []
    err_rf = 0
    err_avg = 0

    for i in range(tnum):
        err,asgn = d_valid(dtrees[i],data)
        sgn.append(asgn)
        err_avg += err

    for n in range(dnum):
        tmp = 0
        for m in range(len(sgn)):
            tmp+=sgn[m][n]

        if tmp*data[n][2] < 0:
            err_rf+=1
        else:
            err_rf = err_rf
    print err_rf
    err_rf = float(err_rf)/dnum
    err_avg = float(err_avg)/float(dnum*tnum)

    print 'E(RF) = ', err_rf
    print 'E(Avg) = ', err_avg


def pruned(data):

    dnum = len(data)

    if dnum > 0:
        
        dim = len(data[0])    
        tmp = [0,0]
        little_t = tree(tmp,0)
        little_t.pop()
        err = 0
       
        ds = mla.decision_stump()
        d_res,s,axis,thresh, gidx  = ds.tds_branch(data)

        little_t[0][1] = [d_res,s,axis,thresh]
        tmp_0 = 0
        tmp_1 = 0
        for i in range(len(d_res[0])):
             tmp_0 += d_res[0][i][2]
        for i in range(len(d_res[1])):
             tmp_1 += d_res[1][i][2]
           
        if tmp_0 > 0:         
            little_t[1][1] = 1.
            little_t[2][1] = -1.
        else:
            little_t[1][1] = -1.
            little_t[2][1] = 1.
            
        
        little_t[1][0] = -1
        little_t[1][2] = -1
        little_t[2][0] = -1
        little_t[2][2] = -1

##        print d_res[0],'\n'
##        print d_res[1],'\n'
##        print s,axis,thresh,tmp_0,tmp_1

##        for n in range(len(valdat)):
##            sgn = little_t[0][1][1]*(valdat[n][little_t[0][1][2]] - little_t[0][1][3])
##            if sgn > 0 and little_t[1][1]*valdat[n][2]<0:
##                err += 1
##            if sgn <= 0 and little_t[2][1]*valdat[n][2]<0:
##                err += 1

        return little_t

        
    else:
        raise ValueError('length of data should not be zero')
    
    
            

    

       

    


    
                    
                        
                        
                        

               
               
                
    
    
   












