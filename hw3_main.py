import d_tree as tree
import numpy as np


traindat  = np.loadtxt('hw3_train.dat')
testdat  = np.loadtxt('hw3_test.dat')
minidat = np.loadtxt('test.dat')
tree_1 = tree.tree_ds(traindat)
err,sgn = tree.d_valid(tree_1,testdat)


##for time in range(100):
##    bs = tree.bootstrap(traindat,300)
##    bstree = []
##    for t in range(len(bs)):
##        tdata = []
##        tdata = np.copy(bs[t])
##        bstree.append(tree.tree_ds(tdata))
##
##    #print len(bstree)
##
##    tree.rndforest(testdat,bstree)

##err_total = 0
##
##for time in range(100):
##    bs = []
##    bstree = []
##    bs = tree.bootstrap(traindat,300)
##   
##    for t in range(len(bs)):
##        tdata = []
##        tdata = np.copy(bs[t])
##        bstree.append(tree.pruned(tdata))
##
##    ds_err = 0
##    dat = np.copy(traindat)
##
##    for n in range(len(dat)):
##        sgn = 0
##        for i in range(len(bstree)):
##            tmp = bstree[i][0][1][1]*(dat[n][bstree[i][0][1][2]] - bstree[i][0][1][3])
##            if tmp > 0:
##                sgn += bstree[i][1][1]
##            if  tmp <= 0:
##                sgn += bstree[i][2][1]
##        sgn = sgn*dat[n][2]
##
##        if sgn < 0:
##            ds_err += 1
##    print float(ds_err)/100
##    err_total += float(ds_err)/100
##        
##print  float(err_total)/100   


##n_rule = 0
##
##for n in range(len(tree_1)):
##    if tree_1[n][0] != [] and tree_1[n][0] != -1 and tree_1[n][2] != -1 and len(tree_1[n][1]) > 1:
##        n_rule += 1
##        print '[0] =',tree_1[n][1][0][0],'\n'
##        print '[1] =',tree_1[n][1][0][1],'\n'
##
##print 'n_rule = ', n_rule

##for n in range(len(tree_1)):
##    if tree_1[n][0] == [] and len(tree_1[n][1]) > 1:
##        print tree_1


    
