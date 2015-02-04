import numpy as np
import ml_algorithm as mla


def node_c(c = 2):
    node = [[]]
    for i in range(c):
        node.append([])
    return node


def tree(data):
    global tree_g
    global itr

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
        tree(data)
    
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
    fg_tree = tree(data)
    ds = mla.decision_stump()

    print len(fg_tree)

    for n in range(len(fg_tree)):
        if n != 0:
            tmp_dat = fg_tree[n][1]
            
        if len(tmp_dat) > 1:
            
            d_res,s,axis,thresh  = ds.tds_branch(tmp_dat)
            fg_tree[n][1] = [d_res,s,axis,thresh]

            if fg_tree[n][0] != []:
                fg_tree[fg_tree[n][0]][1] = d_res[0]
            if fg_tree[n][2] != []:
                fg_tree[fg_tree[n][2]][1] = d_res[1]
                
        elif len(tmp_dat) == 1:
            fg_tree[n][1] = fg_tree[n][1][0][2]
            fg_tree[n][0] = -1
            fg_tree[n][2] = -1
        
    
    for i in  range(len(fg_tree)):
        print fg_tree[i],'\n'
    
   
tree_g = [[]]
traindat  = np.loadtxt('test.dat')
tree_1 = tree_ds(traindat)
#print tree_1










