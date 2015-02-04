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
        itr = len(data)
        while n_node != 0:
            itr += n_node/2
            n_node = n_node/2

            
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
    num = len(data)
    fg_tree = tree(data)
    ds = mla.decision_stump()
    result  = ds.tds_branch(data)
    print result
    return fg_tree
    
   
tree_g = [[]]
traindat  = np.loadtxt('test.dat')
tree_1 = tree_ds(traindat)
print tree_1










