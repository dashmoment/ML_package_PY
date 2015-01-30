import numpy as np


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
            tree_g[0][1] = data
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
        

    
    
     

def tree2(data):
    val_array = [[]]
    num = 0
    d_size = len(data)
    itr_times = d_size
    

    while (d_size/2) > 0:
        itr_times += d_size/2
        d_size = d_size/2

 
    if len(val_array) == 1:
        num = 0
        val_array[0] = node_c()
        val_array[0][1] = data

    for n in range(itr_times):

        thresh = len(val_array[num][1])/2 

        if thresh > 0:    
            nodel = node_c()
            noder = node_c()
            for i in range(len(val_array[num][1])):
                if i < thresh:
                    nodel[1].append(val_array[num][1][i])
                else:
                    noder[1].append(val_array[num][1][i])
               
            val_array.append(nodel)
            val_array.append(noder)     
            val_array[num][0] = len(val_array) - 2   
            val_array[num][2] = len(val_array) - 1

        if  num < len(val_array) - 1:
            num+=1
        else:
            break
        
        

    print val_array,'\n'
       
raw = [1,2,3,4,5,6,7,8]
tree_g = [[]]

tree1 = tree(raw)
tree_r(tree1)
print tree1







