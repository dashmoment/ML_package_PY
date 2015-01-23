import sys
sys.path.append('D:\libsvm-3.20\python')
from svmutil import *
import numpy as np


##dat = np.loadtxt('featurestrain.csv')
##val_dat = np.loadtxt('featurestest.csv')
##
##label = []
##label_val = []
##datatrain = []
##dataval = []
##
##digit = 0
##
##for i in range(len(dat)):
##    label.append(dat[i][0] == digit and 1 or -1)
##
##for i in range(len(val_dat)):
##    label_val.append(dat[i][0] == digit and 1 or -1)
##
##for i in range(len(label)):
##    datatrain.append([label[i], '1:'+ str(dat[i][1]), '2:'+ str(dat[i][2])])
##
##for i in range(len(label_val)):
##    dataval.append([label_val[i], '1:'+ str(val_dat[i][1]), '2:'+ str(val_dat[i][2])])
##
##    
##np.savetxt('train.txt', datatrain, delimiter=" ", fmt="%s")
##np.savetxt('valid.txt', dataval, delimiter=" ", fmt="%s")

###Build SVM model####

##y, x = svm_read_problem('train.txt')
##yv,xv = svm_read_problem('valid.txt')
##
###m = svm_train(y, x, '-c 0.01 -t 0')
##m = svm_train(y, x, '-c 0.1 -t 2 -g 10000')
##
##p_label, p_acc, p_val = svm_predict(yv, xv, m)
##print 'C = 0.1',p_acc
##
##svm_cof = m.get_sv_coef()
##sv = m.get_SV()

##Q17 sum of alpha
##sum_alpha = 0
##sv_num = 0
##
##for j in range(len(svm_cof)):
##    if svm_cof[j][0] > 0:
##        sum_alpha += svm_cof[j][0]
##        #print svm_cof[j][0]
##        sv_num +=1 
##    else:
##        sum_alpha += (-1.)*svm_cof[j][0]
##        #print svm_cof[j][0]
##        sv_num +=1 
        

#print sv_num
##end Q17 sum of alpha

##ww = []
##
##ww.append(sum(svm_cof[i][0]*sv[i][1] for i in range(len(svm_cof))))
##ww.append(sum(svm_cof[i][0]*sv[i][2] for i in range(len(svm_cof))))
##
##margin = np.sqrt(pow(ww[0],2)+pow(ww[1],2))

#print ww,len(svm_cof)
#print '1/margin = ', 1./margin  #Q15 ans

y2 = [-1,-1,-1,1,1,1,1]
x2 = [[1,0],[0,1],[0,-1],[-1,0],[0,2],[0,-2],[-2,0]]

m2 = svm_train(y2, x2, '-c 0.0001 -t 1 -d 2 -g 1 -r 1')
p_label2, p_acc2, p_val2 = svm_predict(y2, x2, m2)

svm_cof2 = m2.get_sv_coef()
sv2 = m2.get_SV()

print svm_cof2

sum_alpha = 0
sv_num = 0

for j in range(len(svm_cof2)):
    if svm_cof2[j][0] > 0:
        sum_alpha += svm_cof2[j][0]
        #print svm_cof[j][0]
        sv_num +=1 
    else:
        sum_alpha += (-1.)*svm_cof2[j][0]
        #print svm_cof[j][0]
        sv_num +=1
print sum_alpha









