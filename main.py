import numpy as np
import matplotlib.pyplot as plt
import pla 

pla_dat = np.loadtxt('pla_data.csv',delimiter=',')

x,y = np.array(pla_dat[:,:2]),pla_dat[:,2].astype(np.int)

new_pla = pla.find_pla([-x[0][1]/x[0][0],1],x,y)

for i in range(10):
    new_pla.update()
    new_pla.plot()
