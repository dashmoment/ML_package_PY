import numpy as np
import matplotlib.pyplot as plt
import pla
import randomizer2 as ran

##pla_dat = np.loadtxt('pla_data.csv',delimiter=',')
##
##x,y = np.array(pla_dat[:,:2]),pla_dat[:,2].astype(np.int)
##
##new_pla = pla.find_pla([-x[0][1]/x[0][0],1],x,y)
##
##for i in range(10):
##    new_pla.update()
##    new_pla.plot()

##gausssian = ran.gaussion_2D(0,10,-100,100)
##res = gausssian.g2d_build()
##gausssian.g2d_plot(res, "Gaussian distribution")
##gausssian.g2d_writefile(res)
##result = gausssian.g2d_readfile()
##print result

g_3d = ran.gaussion_3D(0,0,5,5,0.5,-50,50)
g_3d.g3d_build()
