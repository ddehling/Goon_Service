import numpy as np


for n in range(7):
    rads=3.0
    radb=100.0
    radb=103.0
    ang=np.pi/6*n
    x=np.round(rads*np.cos(ang),2)
    y=np.round(rads*np.sin(ang),2)
    a=np.round(radb*np.cos(ang),2)
    b=np.round(radb*np.sin(ang),2)
    print([x,y,0])
    print([a,b,100])