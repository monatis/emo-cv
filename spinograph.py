import cv2
import numpy as np
import random as rd

width = 500
height = 500
canvas = np.zeros((width,height,3),dtype='uint8')
shifter = lambda x : (x[0]+width//2,-x[1]+height//2) 
class Graph:
    def __init__(self,outter_r,inner_r,h):
        self.outter_r =outter_r
        self.inner_r=inner_r
        self.h = h

    def graphPoint(self,t):
        x = (self.outter_r-self.inner_r)*np.cos(t) + self.h *np.cos((self.outter_r-self.inner_r)/self.inner_r*t)
        y = (self.outter_r-self.inner_r)*np.sin(t) + self.h *np.sin((self.outter_r-self.inner_r)/self.inner_r*t)
        return (int(x),int(y))

    def setTimeArray(self,start = 0,stop=100,num=100):
        self.array = np.linspace(start,stop,num)

    def graphIt(self,color=(50,50,50),linewidth=1,step=0.1,t0 = 0):
        t1 = t0 + step
        while True:
            
            cv2.line(canvas,shifter(self.graphPoint(t1)),shifter(self.graphPoint(t1+step)),color,linewidth)
            t1 += step
            cv2.imshow('Spinograf',canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

for i in range(1,20):
    Graph(rd.randint(100,300),rd.randint(50,100),rd.randint(1,50)).graphIt(color=(i*rd.randint(0,255//i),i*rd.randint(0,255//i),i*rd.randint(0,255//i)))
    
