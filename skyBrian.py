#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:15:31 2020

@author: gerard
"""

import skyTracker
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as ss

#images = skyTracker.alignedImages
#starCoords = skyTracker.starCoords 


def getBrians(images, starCoords): #Corba de llum
    ohHiBrian = []
    tmp = np.array(images)
    n = 10
    for star in starCoords:
        star_value = []
        xl = max(0, min(int(star[1])-n, len(images[0])))
        xr = max(0, min(int(star[1])+n+1, len(images[0])))
        yl = max(0, min(int(star[0])-n, len(images[0][0])))
        yr = max(0, min(int(star[0])+n+1, len(images[0][0])))
        maskFrame = tmp[:,xl:xr, yl:yr]
        #print("Mask shape:", maskFrame.shape)
        #skyCleaner.imshow_all(maskFrame[:5])
        for i in range(len(maskFrame)):
            star_value.append(np.mean(maskFrame[i]))
        ohHiBrian.append(star_value)
        
    return ohHiBrian

def getBetterBrians(lightCurve): #Corba de llum mitjana veins
    ohHiBrian = []
    for light in lightCurve:
        res = []
        for i in range(1,len(light)-1):
            mean = np.mean(light[i-1:i+1])
            res.append(mean)
        #skyTracker.skyCleaner.time_serie_plot(res, indx)
        ohHiBrian.append(res)
    return ohHiBrian
    

def getAdris(myLovelyBrians): #Corba de llum - corba mitjana
    adriCurves = np.array(myLovelyBrians)
    meanAdri = np.mean(adriCurves, 0)
    adriRes = adriCurves - meanAdri
    return adriRes

def derivAdris(adriRes): #Derivada de la corba de llum
    dx = np.array([-1,0,1]) * 0.5
    for adri in adriRes:
        deriv = ss.convolve(adri, dx)
        plt.plot(deriv)
    plt.show()

def getCurves(images, starCoords):
    myLovelyBrians = getBrians(images, starCoords)
    myBetterBrians = getBetterBrians(myLovelyBrians)
    myAdrians = getAdris(myBetterBrians)
    idx = 86
    plt.figure(1), plt.plot(myLovelyBrians[idx]), plt.plot(myBetterBrians[idx]), plt.plot(myAdrians[idx])
    plt.figure(2), plt.scatter(np.arange(len(myLovelyBrians[idx])), myLovelyBrians[idx]),
    plt.scatter(np.arange(len(myBetterBrians[idx])), myBetterBrians[idx]), 
    plt.scatter(np.arange(len(myAdrians[idx])), myAdrians[idx])

#derivAdris(myAdrians)

# Visualization mask
#numpyIm = np.array(images)
#x, y = int(starCoords[86][0]), int(starCoords[86][1])
#tmp = numpyIm[:, y-5:y+5, x-5:x+5]
#skyCleaner.imshow_all(tmp[:10])

#plt.plot(myLovelyBrians[86]); plt.plot(myBetterBrians(myLovelyBrians[86], 0.5)), plt.plot(adriRes[86])


