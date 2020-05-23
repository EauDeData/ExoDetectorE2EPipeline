#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 19:14:52 2020

@author: gerard
"""

#### Image processing ####
import cv2
import matplotlib.pyplot as plt
from PIL import Image as pil
import seaborn as sns
import png

#### Data sctructure and numerical support #####
import numpy as np
from scipy import signal as ss

#### OS Support ####
import os

### Astro data support ###
from astropy.io import fits

#### Codes ####
import geneticCleaner


#### Data Flags for classification and usage #####

#FLAT_FRAMES_EXPTIME = '5s'
#FOLDER = './wasp52b/' #Where the observation is placed (wasp52b) (cleanObservations)
#IMAGENAME_FLAG = 'TRE' #'TRE' #How we will identify images on a folder ('')
#FDO_FLAG = 'CDO' #Flat Dark frames identifier
#FFF_FLAG = 'FFF' #Flat Frames identifier
#DF_FLAG = 'CDO' #Dark Frames identifier


def loadImages(fileObs, fileDF, fileFF, fileFDF):
    print("Loading images...")
    obs = os.listdir(fileObs)
    obs.sort()
    df = os.listdir(fileDF)
    df.sort()
    ff = os.listdir(fileFF)
    ff.sort()
    fdf = os.listdir(fileFDF)
    fdf.sort()
    obs = [fileObs + name for name in obs]
    df = [fileDF + name for name in df]
    ff = [fileFF + name for name in ff]
    fdf = [fileFDF + name for name in fdf]
    images = [fits.open(image)[0].data for image in obs] #les imatges
    df = [fits.open(image)[0].data for image in df]
    ff = [fits.open(image)[0].data for image in ff] #Flat frames son frames totalment iluminats per trobar pixels morts (trobarem pols a la ccd)
    fdf = [fits.open(image)[0].data for image in fdf] #Flat dark frames son els dark frams de l'exposició dels flat frames
    return images, df, ff, fdf

def save16bits(filename, img): #.png
    with open(filename, 'wb') as f:
        writer = png.Writer(width=img.shape[1], height=img.shape[0], bitdepth=16, greyscale=True)
        zgray2list = img.tolist()
        writer.write(f, zgray2list)

def cleanImages(images, df, ff, fdf, save = False):
    images = np.array(images, dtype=np.float64)

    # ff
    meanff = np.mean(ff, axis=0)
    
    # fdf
    meanfdf = np.mean(fdf, axis=0)
    
    print("Genetically cleaning images...")
    gene = geneticCleaner.epoches(100, images[0], meanff, meanfdf)
    gene = gene[0]
    print(gene)
    epsilon = 0.00000001
    
    cleanIm = []
    if save == True:
        try:
            os.mkdir('./cleanImages/')
            n = 0
        except:
            n = 0
            
    for im1 in images:
        res = meanff / (gene[1] * meanfdf + epsilon)
        res = im1 / (gene[0] * res + epsilon)
        res = res - res.min()
        res = res / res.max()
        res = res * (2**16-1)
        cleanIm.append(np.uint16(res))
        if save == True:
            save16bits('./cleanImages/prova' + str(n) + '.png', np.uint16(res))
            n += 1
        
    print("Done!")
    return cleanIm

