#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 19:14:52 2020

@author: gerard
"""
""" NOTES
- A prtir de la 100 inclosa estan netes, abans brutes jasjas
"""
#### Image processing ####
import cv2
import matplotlib.pyplot as plt
from PIL import Image as pil
import seaborn as sns

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
    fdf = [fileFDF + name for name in fdf]
    ff = [fileFF + name for name in ff]
    df = [fileDF + name for name in df]
    images = [fits.open(image)[0].data for image in obs] #les imatges
    fdf = [fits.open(image)[0].data for image in fdf] #Flat dark frames son els dark frams de l'exposició dels flat frames
    ff = [fits.open(image)[0].data for image in ff] #Flat frames son frames totalment iluminats per trobar pixels morts (trobarem pols a la ccd)
    df = [fits.open(image)[0].data for image in df]
    print("Done!")
    return images, fdf, ff, df

def cleanImages(images, df, ff, fdf):
    # Image
    im1 = np.array(images[0], dtype=float)
    
    # ff
    meanff = np.mean(ff, axis=0)
    
    # fdf
    meanfdf = np.mean(fdf, axis=0)
    
    gene = geneticCleaner.epoches(100, im1, meanff, meanfdf)
    gene = gene[0]
    
    epsilon = 0.00000001
    
    res = meanff / (gene[2] * meanfdf + epsilon + gene[3])
    res = im1 / (gene[0] * res + epsilon + gene[1])
    res = res - res.min()
    res = res / res.max()
    res = res * (2**16-1)
    
    return res



