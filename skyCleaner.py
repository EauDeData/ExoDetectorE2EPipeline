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

#### Data sctructure and numerical support #####
import numpy as np

#### OS Support ####
import os

### Astro data support ###
from astropy.io import fits

#### Data Flags for classification and usage #####

FLAT_FRAMES_EXPTIME = '5s'
FOLDER = './wasp52b/' #Where the observation is placed
IMAGENAME_FLAG = 'TRE' #How we will identify images on a folder
FDO_FLAG = 'CDO' #Flat Dark frames identifier
FFF_FLAG = 'FFF' #Flat Frames identifier
DF_FLAG = 'CDO' #Dark Frames identifier

def imshow(im, c = 'gray', lut = False):
    if lut:
        histim = loadHist(im)
        cumhist = cumHist(histim)
        im_luted = cumhist[im]
        plt.imshow(im_luted, cmap = c)
        plt.show()
        return True
    plt.imshow(im, cmap = c)
    plt.show()

def loadHist(im):
    maxim = im.max()
    minim = im.min()
    im = im-minim
    hist = np.zeros(maxim+1)
    vals = im.reshape(-1)
    for value in vals:
        hist[value] += 1
    return hist

def cumHist(hist):
    cummies = np.zeros(len(hist))
    cummies[0] = hist[0]
    for i in range(1, len(cummies)):
        cummies[i] = cummies[i-1] + hist[i]
    return cummies

##### Image batching ####

allFiles = os.listdir(FOLDER) #All docs names in the folder
images = list(filter(lambda x: IMAGENAME_FLAG in x, allFiles)) #List of the images
fdf = list(filter(lambda x: FDO_FLAG in x and FLAT_FRAMES_EXPTIME in x, allFiles)) #List of flat dark frames
ff = list(filter(lambda x: FFF_FLAG in x, allFiles)) #List of flat frames
df = list(filter(lambda x: DF_FLAG in x and not FLAT_FRAMES_EXPTIME in x, allFiles)) #List of dark frames

print(" ---- Data Summary -----")
print("We have identified {} files as {}".format(len(images), 'images'))
print("We have identified {} files as {}".format(len(fdf), 'flat dark frames'))
print("We have identified {} files as {}".format(len(ff), 'flat frames'))
print("We have identified {} files as {}".format(len(df), 'dark frames'))

images = [FOLDER + name for name in images]
fdf = [FOLDER + name for name in fdf]
ff = [FOLDER + name for name in ff]
df = [FOLDER + name for name in df]

images = [fits.open(image)[0].data for image in images] #les imatges
fdf = [fits.open(image)[0].data for image in fdf] #Flat dark frames son els dark frams de l'exposició dels flat frames
ff = [fits.open(image)[0].data for image in ff] #Flat frames son frames totalment iluminats per trobar pixels morts (trobarem pols a la ccd)
df = [fits.open(image)[0].data for image in df] #Dark frames són les imatges apuntant a la foscar (trobar el soroll de fons)

#### Test zone ####
imshow(images[10], lut = 'Sí, si us plau') #Amb laLUT passada per que es vegi guai l'imatge
imshow(fdf[10])
imshow(ff[10], lut = True)
imshow(df[10], lut = True)

#TODO: Neteja del senyal
# NETEJAR DF
nframes = 30
meanIm = sum(df[:nframes])/nframes
# Convert to range 0-255
meanIm = (meanIm - meanIm.min())
meanIm = 255 * meanIm/meanIm.max()
# Double threshold to get dark and bright peaks
light =  (40 > meanIm) + (meanIm > 210)
# Remove noise
res = images[10] - meanIm * light
imshow(res.astype(int), lut = True)
imshow(images[10] != res)






