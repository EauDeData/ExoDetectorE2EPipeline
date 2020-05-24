#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 19:14:52 2020

@author: gerard
"""

#### Image processing ####
import png

#### Data sctructure and numerical support #####
import numpy as np

#### OS Support ####
import os

### Astro data support ###
from astropy.io import fits

#### Codes ####
import geneticCleaner


def loadImages(fileObs, fileDF, fileFF, fileFDF):
    """
    params:
        fileObs: filename of the folder containing the observation frames
        fileDF: filename of the folder containing the dark frames
        fileFF: filename of the folder containing the flat frames
        fileFDF: filename of the folder containing the flat dark frames
    return
        list containing the images loaded as numpy matrixs (dtype=int16)
    """
    
    print("Loading images...")
    if (fileDF == fileFF and fileFF == fileFDF):
        obs = os.listdir(fileObs)
        obs.sort()
        obs = [fileObs + name for name in obs]
        images = [fits.open(image)[0].data for image in obs] #les imatges
        print("Done!")
        return images, [], [], []
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
    print("Done!")
    return images, df, ff, fdf

def save16bits(filename, img): #.png
    """
    params:
        filename: name of the image to save
        img: image to be saved
    return
        image saved as np.uint16
    """
    with open(filename, 'wb') as f:
        writer = png.Writer(width=img.shape[1], height=img.shape[0], bitdepth=16, greyscale=True)
        zgray2list = img.tolist()
        writer.write(f, zgray2list)

def cleanImages(images, df, ff, fdf, save = False):
    """
    params:
        images: list of images to be cleaned
        df: list of dark frames to clean the images
        ff: list of flat frames to clean the images
        fdf: list of flat dark frames to clean the flat frames
        save: determines if the clean images should be saved
    return
        images cleaned using a genetically trained model
    """
    
    images = np.array(images, dtype=np.float64)

    # ff
    meanff = np.mean(ff, axis=0)
    
    # fdf
    meanfdf = np.mean(fdf, axis=0)
    
    print("Genetically cleaning images...")
    gene = geneticCleaner.epoches(100, images[0], meanff, meanfdf)
    gene = gene[0]
    epsilon = 0.00000001
    
    cleanIm = []
    if save == True:
        try:
            os.mkdir('./cleanImages/')
            n = 0
        except:
            n = 0
            
    for im1 in images:
        res = meanff / (abs(gene[2]) * meanfdf + epsilon)
        res = im1 / (abs(gene[0]) * res + epsilon)
        res = res - res.min()
        res = res / res.max()
        res = res * (2**16-1)
        cleanIm.append(np.uint16(res))
        if save == True:
            save16bits('./cleanImages/prova' + str(n) + '.png', np.uint16(res))
            n += 1
        
    print("Done!")
    return cleanIm

