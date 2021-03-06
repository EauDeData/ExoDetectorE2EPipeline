#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:16:52 2020

@author: gerard
"""

import tkinter as Tk
import skyCleaner, skyAlignment, skyTracker, skyCurve, classData
from classData import Parser
from PltInterface import Plot
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle as pk

class CreationMenu:
    """
    Interface:
        There are thre available options:
            1. Load images: if only the observations are used the program will assume
            that the frames are already clean. If all the folders are specified, the
            program will clean the images first.
            2. Get light curve: select the n stars you want to get the curve light from
            and then press 'esc' or the middle button of your mouse. After that, a plot
            will appea in your screen.
            3. Classify star: select the n stars you want to classiy. After that, a plot
            will appea in your screen with the corresponding labels. 
        Requeriments:
            - You must download the file 'model.h5' to load the weights of the classifier
    """
    
    def __init__(self):
        self.master = Tk.Tk()
        self.master.title("EDE2EP - ExoDetectorE2EPipeline")
        self.master.geometry("700x400")
        self.master.resizable(width=False, height=False)

        self.observation = Tk.Label(self.master, text="Folder containing the observation:", width=50)
        self.observation.pack()

        self._input_observation = Tk.Entry(self.master)
        self._input_observation.pack()
        
        self.df = Tk.Label(self.master, text="Folder containing the dark frames:", width=50) #flat dark 
        self.df.pack()

        self._input_df = Tk.Entry(self.master)
        self._input_df.pack()

        self.ff = Tk.Label(self.master, text="Folder containing the flat frames:", width=50)
        self.ff.pack()

        self._input_ff = Tk.Entry(self.master)
        self._input_ff.pack()

        self.fdf = Tk.Label(self.master, text="Folder containing the flat dark frames:", width=50)
        self.fdf.pack()

        self._input_fdf = Tk.Entry(self.master)
        self._input_fdf.pack()
        
        self._load = Tk.Button(self.master, command = self.load_button, text = "Load images", bg = 'Orchid1')
        self._load.pack()
        self._load.place(x=300, y=140)

        self._calculate_lc = Tk.Button(self.master, command = self.calculate_button, text = "Get light curve from star", bg = 'Orchid2')
        self._calculate_lc.pack()
        self._calculate_lc.place(x=260, y=175)
        
        self._detect_exo = Tk.Button(self.master, command = self.detect_button, text = "Classify star", bg = 'Orchid3')
        self._detect_exo.pack()
        self._detect_exo.place(x=295, y=215)

        self._canvas = Tk.Canvas(self.master, height=100, width=600, bg="snow", highlightbackground="gainsboro")
        self.set_canvas_text()
        self._canvas.place(x=50, y=255)
        
        self.quit = Tk.Button(master=self.master, text="Quit", command=self._quit, width=50, bg='snow', )
        self.quit.pack(side=Tk.BOTTOM)

        self.master.mainloop()

    def set_canvas_text(self, text = '\n\tBenvingudes!'
                        '\n\n\tIntroduiu les dades per obtenir un resultat!'):

        self._canvas.delete('Out')
        self._canvas.create_text(220,30,tag = 'Out', text = text)

        return None
    
    def load_button(self):
        obs = str(self._input_observation.get()) if str(self._input_observation.get())[-1] == '/' else str(self._input_observation.get())+'/'
        df = '' if str(self._input_df.get()) == str(self._input_ff.get()) else str(self._input_df.get()) if str(self._input_df.get())[-1] == '/' else str(self._input_df.get())+'/'
        ff = '' if str(self._input_ff.get()) == str(self._input_df.get()) else str(self._input_ff.get()) if str(self._input_ff.get())[-1] == '/' else str(self._input_ff.get())+'/'
        fdf = '' if str(self._input_fdf.get()) == str(self._input_df.get()) else str(self._input_fdf.get()) if str(self._input_fdf.get())[-1] == '/' else str(self._input_fdf.get())+'/'
        self.images, df, ff, fdf = skyCleaner.loadImages(obs, df, ff, fdf)
        if df == []:
            cleanImages = self.images
        else:
            cleanImages = skyCleaner.cleanImages(self.images, df, ff, fdf)
        self.alignedImages = skyAlignment.cross_correlation_fourier(cleanImages)
        self.starCoords , self.viewCoords = skyTracker.getStarCoords(self.alignedImages[0])
        self.lightCurves = skyCurve.getCurves(self.images, self.starCoords)
        
        

    def calculate_button(self):
        #Calcularà la corva de llum de les estrelles seleccionades

        images = self.alignedImages
        betterLightCurve = self.lightCurves
        plt.title("Select the stars you want to get the light curve from:")
        plt.figure(1,figsize=(10,10))
        #plt.scatter(self.viewCoords[0], self.viewCoords[1], s=80, facecolors='none', edgecolors='r')
        plt.imshow(images[0], cmap="gray")
        coords = plt.ginput(-1)
        plt.draw()
        plt.close(1)
        coords = np.array(coords)
        starCoords = np.array(self.starCoords)
        selectedCurves = []
        selectedStars = []
        for star in coords:
            dist = np.sqrt(np.sum(np.power((starCoords - star), 2),1))
            idx = np.argmin(dist)
            selectedCurves.append(betterLightCurve[idx])
            selectedStars.append(starCoords[idx])
        print("N stars:", len(selectedCurves))
        Plot(selectedCurves, selectedStars)
        
        return None
            
    def detect_button(self):
        #Detectara la possible estrella amb un exoplaneta
        images = self.alignedImages
        betterLightCurve = self.lightCurves
        plt.title("Select the stars you want classify:")
        plt.figure(1,figsize=(10,10))
        #plt.scatter(self.viewCoords[0], self.viewCoords[1], s=80, facecolors='none', edgecolors='r')
        plt.imshow(images[0], cmap="gray")
        coords = plt.ginput(-1)
        plt.draw()
        plt.close(1)
        coords = np.array(coords)
        starCoords = np.array(self.starCoords)
        selectedCurves = []
        selectedStars = []
        for star in coords:
            dist = np.sqrt(np.sum(np.power((starCoords - star), 2),1))
            idx = np.argmin(dist)
            selectedCurves.append(betterLightCurve[idx])
            selectedStars.append(starCoords[idx])
        
        #tf.autograph.set_verbosity(10)
        obj = classData.get_obj()
        model = classData.create_model(obj)
        model.load_weights('model.h5')
        idxToClass = {0: 64, 1: 65, 2: 67, 3: 6, 4: 42, 5: 15, 6: 16, 7: 52, 
        8: 53, 9: 88, 10: 90, 11: 92, 12: 62, 13: 95}
        classToName = {92: "Regular Star", 88:"Patient Variable Star",
                       42: "Supernova", 90:"Supernova (II)", 65:"TGM", 
                       16:"Binary Star or Exoplanetary Transit", 67:"?", 
                       95:"?", 62:"?", 15: "?", 52: "?", 6:"?", 64:"?", 53:"?"}
        
        tmp = []
        for sc in selectedCurves:
            longCurve = np.zeros(352)
            longCurve[:len(sc)] = sc
            tmp.append(longCurve)
        print(len(tmp))
        print(len(tmp[0]))
        
        x = np.array(tmp) - np.min(tmp)
        x = tf.keras.preprocessing.sequence.pad_sequences(x)
        prediction = model.predict(x)
        starClass = []
        for pred in prediction:
            newclass = classToName[idxToClass[np.argmax(pred)]]
            starClass.append(newclass)
        
        colors = plt.rcParams["axes.prop_cycle"]()
        plt.imshow(images[0], cmap="gray")
        labels = [str(selectedStars[x]) + " - category: " + str(starClass[x]) for x in range(len(selectedStars))]
        for i in range(len(selectedStars)):
            c = next(colors)["color"]
            print("Selected Star:", selectedStars[i][0], selectedStars[i][1])
            plt.scatter(int(selectedStars[i][0]), int(selectedStars[i][1]), s=80, facecolors='none', edgecolors=c, label=labels[i])
        plt.legend()
        plt.show()

        return None
    
    def _quit(self):
        self.master.quit()     # stops mainloop
        self.master.destroy()
    
