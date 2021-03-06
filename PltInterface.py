#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 21:17:02 2020

@author: gerard
"""


import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as Tk
import matplotlib.pyplot as plt

class Plot:
    """
    Plot Interface:
        This object is used to combine matplotlib plots with the Tkinter interfice
    """
    def __init__(self, curveList, starList):
        self.master = Tk.Tk()
        self.master.resizable(width=False, height=False)
        self.plot(curveList, starList)

    def plot(self, curveList, starList):
        colors = plt.rcParams["axes.prop_cycle"]()
        fig = Figure(figsize=(10,10))
        n = len(curveList)
        starList = [str(x) for x in starList]
        for i in range(n):
            c = next(colors)["color"]
            plotfig = fig.add_subplot((n//2)+1, 2, i + 1)
            plotfig.plot(curveList[i],label=starList[i],color=c)
        fig.legend()
        canvas = FigureCanvasTkAgg(fig, master=self.master)
        canvas.get_tk_widget().pack()
        canvas.draw()
                
        return None
        

    

