#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 19:47:00 2020

@author: siddhi
"""

import tkinter as tk
import train


if __name__ == '__main__':
    
    r = tk.Tk()
    
    w = tk.Label(r,bg="black",fg="white", text="Face Recognition Attendance System"
                 ,font=("Helvetica", 20))
    w.pack()
    
    b1 = tk.Button(r, text='Take images of a new student',font=("Helvetica", 16), 
                   width=50,command=train.TakeImages, bg="blue",fg="yellow") 
    
    
    
    
    b2 = tk.Button(r, text='Train images',font=("Helvetica", 16), 
                   width=50,command=train.TrainImages, bg="blue",fg="yellow")
    
    b3 = tk.Button(r, text='Recognize face and mark attendance',font=("Helvetica", 16), 
                   width=50,command=train.TrackImages, bg="blue",fg="yellow")
    
    b1.pack()
    b2.pack()
    b3.pack()
    
    r.mainloop()
    
    