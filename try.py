#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 06:54:48 2020

@author: siddhi
"""
import cv2,time

cam = cv2.VideoCapture(0)
time.sleep(10)
cam.release()