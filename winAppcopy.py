import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
from faceDetection import face_detector
import numpy as np
from typing import Tuple
from featureExtraction import feature_extraction
import time
import os


dataset_path = 'storage/dataset.npz'


class MainUI(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.columnconfigure(0,weight=1)
        self.columnconfigure(1,weight=1)
        self.columnconfigure(2,weight=1)
        self.rowconfigure(0,weight=1)
        self.rowconfigure(1,weight=1)
        self.rowconfigure(2,weight=1)
        self.title_container = tk.Frame(self)
        self.setting_container = tk.Frame(self)
        self.center_container = tk.Frame(self)
        self.option_container = tk.Frame(self)
        self.bottom_container = tk.Frame(self) 
        self.title_container.grid(row=0,column=0,columnspan=3)
        self.setting_container.grid(row=1,column=0)
        self.center_container.grid(row=1,column=1)
        self.option_container.grid(row=1,column=2)
        self.bottom_container.grid(row=2,column=0,columnspan=3)
        self.webcam = WebCam(self.center_container)


class WebCam(object):
    def __init__(self,master):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.background = tk.Canvas(self.frame)
        self.background.pack()
        # self.ds_face, self.ds_feature, self.ds_label = self.load_dataset()
        self.video_source = 0
        self.video_source = 'C:/Users/TrongTN/Downloads/1.mp4'
        self.vid = cv2.VideoCapture(self.video_source)
        self.update()

    def update(self):
        is_true, frame = self.get_frame()
        if is_true:
            self.background.configure(width=frame.shape[1], height=frame.shape[0])
            self.background_photo = ImageTk.PhotoImage(image = Image.fromarray(frame))
            self.background.create_image(0,0,image=self.background_photo)
        self.frame.after(15, self.update)

    def get_frame(self):
        if self.vid.isOpened():
            is_true, frame = self.vid.read()
            # frame = cv2.resize(frame, (self.winfo_screenwidth(),self.winfo_screenheight()))
            frame = cv2.resize(frame, (640,480))
            if is_true:
                return (is_true, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (is_true, None)
        else:
            return (is_true, None)



if __name__ == '__main__':
    MainUI().mainloop()