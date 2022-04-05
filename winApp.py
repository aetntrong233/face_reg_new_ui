import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from tkinter import font as tkfont
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
from faceDetection import face_detector
import numpy as np
from typing import Tuple
from featureExtraction import feature_extraction
import time
import os


dataset_path = 'storage/dataset.npz'

# init: tk.Tk
# return: ttk.Style
# return a dark style to the window
def darkstyle(root):     
    style = ttk.Style(root)
    root.tk.call('source', 'storage/something/new_theme/forest-dark.tcl')
    style.theme_use('forest-dark')
    return style


# class main ui
class MainUI(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        style = darkstyle(self)
        self.wm_attributes('-transparentcolor','grey')
        self.resizable(0,0)
        self.iconphoto(False, ImageTk.PhotoImage(file=r'storage/something/facerecog.png'))
        self.title("Face Recognizer")
        self.protocol("WM_DELETE_WINDOW", self.close)
        # self.overrideredirect(True)
        self.win_w = self.winfo_screenwidth()
        self.win_h = self.winfo_screenheight()
        self.geometry('{}x{}'.format(int(0.75*self.win_w),int(0.70*self.win_h)))
        self.clicked_font = tkfont.Font(family='Helvetica', size=16, weight="bold")
        self.normal_font = tkfont.Font(family='Helvetica', size=16, weight="normal")
        # custom title bar
        # self.container_title_init()
        # top frame
        self.container_top_init()
        # bottom frame
        self.container_bottom_init()
        # setting frame 
        self.container_setting_init()
        # camera center frame
        self.container_center_init()
        # option frame
        self.container_option_init()

    # # init:
    # # return:
    # # declare title frame init widget
    # def container_title_init(self):
    #     self.container_title = tk.Frame(self,bg='#6e9b43',height=int(self.win_h*0.05))
    #     self.container_title.pack_propagate(0)
    #     self.container_title.pack(side=TOP,fill=X)
    #     self.btn_close = tk.Button(self.container_title,text=' x ',bg='#6e9b43',fg='white',command=self.close)
    #     self.btn_close.pack(side=RIGHT,padx=(0,15))

    # init:
    # return:
    # close window
    def  close(self):
        if messagebox.askokcancel("Quit", "Are you sure?"):
            self.destroy()

    # init:
    # return:
    # close window
    def container_top_init(self):
        self.container_top = tk.Frame(self,bg='#6e9b43',height=int(self.win_h*0.1))
        self.container_top.pack_propagate(0)
        self.container_top.pack(side=TOP,fill=X)
        self.recognition_label = tk.Label(self.container_top,bg='#6e9b43',fg='white',text='Recogniton',font=self.clicked_font)
        self.recognition_label.pack(side=LEFT,fill=BOTH,expand=True)
        self.recognition_label.bind("<Button-1>",self.recognition_clicked)
        self.registration_label = tk.Label(self.container_top,bg='#6e9b43',fg='white',text='Registration',font=self.normal_font)
        self.registration_label.pack(side=LEFT,fill=BOTH,expand=True)
        self.registration_label.bind("<Button-1>",self.registration_clicked)
        self.training_label = tk.Label(self.container_top,bg='#6e9b43',fg='white',text='Model Training',font=self.normal_font)
        self.training_label.pack(side=LEFT,fill=BOTH,expand=True)
        self.training_label.bind("<Button-1>",self.traning_clicked)
        self.setting_label = tk.Label(self.container_top,bg='#6e9b43',fg='white',text='Setting',font=self.normal_font)
        self.setting_label.pack(side=LEFT,fill=BOTH,expand=True)
        self.setting_label.bind("<Button-1>",self.setting_clicked)

    def recognition_clicked(self, event):
        self.recognition_label.configure(font=self.clicked_font)
        self.registration_label.configure(font=self.normal_font)
        self.training_label.configure(font=self.normal_font)
        self.setting_label.configure(font=self.normal_font)
        self.show_left_frame('LeftFrame1')
        self.show_right_frame('RightFrame1')
        self.show_center_frame('WebCam')

    def registration_clicked(self, event):
        self.recognition_label.configure(font=self.normal_font)
        self.registration_label.configure(font=self.clicked_font)
        self.training_label.configure(font=self.normal_font)
        self.setting_label.configure(font=self.normal_font)
        self.show_left_frame('LeftFrame2')
        self.show_right_frame('RightFrame2')
        self.show_center_frame('RegistrationPage')

    def traning_clicked(self, event):
        self.recognition_label.configure(font=self.normal_font)
        self.registration_label.configure(font=self.normal_font)
        self.training_label.configure(font=self.clicked_font)
        self.setting_label.configure(font=self.normal_font)
        self.show_left_frame('LeftFrame3')
        self.show_right_frame('RightFrame3')
        self.show_center_frame('TrainingPage')

    def setting_clicked(self, event):
        self.recognition_label.configure(font=self.normal_font)
        self.registration_label.configure(font=self.normal_font)
        self.training_label.configure(font=self.normal_font)
        self.setting_label.configure(font=self.clicked_font)
        self.show_left_frame('LeftFrame4')
        self.show_right_frame('RightFrame4')
        self.show_center_frame('SettingPage')

    # init:
    # return:
    # declare setting frame init widget
    def container_setting_init(self):
        self.container_setting = tk.Frame(self,bg='#c8c8c8',width=int(self.win_w*0.125),height=int(self.win_h*0.5))
        self.container_setting.pack_propagate(0)
        self.container_setting.pack(side=LEFT)
        self.left_frames = {}
        for F in (LeftFrame1,LeftFrame2,LeftFrame3,LeftFrame4):
            page_name = F.__name__
            left_frame = F(self.container_setting,self)
            self.left_frames[page_name] = left_frame
        self.last_left_frame = left_frame
        self.show_left_frame('LeftFrame1')

    # init:
    # return:
    # switch left frame view
    def show_left_frame(self, page_name):
        self.last_left_frame.pack_forget()
        frame = self.left_frames[page_name]
        self.last_left_frame = frame
        frame.pack()
        frame.tkraise()
    
    # init:
    # return:
    # declare center frame init widget
    def container_center_init(self):
        self.container_center = tk.Frame(self,bg='White',width=int(self.win_w*0.5),height=int(self.win_h*0.5))
        self.container_center.pack_propagate(0)
        self.container_center.pack(side=LEFT)
        self.enable_cam = False
        self.center_frames = {}
        for F in (WebCam,RegistrationPage,TrainingPage,SettingPage):
            page_name = F.__name__
            center_frame = F(self.container_center,self)
            self.center_frames[page_name] = center_frame
        self.last_center_frame = center_frame
        self.show_center_frame('WebCam')

    # init:
    # return:
    # switch center frame view
    def show_center_frame(self, page_name):
        if page_name == 'WebCam':
            self.enable_cam = True
        else:
            self.enable_cam = False
        self.last_center_frame.pack_forget()
        frame = self.center_frames[page_name]
        self.last_center_frame = frame
        frame.pack()
        frame.tkraise()

    # init:
    # return:
    # declare center frame init widget
    def container_option_init(self):
        self.container_option = tk.Frame(self,bg='#d4e8c5',width=int(self.win_w*0.125),height=int(self.win_h*0.5))
        self.container_option.pack_propagate(0)
        self.container_option.pack(side=LEFT)
        self.right_frames = {}
        for F in (RightFrame1,RightFrame2,RightFrame3,RightFrame4):
            page_name = F.__name__
            right_frame = F(self.container_option,self)
            self.right_frames[page_name] = right_frame
        self.last_right_frame = right_frame
        self.show_right_frame('RightFrame1')

    # init:
    # return:
    # switch center frame view
    def show_right_frame(self, page_name):
        self.last_right_frame.pack_forget()
        frame = self.right_frames[page_name]
        self.last_right_frame = frame
        frame.pack()
        frame.tkraise()

    # init:
    # return:
    # declare center frame init widget
    def container_bottom_init(self):
        self.container_bottom = tk.Frame(self,bg='#6b6b6b',height=int(self.win_h*0.1))
        self.container_bottom.pack_propagate(0)
        self.container_bottom.pack(side=BOTTOM,fill=X)


# class webcam
class WebCam(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        self.bg_layer = tk.Canvas(self)
        self.bg_layer.pack()
        self.ds_face, self.ds_feature, self.ds_label, self.ds_id = load_dataset()
        self.video_source = 0
        self.video_source = 'C:/Users/TrongTN/Downloads/1.mp4'
        self.vid = cv2.VideoCapture(self.video_source)
        if self.vid is None or not self.vid.isOpened():
            raise ValueError("Unable to open this camera \n Select another video source", self.video_source)
        self.bg_layer_update()

    # init: 
    # return:
    # loop to update new show frame
    def bg_layer_update(self):
        if self.master.enable_cam:
            is_true, frame = self.get_frame()
            if is_true:
                bbox_layer = self.get_bbox_layer()
                combine_layer = roi(frame,bbox_layer)
                self.bg_layer.configure(width=frame.shape[1], height=frame.shape[0])
                self.bg_layer_photo = ImageTk.PhotoImage(image = Image.fromarray(combine_layer))
                self.bg_layer.create_image(frame.shape[1]//2,frame.shape[0]//2,image=self.bg_layer_photo)
        self.after(15, self.bg_layer_update)

    # init: 
    # return: 
    # create bbox and info frame
    def get_bbox_layer(self):
        is_true, frame = self.get_frame()
        if is_true:
            face_list, face_location_list = face_detector(frame)
            blank_image = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)
            if face_list and face_location_list:
                for i,(x,y,w,h) in enumerate(face_location_list):
                    bbox_layer = draw_bbox(blank_image,(x,y,w,h), (0,255,0), 2, 10)
                    feature, label, prob = self.classifier(face_list[i])
                    info = '%s' % (label)
                    text_size = 24
                    if (y-text_size>=10):
                        left_corner = (x,y-text_size)
                    else:
                        left_corner = (x,y+h+text_size)
                    bbox_layer = cv2_img_add_text(bbox_layer, info, left_corner, (0,255,0))
                    # bbox_layer = cv2_img_add_text(bbox_layer, time.strftime("%d-%m-%y-%H-%M-%S"), (0,frame.shape[0]-text_size), (0,0,255))
            else:
                bbox_layer = blank_image
        return bbox_layer

    # init: 
    # return: 
    # get frame from webcam
    def get_frame(self):
        if self.vid.isOpened():
            is_true, frame = self.vid.read()
            if frame.shape[1] > self.master.win_w*0.5 or frame.shape[0] > self.master.win_h*0.5:
                scale_x = (self.master.win_w*0.5)/frame.shape[1]
                scale_y = (self.master.win_h*0.5)/frame.shape[0]
                scale = min(scale_x, scale_y)
                frame = cv2.resize(frame, (int(frame.shape[1]*scale),int(frame.shape[0]*scale)))
            if is_true:
                return (is_true, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (is_true, None)
        else:
            return (is_true, None)

    # init: 
    # return: 
    # face classify
    def classifier(self, face_pixels):
        audit_feature = feature_extraction(face_pixels)
        if not self.ds_face or not self.ds_feature or not self.ds_label or not self.ds_id:
            return audit_feature, 'Unknown', 0.0
        probability_list = []
        for feature in self.ds_feature:
            if audit_feature.size == feature.size:
                probability = np.dot(audit_feature, feature)/(np.linalg.norm(audit_feature)*np.linalg.norm(feature))
            else:
                probability = 0.0
            probability_list.append(probability)
        max_prob = np.max(probability_list)
        max_index = probability_list.index(max_prob)
        if max_prob >= 0.85:
            label = self.ds_label[max_index]
        else:
            label = 'Unknown'
        return audit_feature, label, max_prob*100

    # init: 
    # return: 
    # release webcam when destroy class
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


# init: 
# return: 
# 
def roi(lower, upper):
        alpha_u = upper / 255.0
        alpha_l = 1.0 - alpha_u
        if upper.shape == lower.shape:
            return (alpha_u * upper[:, :] + alpha_l * lower[:, :]).astype('uint8')
        else:
            return lower


# init: 
# return: 
# draw bouding box
def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2, length=10):
    (x,y,w,h) = bbox
    image = cv2.line(image, (x,y),(x+length,y),color,thickness)
    image = cv2.line(image, (x,y),(x,y+length),color,thickness)
    image = cv2.line(image, (x+w-length,y), (x+w,y),color,thickness)
    image = cv2.line(image, (x+w,y),(x+w,y+length),color,thickness)
    image = cv2.line(image, (x,y+h),(x+length,y+h),color,thickness)
    image = cv2.line(image, (x,y+h),(x,y+h-length),color,thickness)
    image = cv2.line(image, (x+w,y+h),(x+w-length,y+h),color,thickness)
    image = cv2.line(image, (x+w,y+h),(x+w,y+h-length),color,thickness)
    return image


# init: 
# return: 
# add text to cv2 image
def cv2_img_add_text(img, text, left_corner: Tuple[int, int], text_rgb_color=(255, 0, 0), text_size=24, font=r'storage/something/arial.ttc', **option):
    pil_img = img
    if isinstance(pil_img, np.ndarray):
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_text = ImageFont.truetype(font=font, size=text_size, encoding=option.get('encoding', 'utf-8'))
    draw.text(left_corner, text, text_rgb_color, font=font_text)
    cv2_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    if option.get('replace'):
        img[:] = cv2_img[:]
        return None
    return cv2_img


# init: 
# return: 
# load dataset
def load_dataset():
    if os.path.isfile(dataset_path):
        dataset = np.load(dataset_path)
        face_list = []
        feature_list = []
        label_list = []
        id_list = []
        for i in range(dataset['face_ds'].shape[0]):           
            face_list.append(dataset['face_ds'][i])
            feature_list.append(dataset['feature_ds'][i])
            label_list.append(dataset['label_ds'][i])
            id_list.append(dataset['id'][i])
        return face_list, feature_list, label_list, id_list
    else:
        return [], [], [], []


# class registration page
class RegistrationPage(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        tk.Label(self,text='regis').pack()


# class registration page
class TrainingPage(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        tk.Label(self,text='Training').pack()


class SettingPage(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        tk.Label(self,text='Setting').pack()


class LeftFrame1(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        tk.Label(self,text='LeftFrame1').pack()


class LeftFrame2(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        tk.Label(self,text='LeftFrame2').pack()


class LeftFrame3(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        tk.Label(self,text='LeftFrame3').pack()


class LeftFrame4(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        tk.Label(self,text='LeftFrame4').pack()


class RightFrame1(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        tk.Label(self,text='RightFrame1').pack()


class RightFrame2(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        tk.Label(self,text='RightFrame2').pack()


class RightFrame3(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        tk.Label(self,text='RightFrame3').pack()


class RightFrame4(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        tk.Label(self,text='RightFrame4').pack()


if __name__ == '__main__':
    MainUI().mainloop()