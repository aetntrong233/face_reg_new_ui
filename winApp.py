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


# def darkstyle(root):
#     ''' Return a dark style to the window'''
    
#     style = ttk.Style(root)
#     root.tk.call('source', 'storage/something/Forest-ttk-theme/forest-dark.tcl')
#     style.theme_use('forest-dark')
#     return style


class MainUI(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        # style = darkstyle(self)
        self.columnconfigure(0, minsize=60)
        self.columnconfigure(2, minsize=115)
        self.container_setting = ttk.Frame(self)
        self.container_setting.grid(column=0)
        self.container_canvas = ttk.Frame(self)
        self.container_canvas.grid(column=1)
        self.container_mode = ttk.Frame(self)
        self.container_mode.grid(column=2)
        self.iconphoto(False, ImageTk.PhotoImage(file=r'storage/something/facerecog.png'))
        self.title("Face Recognizer")
        self.full_width= self.winfo_screenwidth()               
        self.full_height= self.winfo_screenheight()        
        self.geometry('815x480')
        self.resizable(0,0)
        # self.minsize(625, 600)
        # self.maxsize(self.full_width, self.full_height)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.video_source = 0
        self.video_source = 'C:/Users/TrongTN/Downloads/1.mp4'
        self.vid = cv2.VideoCapture(self.video_source)
        if self.vid is None or not self.vid.isOpened():
            raise ValueError("Unable to open this camera \n Select another video source", self.video_source)
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.x = self.winfo_x()
        self.y = self.winfo_y()
        self.background = tk.Canvas(self.container_canvas)
        self.background.grid()
        self.bbox_layer = np.zeros((480,640,3), np.uint8)
        self.ds_face, self.ds_feature, self.ds_label = load_dataset()
        self.update()
        self.background.bind("<Button-1>", self.background_clicked)
        self.mainloop()

    def background_clicked(self, event):
        face_list, feature_list, face_location_list, label_list = self.face_list, self.feature_list, self.face_location_list, self.label_list
        if face_list and feature_list and face_location_list and label_list:
            for i,(x,y,w,h) in enumerate(face_location_list):
                if event.x in range(x,x+w+1) and event.y in range(y,y+h+1):
                    # print('Type name of this user')
                    self.current_face = face_list[i]
                    self.current_feature = feature_list[i]
                    self.current_location = face_location_list[i]
                    self.current_label = label_list[i] 
                    self.popup()                   

    def popup(self):
        self.w=popupWindow(self)
        self.background.unbind("<Button-1>")
        self.wait_window(self.w.top)
        try:
            if self.w.value == '':
                messagebox.showwarning('Warning','Name can not be empty')
            elif self.w.value == 'None':
                pass
            elif self.w.value != self.current_label and self.current_label!= 'Unknown':
                if messagebox.askyesno('Noitice','This face look very similar {}.\n Are you sure to add this face with label {}'.format(self.current_label, self.w.value)):
                    self.append_dataset(self.w.value)
                    print(self.w.value)
            else:
                self.append_dataset(self.w.value)
                print(self.w.value)
        except Exception as e:
            print(e)
        self.background.bind("<Button-1>", self.background_clicked)

    def update(self):
        is_true, frame = self.get_frame()
        if is_true:
            self.add_info_bbox()
            combine_frame = self.roi(frame, self.bbox_layer)
            self.background.configure(width=frame.shape[1], height=frame.shape[0])
            self.background_photo = ImageTk.PhotoImage(image = Image.fromarray(combine_frame))
            self.background.create_image(frame.shape[1]//2,frame.shape[0]//2,image=self.background_photo)
        self.after(15, self.update)

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Are you sure?"):
            self.destroy()

    def get_frame(self):
        if self.vid.isOpened():
            is_true, frame = self.vid.read()
            # frame = cv2.resize(frame, (self.winfo_screenwidth(),self.winfo_screenheight()))
            # frame = cv2.imread(r'C:\Trong\python\st\data\Ly\3.jpeg')
            frame = cv2.resize(frame, (640,480))
            if is_true:
                return (is_true, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (is_true, None)
        else:
            return (is_true, None)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
    
    def add_info_bbox(self):
        is_true, frame = self.get_frame()
        if is_true:
            self.face_list, self.face_location_list = face_detector(frame)
            blank_image = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)
            if self.face_list and self.face_location_list:
                self.feature_list = []
                self.label_list = []
                for i,(x,y,w,h) in enumerate(self.face_location_list):
                    self.bbox_layer = cv2.rectangle(blank_image,(x,y),(x+w,y+h), (0,255,0), 2)
                    feature, label, prob = self.classifier(self.face_list[i])
                    self.feature_list.append(feature)
                    self.label_list.append(label)
                    info = '%s' % (label)
                    text_size = 24
                    if (y-text_size>=0):
                        left_corner = (x,y-text_size)
                    else:
                        left_corner = (x,y+h+text_size)
                    self.bbox_layer = cv2_img_add_text(self.bbox_layer, info, left_corner, (0,255,0))
                    self.bbox_layer = cv2_img_add_text(self.bbox_layer, time.strftime("%d-%m-%y-%H-%M-%S"), (0,frame.shape[0]-text_size), (0,0,255))
            else:
                self.bbox_layer = blank_image

    def roi(self, lower, upper):
        alpha_u = upper / 255.0
        alpha_l = 1.0 - alpha_u
        if upper.shape == lower.shape:
            return (alpha_u * upper[:, :] + alpha_l * lower[:, :]).astype('uint8')
        else:
            return lower

    def classifier(self, face_pixels):
        audit_feature = feature_extraction(face_pixels)
        if not self.ds_face or not self.ds_feature or not self.ds_label:
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

    def append_dataset(self, label):
        self.ds_face.append(self.current_face)
        self.ds_feature.append(self.current_feature)
        self.ds_label.append(label)
        np.savez(dataset_path, face_ds=self.ds_face, feature_ds=self.ds_feature, label_ds=self.ds_label)

    def index_remove(self, index):
        del self.ds_face[index]
        del self.ds_feature[index]
        del self.ds_label[index]
        np.savez(dataset_path, face_ds=self.ds_face, feature_ds=self.ds_feature, label_ds=self.ds_label)

    def user_remove(self, label):
        indexes = [i for i,x in enumerate(self.ds_label) if x == label]
        for index in indexes:
            del self.ds_face[index]
            del self.ds_feature[index]
        self.ds_label.remove(label)
        np.savez(dataset_path, face_ds=self.ds_face, feature_ds=self.ds_feature, label_ds=self.ds_label)
        

class popupWindow(object):
    def __init__(self,master):
        self.master = master
        top=self.top=tk.Toplevel(master)
        top.wm_overrideredirect(True)
        x,y,w,h = master.current_location
        win_x = master.x + x + 60
        win_y = master.y + y - 0
        top.geometry(f'+{win_x}+{win_y}')
        top.config(bg='')
        # self.l=tk.Label(top,text="Enter User name")
        # self.l.pack()
        self.e=ttk.Entry(top)
        self.e.bind("<Return>", self.enter)
        self.e.bind("<Escape>", self.esc)
        master.bind("<Button-1>", self.esc)
        self.e.pack()
        # self.b=tk.Button(top,text='Ok',command=self.cleanup)
        # self.b.pack(side=tk.LEFT,padx=(10,10),pady=(5,0))
        # self.b1=tk.Button(top,text='Cancel',command=self.destroy)
        # self.b1.pack(side=tk.RIGHT,padx=(10,10),pady=(5,0))
    def cleanup(self):
        self.value=self.e.get()
        self.top.destroy()
    def destroy(self):
        self.value='None'
        self.top.destroy()
    def enter(self,event):
        self.value=self.e.get()
        self.top.destroy()
    def esc(self,event):
        self.value='None'
        self.top.destroy()

        
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


def load_dataset():
    if os.path.isfile(dataset_path):
        dataset = np.load(dataset_path)
        face_list = []
        feature_list = []
        label_list = []
        for i in range(dataset['face_ds'].shape[0]):           
            face_list.append(dataset['face_ds'][i])
            feature_list.append(dataset['feature_ds'][i])
            label_list.append(dataset['label_ds'][i])
        return face_list, feature_list, label_list
    else:
        return [], [], []


__main__ = MainUI()