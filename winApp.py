import tkinter as tk
from tkinter import *
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
from landmarkDetection import get_landmark
from faceAngle import get_face_angle
from setting import *
import functools
import random
from faceDivider import face_divider
# from gtts import gTTS
# from playsound import playsound
import datetime
from maskDetection import mask_detector


dataset_path = 'storage/dataset.npz'


def darkstyle(root):     
    style = ttk.Style(root)
    root.tk.call('source', 'storage/something/new_theme/forest-dark.tcl')
    style.theme_use('forest-dark')
    style.configure('Treeview',background=COLOR[0],fieldbackground=COLOR[0],foreground=COLOR[4])
    return style


# class main ui
class MainUI(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        style = darkstyle(self)
        self.resizable(0,0)
        self.iconphoto(False, ImageTk.PhotoImage(file=r'storage/something/facerecog.png'))
        self.title("Face Recognizer")
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.win_w = self.winfo_screenwidth()
        self.win_h = self.winfo_screenheight()
        self.geometry('{}x{}'.format(int(0.75*self.win_w),int(0.75*self.win_h)))
        self.ds_face, self.ds_feature, self.ds_feature_masked, self.ds_label, self.ds_id = load_dataset()
        self.is_mask_recog = IS_MASK_RECOG
        self.used_users = []
        self.used_ids = []
        self.used_timestamps = []
        self.in_ids = []
        self.out_ids = []
        self.last_check = datetime.datetime.utcnow().strftime('%Y-%m-%d')
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

    def new_day_reset(self):
        now = datetime.datetime.utcnow().strftime('%Y-%m-%d')
        if self.last_check != now:
            self.last_check = now
            self.in_ids = []
            self.out_ids = []

    def container_top_init(self):
        self.container_top = ttk.Frame(self,height=int(self.win_h*0.125))
        self.container_top.pack_propagate(0)
        self.container_top.pack(side=TOP,fill=X)
        self.lb_list = []
        self.face_recog_icon = ImageTk.PhotoImage(Image.open('storage/something/face-id.png').resize((64,64),Image.ANTIALIAS))
        self.face_regis_icon = ImageTk.PhotoImage(Image.open('storage/something/face-recognition.png').resize((64,64),Image.ANTIALIAS))
        self.info_icon = ImageTk.PhotoImage(Image.open('storage/something/personal-information.png').resize((64,64),Image.ANTIALIAS))
        self.lb_list.append(tk.Label(self.container_top,text='Recogniton'))
        self.lb_list[0]["compound"] = BOTTOM
        self.lb_list[0]["image"]=self.face_recog_icon
        self.lb_list[0].pack(side=LEFT,fill=BOTH,expand=True)
        self.lb_list[0].bind("<Button-1>",self.recognition_clicked)
        self.lb_list.append(tk.Label(self.container_top,text='Registration'))
        self.lb_list[1]["compound"] = BOTTOM
        self.lb_list[1]["image"]=self.face_regis_icon
        self.lb_list[1].pack(side=LEFT,fill=BOTH,expand=True)
        self.lb_list[1].bind("<Button-1>",self.registration_clicked)
        self.lb_list.append(tk.Label(self.container_top,text='Model Training'))
        # self.lb_list[2].pack(side=LEFT,fill=BOTH,expand=True)
        self.lb_list[2].bind("<Button-1>",self.traning_clicked)
        self.lb_list.append(tk.Label(self.container_top,text='Information'))
        self.lb_list[3]["compound"] = BOTTOM
        self.lb_list[3]["image"]=self.info_icon
        self.lb_list[3].pack(side=LEFT,fill=BOTH,expand=True)
        self.lb_list[3].bind("<Button-1>",self.setting_clicked)
        for lb in self.lb_list:
            lb.configure(font=NORMAL_FONT,anchor=CENTER,fg=BLACK,highlightbackground=BLUE_GRAY[6],highlightthickness=1)
        self.lb_clicked(0)

    def lb_clicked(self, index):
        for i,lb in enumerate(self.lb_list):
            if i == index:
                lb.configure(borderwidth=1, relief="ridge",bg=TEAL[8])
            else:
                lb.configure(borderwidth=1, relief="flat",bg=TEAL[3])

    def recognition_clicked(self, event):
        self.lb_clicked(0)
        self.show_left_frame('LeftFrame1')
        self.show_right_frame('RightFrame1')
        self.show_center_frame('WebCam')

    def registration_clicked(self, event):
        self.lb_clicked(1)
        self.show_left_frame('LeftFrame2')
        self.show_right_frame('RightFrame2')
        self.show_center_frame('RegistrationPage')

    def traning_clicked(self, event):
        self.lb_clicked(2)
        self.show_left_frame('LeftFrame3')
        self.show_right_frame('RightFrame3')
        self.show_center_frame('TrainingPage')

    def setting_clicked(self, event):
        self.lb_clicked(3)
        self.show_left_frame('LeftFrame4')
        self.show_right_frame('RightFrame4')
        self.show_center_frame('SettingPage')

    def container_setting_init(self):
        self.container_setting = ttk.Frame(self,width=int(self.win_w*0.125),height=int(self.win_h*0.5))
        self.container_setting.pack_propagate(0)
        self.container_setting.pack(side=LEFT)
        self.left_frames = {}
        for F in (LeftFrame1,LeftFrame2,LeftFrame3,LeftFrame4):
            page_name = F.__name__
            left_frame = F(self.container_setting,self)
            self.left_frames[page_name] = left_frame
            left_frame.configure(bg=COLOR[0])
        self.last_left_frame = left_frame
        self.show_left_frame('LeftFrame1')

    def show_left_frame(self, page_name):
        self.last_left_frame.pack_forget()
        frame = self.left_frames[page_name]
        self.last_left_frame = frame
        frame.pack(fill=BOTH,expand=True)
        frame.tkraise()
    
    def container_center_init(self):
        self.container_center = ttk.Frame(self,width=int(self.win_w*0.5),height=int(self.win_h*0.5))
        self.container_center.pack_propagate(0)
        self.container_center.pack(side=LEFT)
        self.center_frames = {}
        for F in (WebCam,RegistrationPage,TrainingPage,SettingPage):
            page_name = F.__name__
            center_frame = F(self.container_center,self)
            self.center_frames[page_name] = center_frame
            center_frame.configure(style='Card',padding=(5,6,7,8))
        self.last_center_frame = center_frame
        self.show_center_frame('WebCam')

    def show_center_frame(self, page_name):
        frame = self.center_frames[page_name]
        if self.last_center_frame != frame:
            try:
                self.last_center_frame.enable_loop = False
            except Exception as e:
                pass
            self.last_center_frame.pack_forget()
            try:
                frame.enable_loop = True
                frame.loop()
            except Exception as e:
                pass
            self.last_center_frame = frame
            frame.pack(fill=BOTH,expand=True)
            frame.tkraise()
        try:
            frame.default()
        except Exception as e:
                pass

    def container_option_init(self):
        self.container_option = ttk.Frame(self,width=int(self.win_w*0.125),height=int(self.win_h*0.5))
        self.container_option.pack_propagate(0)
        self.container_option.pack(side=LEFT)
        self.right_frames = {}
        for F in (RightFrame1,RightFrame2,RightFrame3,RightFrame4):
            page_name = F.__name__
            right_frame = F(self.container_option,self)
            self.right_frames[page_name] = right_frame
            right_frame.configure(bg=COLOR[0])
        self.last_right_frame = right_frame
        self.show_right_frame('RightFrame1')

    def show_right_frame(self, page_name):
        self.last_right_frame.pack_forget()
        frame = self.right_frames[page_name]
        self.last_right_frame = frame
        frame.pack(fill=BOTH,expand=True)
        frame.tkraise()

    def container_bottom_init(self):
        self.container_bottom = tk.Frame(self,height=int(self.win_h*0.125))
        self.container_bottom.configure(bg=TEAL[8],highlightbackground=BLUE_GRAY[6],highlightthickness=1)
        self.container_bottom.pack_propagate(0)
        self.container_bottom.pack(side=BOTTOM,fill=X)
        label = tk.Label(self.container_bottom,font=NORMAL_FONT,bg=BLUE_GRAY[6],fg=COLOR[4],anchor=CENTER)
        label.configure(text=tthd)
        label.pack(fill=BOTH,expand=True)


# class webcam
class WebCam(ttk.Frame):
    def __init__(self,container,master):
        ttk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        self.bg_layer = tk.Canvas(self)
        self.bg_layer.pack(anchor=CENTER)
        self.video_source = 0
        # self.video_source = 'C:/Users/TrongTN/Downloads/1.mp4'
        # self.video_source = 'C:/Users/TrongTN/Downloads/y2mate.com - Know How to Wear Your Face Mask Correctly_1080pFHR.mp4'
        self.vid = cv2.VideoCapture(self.video_source)
        if self.vid is None or not self.vid.isOpened():
            raise ValueError("Unable to open this camera. Select another video source", self.video_source)
        self.enable_loop = False
        self.loop()

    def loop(self):
        if self.enable_loop:
            is_true, frame = self.get_frame()
            if is_true:
                bbox_layer = self.get_bbox_layer()
                combine_layer = roi(frame,bbox_layer)
                self.bg_layer.configure(width=frame.shape[1], height=frame.shape[0])
                self.bg_layer_photo = ImageTk.PhotoImage(image = Image.fromarray(combine_layer))
                self.bg_layer.create_image(frame.shape[1]//2,frame.shape[0]//2,image=self.bg_layer_photo)
            self.after(15, self.loop)

    def get_bbox_layer(self):
        is_true, frame = self.get_frame()
        if is_true:
            face_list, face_location_list = face_detector(frame)
            blank_image = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)
            if face_list and face_location_list:
                self.cf_ids = []
                bbox_layer = blank_image.copy()
                for i,(x,y,w,h) in enumerate(face_location_list):
                    bbox_layer = draw_bbox(bbox_layer,(x,y,w,h), (0,255,0), 2, 10)
                    face_alignment, face_parts, face_angle = get_face(frame,(x,y,w,h))
                    self.master.is_mask_recog = mask_detector(face_alignment)[0]
                    if self.master.is_mask_recog:
                        face = face_parts
                    else:
                        face = face_alignment
                    label, prob = self.classifier(face, self.master.is_mask_recog, True)
                    info = '%s' % (label)
                    text_size = 24
                    if (y-text_size>=10):
                        left_corner = (x,y-text_size)
                    else:
                        left_corner = (x,y+h+text_size)
                    bbox_layer = cv2_img_add_text(bbox_layer, info, left_corner, (0,255,0))
                    # bbox_layer = roi(bbox_layer,bbox_text_layer)
                    # bbox_layer = cv2_img_add_text(bbox_layer, time.strftime("%d-%m-%y-%H-%M-%S"), (0,frame.shape[0]-text_size), (0,0,255))
                hello_labels = []
                bye_labels = []
                for id in self.cf_ids:
                    if id not in self.master.in_ids:
                        indexes = [j for j,x in enumerate(self.master.ds_id) if x == id]
                        label = self.master.ds_label[indexes[0]]
                        self.master.in_ids.append(id)
                        hello_labels.append(label)
                    else:
                        if self.bye_time_check():
                            if id not in self.master.out_ids:
                                indexes = [j for j,x in enumerate(self.master.ds_id) if x == id]
                                label = self.master.ds_label[indexes[0]]
                                self.master.out_ids.append(id)
                                bye_labels.append(label)
                if hello_labels:
                    self.speech(hello_labels)
                if bye_labels:
                    self.speech(bye_labels, False)
            else:
                bbox_layer = blank_image
        return bbox_layer

    def speech(self, label_list, is_hello=True):
        if is_hello:
            text = 'Xin chào'
        else:
            text = 'Tạm biệt'
        for lb in label_list:
            text += ' '+lb
            if not lb == label_list[-1]:
                text += ','
        # text2speech(text)

    def bye_time_check(self):
        now = datetime.datetime.now()
        today_5pm = now.replace(hour=17,minute=0,second=0,microsecond=0)
        if now < today_5pm:
            return False
        else:
            return True

    def get_frame(self):
        self.master.new_day_reset()
        if self.vid.isOpened():
            is_true, frame = self.vid.read()
            # frame  = cv2.imread(r'C:\Users\TrongTN\Downloads\15-01.png')
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

    def classifier(self, face_pixels, is_mask_recog=False, add_extender=False):
        if add_extender:
            if is_mask_recog:
                extender = ' (mask)'
            else:
                extender = ' (without mask)'
        else:
            extender = ''
        if not self.master.ds_face or not self.master.ds_feature or not self.master.ds_feature_masked or not self.master.ds_label or not self.master.ds_id:
            return 'Unknown'+extender, 0.0    
        max_prob = 0.0
        probability_list = []
        if is_mask_recog:
            audit_feature = feature_extraction(face_pixels[2])
        else:
            audit_feature = feature_extraction(face_pixels)
        if is_mask_recog:
            ds_feature = self.master.ds_feature_masked
            for feature in ds_feature:
                for i in PART_CHECK:
                    probability_list_ = []
                    if audit_feature.size == feature[i].size:
                        probability = np.dot(audit_feature, feature[i])/(np.linalg.norm(audit_feature)*np.linalg.norm(feature[i]))
                    else:
                        probability = 0.0
                    probability_list_.append(probability)
                probability_list.append(np.mean(probability_list_))              
        else:
            ds_feature = self.master.ds_feature
            for feature in ds_feature:
                if audit_feature.size == feature.size:
                    probability = np.dot(audit_feature, feature)/(np.linalg.norm(audit_feature)*np.linalg.norm(feature))
                else:
                    probability = 0.0
                probability_list.append(probability)
        max_prob = np.max(probability_list)
        max_index = probability_list.index(max_prob)
        if max_prob >= 0.80:
            label = self.master.ds_label[max_index]
            id = self.master.ds_id[max_index]   
            t = time.strftime("%d-%m-%y-%H-%M-%S")
            self.master.used_users.append(label)
            self.master.used_ids.append(id)
            self.master.used_timestamps.append(t)
            self.cf_ids.append(id)
            try:
                self.master.right_frames['RightFrame1'].update()
            except Exception as e:
                pass
        else:
            label = 'Unknown'
        return label+extender, max_prob*100

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


def roi(lower, upper):
        alpha_u = upper / 255.0
        alpha_l = 1.0 - alpha_u
        if upper.shape == lower.shape:
            return (alpha_u * upper[:, :] + alpha_l * lower[:, :]).astype('uint8')
        else:
            return lower


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


def cv2_img_add_text(img, text, left_corner: Tuple[int, int], text_rgb_color=(255, 0, 0), text_size=24, font='storage/something/arial.ttc', **option):
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
        feature_masked_list = []
        label_list = []
        id_list = []
        for i in range(dataset['face_ds'].shape[0]):           
            face_list.append(dataset['face_ds'][i])
            feature_list.append(dataset['feature_ds'][i])
            feature_masked_list.append(dataset['feature_masked_ds'][i])
            label_list.append(dataset['label_ds'][i])
            id_list.append(dataset['id_ds'][i])
        return face_list, feature_list, feature_masked_list, label_list, id_list
    else:
        return [], [], [], [], []


def append_dataset(master, face, feature, feature_masked, label, id):
    master.ds_face.append(face)
    master.ds_feature.append(feature)
    master.ds_feature_masked.append(feature_masked)
    master.ds_label.append(label)
    master.ds_id.append(id)
    np.savez(dataset_path, face_ds=master.ds_face,feature_ds=master.ds_feature,feature_masked_ds=master.ds_feature_masked,label_ds=master.ds_label,id_ds=master.ds_id)
    master.right_frames['RightFrame2'].user_list_frame.reload_user_list()


def index_remove(master, index):
    del master.ds_face[index]
    del master.ds_feature[index]
    del master.ds_feature_masked[index]
    del master.ds_label[index]
    del master.ds_id[index]
    np.savez(dataset_path, face_ds=master.ds_face,feature_ds=master.ds_feature,feature_masked_ds=master.ds_feature_masked,label_ds=master.ds_label,id_ds=master.ds_id)
    master.right_frames['RightFrame2'].user_list_frame.reload_user_list()


def user_remove(master, id):
    indexes = [i for i,x in enumerate(master.ds_id) if x == id]
    indexes = sorted(indexes,reverse=True)
    for index in indexes:
        del master.ds_face[index]
        del master.ds_feature[index]
        del master.ds_feature_masked[index]
        del master.ds_label[index]
        del master.ds_id[index]
    np.savez(dataset_path, face_ds=master.ds_face,feature_ds=master.ds_feature,feature_masked_ds=master.ds_feature_masked,label_ds=master.ds_label,id_ds=master.ds_id)
    master.right_frames['RightFrame2'].user_list_frame.reload_user_list()


# class registration page
class RegistrationPage(ttk.Frame):
    def __init__(self,container,master):
        ttk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        self.webcam_frame = master.center_frames['WebCam']
        self.info_frame = ttk.Frame(self)
        self.info_frame_init()
        self.info_frame.pack(expand=True)
        self.add_user_frame = tk.Frame(self)
        self.add_user_frame_init()
        self.camera_frame = tk.Frame(self)
        self.camera_frame_init()
        self.new_user_faces = []
        self.face_parts = []
        self.labels = []
        self.pitchs = ['Center','Up','Down']
        self.yawns = ['Straight','Left','Right']
        for pitch in self.pitchs:
            for yawn in self.yawns:
                self.labels.append(pitch+'_'+yawn)
        for i in range(9):
            self.new_user_faces.append(None)
            self.face_parts.append(None)
        self.enable_loop = False
        self.enable_get_face = False
        self.loop()

    def info_frame_init(self):
        info_lb = ttk.Label(self.info_frame,font=NORMAL_FONT)
        info_lb.pack(fill=BOTH,expand=True)
        info_lb['text']='Please click Add new user to create new user\nor click to user name to change user data'

    def add_user_frame_init(self):
        ttk.Label(self.add_user_frame,text='Add new user',font=BOLD_FONT,anchor=CENTER).pack(side=TOP,fill=BOTH,pady=20)
        user_name_frame = ttk.Frame(self.add_user_frame)
        user_name_frame.pack(side=TOP,fill=BOTH,expand=True)
        ttk.Label(user_name_frame,text='User name',font=BOLD_FONT).pack(side=LEFT,fill=BOTH,pady=20,padx=15)
        self.user_name_var = tk.StringVar()
        user_name_entry = ttk.Entry(user_name_frame, textvariable=self.user_name_var)
        user_name_entry.pack(side=LEFT,fill=BOTH,pady=20)
        button_frame = ttk.Frame(self.add_user_frame)
        button_frame.pack(side=TOP,fill=BOTH,expand=True)
        self.ok_btn = ttk.Button(button_frame,text='Ok',command=self.ok_clicked)
        self.ok_btn.pack(side=LEFT,fill=BOTH,pady=20)
        self.cancel_btn = ttk.Button(button_frame,text='Cancel',command=self.cancel_clicked)
        self.cancel_btn.pack(side=RIGHT,fill=BOTH,pady=20)
    
    def ok_clicked(self):
        self.username = self.user_name_var.get()
        if self.username == '':
            messagebox.showwarning('Warning','User name cannot be empty!')
            self.user_name_var.set('')
        elif self.username == 'None':
            messagebox.showwarning('Warning','User name cannot be None!')
            self.user_name_var.set('')
        else:
            self.id = 0
            while(self.id in self.master.ds_id):
                self.id += 1
            self.master.right_frames['RightFrame2'].user_list_frame.pack_forget()
            self.master.right_frames['RightFrame2'].register_status_frame.pack(fill=BOTH,expand=True)
            self.add_user_frame.pack_forget()
            self.camera_frame.pack(expand=True)
            for i in range(9):
                self.new_user_faces[i] = None
                self.face_parts[i] = None
            self.enable_get_face = True
            self.master.left_frames['LeftFrame2'].chosen_lb(2)
            
    def cancel_clicked(self):
        self.add_user_frame.pack_forget()
        self.info_frame.pack(expand=True)
        self.username = ''
        for i in range(9):
            self.new_user_faces[i] = None
            self.face_parts[i] = None
        self.enable_get_face = False
        self.master.left_frames['LeftFrame2'].chosen_lb(0)

    def camera_frame_init(self):
        self.bg_layer = tk.Canvas(self.camera_frame)
        self.bg_layer.pack(anchor=CENTER)

    def default(self):
        self.master.left_frames['LeftFrame2'].chosen_lb(0)
        self.username = ''
        self.user_name_var.set('')
        self.add_user_frame.pack_forget()
        self.camera_frame.pack_forget()
        self.info_frame.pack(expand=True)
        self.master.right_frames['RightFrame2'].register_status_frame.pack_forget()
        self.master.right_frames['RightFrame2'].user_list_frame.pack(fill=BOTH,expand=True)
        for i in range(9):
            self.new_user_faces[i] = None
            self.face_parts[i] = None
            self.master.right_frames['RightFrame2'].register_status_frame.status[i].configure(text='...')
        self.enable_get_face = False
    
    def loop(self):
        if self.enable_loop:
            is_true, frame = self.webcam_frame.get_frame()
            if is_true:
                bbox_layer, bbox_frame, bbox_location = self.get_bbox_layer(frame)
                combine_layer = roi(frame,bbox_layer)
                if self.enable_get_face:
                    face_list, face_location_list = face_detector(bbox_frame)
                    if face_list and face_location_list:
                        face_alignment, face_parts, face_angle, face_bbox_layer = get_face(bbox_frame, face_location_list[0], True)
                        (x,y,w,h) = bbox_location
                        croped_combine_layer = roi(combine_layer[y:y+h,x:x+w],face_bbox_layer)
                        combine_layer[y:y+h,x:x+w] = croped_combine_layer
                        for i,label in enumerate(self.labels):
                            if self.check_face_angle(face_angle) == label:
                                if self.new_user_faces[i] is None:
                                    self.new_user_faces[i] = face_alignment
                                    self.face_parts[i] = face_parts
                        ct = 0
                        for i,new_user_face in enumerate(self.new_user_faces):
                            if new_user_face is None:
                                self.master.right_frames['RightFrame2'].register_status_frame.status[i].configure(text='...')
                            else:
                                ct += 1
                                self.master.right_frames['RightFrame2'].register_status_frame.status[i].configure(text='ok')
                        if ct == 9:
                            self.master.left_frames['LeftFrame2'].chosen_lb(3)
                            user_remove(self.master, self.id)
                            for i,new_user_face in enumerate(self.new_user_faces):
                                feature = feature_extraction(new_user_face)
                                feature_masked = []
                                for j in range(7):
                                    feature_masked.append(feature_extraction(self.face_parts[i][j]))
                                try:
                                    append_dataset(self.master, new_user_face, feature, feature_masked, self.username, self.id)
                                except Exception as e:
                                    print(e)
                            self.default()
                self.bg_layer.configure(width=frame.shape[1], height=frame.shape[0])
                self.bg_layer_photo = ImageTk.PhotoImage(image = Image.fromarray(combine_layer))
                self.bg_layer.create_image(frame.shape[1]//2,frame.shape[0]//2,image=self.bg_layer_photo)
            self.after(15, self.loop)

    def get_bbox_layer(self, frame, bbox_size = (400,400)):
        blank_image = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)
        center_x = frame.shape[1]/2
        center_y = frame.shape[0]/2
        w,h = bbox_size
        x = int(center_x - w/2)
        y = int(center_y - h/2)
        if bbox_size[0] >= frame.shape[0] or bbox_size[1] >= frame.shape[1] or bbox_size < (150,150):
            return blank_image, frame.copy()
        bbox_layer = draw_bbox(blank_image,(x,y,w,h), (0,255,0), 2, 10)
        bbox_frame = frame.copy()[y:y+h,x:x+w]
        return bbox_layer, bbox_frame, (x,y,w,h)

    def check_face_angle(self, face_angle):
        pitch = ''
        yawn = ''
        if -5.0 <= face_angle[1] <= 5.0:
            pitch = self.pitchs[0]
        elif face_angle[1] > 15.0:
            pitch = self.pitchs[1]
        elif face_angle[1] < -15.0:
            pitch = self.pitchs[2]
        else:
            if face_angle[1] > 0:
                pitch = 'Slightly'+self.pitchs[1]
            else:
                pitch = 'Slightly'+self.pitchs[2]
        if face_angle[2][0] <= 10.0:
            yawn = self.yawns[0]
        elif face_angle[2][0] > 20.0 and face_angle[2][1] == 'left':
            yawn = self.yawns[1]
        elif face_angle[2][0] > 20.0 and face_angle[2][1] == 'right':
            yawn = self.yawns[2]
        else:
            if face_angle[2][1] == 'left':
                yawn = 'Slightly'+self.yawns[1]
            else:
                yawn = 'Slightly'+self.yawns[2]
        return pitch+'_'+yawn


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


NOSE_CENTER_POINT = 5


def euclidean_distance(point1, point2):
    return np.sqrt(pow((point2[0]-point1[0]),2)+pow((point2[1]-point1[1]),2))


def get_face(frame,face_location,get_bbox_layer=False):
    (x,y,w,h) = face_location
    face = frame.copy()[y:y+h, x:x+w]
    landmark, score = get_landmark(face)
    landmark_ = []
    for point in landmark:
        point_x = int(x+point[0]*face.shape[1])
        point_y = int(y+point[1]*face.shape[0])
        landmark_.append((point_x,point_y))
    face_angle = get_face_angle(landmark_)
    rotate_frame = rotate_image(frame.copy(),face_angle[0])
    face_alignment = cv2.resize(rotate_frame.copy()[y:y+h, x:x+w], (224,224))
    scale_x = face_alignment.shape[0]/rotate_frame.shape[0]
    scale_y = face_alignment.shape[1]/rotate_frame.shape[1]
    landmark__ = []
    for point in landmark:
        point_x = int(point[0]*rotate_frame.shape[0]*scale_x)
        point_y = int(point[1]*rotate_frame.shape[1]*scale_y)
        landmark__.append((point_x,point_y))
    face_parts = face_divider(face_alignment, landmark__)
    if not get_bbox_layer:
        return face_alignment, face_parts, face_angle 
    else:
        blank_image = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)
        layer = blank_image.copy()
        layer = draw_bbox(blank_image,face_location)
        return face_alignment, face_parts, face_angle, layer


# class registration page
class TrainingPage(ttk.Frame):
    def __init__(self,container,master):
        ttk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        ttk.Label(self,text='Training').pack()


class SettingPage(ttk.Frame):
    def __init__(self,container,master):
        ttk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        ttk.Label(self,text='Infomation',font=BOLD_FONT,anchor=CENTER).pack(fill=X,ipady=10)
        ttk.Label(self,text='Version: 0.1',font=NORMAL_FONT,anchor=W).pack(fill=X,ipady=10)
        ttk.Label(self,text='Date: 2022-04-14 11:00:00 AM',font=NORMAL_FONT,anchor=W).pack(fill=X,ipady=10)
        ttk.Label(self,text='Code IDE: Visual Code Studio v1.66.2',font=NORMAL_FONT,anchor=W).pack(fill=X,ipady=10)
        ttk.Label(self,text='Machine Learning Back End: Tensorflow v2.8.0',font=NORMAL_FONT,anchor=W).pack(fill=X,ipady=10)
        ttk.Label(self,text='Programming Language: Python 3.9.',font=NORMAL_FONT,anchor=W).pack(fill=X,ipady=10)
        ttk.Label(self,text='Python GUI Library: Tkinter v0.0.1',font=NORMAL_FONT,anchor=W).pack(fill=X,ipady=10)
        # ttk.Label(self,text='Face Detection Model Architecture: SSD-like with a custom encoder',font=NORMAL_FONT,anchor=W).pack(fill=X,ipady=10)
        ttk.Label(self,text='Face Detection Model Architecture: DNN',font=NORMAL_FONT,anchor=W).pack(fill=X,ipady=10)
        ttk.Label(self,text='Face Landmark Detection Model Architecture: MobileNetV2-like with customized blocks',font=NORMAL_FONT,anchor=W).pack(fill=X,ipady=10)
        ttk.Label(self,text='Face Feature Extraction Model Architecture: VGGFace',font=NORMAL_FONT,anchor=W).pack(fill=X,ipady=10)


class LeftFrame1(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master


class LeftFrame2(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        self.lb_list = []
        self.lb_list.append(tk.Label(self,text='Tutorial'))
        self.lb_list.append(tk.Label(self,text='Step 1: Enter Username'))
        self.lb_list.append(tk.Label(self,text='Step 2: Add User Data'))
        self.lb_list.append(tk.Label(self,text='Step 3: Wait processing'))
        for lb in self.lb_list:
            lb.configure(font=NORMAL_FONT,anchor=W,bg=COLOR[0],fg=COLOR[4])
            lb.pack(side=TOP,fill=X,ipady=10)
        self.chosen_lb(0)
    
    def chosen_lb(self, index):
        for i,lb in enumerate(self.lb_list):
            if i == index:
                lb.configure(font=BOLD_FONT,bg=COLOR[1])
            else:
                lb.configure(font=NORMAL_FONT,bg=COLOR[0])
                

class LeftFrame3(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master


class LeftFrame4(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master


class RightFrame1(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        self.ct = 0
        tk.Label(self,text='Timestamps recording',font=BOLD_FONT,bg=COLOR[0],fg=COLOR[4]).pack(side=TOP,fill=X)
        self.frame = tk.Frame(self,bg=COLOR[0])
        self.frame.pack(side=BOTTOM,fill=BOTH,expand=True)
        self.scrollbarx = ttk.Scrollbar(self.frame,orient=HORIZONTAL)
        self.scrollbary = ttk.Scrollbar(self.frame,orient=VERTICAL)
        self.treeview = ttk.Treeview(self.frame,columns=("Id", "Name", "Time"))
        self.treeview.configure(height=100,selectmode="extended",xscrollcommand=self.scrollbarx.set,yscrollcommand=self.scrollbary.set)
        self.scrollbarx.config(command=self.treeview.xview)
        self.scrollbary.config(command=self.treeview.yview)
        self.scrollbary.pack(side=RIGHT,fill=Y)
        self.scrollbarx.pack(side=BOTTOM,fill=X)
        self.treeview.heading('Id', text="Id", anchor=W)
        self.treeview.heading('Name', text="Name", anchor=W)
        self.treeview.heading('Time', text="Time", anchor=W)
        self.treeview.column('#0', stretch=NO, minwidth=0, width=0)
        self.treeview.column('#1', stretch=NO, minwidth=0, width=30)
        self.treeview.column('#2', stretch=NO, minwidth=0, width=60)
        self.treeview.column('#3', stretch=NO, minwidth=0, width=120)
        self.treeview.pack()

    def update(self):
        self.treeview.insert('', 0, value=(self.master.used_ids[-1],self.master.used_users[-1],self.master.used_timestamps[-1]))


class RightFrame2(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        self.user_list_frame = UserList(self,master)
        self.user_list_frame.configure(bg=COLOR[0])
        self.user_list_frame.pack(fill=BOTH,expand=True)
        self.register_status_frame = RegisterStatus(self,master)
        self.user_list_frame.reload_user_list()


class RightFrame3(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master


class RightFrame4(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master


class RegisterStatus(ttk.Frame):
    def __init__(self,container,master):
        ttk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        labels = []
        pitchs = ['Center','Up','Down']
        yawns = ['Straight','Left','Right']
        for pitch in pitchs:
            for yawn in yawns:
                labels.append(pitch+' '+yawn)
        ttk.Label(self,text='Register Status',font=BOLD_FONT,).pack(side=TOP,fill=BOTH)
        self.left_frame = ttk.Frame(self)
        self.left_frame.pack(side=LEFT,fill=BOTH,expand=True)
        self.right_frame = ttk.Frame(self)
        self.right_frame.pack(side=RIGHT,fill=BOTH,expand=True)
        self.status = []
        for i,label in enumerate(labels):
            ttk.Label(self.left_frame,text=label,font=NORMAL_FONT).pack(side=TOP,fill=BOTH)
            self.status.append(ttk.Label(self.right_frame,text='...',font=NORMAL_FONT))
            self.status[i].pack(side=TOP,fill=BOTH)        


class UserList(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        self.bin_icon = ImageTk.PhotoImage(Image.open('storage/something/bin.png').resize((20,20),Image.ANTIALIAS))
        tk.Label(self,text='User List',font=BOLD_FONT,bg=COLOR[0],fg=COLOR[4]).pack(side=TOP,fill=BOTH,ipady=5)
        self.add_icon = ImageTk.PhotoImage(Image.open('storage/something/add-user.png').resize((30,30),Image.ANTIALIAS))
        self.add_new_user_lb = tk.Label(self,text='Add new user   ',font=NORMAL_FONT,bg=COLOR[0],fg=COLOR[4])
        self.add_new_user_lb["compound"] = RIGHT
        self.add_new_user_lb["image"]=self.add_icon
        self.add_new_user
        create_tool_tip(self.add_new_user_lb,COLOR[1],COLOR[0])
        self.add_new_user_lb.pack(side=TOP,fill=BOTH,ipady=5)
        self.add_new_user_lb.bind('<Button-1>', self.add_new_user)
        self.main_frame = tk.Frame(self,bg=COLOR[0])
        self.main_frame.pack(side=TOP,fill=BOTH,expand=True)
        self.scrollbar = ttk.Scrollbar(self.main_frame,orient='vertical')
        self.scrollbar.pack(side=RIGHT,fill=Y)
        self.canvas = tk.Canvas(self.main_frame,yscrollcommand=self.scrollbar.set,bg=COLOR[0])
        self.canvas.pack(side=LEFT,fill=BOTH,expand=True)
        self.scrollbar.config(command=self.canvas.yview)
        self.frame = tk.Frame(self.canvas,bg=COLOR[0])
        self.frame.pack(fill=BOTH,expand=True)
        self.frames = []
        self.choose_user_btns = []
        self.delete_user_btns = []

    def add_new_user(self, event):
        self.master.center_frames['RegistrationPage'].user_name_var.set('')
        self.master.center_frames['RegistrationPage'].info_frame.pack_forget()
        self.master.center_frames['RegistrationPage'].camera_frame.pack_forget()
        self.master.center_frames['RegistrationPage'].add_user_frame.pack(expand=True)
        self.master.left_frames['LeftFrame2'].chosen_lb(1)

    def choose_user(self, id, label, event):
        self.master.center_frames['RegistrationPage'].username = label
        self.master.center_frames['RegistrationPage'].id = id
        self.container.user_list_frame.pack_forget()
        self.container.register_status_frame.pack(fill=BOTH,expand=True)
        self.master.center_frames['RegistrationPage'].add_user_frame.pack_forget()
        self.master.center_frames['RegistrationPage'].info_frame.pack_forget()
        self.master.center_frames['RegistrationPage'].camera_frame.pack(expand=True)
        for i in range(9):
            self.master.center_frames['RegistrationPage'].new_user_faces[i] = None
            self.master.center_frames['RegistrationPage'].face_parts[i] = None
        self.master.center_frames['RegistrationPage'].enable_get_face = True
        self.master.left_frames['LeftFrame2'].chosen_lb(2)

    def delete_user(self, id, event):
        user_remove(self.master, id)

    def reload_user_list(self):   
        for frame in self.frames:         
            for frame in self.frames:
                for widget in frame.winfo_children():
                    widget.destroy()
        self.choose_user_btns = []
        self.delete_user_btns = []
        if self.master.ds_id:
            for i,id_ in enumerate(list(dict.fromkeys(self.master.ds_id))):
                self.frames.append(tk.Frame(self.frame,bg=COLOR[0]))
                self.frames[i].pack(fill=X,side=TOP)
                indexes = [j for j,x in enumerate(self.master.ds_id) if x == id_]
                label = self.master.ds_label[indexes[0]]
                self.choose_user_btns.append(tk.Label(self.frames[i],text=label))
                self.choose_user_btns[i].configure(font=NORMAL_FONT,anchor=W,bg=COLOR[0],fg=COLOR[4])
                self.choose_user_btns[i].pack(side=LEFT,fill=X,ipady=5,expand=True)
                self.choose_user_btns[i].bind('<Button-1>', functools.partial(self.choose_user,id_,label))
                icon_img = ImageTk.PhotoImage(Image.fromarray(cv2.resize(self.master.ds_face[random.choice(indexes)],(100,100))))
                create_tool_tip(self.choose_user_btns[i],COLOR[1],COLOR[0],'{} (id:{})'.format(label,id_),icon_img)
                self.delete_user_btns.append(tk.Label(self.frames[i],image=self.bin_icon))
                self.delete_user_btns[i].configure(anchor=CENTER,bg=COLOR[0])
                self.delete_user_btns[i].pack(side=LEFT,fill=X,ipady=5,ipadx=5)
                self.delete_user_btns[i].bind('<Button-1>', functools.partial(self.delete_user,id_))
                create_tool_tip(self.delete_user_btns[i],'red',COLOR[0])


class ToolTip(object):
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0
    
    def showtip(self,text=None,image=None):
        self.text = text
        self.image = image
        if self.tipwindow or not (text or image):
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 57
        y = y + cy + self.widget.winfo_rooty() +27
        self.tipwindow = tw = Toplevel(self.widget,background=TOOLTIP_BG,relief=SOLID,borderwidth=1)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        if self.text:
            label = Label(tw,text=self.text,justify=LEFT,background=TOOLTIP_BG,fg=TOOLTIP_FG,font=TOOLTIP_FONT)
            label.pack()
        if self.image:
            image = Label(tw,image=image,justify=LEFT,background=TOOLTIP_BG,relief=SOLID)
            image.pack()
    
    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

def create_tool_tip(widget, color1=COLOR[0], color2=COLOR[0], text=None, image=None):
    tool_tip = ToolTip(widget)  
    def enter(event):
        widget.configure(bg=color1)
        tool_tip.showtip(text, image)    
    def leave(event):
        widget.configure(bg=color2)
        tool_tip.hidetip()
    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)


mp3_path = 'storage/speech.mp3'


# def text2speech(text):
#     output = gTTS(text,lang="vi", slow=False)
#     output.save(mp3_path)
#     playsound(mp3_path, False)
#     print(text)
#     os.remove(mp3_path)


if __name__ == '__main__':
    MainUI().mainloop()