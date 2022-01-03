# -*- coding: utf-8 -*-

from PyQt5.QtCore import QTimer, QRect, QCoreApplication, Qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QPushButton, QLineEdit, QSlider, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
import cv2
# from Ui_main import Ui_MainWindow
import sys
import os
import numpy as np

from face_swap_model import FaceSwapModel
from face_detector2 import FaceDetectorModel

class MainWindow(QMainWindow):
    def __init__(self, parent=None): 
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle('image_loader')
        self.resize(1020,400)
        # self.setMinimumHeight(300)
        # self.setMinimumWidth(400)
        # self.status=self.statusBar()
        # self.status.showMessage('这是状态栏提示',4000)
        
        # video path
        self.open_path_text = QLineEdit(self)
        self.open_path_text.setGeometry(QRect(20, 20, 280, 20))
        self.open_path_text.setText("如果不选视频源，默认打开camera！")
        self.open_path_but = QPushButton(self)
        self.open_path_but.setGeometry(QRect(310, 16, 120, 30))
        self.open_path_but.setText("打开视频源")
        self.open_path_but.clicked.connect(self.open_event)
        self.video_file_path = None

        # style img path
        self.bg_text = QLineEdit(self)
        self.bg_text.setGeometry(QRect(500, 20, 280, 20))
        self.bg_text.setText("如果不选风格图片，采用指定风格！")
        self.bg_but = QPushButton(self)
        self.bg_but.setGeometry(QRect(790, 16, 120, 30))
        self.bg_but.setText("选择目标风格图片")
        self.bg_but.clicked.connect(self.bg_event)
        self.background_img = np.zeros((360, 640, 3), dtype=np.uint8)

        self.style1_label=QLabel("style1", self)
        self.style1_label.setGeometry(QRect(20, 50, 100, 30))
        self.sp_style1 = QSlider(Qt.Horizontal, self)
        self.sp_style1.setMinimum(-200)
        self.sp_style1.setMaximum(200)
        self.sp_style1.setSingleStep(0.5)
        self.sp_style1.setValue(0.01)
        self.sp_style1.setTickPosition(QSlider.TicksBelow)
        self.sp_style1.setTickInterval(20)
        self.sp_style1.setGeometry(QRect(100, 50, 600, 30))
        self.sp_style1.valueChanged.connect(self.style1_change)
        self.style1 = 0.01

        self.style2_label=QLabel("style2", self)
        self.style2_label.setGeometry(QRect(20, 80, 100, 30))
        self.sp_style2 = QSlider(Qt.Horizontal, self)
        self.sp_style2.setMinimum(-200)
        self.sp_style2.setMaximum(200)
        self.sp_style2.setSingleStep(0.5)
        self.sp_style2.setValue(0.01)
        self.sp_style2.setTickPosition(QSlider.TicksBelow)
        self.sp_style2.setTickInterval(20)
        self.sp_style2.setGeometry(QRect(100, 80, 600, 30))
        self.sp_style2.valueChanged.connect(self.style2_change)
        self.style2 = 0.01

        self.style3_label=QLabel("style3", self)
        self.style3_label.setGeometry(QRect(20, 110, 100, 30))
        self.sp_style3 = QSlider(Qt.Horizontal, self)
        self.sp_style3.setMinimum(-200)
        self.sp_style3.setMaximum(200)
        self.sp_style3.setSingleStep(0.5)
        self.sp_style3.setValue(0.01)
        self.sp_style3.setTickPosition(QSlider.TicksBelow)
        self.sp_style3.setTickInterval(20)
        self.sp_style3.setGeometry(QRect(100, 110, 600, 30))
        self.sp_style3.valueChanged.connect(self.style3_change)
        self.style3 = 0.01

        self.style4_label=QLabel("style4", self)
        self.style4_label.setGeometry(QRect(20, 140, 100, 30))
        self.sp_style4 = QSlider(Qt.Horizontal, self)
        self.sp_style4.setMinimum(-200)
        self.sp_style4.setMaximum(200)
        self.sp_style4.setSingleStep(0.5)
        self.sp_style4.setValue(0.01)
        self.sp_style4.setTickPosition(QSlider.TicksBelow)
        self.sp_style4.setTickInterval(20)
        self.sp_style4.setGeometry(QRect(100, 140, 600, 30))
        self.sp_style4.valueChanged.connect(self.style4_change)
        self.style4 = 0.01

        self.style5_label=QLabel("style5", self)
        self.style5_label.setGeometry(QRect(20, 170, 100, 30))
        self.sp_style5 = QSlider(Qt.Horizontal, self)
        self.sp_style5.setMinimum(-200)
        self.sp_style5.setMaximum(200)
        self.sp_style5.setSingleStep(0.5)
        self.sp_style5.setValue(0.01)
        self.sp_style5.setTickPosition(QSlider.TicksBelow)
        self.sp_style5.setTickInterval(20)
        self.sp_style5.setGeometry(QRect(100, 170, 600, 30))
        self.sp_style5.valueChanged.connect(self.style5_change)
        self.style5 = 0.01

        self.style6_label=QLabel("style6", self)
        self.style6_label.setGeometry(QRect(20, 200, 100, 30))
        self.sp_style6 = QSlider(Qt.Horizontal, self)
        self.sp_style6.setMinimum(-200)
        self.sp_style6.setMaximum(200)
        self.sp_style6.setSingleStep(0.5)
        self.sp_style6.setValue(0.01)
        self.sp_style6.setTickPosition(QSlider.TicksBelow)
        self.sp_style6.setTickInterval(20)
        self.sp_style6.setGeometry(QRect(100, 200, 600, 30))
        self.sp_style6.valueChanged.connect(self.style6_change)
        self.style6 = 0.01

        self.style7_label=QLabel("style7", self)
        self.style7_label.setGeometry(QRect(20, 230, 100, 30))
        self.sp_style7 = QSlider(Qt.Horizontal, self)
        self.sp_style7.setMinimum(-200)
        self.sp_style7.setMaximum(200)
        self.sp_style7.setSingleStep(0.5)
        self.sp_style7.setValue(0.01)
        self.sp_style7.setTickPosition(QSlider.TicksBelow)
        self.sp_style7.setTickInterval(20)
        self.sp_style7.setGeometry(QRect(100, 230, 600, 30))
        self.sp_style7.valueChanged.connect(self.style7_change)
        self.style7 = 0.01

        self.style8_label=QLabel("style8", self)
        self.style8_label.setGeometry(QRect(20, 260, 100, 30))
        self.sp_style8 = QSlider(Qt.Horizontal, self)
        self.sp_style8.setMinimum(-200)
        self.sp_style8.setMaximum(200)
        self.sp_style8.setSingleStep(0.5)
        self.sp_style8.setValue(0.01)
        self.sp_style8.setTickPosition(QSlider.TicksBelow)
        self.sp_style8.setTickInterval(20)
        self.sp_style8.setGeometry(QRect(100, 260, 600, 30))
        self.sp_style8.valueChanged.connect(self.style8_change)
        self.style8 = 0.01

        # show frame on label
        self.label=QLabel(self)
        # self.label.setAutoFillBackground(True)
        self.label.move(0, 280)

        # timer
        self.timer_camera = QTimer(self)
        self.cap = None
        self.timer_camera.timeout.connect(self.show_pic)
        self.timer_camera.start(30)

        self.face_swap_model = FaceSwapModel()
        # self.face_detector2 = FaceDetector()
        self.face_detector = FaceDetectorModel()

        self.write_video = False
        self.video_writer_created = False


    def __del__(self):
        if self.video_file_path is not None and self.write_video:
            self.videoWriter.release()
            os.system('ffmpeg -i ' + self.video_file_path + ' -f mp3 cr.mp3')
            os.system('ffmpeg -i cr.mp4 -i cr.mp3 -vcodec copy cr2.mp4')

    def open_event(self):
        _translate = QCoreApplication.translate
        directory1 = QFileDialog.getOpenFileName(None, "选择文件", "H:/")
        print(directory1)  # 打印文件夹路径
        self.video_file_path = directory1[0]
        self.open_path_text.setText(_translate("Form", directory1[0]))
        self.cap = None

    def bg_event(self):
        _translate = QCoreApplication.translate
        directory1 = QFileDialog.getOpenFileName(None, "选择文件", "H:/")
        print(directory1)  # 打印文件夹路径
        bg_img_path = directory1[0]
        self.bg_text.setText(_translate("Form", bg_img_path))

        # background img src
        self.background_img = cv2.imread(bg_img_path)

    def style1_change(self):
        self.style1 = self.sp_style1.value()/100
        self.style1_label.setText("style1:"+str(self.style1))
    def style2_change(self):
        self.style2 = self.sp_style2.value()/100
        self.style2_label.setText("style1:"+str(self.style2))
    def style3_change(self):
        self.style3 = self.sp_style3.value()/100
        self.style3_label.setText("style1:"+str(self.style3))
    def style4_change(self):
        self.style4 = self.sp_style4.value()/100
        self.style4_label.setText("style1:"+str(self.style4))
    def style5_change(self):
        self.style5 = self.sp_style5.value()/100
        self.style5_label.setText("style1:"+str(self.style5))
    def style6_change(self):
        self.style6 = self.sp_style6.value()/100
        self.style6_label.setText("style1:"+str(self.style6))
    def style7_change(self):
        self.style7 = self.sp_style7.value()/100
        self.style7_label.setText("style1:"+str(self.style7))
    def style8_change(self):
        self.style8 = self.sp_style8.value()/100
        self.style8_label.setText("style1:"+str(self.style8))

    def show_img(self, camera_rgb_frame):
        height = np.shape(camera_rgb_frame)[0]
        width = np.shape(camera_rgb_frame)[1]
        if width < 1024:
            self.resize(1024, height+280)
        else:
            self.resize(width, height+280)
        self.label.resize(width, height)
        showImage = QImage(camera_rgb_frame, width, height, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(showImage))

    def show_pic(self):
        # video src
        if self.cap is None:
            if self.video_file_path is not None:   #video from file
                self.cap = cv2.VideoCapture(self.video_file_path)
                self.video_writer_created = False
            else:                           #video from camera
                self.cap = cv2.VideoCapture(0)

        success, camera_frame=self.cap.read()
        if success and camera_frame is not None:
            new_width = int(np.shape(camera_frame)[1]/3.)
            camera_frame = camera_frame[:, new_width:new_width*2,:]
            zoom = 2
            if np.shape(camera_frame)[1] < 540 or np.shape(camera_frame)[0] < 540:
                zoom = 1
            tar_width = (int)(np.shape(camera_frame)[1]/zoom)
            tar_height = (int)(np.shape(camera_frame)[0]/zoom)
            tar_width = int((tar_width + 3)/4.)*4
            tar_height = int((tar_height + 3)/4.)*4

            camera_frame = cv2.resize(camera_frame, (tar_width, tar_height))
            camera_frame = cv2.flip(camera_frame, 1)
            camera_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)

            write_frame = np.zeros((np.shape(camera_frame)[0], np.shape(camera_frame)[1]*2, 3), dtype=np.uint8)
            write_frame[:,0:np.shape(camera_frame)[1],:] = camera_frame

            if self.video_writer_created == False and self.write_video:
                w_fps = self.cap.get(cv2.CAP_PROP_FPS)  
                w_size = (np.shape(camera_frame)[1]*2, np.shape(camera_frame)[0])  
                # self.videoWriter = cv2.VideoWriter('cr.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), w_fps, w_size)
                # self.videoWriter = cv2.VideoWriter('cr.mp4', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), w_fps, w_size)
                self.videoWriter = cv2.VideoWriter('cr.mp4', cv2.VideoWriter_fourcc('X', '2', '6', '4'), w_fps, w_size)
                self.video_writer_created = True
            
            left_up_x = 0
            left_up_y = 0
            right_down_x = 0
            right_down_y = 0
            detections = self.face_detector.ProcessOneFrame(camera_frame)
            if len(detections) > 0:
                left_up_x,left_up_y,right_down_x,right_down_y = detections[0]
            else:
                self.show_img(camera_frame)
                if self.write_video:
                    write_frame[:,np.shape(camera_frame)[1]:np.shape(camera_frame)[1]*2,:] = camera_frame
                    write_frame = cv2.cvtColor(write_frame, cv2.COLOR_RGB2BGR)
                    self.videoWriter.write(write_frame)
                return

            rect_width = right_down_y - left_up_y + 1
            left_up_x_final = left_up_x
            right_down_x_final = right_down_x
            left_up_y_final = left_up_y
            right_down_y_final = right_down_y

            if left_up_x_final < 0:
                left_up_x_final = 0
            if left_up_y_final < 0:
                left_up_y_final = 0
            if right_down_x_final > tar_width:
                right_down_x_final = tar_width-1
            if right_down_y_final > tar_height:
                right_down_y_final = tar_height-1

            left_up_x_in_256 = int((left_up_x_final - left_up_x)/rect_width*256.)
            left_up_y_in_256 = int((left_up_y_final - left_up_y)/rect_width*256.)
            right_down_x_in_256 = 255 - int((right_down_x - right_down_x_final)/rect_width*256.)
            right_down_y_in_256 = 255 - int((right_down_y - right_down_y_final)/rect_width*256.)
            # print(left_up_x_in_256, left_up_y_in_256, right_down_x_in_256, right_down_y_in_256)
            face_rect_rgb_img = camera_frame[left_up_y_final:right_down_y_final+1, left_up_x_final:right_down_x_final+1, :]
            out  = self.face_swap_model.ProcessOneFrame(face_rect_rgb_img, left_up_x_in_256, left_up_y_in_256, right_down_x_in_256, right_down_y_in_256)
            camera_frame[left_up_y_final:right_down_y_final+1, left_up_x_final:right_down_x_final+1, :] = out
            self.show_img(camera_frame)

            if self.write_video:
                write_frame[:,np.shape(camera_frame)[1]:np.shape(camera_frame)[1]*2,:] = camera_frame
                write_frame = cv2.cvtColor(write_frame, cv2.COLOR_RGB2BGR)
                self.videoWriter.write(write_frame)


if __name__=='__main__':
    app=QApplication(sys.argv)
    window=MainWindow()
    window.show()
    sys.exit(app.exec_())