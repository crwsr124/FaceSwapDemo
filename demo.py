# -*- coding: utf-8 -*-

from PyQt5.QtCore import QTimer, QRect, QCoreApplication, Qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QPushButton, QLineEdit, QSlider, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
import cv2
# from Ui_main import Ui_MainWindow
import sys
import numpy as np

from face_detector import FaceDetector
from face_swap_model import FaceSwapModel

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

        # background img path
        self.bg_text = QLineEdit(self)
        self.bg_text.setGeometry(QRect(500, 20, 280, 20))
        self.bg_text.setText("如果不选风格图片，采用指定风格！")
        self.bg_but = QPushButton(self)
        self.bg_but.setGeometry(QRect(790, 16, 120, 30))
        self.bg_but.setText("选择目标风格图片")
        self.bg_but.clicked.connect(self.bg_event)
        self.background_img = np.zeros((360, 640, 3), dtype=np.uint8)

        self.sp_label=QLabel("容差：0.5", self)
        self.sp_label.setGeometry(QRect(20, 50, 100, 30))
        self.sp = QSlider(Qt.Horizontal, self)
        self.sp.setMinimum(0)
        self.sp.setMaximum(200)
        self.sp.setSingleStep(1)
        self.sp.setValue(100)
        self.sp.setTickPosition(QSlider.TicksBelow)
        self.sp.setTickInterval(10)
        self.sp.setGeometry(QRect(120, 50, 600, 30))
        self.sp.valueChanged.connect(self.slider_val_change)
        self.fcapacity = 0.5

        # show frame on label
        self.label=QLabel(self)
        # self.label.setAutoFillBackground(True)
        self.label.move(0, 100)

        # timer
        self.timer_camera = QTimer(self)
        self.cap = None
        self.timer_camera.timeout.connect(self.show_pic)
        self.timer_camera.start(10)

        self.face_swap_model = FaceSwapModel()
        self.face_detector = FaceDetector()

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

    def slider_val_change(self):
        self.fcapacity = self.sp.value()/200
        self.sp_label.setText("容差："+str(self.fcapacity))
        # self.matting.SetCapacity(self.fcapacity)

    def show_img(self, camera_rgb_frame):
        height = np.shape(camera_rgb_frame)[0]
        width = np.shape(camera_rgb_frame)[1]
        if width < 1024:
            self.resize(1024, height+100)
        else:
            self.resize(width, height+100)
        self.label.resize(width, height)
        showImage = QImage(camera_rgb_frame, width, height, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(showImage))
        self.timer_camera.start(10)

    def show_pic(self):
        # video src
        if self.cap is None:
            if self.video_file_path is not None:   #video from file
                self.cap = cv2.VideoCapture(self.video_file_path)
                # self.matting.Reset()
            else:                           #video from camera
                self.cap = cv2.VideoCapture(0)
                # self.matting.Reset()

        success, camera_frame=self.cap.read()
        if success and camera_frame is not None:
            tar_width = (int)(np.shape(camera_frame)[1]/2.5)
            tar_height = (int)(np.shape(camera_frame)[0]/2.5)
            camera_frame = cv2.resize(camera_frame, (tar_width, tar_height))
            camera_frame = cv2.flip(camera_frame, 1)
            camera_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
            
            left_up_x = 0
            left_up_y = 0
            right_down_x = 0
            right_down_y = 0
            detections, _ = self.face_detector.run(camera_frame)
            if len(detections) > 0:
                left_up_x,left_up_y,right_down_x,right_down_y,_ = detections[0].astype(int)
            else:
                self.show_img(camera_frame)
                return

            center_x = (left_up_x + right_down_x)//2
            # center_y = (left_up_y + right_down_y)//2
            center_y = int(left_up_y*0.6 + right_down_y*0.4)
            half_length = (right_down_y - left_up_y)//2
            # new_half_length = 3*half_length//2
            new_half_length = (int)(1.6*half_length)
            left_up_x = center_x - new_half_length
            right_down_x = center_x + new_half_length
            left_up_y = center_y - new_half_length
            right_down_y = center_y + new_half_length
            rect_width = right_down_y - left_up_y

            left_up_x_final = left_up_x
            right_down_x_final = right_down_x
            left_up_y_final = left_up_y
            right_down_y_final = right_down_y

            if left_up_x_final < 0:
                left_up_x_final = 0
            if left_up_y_final < 0:
                left_up_y_final = 0
            if right_down_x_final > tar_width:
                right_down_x_final = tar_width
            if right_down_y_final > tar_height:
                right_down_y_final = tar_height

            left_up_x_in_256 = int((left_up_x_final - left_up_x)/rect_width*256.)
            left_up_y_in_256 = int((left_up_y_final - left_up_y)/rect_width*256.)
            right_down_x_in_256 = 256 - int((right_down_x - right_down_x_final)/rect_width*256.)
            right_down_y_in_256 = 256 - int((right_down_y - right_down_y_final)/rect_width*256.)
            print(left_up_x_in_256, left_up_y_in_256, right_down_x_in_256, right_down_y_in_256)
            face_rect_rgb_img = camera_frame[left_up_y_final:right_down_y_final, left_up_x_final:right_down_x_final, :]
            out  = self.face_swap_model.ProcessOneFrame(face_rect_rgb_img, left_up_x_in_256, left_up_y_in_256, right_down_x_in_256, right_down_y_in_256)
            camera_frame[left_up_y_final:right_down_y_final, left_up_x_final:right_down_x_final, :] = out
            self.show_img(camera_frame)


if __name__=='__main__':
    app=QApplication(sys.argv)
    window=MainWindow()
    window.show()
    sys.exit(app.exec_())