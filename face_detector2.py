
import numpy as  np
import sys
sys.path.append("./model")
from model.landmarks_detector import LandmarksDetector



class FaceDetectorModel:
    def __init__(self, params = None):
        self.landmarks_detector = LandmarksDetector("model/shape_predictor_68_face_landmarks.dat")
    
    def ProcessOneFrame(self, rgb_img_uint8):
        detections = []
        for i, face_landmarks in enumerate(self.landmarks_detector.get_landmarks(rgb_img_uint8), start=1):
            try:
                lm = np.array(face_landmarks)
                lm_chin          = lm[0  : 17]  # left-right
                lm_eyebrow_left  = lm[17 : 22]  # left-right
                lm_eyebrow_right = lm[22 : 27]  # left-right
                lm_nose          = lm[27 : 31]  # top-down
                lm_nostrils      = lm[31 : 36]  # top-down
                lm_eye_left      = lm[36 : 42]  # left-clockwise
                lm_eye_right     = lm[42 : 48]  # left-clockwise
                lm_mouth_outer   = lm[48 : 60]  # left-clockwise
                lm_mouth_inner   = lm[60 : 68]  # left-clockwise

                # Calculate auxiliary vectors.
                eye_left     = np.mean(lm_eye_left, axis=0)
                eye_right    = np.mean(lm_eye_right, axis=0)
                eye_avg      = (eye_left + eye_right) * 0.5
                eye_to_eye   = eye_right - eye_left
                mouth_left   = lm_mouth_outer[0]
                mouth_right  = lm_mouth_outer[6]
                mouth_avg    = (mouth_left + mouth_right) * 0.5
                eye_to_mouth = mouth_avg - eye_avg
                chin_lowest = lm_chin[8]
                eye_to_chin = chin_lowest - eye_avg

                # eye_to_eye_dis = math.sqrt(eye_to_eye[0]*eye_to_eye[0] + eye_to_eye[1]*eye_to_eye[1])
                # eye_to_mouth_dis = math.sqrt(eye_to_mouth[0]*eye_to_mouth[0] + eye_to_mouth[1]*eye_to_mouth[1])
                # eye_to_chin_dis = math.sqrt(eye_to_chin[0]*eye_to_chin[0] + eye_to_chin[1]*eye_to_chin[1])
                eye_to_chin_dis = np.max(eye_to_chin)
                # center = eye_avg + eye_to_chin * 0.315
                center = eye_avg + eye_to_chin * 0.15
                center = center.astype(int)

                height, width, channels = np.shape(rgb_img_uint8)[0], np.shape(rgb_img_uint8)[1], np.shape(rgb_img_uint8)[2]

                half_side_lengh = eye_to_chin_dis*1.3
                half_side_lengh = np.ceil(half_side_lengh)
                half_side_lengh = half_side_lengh.astype(np.int64)
                # print("half_side_lengh", half_side_lengh)

                # crop cordinate 
                x_left = center[0] - half_side_lengh
                x_right = center[0] + half_side_lengh
                y_up = center[1] - half_side_lengh
                y_down = center[1] + half_side_lengh

                detections.append([x_left, y_up, x_right, y_down])

            except:
                print("Exception in face crop!")
        return detections
        