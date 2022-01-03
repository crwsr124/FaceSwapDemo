import torch
from torchvision import transforms
import numpy as  np
import cv2
import time
from model.img_process_util import USMSharp

import sys
sys.path.append("./model")
# import model.downsample
# import model.mobilenetv3
# import model.modulated_conv2d
# import model.crnet_small

def denorm(x):
    return x * 0.5 + 0.5

def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1,2,0)

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=True)
    ])

class FaceSwapModel:
    def __init__(self, params = None):
        self.model_path = 'model/'
        self.encoder = torch.load('model/encoder.pkl',map_location=torch.device('cpu'))
        self.decoder = torch.load('model/decoder.pkl',map_location=torch.device('cpu'))
        self.decoder.eval()
        self.encoder.eval()
        self.style = torch.randn(1, 8)
        self.frame_num = 0
        self.sharper = USMSharp()
        self.mask_alpha = cv2.imread("alpha.png")/255.

    def __del__(self):
        cv2.destroyAllWindows()
    
    def ProcessOneFrame(self, rgb_img_uint8, left_up_x_in_256, left_up_y_in_256, right_down_x_in_256, right_down_y_in_256):
        self.frame_num = self.frame_num+1

        frame = np.zeros(shape=(256, 256, 3), dtype=np.uint8)
        tmp = cv2.resize(rgb_img_uint8, dsize=(right_down_x_in_256-left_up_x_in_256+1, right_down_y_in_256-left_up_y_in_256+1))
        frame[left_up_y_in_256:right_down_y_in_256+1, left_up_x_in_256:right_down_x_in_256+1, :] = tmp

        img_tensor = transform(frame)
        img_tensor = torch.reshape(img_tensor, (1, 3, 256, 256))
        if self.frame_num%30 == 0:
            self.style = torch.randn(1, 8)
            # self.style = torch.clamp(self.style, -0.6, 0.6)
            # self.style = 0*self.style
        start_time = time.time()
        content, _ = self.encoder(img_tensor)
        out, alpha = self.decoder(content, self.style)
        elapse_time = time.time() - start_time 
        # print("time_cost:", elapse_time)
        
        out = torch.reshape(out, (3, 256, 256))
        out = tensor2numpy(denorm(out))
        out = np.clip(out, 0, 1)
        out = out*255.
        out = out.astype(np.uint8)

        alpha = torch.reshape(alpha, (256, 256))
        alpha = alpha.detach().cpu().numpy()
        alpha = np.reshape(alpha, (256, 256, 1))
        alpha = alpha * self.mask_alpha
        out = alpha*out + (1.-alpha)*frame
        out = out.astype(np.uint8)

        temp = out[left_up_y_in_256:right_down_y_in_256+1, left_up_x_in_256:right_down_x_in_256+1, :]
        out = cv2.resize(temp, dsize=(np.shape(rgb_img_uint8)[1], np.shape(rgb_img_uint8)[0]))
        # cv2.imshow("out", out)
        return out
        