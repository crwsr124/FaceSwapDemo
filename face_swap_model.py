import torch
from torchvision import transforms
import numpy as  np
import cv2
import time

import sys
sys.path.append("./model")
# import model.downsample
# import model.mobilenetv3
# import model.modulated_conv2d
# import model.crnet_small

def normalize(x):
    mean = np.mean(x[:])
    std = np.std(x[:])
    return (x-mean)/std

def denormalize(x):
    min = np.min(x[:])
    max = np.max(x[:])
    return ((x-min)/max)

def denorm(x):
    return x * 0.5 + 0.5

def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1,2,0)

def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

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

    def __del__(self):
        cv2.destroyAllWindows()
    
    def ProcessOneFrame(self, rgb_img_uint8, left_up_x_in_256, left_up_y_in_256, right_down_x_in_256, right_down_y_in_256):
        self.frame_num = self.frame_num+1

        frame = np.zeros(shape=(256, 256, 3), dtype=np.uint8)
        tmp = cv2.resize(rgb_img_uint8, dsize=(right_down_x_in_256-left_up_x_in_256, right_down_y_in_256-left_up_y_in_256))
        frame[left_up_y_in_256:right_down_y_in_256, left_up_x_in_256:right_down_x_in_256, :] = tmp
        cv2.imshow("tmp", tmp)
        cv2.imshow("kkk", frame)

        img_tensor = transform(frame)
        img_tensor = torch.reshape(img_tensor, (1, 3, 256, 256))
        
        if self.frame_num%20 == 0:
            self.style = torch.randn(1, 8)
            # style = torch.clamp(style, -0.6, 0.6)
            # style = torch.tensor({0,0,0,0,0,0,0,0})
            
        start_time = time.time()
        content, _ = self.encoder(img_tensor)
        out, alpha = self.decoder(content, self.style)
        elapse_time = time.time() - start_time 
        # print("time_cost:", elapse_time)

        # out = out.detach().numpy()
        # out = np.reshape(out, (3, 256, 256))
        # out = out.transpose((1,2,0))
        out = torch.reshape(out, (3, 256, 256))
        out = tensor2numpy(denorm(out))
        out = np.clip(out, 0, 1)
        out = out*255.
        out = out.astype(np.uint8)


        alpha = torch.reshape(alpha, (256, 256))
        # print(out)
        # alpha = denorm(alpha).detach().cpu().numpy()
        alpha = alpha.detach().cpu().numpy()
        # print(np.min(alpha))
        alpha2 = cv2.blur(alpha, (7, 7))
        alpha2 = np.reshape(alpha2, (256, 256, 1))

        for j in range(30):
            alpha2[255-j, :, :] = (0 + j)/30.*alpha2[255-j, :, :]
        for j in range(10):
            alpha2[j, :, :] = (0 + j)/10.*alpha2[j, :, :]

        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out = alpha2*out + (1.-alpha2)*frame
        # out = out.astype(np.uint8)

        temp = out[left_up_y_in_256:right_down_y_in_256, left_up_x_in_256:right_down_x_in_256, :]
        out = cv2.resize(temp, dsize=(np.shape(rgb_img_uint8)[1], np.shape(rgb_img_uint8)[0]))

        
        return out
        