import numpy as  np
import cv2



alpha = np.ones(shape=(256, 256, 1), dtype=np.float64)


# case 1
# for j in range(30):
#     alpha[255-j, :, :] = (0 + j)/30.*alpha[255-j, :, :]
# for j in range(30):
#     alpha[j, :, :] = (0 + j)/30.*alpha[j, :, :]
# for j in range(30):
#     alpha[:, j, :] = (0 + j)/30.*alpha[:, j, :]
# for j in range(30):
#     alpha[:, 255-j, :] = (0 + j)/30.*alpha[:, 255-j, :]

# case 2
# for j in range(30):
#     alpha[255-j, :, :] = (0 + j)/30.*alpha[255-j, :, :]
# for j in range(10):
#     alpha[j, :, :] = (0 + j)/10.*alpha[j, :, :]
# for j in range(10):
#     alpha[:, j, :] = (0 + j)/10.*alpha[:, j, :]
# for j in range(10):
#     alpha[:, 255-j, :] = (0 + j)/10.*alpha[:, 255-j, :]

for i in range(0, 256):
    for j in range(0, 256):
        x = np.array([i - 255./2], dtype=np.float64)
        y = np.array([j - 255./2], dtype=np.float64)

        rad_min = 255./2 - 30.
        rad = np.sqrt(x*x + y*y)
        if rad > rad_min:
            if np.abs(y) > np.abs(x):
                rad_max = (255./2) / np.abs(y) * rad
            else:
                rad_max = (255./2) / np.abs(x) * rad
            point = (rad_max - rad) / (rad_max - rad_min)
            alpha[j, i, 0] = point

alpha = cv2.blur(alpha, (3, 3), borderType=cv2.BORDER_CONSTANT)
alpha = np.clip(alpha, 0, 1)
alpha = (alpha*255).astype(np.uint8)
print("------------", alpha[255, 128])

cv2.imwrite("alpha.png", alpha)
cv2.imshow("alpha", alpha)
cv2.waitKey(0)
cv2.destroyAllWindows()