import numpy as np
import cv2
import torch
from collections import deque
from TextRecognitionModule.MNIST.digitmodel import DigitRecognizer
from AirWritingModule.model import AirWritingModule


def normalize_img(img):
    img = img.astype(np.float32) / 255.0
    img -= img.mean()
    return img


def normalize_num(img):
    img = img.astype(np.float32) / 255.0
    return img


label_order = ['SingleOne', 'SingleTwo', 'SingleThree', 'SingleFour', 'SingleFive',
               'SingleSix', 'SingleSeven', 'SingleEight', 'SingleNine', 'SingleGood', 'SingleBad']

location = (30, 440)
font = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 2

pts = deque(maxlen=1000)

# Load Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

AWM = AirWritingModule().to(device)
AWM.load_state_dict(torch.load('AWCheckPoint.pt'))
AWM.eval()
t = 0

# Load MNIST
TRM = DigitRecognizer().to(device)
TRM.load_state_dict(torch.load('TRDigitCheckPoint.pt', map_location=device))
TRM.eval()
# Prepare Camera
cap = cv2.VideoCapture(0)

save_img = np.zeros((480, 640)).astype('uint8')

while True:

    retval, color_frame = cap.read()
    color_frame = cv2.flip(color_frame, 1)

    # Convert images to numpy arrays
    imgs = np.zeros((28, 28))
    color_image = np.asanyarray(color_frame)
    input_image = cv2.resize(color_image, (160, 120))
    input_image = normalize_img(input_image)
    input_image = input_image.transpose((2, 0, 1))
    input_image = torch.from_numpy(input_image).float()
    input_image = input_image.to(device)
    input_image = torch.reshape(input_image, (1, 3, 120, 160))
    preds_h, preds_c = AWM(input_image)
    y, x = np.where(preds_h[0, 1, :, :].cpu() == preds_h[0, 1, :, :].max())
    if bool(preds_h[0, 1, :, :].max() > 0.5) and int(preds_c.argmax()) == 0 or int(preds_c.argmax()) == 8:
        cv2.putText(color_image, 'Write', (30, 80), font, fontscale, (0, 255, 0), 10)
        pts.appendleft((x * 8, y * 8))
        for i in range(1, len(pts)):
            if pts[i-1] is None or pts[i] is None:
                continue
            thick = int(10)
            cv2.line(color_image, pts[i-1], pts[i], (0, 0, 255), thick)
            cv2.line(save_img, pts[i-1], pts[i], (255, 255, 255), thick)
        # show_image = cv2.circle(color_image, (x * 8, y * 8), 5, (255, 0, 0), -1)
    elif int(preds_c.argmax()) == 4:
        cv2.putText(color_image, 'Store', (30, 80), font, fontscale, (0, 255, 0), 10)
        cv2.imwrite('./digit_test.png', save_img)
        # cv2.imwrite('./Real/'+str(time.time())+'.png', save_img)
    elif int(preds_c.argmax()) == 5:
        cv2.putText(color_image, 'Delete', (30, 80), font, fontscale, (0, 255, 0), 10)
        pts = deque(maxlen=1000)
        save_img = np.zeros((480, 640)).astype('uint8')
    elif int(preds_c.argmax()) == 2:
        cv2.putText(color_image, 'Recognize', (30, 80), font, fontscale, (0, 255, 0), 10)
        demo_data = cv2.imread('./digit_test.png', cv2.IMREAD_GRAYSCALE)
        temp = np.where(demo_data==255)
        c1, c2, c3, c4 = temp[0].min(), temp[0].max(), temp[1].min(), temp[1].max()
        if c2-c1 >= c4-c3:
            imgc = demo_data[c1-t:c2+t, int(c3/2+c4/2-(c2/2-c1/2))-t:int(c3/2+c4/2+(c2/2-c1/2))+t]
        else:
            imgc = demo_data[int(c1/2+c2/2-(c4/2-c3/2))-t:int(c1/2+c2/2+(c4/2-c3/2))+t, c3-t:c4+t]
        imgt = cv2.resize(imgc, (20, 20))
        imgc = np.zeros((1, 1, 28, 28))
        imgc[:, :, 4:24, 4:24] = imgt
        imgs = imgc[0, 0, :, :]
        imgc = normalize_num(imgc)
        imgc = torch.from_numpy(imgc).float()
        imgc = torch.reshape(imgc, (1, 1, 28, 28))
        pred_n = TRM(imgc.to(device))
        pred_n = pred_n.argmax()
        cv2.putText(color_image, str(pred_n.item()), location, font, fontscale, (255, 0, 0), 10)    

    if preds_h.max() < 0.5:
        print("prepare")
        cv2.putText(color_image, 'Prepare', (30, 80), font, fontscale, (0, 255, 0), 10)
    else:
        print(label_order[int(preds_c.argmax())])
    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', color_image)
    cv2.imshow('Written Num', imgs)
    cv2.waitKey(1)