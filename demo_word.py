import sys
sys.path.append('./dtrb')
import numpy as np
import cv2
import torch
from collections import deque
from dtrb.model import Model
from dtrb.utils import AttnLabelConverter
from dtrb.dataset import RawDataset, AlignCollate
from AirWritingModule.model import AirWritingModule
import argparse
import torch.nn.functional as F
import pdb


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', default='./dtrb_test', help='path to image_folder which contains text images')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
parser.add_argument('--saved_model', default='TRWordCheckPoint.pth', help="path to saved_model to evaluation")
""" Data processing """
parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
parser.add_argument('--rgb', action='store_true', help='use rgb input')
parser.add_argument('--character', type=str, default='abcdefghijklmnopqrstuvwxyz', help='character label')
parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
""" Model Architecture """
parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
parser.add_argument('--Prediction', type=str, default='Attn', help='Prediction stage. CTC|Attn')
parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

opt = parser.parse_args()

converter = AttnLabelConverter(opt.character)
opt.num_class = len(converter.character)

if opt.rgb:
    opt.input_channel = 3
TRM = Model(opt)
TRM = torch.nn.DataParallel(TRM).to(device)
TRM.load_state_dict(torch.load(opt.saved_model, map_location=device))
TRM.eval()

AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)

def normalize_img(img):
    img = img.astype(np.float32) / 255.0
    img -= img.mean()
    return img


label_order = ['SingleOne', 'SingleTwo', 'SingleThree', 'SingleFour', 'SingleFive',
               'SingleSix', 'SingleSeven', 'SingleEight', 'SingleNine', 'SingleGood', 'SingleBad']


pts = deque(maxlen=1000)

th = 30
# Load Model
AWM = AirWritingModule().to(device)
AWM.load_state_dict(torch.load('AWCheckPoint.pt'))
AWM.eval()

# Prepare Camera
cap = cv2.VideoCapture(0)

save_img = np.zeros((480, 640)).astype('uint8')

location = (30, 440)
font = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 2


while True:

    retval, color_image = cap.read()
    color_image = cv2.flip(color_image, 1)

    # Convert images to numpy arrays
    input_image = cv2.resize(color_image, (160, 120))
    input_image = normalize_img(input_image)
    input_image = input_image.transpose((2, 0, 1))
    input_image = torch.from_numpy(input_image).float()
    input_image = input_image.to(device)
    input_image = torch.reshape(input_image, (1, 3, 120, 160))
    preds_h, preds_c = AWM(input_image)
    y, x = np.where(preds_h[0, 1, :, :].cpu() == preds_h[0, 1, :, :].max())
    if bool(preds_h[0, 1, :, :].max() > 0.2) and int(preds_c.argmax()) == 0:
        cv2.putText(color_image, 'Write', (30, 80), font, fontscale, (0, 255, 0), 10)
        pts.appendleft((x * 8, y * 8))
        for i in range(1, len(pts)):
            if pts[i-1] is None or pts[i] is None:
                continue
            thick = int(13)
            cv2.line(color_image, pts[i-1], pts[i], (0, 0, 255), thick)
            cv2.line(save_img, pts[i-1], pts[i], (255, 255, 255), thick)
        # show_image = cv2.circle(color_image, (x * 8, y * 8), 5, (255, 0, 0), -1)
    elif int(preds_c.argmax()) == 4:
        cv2.putText(color_image, 'Store', (30, 80), font, fontscale, (0, 255, 0), 10)
        # temp = np.where(save_img==255)
        # c1, c2, c3, c4 = temp[0].min(), temp[0].max(), temp[1].min(), temp[1].max()
        # imgc = save_img[c1-th:c2+th, c3-th:c4+th]
        # imgs = cv2.resize(imgc, (100, 32))
        cv2.imwrite('./dtrb_test/test.png', save_img)
    elif int(preds_c.argmax()) == 5:
        cv2.putText(color_image, 'Delete', (30, 80), font, fontscale, (0, 255, 0), 10)
        pts = deque(maxlen=1000)
        save_img = np.zeros((480, 640)).astype('uint8')
        
    elif int(preds_c.argmax()) == 2:
        cv2.putText(color_image, 'Recognize', (30, 80), font, fontscale, (0, 255, 0), 10)
        demo_data = RawDataset(root=opt.image_folder, opt=opt)
        demo_loader = torch.utils.data.DataLoader(
            demo_data, batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_demo, pin_memory=True)
        with torch.no_grad():
            for image_tensors, image_path_list in demo_loader:
                batch_size = image_tensors.size(0)
                image = image_tensors.to(device)
                # For max length prediction
                length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
                text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
                preds = TRM(image, text_for_pred, is_train=False)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)
                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                    if 'Attn' in opt.Prediction:
                        pred_EOS = pred.find('[s]')
                        pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                        pred_max_prob = pred_max_prob[:pred_EOS]

        cv2.putText(color_image, pred, location, font, fontscale, (255, 0, 0), 10)

    if preds_h.max() < 0.2:
        print("prepare")
        cv2.putText(color_image, 'Prepare', (30, 80), font, fontscale, (0, 255, 0), 10)
    else:
        print(label_order[int(preds_c.argmax())])
    cv2.imshow('Air-Text', color_image)
    thu = np.asarray(preds_h[0,0,:,:].detach().cpu().numpy())
    ind = np.asarray(preds_h[0,1,:,:].detach().cpu().numpy())
    mid = np.asarray(preds_h[0,2,:,:].detach().cpu().numpy())
    rin = np.asarray(preds_h[0,3,:,:].detach().cpu().numpy())
    pin = np.asarray(preds_h[0,4,:,:].detach().cpu().numpy())
    fin_temp = np.ones((45,80)).astype('float32')
    cv2.imshow('Fingertip Heatmaps', np.vstack((thu, fin_temp, ind, fin_temp, mid, fin_temp, rin, fin_temp, pin)))
    cv2.waitKey(1)
