import os
import datetime
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from data import FingerTipGestureDataset
from model import AirWritingModule
import torch.nn as nn
import tensorboardX
import logging
from torchsummary import summary
import pdb
import time


logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Train FingertipDetection')

    # Dataset & Data & Training
    parser.add_argument('--dataset-path', type=str, default='../../dataset', help='Path to dataset')
    parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--num-workers', type=int, default=4, help='Dataset workers')

    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--val-batches', type=int, default=8, help='Validation Batches')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--wd', type=float, default=0.0005, help='Weight Decay')

    # Logging etc.
    parser.add_argument('--outdir', type=str, default='output/', help='Training Output Directory')
    parser.add_argument('--logdir', type=str, default='tensorboard/', help='Log directory')
    parser.add_argument('--train-log-step', type=int, default=1000, help='Train Log Step')

    args = parser.parse_args()
    return args


args = parse_args()


def train(epoch, model, device, train_data, optimizer, criterion_h, criterion_c, train_log_step):

    results = {
        'loss_h' : 0,
        'loss_c' : 0
    }

    model.train()
    total_train = len(train_data)

    tl = []
    for i, (x, yh, yc) in enumerate(train_data):
        xd = x.to(device)
        yhd = yh.to(device)
        ycd = yc.to(device)
        st = time.time()
        pred_h, pred_c = model(xd)
        end = time.time()
        tl.append(end-st)
        loss_h = criterion_h(pred_h, yhd)
        loss_c = criterion_c(pred_c, ycd)
        loss = loss_h + loss_c
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % train_log_step == 0:
            logging.info('Epoch [{}/{}], Step [{}/{}], Loss_H: {:.4f}, Loss_C: {:.4f}'
                         .format(epoch, args.epochs, i, total_train, loss_h.item() / args.batch_size, loss_c.item()))
            print('average : ' + str(sum(tl)/len(tl)))
        try:
            results['loss_h'] += loss_h.cpu().item() / args.batch_size
            results['loss_c'] += loss_c.cpu().item()
        except:
            pdb.set_trace()
    results['loss_h'] /= total_train
    results['loss_c'] /= total_train

    return results


def validate(model, device, val_data, criterion_h, criterion_c):

    results = {
        'loss_h' : 0,
        'loss_c' : 0,
        'accuracy' : 0,
        'pixel' : 0
    }

    model.eval()
    total_valid = len(val_data)

    with torch.no_grad():
        for i, (x, yh, yc) in enumerate(val_data):
            xd = x.to(device)
            yhd = yh.to(device)
            ycd = yc.to(device)
            pred_h, pred_c = model(xd)
            loss_h = criterion_h(pred_h, yhd)
            loss_c = criterion_c(pred_c, ycd)

            if pred_c.argmax().item() == yc.item():
                results['accuracy'] += 1 / total_valid

            num_fin = 0
            pixel_buffer = 0
            for j in range(pred_h.shape[1]):
                pixel_temp = pred_h[0, j, :, :].cpu()
                gt_temp = yh[0, j, :, :]
                if gt_temp.max() != 0:
                    num_fin += 1
                    pred_y, pred_x = np.unravel_index(np.argmax(pixel_temp, axis=None), pixel_temp.shape)
                    gt_y, gt_x = np.unravel_index(np.argmax(gt_temp, axis=None), gt_temp.shape)
                    pixel_buffer += np.sqrt((gt_y-pred_y) ** 2 + (gt_x-pred_x) ** 2)
            results['pixel'] += pixel_buffer / num_fin / total_valid * 8
            results['loss_h'] += loss_h.cpu().item() / total_valid
            results['loss_c'] += loss_c.cpu().item() / total_valid

    return results


def run():
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    save_folder = os.path.join(args.outdir, dt)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    tb = tensorboardX.SummaryWriter(os.path.join(args.logdir, dt))

    logging.info('Loading Dataset')
    train_dataset = FingerTipGestureDataset(args.dataset_path, start=0.0, end=args.split)
    train_data = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataset = FingerTipGestureDataset(args.dataset_path, start=args.split, end=1.0)
    val_data = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    logging.info('Loading Dataset Finished')

    logging.info('Loading Network')
    model = AirWritingModule()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion_h = nn.MSELoss(reduction='sum')
    criterion_c = nn.CrossEntropyLoss()
    logging.info('Loading Network Finished')
    summary(model, (3, 120, 160))

    for epoch in range(args.epochs):
        logging.info('Beginning Epoch {:02d}'.format(epoch))
        train_results = train(epoch, model, device, train_data, optimizer, criterion_h, criterion_c, args.train_log_step)
        tb.add_scalar('loss_h/train_loss', train_results['loss_h'], epoch)
        tb.add_scalar('loss_c/train_loss', train_results['loss_c'], epoch)

        logging.info('Validating...')
        test_results = validate(model, device, val_data, criterion_h, criterion_c)
        logging.info('Validataion Loss_H: {:.4f}, LOSS_C: {:.4f}, Accuracy: {:.4f}, Pixel:{:.4f}'.format(test_results['loss_h'], test_results['loss_c'], test_results['accuracy'], test_results['pixel']))
        tb.add_scalar('loss_h/val_loss', test_results['loss_h'], epoch)
        tb.add_scalar('loss_c/val_loss', test_results['loss_c'], epoch)
        tb.add_scalar('accuracy', test_results['accuracy'], epoch)
        tb.add_scalar('pixel', test_results['pixel'], epoch)
        torch.save(model.state_dict(), os.path.join(save_folder, 'epoch_%02d.pt' % (epoch)))

if __name__ == '__main__':
    run()