import os
import sys
import json
from datetime import datetime
from statistics import mean
import argparse

import numpy as np
import cv2

import torch
import torchvision

import datasets.transforms as tf

from models.dla.pose_dla_dcn import get_pose_net
from models.erfnet.erfnet import ERFNet
from models.enet.ENet import ENet

from utils.affinity_fields import decodeAFs
from utils.visualize import tensor2image, create_viz

parser = argparse.ArgumentParser('Options for inference with LaneAF models in PyTorch...')
parser.add_argument('--dataset-dir', type=str, default=None, help='path to dataset')
parser.add_argument('--output-dir', type=str, default=None, help='output directory for model and logs')
parser.add_argument('--snapshot', type=str, default=None, help='path to pre-trained model snapshot')
parser.add_argument('--split', type=str, default='test', help='dataset split to evaluate on (train/val/test)')
parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--no-cuda', action='store_true', default=False, help='do not use cuda for training')
parser.add_argument('--save-viz', action='store_true', default=True,
                    help='save visualization depicting intermediate and final results')
args = parser.parse_args()
args.backbone = "dla34"

global mean,std
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# check args
if args.dataset_dir is None:
    assert False, 'Path to dataset not provided!'
if args.snapshot is None:
    assert False, 'Model snapshot not provided!'
if args.split is ['train', 'val', 'test']:
    assert False, 'Incorrect dataset split provided!'

# set batch size to 1 for visualization purposes
args.batch_size = 1

# setup args
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.output_dir is None:
    args.output_dir = datetime.now().strftime("%Y-%m-%d-%H:%M-infer")
    args.output_dir = os.path.join('.', 'experiments', 'tusimple', args.output_dir)

# load args used from training snapshot (if available)
if os.path.exists(os.path.join(os.path.dirname(args.snapshot), 'config.json')):
    with open(os.path.join(os.path.dirname(args.snapshot), 'config.json')) as f:
        json_args = json.load(f)
    # augment infer args with training args for model consistency
    if 'backbone' in json_args.keys():
        args.backbone = json_args['backbone']
    else:
        args.backbone = 'dla34'

# set random seed
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 1}

class Process(object):
    def __init__(self,frame):
        self.frame = frame
        self.mean = [0.485, 0.456, 0.406] #[103.939, 116.779, 123.68]
        self.std = [0.229, 0.224, 0.225] #[1, 1, 1]
        self.transforms = torchvision.transforms.Compose([
            tf.GroupRandomScale(size=(0.5, 0.5), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
            tf.GroupNormalize(mean=(self.mean, (0,)), std=(self.std, (1,))),
        ])
global transforms
transforms = torchvision.transforms.Compose([
            tf.GroupRandomScale(size=(0.5, 0.5), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
            tf.GroupNormalize(mean=(mean, (0,)), std=(std, (1,))),
        ])

def preprocess(input_frame):
    frame = cv2.resize(input_frame,(1280,720))
    img = frame.astype(np.float32) / 255.  # (H, W, 3)
    img = cv2.cvtColor(img[16:, :, :], cv2.COLOR_BGR2RGB)
    img, _ = transforms((img, img))

    # convert all outputs to torch tensors
    img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
    img = img.reshape([1,3,352,640])
    seg, mask, af = torch.tensor(float('nan')), torch.tensor(float('nan')), torch.tensor(float('nan'))

    # convert all outputs to torch tensors
    return img, seg, mask, af

# test function
def test(net,videopath):
    net.eval()
    cap = cv2.VideoCapture(videopath)
    for i in range(1250):
        cap.read()
    while True:
        ret,frame = cap.read()
        if ret:
            input_img,input_seg,input_mask,input_af = preprocess(frame)
            if args.cuda:
                input_img = input_img.cuda()
                input_seg = input_seg.cuda()
                input_mask = input_mask.cuda()
                input_af = input_af.cuda()

            st_time = datetime.now()
            # do the forward pass

            outputs = net(input_img)[-1]

            # convert to arrays
            mask_out = tensor2image(torch.sigmoid(outputs['hm']).repeat(1, 3, 1, 1).detach(),
                                    np.array([0.0 for _ in range(3)], dtype='float32'),
                                    np.array([1.0 for _ in range(3)], dtype='float32'))
            img = tensor2image(input_img.detach(), np.array(mean),
                               np.array(std))
            vaf_out = np.transpose(outputs['vaf'][0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))
            haf_out = np.transpose(outputs['haf'][0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))

            # decode AFs to get lane instances
            seg_out = decodeAFs(mask_out[:, :, 0], vaf_out, haf_out, fg_thresh=128, err_thresh=5)
            ed_time = datetime.now()
            if args.save_viz:
                img_out = create_viz(img, seg_out.astype(np.uint8), mask_out, vaf_out, haf_out)
                img_out = cv2.putText(img_out,'fps: %.2f' % (1 / (ed_time - st_time).total_seconds()), (10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                cv2.imshow("Detections", img_out)
                cv2.imshow("mask", cv2.resize(mask_out,(640,480)))
                if cv2.waitKey(10) & 0XFF == ord("q"):
                    break
        else:
            break
    return


if __name__ == "__main__":
    heads = {'hm': 1, 'vaf': 2, 'haf': 1}
    if args.backbone == 'dla34':
        model = get_pose_net(num_layers=34, heads=heads, head_conv=256, down_ratio=4)
    elif args.backbone == 'erfnet':
        model = ERFNet(heads=heads)
    elif args.backbone == 'enet':
        model = ENet(heads=heads)

    model.load_state_dict(torch.load(args.snapshot), strict=True)
    if args.cuda:
        model.cuda()
    print(model)

    test(model,args.dataset_dir)
