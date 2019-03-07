from __future__ import print_function

import argparse
import os
import pickle
import json
import math

import numpy as np
import torch
from scipy import misc
import time
from tqdm import tqdm

from src.i3dpt import I3D

FPS = 25
MAX_INTERVAL = 400
OVERLAP = 25
rgb_pt_checkpoint = 'model/model_rgb.pth'


def get_features(sample, model):
    sample = sample.transpose(0, 4, 1, 2, 3)
    sample_var = torch.autograd.Variable(torch.from_numpy(sample).cuda(), volatile=True)
    out_var = model.extract(sample_var)
    out_tensor = out_var.data.cpu()
    return out_tensor.numpy()


def read_video(video_dir):
    # start = time.time()
    frames = [f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]
    data = []
    for i, frame in enumerate(sorted(frames)):
        I = misc.imread(os.path.join(video_dir, frame), mode='RGB')
        if len(I.shape) == 2:
            I = I[:, :, np.newaxis]
            I = np.concatenate((I, I, I), axis=2)
        I = (I.astype('float32') / 255.0 - 0.5) * 2
        data.append(I)
    res = np.asarray(data)[np.newaxis, :, :, :, :]
    # print("load time: ", time.time() - start)
    return res


def run(args):
    # Run RGB model
    i3d_rgb = I3D(num_classes=400, modality='rgb')
    i3d_rgb.eval()
    i3d_rgb.load_state_dict(torch.load(args.rgb_weights_path))
    i3d_rgb.cuda()

    video_list = os.listdir(args.video_dir)

    for vid in tqdm(video_list):
        video = os.path.join(args.video_dir, vid)
        clip = read_video(video)
        clip_len = clip.shape[1]
        if clip_len <= MAX_INTERVAL:
            features = get_features(clip, i3d_rgb)
        else:
            tmp_1 = 0
            features = []
            while True:
                tmp_2 = tmp_1 + MAX_INTERVAL
                tmp_2 = min(tmp_2, clip_len)
                feat = get_features(clip[:, tmp_1:tmp_2], i3d_rgb)
                features.append(feat)
                if tmp_2 == clip_len:
                    break
                tmp_1 = max(0, tmp_2 - OVERLAP)
            features = np.concatenate(features, axis=1)
        np.save(os.path.join(args.out_dir, args.split, vid), features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Runs inflated inception v1 network on\
    cricket sample from tensorflow demo (generate the network weights with\
    i3d_tf_to_pt.py first)')

    # RGB arguments
    parser.add_argument(
        '--rgb', action='store_true', help='Evaluate RGB pretrained network')
    parser.add_argument(
        '--rgb_weights_path',
        type=str,
        default='model/model_rgb.pth',
        help='Path to rgb model state_dict')
    parser.add_argument(
        '--rgb_sample_path',
        type=str,
        default='data/kinetic-samples/v_CricketShot_g04_c01_rgb.npy',
        help='Path to kinetics rgb numpy sample')

    parser.add_argument('--video_dir', type=str, default='/mnt/kobe-shared/xwang/VATEX/val')
    parser.add_argument('--out_dir', type=str, default="/mnt/bhd/xwang/vatex_features")
    parser.add_argument('--split', type=str, default="train_val", help="train_val | test")
    args = parser.parse_args()
    run(args)
