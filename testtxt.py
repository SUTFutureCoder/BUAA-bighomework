import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse

import sys
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from loader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer,TXTDataset
from torchvision import datasets, models, transforms
import argparse
import tools

ap     = argparse.ArgumentParser(description='Just for eval.')
ap.add_argument('-N','--testImgNamePth', help='Path to Image Name txt file (optional, see readme)')
ap.add_argument('-A','--testAnnoPth', help='Path to the root of your Annotation (optional, see readme)')
ap.add_argument('-I','--testImgPth', help='Path to the root of your test image (optional, see readme)')
ap.add_argument('-M','--model', help='Path to your best model (optional, see readme)')
args = vars(ap.parse_args())

dataset_val = TXTDataset(testImgNamePth=args['testImgNamePth'], testAnnoPth=args['testAnnoPth'], testImgPth =args['testImgPth'], transform=transforms.Compose([Normalizer(), Resizer()]))
sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)
modelretina = torch.load(args['model'])
modelretina.cuda()
modelretina.eval()
AP = tools.evaluate(dataset_val, modelretina)
map = (AP[0][0]+AP[1][0])/2
print("mAp:"+str(map))
