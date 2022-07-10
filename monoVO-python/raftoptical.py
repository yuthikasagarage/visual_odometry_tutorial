import os
import sys
sys.path.append('RAFT/core')

import cv2
import numpy as np
import torch
from argparse import ArgumentParser
from raft import RAFT
from utils import flow_viz


def frame_preprocess(frame, device):
    frame = torch.from_numpy(frame).permute(2, 0, 1).float()
    frame = frame.unsqueeze(0)
    frame = frame.to(device)
    return frame








