import os
import torch
import numpy as np
import pandas as pd
import cv2
import math
import json
import h5py


def l2_loss(user_score, pred_score):
    loss = torch.zeros(user_score.shape[1])
    for user_idx in range(user_score.shape[1]):
        gt_score = torch.from_numpy(np.fromiter(map(lambda x: x, user_score[:, user_idx]), dtype=int))
        loss[user_idx] = ((gt_score - pred_score) ** 2).mean()

    return loss.mean()