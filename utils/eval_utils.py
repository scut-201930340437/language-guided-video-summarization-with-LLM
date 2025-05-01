import os
import torch
import numpy as np
import pandas as pd
import cv2
import math
import json
import h5py


def fill_array_same_len(arr1, arr2, segments_idx):
    print('fill_array')
    arr1 = arr1.astype(np.float32)
    arr2 = arr2.astype(np.float32)

    max_len = max(len(arr1), len(arr2))
    fill_num1 = np.mean(arr1[segments_idx[-2]:])
    fill_num2 = np.mean(arr2[segments_idx[-2]:])

    if len(arr1) < max_len:

        arr1 = np.concatenate((arr1, np.ones(max_len - len(arr1), dtype=int)), axis=0)

    elif len(arr2) < max_len:

        arr2 = np.concatenate((arr2, np.ones(max_len - len(arr2), dtype=np.float32) * np.min(arr2)), axis=0)
    
    
    # indices = torch.topk(torch.from_numpy(arr1).to(torch.float32), k=math.ceil(len(arr1) * gt_sumy_rate))[1]
    # gt_indicator = np.zeros(len(arr1), dtype=int)
    # for index in indices:
    #     gt_indicator[index] = 1
    
    # indices = torch.topk(torch.from_numpy(arr2).to(torch.float32), k=math.ceil(len(arr1) * gt_sumy_rate))[1]
    # pred_indicator = np.zeros(len(arr1), dtype=int)
    # for index in indices:
    #     pred_indicator[index] = 1
    
    return arr1, arr2 # gt_indicator, pred_indicator


def TVSum_segmentation(video_name):
    scores, nFrames = read_TVSum_anno(video_name, 0.15)
    segments = {}

    avg_seg_num = 0

    for score in scores:
        # score_list = score.copy().tolist()

        # score_list = list(set(score_list))

        # score_list.sort()

        # print(score_list)

        segment = []
        for i in range(1, len(score)):
            if score[i] != score[i-1]:
                segment.append(i)
                if i not in segments.keys():
                    segments[i] = 1
                else:
                    segments[i] = segments[i] + 1
        
        avg_seg_num = avg_seg_num + len(segment)
    
    avg_seg_num = avg_seg_num // 20

    segments = sorted(segments.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)

    # print(segments)

    # print(nFrames)
    # print(segments[:avg_seg_num])

    # print(avg_seg_num)

    res = []
    # print(avg_seg_num)
    for i in range(avg_seg_num):
        res.append(segments[i][0])


    res.sort()
    # print(res)
    # print(len(res))
    return res

# res = TVSum_segmentation('iVt07TCkFM0')
# print(res)


def zero_one_knapsack(weight, value, w):
    n = len(weight)

    p = [[0 for j in range(w + 1)] for i in range(n)]
    rec = [[0 for j in range(w + 1)] for i in range(n)]
    for j in range(w + 1):
        if weight[0] <= j:
            p[0][j] = value[0]
            rec[0][j] = 1
    for i in range(1, n):
        for j in range(w + 1):
            if weight[i] <= j and value[i] + p[i-1][j-weight[i]] > p[i-1][j]:
                p[i][j] = value[i] + p[i-1][j-weight[i]]
                rec[i][j] = 1
            else:
                p[i][j] = p[i-1][j]

    # print(p[n-1][w])
    # print("choose item ", end="")

    tmp = w
    res = []
    for i in range(n-1, -1, -1):
        if rec[i][tmp] == 1:
            # print(i, end=" ")
            res.append(i)
            tmp -= weight[i]

    return res

def knapsack(score, segments_idx, budget):
    shot_scores = np.zeros(len(segments_idx) - 1)
    shot_weights = np.zeros(len(segments_idx) - 1, dtype=int)

    for i in range(1, len(segments_idx)):
        seg_beg = segments_idx[i - 1]
        seg_end = segments_idx[i]

        shot_scores[i - 1] = np.mean(score[seg_beg:seg_end])
        shot_weights[i - 1] = seg_end - seg_beg
    

    selected_shot_idx = zero_one_knapsack(shot_weights, shot_scores, budget)

    selected_frames_labels = np.zeros(len(score), dtype=int)

    # print('selected_shot_idx')
    # print(selected_shot_idx)

    for shot_idx in selected_shot_idx:
        seg_beg = segments_idx[shot_idx]
        seg_end = segments_idx[shot_idx + 1] - 1

        for index in range(seg_beg, seg_end + 1):
            selected_frames_labels[index] = 1

    return selected_frames_labels

def frame_score_to_shot_score(score, segments_idx):
    shot_scores = np.zeros(len(segments_idx) - 1)
    for i in range(1, len(segments_idx)):
        seg_beg = segments_idx[i - 1]
        seg_end = segments_idx[i]

        shot_scores[i - 1] = np.mean(score[seg_beg:seg_end])

    shot_level_scores = np.zeros(len(score))

    for i in range(1, len(segments_idx)):
        seg_beg = segments_idx[i - 1]
        seg_end = segments_idx[i]
        for j in range(seg_beg, seg_end):
            shot_level_scores[j] = shot_scores[i - 1]
    
    return shot_level_scores