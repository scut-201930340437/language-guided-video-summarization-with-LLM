'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Demo for the evaluation of video summaries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Package to evaluate and plot summarization results
% on the SumMe dataset
%
%%%%%%%%
% publication: Gygli et al. - Creating Summaries from User Videos, ECCV 2014
% author:      Michael Gygli, PhD student, ETH Zurich,
% mail:        gygli@vision.ee.ethz.ch
% date:        05-16-2014
'''
import scipy.io
import scipy
from scipy.stats import rankdata
import warnings
import numpy as np
import os
import cv2
import torch
import math

import clip
from PIL import Image

from utils.read_utils import read_TVSum_anno
from utils.eval_utils import fill_array_same_len, knapsack, frame_score_to_shot_score

# def evaluate_spearman(att, grad_att):
#     """
#     Function that measures Spearman’s correlation coefficient between target logits and output logits:
#     att: [n, m]
#     grad_att: [n, m]
#     """
#     def _rank_correlation_(att_map, att_gd):
#         n = torch.tensor(att_map.shape[1])
#         upper = 6 * torch.sum((att_gd - att_map).pow(2), dim=1)
#         down = n * (n.pow(2) - 1.0)
#         return (1.0 - (upper / down)).mean(dim=-1)

#     att = att.sort(dim=1)[1]
#     grad_att = grad_att.sort(dim=1)[1]
#     correlation = _rank_correlation_(att.float(), grad_att.float())
#     return correlation

def cal_spearAndken_dif_len(pred_frames_score, pred_indicator, user_score, userIdx):
    pred_frames_score_copy = pred_frames_score.copy()

    gt_score = np.fromiter(map(lambda x: x, user_score[:, userIdx]), dtype=int)
    max_gt_score = np.max(gt_score)
    min_gt_score = np.min(gt_score)

    
    pred_frames_score_key = pred_frames_score.copy()
    pred_frames_score_key = pred_frames_score_key[pred_indicator == 1]
    # 归一化
    pred_frames_score_key = (pred_frames_score_key - np.min(pred_frames_score_key)) / (np.max(pred_frames_score_key) - np.min(pred_frames_score_key) + 1e-6)
    # align-with gt range
    pred_frames_score_key = pred_frames_score_key * (max_gt_score - min_gt_score) + min_gt_score

    # 四舍五入
    for i in range(len(pred_frames_score_key)):
        pred_frames_score_key[i] = int(pred_frames_score_key[i] + 0.5)
    
    pred_frames_score_key = pred_frames_score_key.astype(int)

    index = 0
    for i in range(len(pred_indicator)):
        if pred_indicator[i] == 1:
            pred_frames_score_copy[i] = pred_frames_score_key[index]
            index = index + 1
        else:
            pred_frames_score_copy[i] = 0

    # print('dif_gt_score: ')
    # print(gt_score)
    # print('dif_pred_score:')
    # print(pred_frames_score_copy)

    res1, _ = scipy.stats.spearmanr(gt_score, pred_frames_score_copy)
    res2, _ = scipy.stats.kendalltau(rankdata(-gt_score), rankdata(-pred_frames_score_copy))
    
    return res1, res2


def cal_spearAndken_same_len(pred_frames_score, pred_indicator, user_score, userIdx):
    pred_frames_score_copy = pred_frames_score.copy()

    gt_indicator = np.fromiter(map(lambda x: (1 if x > 0 else 0), user_score[:, userIdx]), dtype=int)
    gt_length = sum(gt_indicator)

    gt_score = np.fromiter(map(lambda x: x, user_score[:, userIdx]), dtype=int)
    gt_score = gt_score[gt_indicator == 1] # -----
    max_gt_score = np.max(gt_score)
    min_gt_score = np.min(gt_score)

    # 转成tensor
    # pred_frames_score_tensor = torch.from_numpy(pred_frames_score_copy.copy()).to(torch.float32)
    
    # 取和gt相同长度的帧
    # pred_indicator = np.zeros(len(pred_frames_score), dtype=int)
    # pred_indices = torch.topk(pred_frames_score_tensor, k=gt_length)[1]
    # for index in pred_indices:
    #     pred_indicator[index] = 1
    
    pred_frames_score_copy = pred_frames_score_copy[pred_indicator == 1]
    # pred_frames_score_copy = pred_frames_score_copy[gt_indicator == 1]

    # 归一化
    pred_frames_score_copy = (pred_frames_score_copy - np.min(pred_frames_score_copy)) / (np.max(pred_frames_score_copy) - np.min(pred_frames_score_copy) + 1e-6)
    # align-with gt range
    pred_frames_score_copy = pred_frames_score_copy * (max_gt_score - min_gt_score) + min_gt_score

    # 四舍五入
    for i in range(len(pred_frames_score_copy)):
        pred_frames_score_copy[i] = int(pred_frames_score_copy[i] + 0.5)
    
    pred_frames_score_copy = pred_frames_score_copy.astype(int)

    max_len = max(len(gt_score), len(pred_frames_score_copy))
    if len(gt_score) < max_len:
        fill_num = gt_score[-1]
        gt_score = np.concatenate((gt_score, np.ones(max_len - len(gt_score), dtype=int) * fill_num), axis=0)
    elif len(pred_frames_score_copy) < max_len:
        fill_num = pred_frames_score_copy[-1]
        pred_frames_score_copy = np.concatenate((pred_frames_score_copy, np.ones(max_len - len(pred_frames_score_copy), dtype=int) * fill_num), axis=0)

    # print('same_gt_score: ')
    # print(gt_score)
    # print('same_pred_score:')
    # print(pred_frames_score_copy)

    res1, _ = scipy.stats.spearmanr(gt_score, pred_frames_score_copy)
    res2, _ = scipy.stats.kendalltau(rankdata(-gt_score), rankdata(-pred_frames_score_copy))
    
    if np.isnan(res1):
        res1 = ''
    if np.isnan(res2):
        res2 = ''
    return res1, res2


def evalute_FClip(gt_indicator, pred_indicator, video_path, videoName, clip_model, clip_preprocess, device, frames_features):
    gt_indices = np.nonzero(gt_indicator)[0].astype(int)
    pred_indices = np.nonzero(pred_indicator)[0].astype(int)

    union_indices = np.union1d(gt_indices, pred_indices)

    capture = cv2.VideoCapture(os.path.join(video_path, videoName + '.mp4'))
    PClip = 0.0
    RClip = 0.0

    frame_idx = 0
    with torch.no_grad():
        while True:
            ret, frame = capture.read()
            if ret and (frame_idx in union_indices) and (frame_idx not in frames_features.keys()):
                frame = clip_preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
                frames_features[frame_idx] = clip_model.encode_image(frame)[0]
            elif not ret:
                break
            frame_idx = frame_idx + 1


        # for frame_idx in union_indices:
        #     if frame_idx not in frames_features.keys():
        #         capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        #         ret, frame = capture.read()
        #         if ret:
        #             frame = clip_preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
        #             frames_features[frame_idx] = clip_model.encode_image(frame)[0]

        
        for pred_frame_idx in pred_indices:
            max_similarity = 0.0
            for gt_frame_idx in gt_indices:
                max_similarity = max(max_similarity, torch.cosine_similarity(frames_features[gt_frame_idx], frames_features[pred_frame_idx], dim=0))
            PClip = PClip + max_similarity
        PClip = PClip / len(pred_indices)

        for gt_frame_idx in gt_indices:
            max_similarity = 0.0
            for pred_frame_idx in pred_indices:
                max_similarity = max(max_similarity, torch.cosine_similarity(frames_features[gt_frame_idx], frames_features[pred_frame_idx], dim=0))
            RClip = RClip + max_similarity
        RClip = RClip / len(gt_indices)
    
    
    # with torch.no_grad():
    #     # PClip
    #     for pred_frame_idx in pred_indices:
    #         capture.set(cv2.CAP_PROP_POS_FRAMES, pred_frame_idx)
    #         ret1, pred_frame = capture.read()
    #         if ret1:
    #             pred_frame = clip_preprocess(Image.fromarray(pred_frame)).unsqueeze(0).to(device)
    #             pred_frame_features = clip_model.encode_image(pred_frame)[0]
                
    #             max_similarity = 0.0

    #             for gt_frame_idx in gt_indices:
    #                 capture.set(cv2.CAP_PROP_POS_FRAMES, gt_frame_idx)
    #                 ret2, gt_frame = capture.read()

    #                 if ret2:
    #                     gt_frame = clip_preprocess(Image.fromarray(gt_frame)).unsqueeze(0).to(device)
    #                     gt_frame_features = clip_model.encode_image(gt_frame)[0]
    #                     max_similarity = max(max_similarity, torch.cosine_similarity(gt_frame_features, pred_frame_features, dim=0))

    #             PClip = PClip + max_similarity
    #     PClip = PClip / len(pred_indices)
        
    #     # RClip
    #     for gt_frame_idx in gt_indices:
    #         capture.set(cv2.CAP_PROP_POS_FRAMES, gt_frame_idx)
    #         ret1, gt_frame = capture.read()
    #         if ret1:
    #             gt_frame = clip_preprocess(Image.fromarray(gt_frame)).unsqueeze(0).to(device)
    #             gt_frame_features = clip_model.encode_image(gt_frame)[0]

    #             max_similarity = 0.0

    #             for pred_frame_idx in pred_indices:
    #                 capture.set(cv2.CAP_PROP_POS_FRAMES, pred_frame_idx)
    #                 ret2, pred_frame = capture.read()

    #                 if ret2:
    #                     pred_frame = clip_preprocess(Image.fromarray(pred_frame)).unsqueeze(0).to(device)
    #                     pred_frame_features = clip_model.encode_image(pred_frame)[0]
    #                     max_similarity = max(max_similarity, torch.cosine_similarity(gt_frame_features, pred_frame_features, dim=0))
                
    #             RClip = RClip + max_similarity
    #     RClip = RClip / len(gt_indices)
    
    capture.release()
    return 2 * PClip * RClip / (PClip + RClip), frames_features



def evaluate_SumMe(pred_frames_score, video_path, video_name, clip_model, clip_preprocess, device, segments_idx):
    HOMEDATA = '/mnt/hdd8T/gw/dataset/SumMeandTVSum/SumMe/GT/'
    # Load GT file
    gt_file = HOMEDATA + '/' + video_name + '.mat'
    gt_data = scipy.io.loadmat(gt_file)
    
    user_score = gt_data.get('user_score')
    nFrames = user_score.shape[0]
    nbOfUsers = user_score.shape[1]
   
    # Check inputs
    # if len(pred_selection) < nFrames:
    #     warnings.warn('Pad selection with %d zeros!' % (nFrames - len(pred_selection)))
    #     pred_selection = np.concatenate((pred_selection, np.zeros(nFrames - len(pred_selection), dtype=int)))

    # elif len(pred_selection) > nFrames:
    #     warnings.warn('Crop selection (%d frames) to GT length' % (len(pred_selection) - nFrames))       
    #     pred_selection = pred_selection[:nFrames]
             
    # 归一化
    pred_frames_score = (pred_frames_score - np.min(pred_frames_score)) / (np.max(pred_frames_score) - np.min(pred_frames_score) + 1e-6)
    # 转成tensor
    pred_frames_score_tensor = torch.from_numpy(pred_frames_score.copy()).to(torch.float32)
    # Compute pairwise f-measure, summary length and recall
    user_intersection = np.zeros(nbOfUsers)
    # user_union = np.zeros(nbOfUsers)
    user_lengths = np.zeros(nbOfUsers, dtype=int)
    gt_lengths = np.zeros(nbOfUsers, dtype=int)
    # spearman
    # user_spearman_indices = np.zeros(nbOfUsers)
    user_spearman_indicator = np.zeros(nbOfUsers)
    user_spearman_dif_len = np.zeros(nbOfUsers)
    user_spearman_same_len = np.zeros(0)
    # kendall
    # user_kendall_indices = np.zeros(nbOfUsers)
    user_kendall_indicator = np.zeros(nbOfUsers)
    user_kendall_dif_len = np.zeros(nbOfUsers)
    user_kendall_same_len = np.zeros(0)
    # FClip
    user_FClip_same_len = np.zeros(nbOfUsers)
    user_FClip_dif_len = np.zeros(nbOfUsers)

    frames_features = {}
    # pred_indicator = pred_selections
    for userIdx in range(nbOfUsers):
        gt_indicator = np.fromiter(map(lambda x: (1 if x > 0 else 0), user_score[:, userIdx]), dtype=int)
        gt_length = np.sum(gt_indicator)
        # print("gt_length: " + str(gt_length))

        
        # pred_indicator = np.zeros(nFrames, dtype=int)
        # pred_indices = torch.topk(pred_frames_score_tensor, k=gt_length)[1]
        # for index in pred_indices:
        #     pred_indicator[index] = 1

        pred_indicator = knapsack(pred_frames_score.copy(), segments_idx, gt_length)

        # gt length != pred length
        user_spearman_dif_len[userIdx], user_kendall_dif_len[userIdx] = cal_spearAndken_dif_len(pred_frames_score, pred_indicator, user_score, userIdx)

        # gt length == pred length
        res1, res2 = cal_spearAndken_same_len(pred_frames_score, pred_indicator, user_score, userIdx)
        if type(res1) != str:
            user_spearman_same_len = np.append(user_spearman_same_len, res1)
        
        if type(res2) != str:
            user_kendall_same_len = np.append(user_kendall_same_len, res2)

        # 计算FClip
        # user_FClip_same_len[userIdx], frames_features = evalute_FClip(gt_indicator, pred_indicator, video_path, videoName, clip_model, clip_preprocess, device, frames_features)
        # user_FClip_dif_len[userIdx], frames_features = evalute_FClip(gt_indicator, pred_selections, video_path, videoName, clip_model, clip_preprocess, device, frames_features)

        # f1-score
        user_intersection[userIdx] = np.sum(gt_indicator * pred_indicator)
        # user_union[userIdx] = sum(np.fromiter(map(lambda x: (1 if x > 0 else 0), gt_indicator + pred_indicator), dtype=int))    
        
        gt_lengths[userIdx] = np.sum(gt_indicator)
        user_lengths[userIdx] = np.sum(pred_indicator)

    recall = user_intersection / gt_lengths
    p = user_intersection / user_lengths

    f_measure = np.zeros(len(p))
    for idx in range(len(p)):
        if p[idx] > 0 or recall[idx] > 0:
            f_measure[idx] = 2 * recall[idx] * p[idx] / (recall[idx] + p[idx])

    max_f_measure = np.max(f_measure)
    f_measure = np.mean(f_measure)
    
    nnz_idx = np.nonzero(pred_selections)
    nbNNZ = len(nnz_idx[0])
         
    summary_rate = float(nbNNZ) / float(len(pred_selections))

    # spearman 
    # spearman_cor_indices = np.mean(user_spearman_indices)
    # max_spearman_cor_indices = np.max(user_spearman_indices)
    
    # spearman_indicator = np.mean(user_spearman_indicator)
    # max_spearman_indicator = np.max(user_spearman_indicator)

    mean_spearman_dif_len = np.mean(user_spearman_dif_len)
    max_spearman_dif_len = np.max(user_spearman_dif_len)

    if len(user_spearman_same_len) == 0:
        mean_spearman_same_len = 0
        max_spearman_same_len = 0
    else:
        mean_spearman_same_len = np.mean(user_spearman_same_len)
        max_spearman_same_len = np.max(user_spearman_same_len)

    # kendall
    # kendall_cor_indices = np.mean(user_kendall_indices)
    # max_kendall_cor_indices = np.max(user_kendall_indices)

    # kendall_indicator = np.mean(user_kendall_indicator)
    # max_kendall_indicator = np.max(user_kendall_indicator)

    mean_kendall_dif_len = np.mean(user_kendall_dif_len)
    max_kendall_dif_len = np.max(user_kendall_dif_len)

    if len(user_kendall_same_len) == 0:
        kendall_same_len = 0
        max_kendall_same_len = 0
    else:
        mean_kendall_same_len = np.mean(user_kendall_same_len)
        max_kendall_same_len = np.max(user_kendall_same_len)

    # FClip
    # mean_FClip_dif_len = np.mean(user_FClip_dif_len)
    # max_FClip_dif_len = np.max(user_FClip_dif_len)

    # mean_FClip_same_len = np.mean(user_FClip_same_len)
    # max_FClip_same_len = np.max(user_FClip_same_len)
     
    return {
        'mean_f1': f_measure, 
        'max_f1': max_f_measure, 

        'mean_spearman_dif_len': mean_spearman_dif_len,
        'max_spearman_dif_len': max_spearman_dif_len,
        'mean_kendall_dif_len': mean_kendall_dif_len,
        'max_kendall_dif_len': max_kendall_dif_len,

        'mean_spearman_same_len': mean_spearman_same_len,
        'max_spearman_same_len': max_spearman_same_len,
        'mean_kendall_same_len': mean_kendall_same_len,
        'max_kendall_same_len': max_kendall_same_len

        # 'spear_indicator': spearman_indicator, 
        # 'max_spear_indicator': max_spearman_indicator, 
        # 'ken_indicator': kendall_indicator, 
        # 'max_ken_indicator': max_kendall_indicator, 

        # 'mean_FClip_dif_len': mean_FClip_dif_len, 
        # 'max_FClip_dif_len': max_FClip_dif_len,
        # 'mean_FClip_same_len': mean_FClip_same_len, 
        # 'max_FClip_same_len': max_FClip_same_len

        # 'summary_rate': summary_rate
    }, pred_indicator


def evaluate_TVSum(pred_frames_score, video_path, video_name, clip_model, clip_preprocess, device, gt_sumy_rate, segments_idx):
    np.set_printoptions(threshold=10000)

    # Load GT
    gt_scores, nFrames = read_TVSum_anno(video_name, gt_sumy_rate)
    nbOfUsers = 20
   
    # Check inputs
    # if len(pred_selection) < nFrames:
    #     warnings.warn('Pad selection with %d zeros!' % (nFrames - len(pred_selection)))
        # pred_selection = np.concatenate(pred_selection, np.zeros(nFrames - len(pred_selection), dtype=int))

    # elif len(pred_selection) > nFrames:
    #     warnings.warn('Crop selection (%d frames) to GT length' % (len(pred_selection) - nFrames))       
    #     pred_selection = pred_selection[:nFrames]
             
    # 归一化
    # pred_frames_score = (pred_frames_score - np.min(pred_frames_score)) / (np.max(pred_frames_score) - np.min(pred_frames_score) + 1e-6)
    # Compute pairwise f-measure, summary length and recall
    user_intersection = np.zeros(nbOfUsers, dtype=int)
    # user_union = np.zeros(nbOfUsers)
    user_length = np.zeros(nbOfUsers, dtype=int)
    gt_length = np.zeros(nbOfUsers, dtype=int)
    # spearman
    user_spearman_dif_len = np.zeros(nbOfUsers)
    user_spearman_same_len = np.zeros(0)
    # kendall
    user_kendall_dif_len = np.zeros(nbOfUsers)
    user_kendall_same_len = np.zeros(0)
    # FClip
    # user_FClip = np.zeros(nbOfUsers)

    frames_features = {}

    if len(pred_frames_score) < nFrames:
        pred_frames_score = np.concatenate((pred_frames_score, np.ones(nFrames - len(pred_frames_score)) * np.mean(pred_frames_score[segments_idx[-2:]])), axis=0)

    for userIdx in range(nbOfUsers):
        # gt_score
        gt_score = gt_scores[userIdx].copy()
        # pred_score
        pred_frames_score_copy = pred_frames_score.copy()

        # gt的长度和pred不一样(opencv读出的数与标注的帧数不一样)
        if len(gt_score) != len(pred_frames_score_copy):
            gt_score, pred_frames_score_copy = fill_array_same_len(gt_score, pred_frames_score_copy, segments_idx)
        
        # 0/1背包
        segments_idx[-1] = len(gt_score)
        gt_indicator = knapsack(gt_score, segments_idx, math.ceil(gt_sumy_rate * len(gt_score)))
        pred_indicator = knapsack(pred_frames_score_copy, segments_idx, math.ceil(gt_sumy_rate * len(pred_frames_score_copy)))

        # gt_score = frame_score_to_shot_score(gt_score, segments_idx)
        # max_gt_score = np.max(gt_score)
        # min_gt_score = np.min(gt_score)

        pred_frames_score_copy = frame_score_to_shot_score(pred_frames_score_copy, segments_idx)
        # 归一化
        # pred_frames_score_copy = (pred_frames_score_copy - np.min(pred_frames_score_copy)) / (np.max(pred_frames_score_copy) - np.min(pred_frames_score_copy) + 1e-6)
        # align-with gt range
        # pred_frames_score_copy = pred_frames_score_copy * (max_gt_score - min_gt_score) + min_gt_score

        # 四舍五入
        # for i in range(len(pred_frames_score_copy)):
        #     pred_frames_score_copy[i] = int(pred_frames_score_copy[i] + 0.5)
        
        # pred_frames_score_copy = pred_frames_score_copy.astype(int)

        # gt_score = gt_score * gt_indicator
        # pred_frames_score_copy = pred_frames_score_copy * pred_indicator

        gt_score = (gt_score - np.min(gt_score)) / (np.max(gt_score) - np.min(gt_score) + 1e-6)
        pred_frames_score_copy = (pred_frames_score_copy - np.min(pred_frames_score_copy)) / (np.max(pred_frames_score_copy) - np.min(pred_frames_score_copy) + 1e-6)

        gt_rank = rankdata(-gt_score)
        pred_rank = rankdata(-pred_frames_score_copy)

        # gt_rank = (gt_rank - np.min(gt_rank)) / (np.max(gt_rank) - np.min(gt_rank) + 1e-6)
        # pred_rank = (pred_rank - np.min(pred_rank)) / (np.max(pred_rank) - np.min(pred_rank) + 1e-6)

        # print('gt_length:', len(gt_score))
        # print('gt_rank.')
        # print(gt_rank)

        # print('pred_length:', len(pred_frames_score_copy))
        # print('pred_rank.')
        # print(pred_rank)

        user_spearman_dif_len[userIdx], _ = scipy.stats.spearmanr(gt_score, pred_frames_score_copy)
        user_kendall_dif_len[userIdx], _ = scipy.stats.kendalltau(gt_rank, pred_rank)


        # selected scores
        # gt_score = gt_scores[userIdx].copy()
        # pred_frames_score_copy = pred_frames_score.copy()
        # if len(gt_score) != len(pred_frames_score_copy):
        #     gt_score, pred_frames_score_copy = fill_array_same_len(gt_score, pred_frames_score_copy)
        
        

        gt_score = gt_score[gt_indicator == 1]
        gt_score = (gt_score - np.min(gt_score)) / (np.max(gt_score) - np.min(gt_score) + 1e-6)
        # max_gt_score = np.max(gt_score)
        # min_gt_score = np.min(gt_score)

        pred_frames_score_copy = pred_frames_score_copy[pred_indicator == 1]
        # pred_frames_score_copy = pred_frames_score_copy[gt_indicator == 1]

        # 归一化
        pred_frames_score_copy = (pred_frames_score_copy - np.min(pred_frames_score_copy)) / (np.max(pred_frames_score_copy) - np.min(pred_frames_score_copy) + 1e-6)
        # align-with gt range
        # pred_frames_score_copy = pred_frames_score_copy * (max_gt_score - min_gt_score) + min_gt_score
        
        # 四舍五入
        # for i in range(len(pred_frames_score_copy)):
        #     pred_frames_score_copy[i] = int(pred_frames_score_copy[i] + 0.5)
        
        # pred_frames_score_copy = pred_frames_score_copy.astype(int)

        max_len = max(len(gt_score), len(pred_frames_score_copy))
        if len(gt_score) < max_len:
            fill_num = 0.0
            gt_score = np.concatenate((gt_score, np.ones(max_len - len(gt_score), dtype=np.float32) * fill_num), axis=0)
        elif len(pred_frames_score_copy) < max_len:
            fill_num = 0.0
            pred_frames_score_copy = np.concatenate((pred_frames_score_copy, np.ones(max_len - len(pred_frames_score_copy), dtype=np.float32) * fill_num), axis=0)

        gt_rank = rankdata(-gt_score)
        pred_rank = rankdata(-pred_frames_score_copy)

        # gt_rank = (gt_rank - np.min(gt_rank)) / (np.max(gt_rank) - np.min(gt_rank) + 1e-6)
        # pred_rank = (pred_rank - np.min(pred_rank)) / (np.max(pred_rank) - np.min(pred_rank) + 1e-6)

        # print('gt_length:', len(gt_score))
        # print('gt_rank.')
        # print(gt_rank)

        # print('pred_length:', len(pred_frames_score_copy))
        # print('pred_rank.')
        # print(pred_rank)

        res1, _ = scipy.stats.spearmanr(gt_score, pred_frames_score_copy)
        res2, _ = scipy.stats.kendalltau(gt_rank, pred_rank)
        if not np.isnan(res1):
            user_spearman_same_len = np.append(user_spearman_same_len, res1)
        if not np.isnan(res2):
            user_kendall_same_len = np.append(user_kendall_same_len, res2)

        # 计算FClip
        # user_FClip[userIdx], frames_features = evalute_FClip(gt_indicator, pred_indicator, video_path, videoName, clip_model, clip_preprocess, device, frames_features)

        # f1-score
        user_intersection[userIdx] = np.sum(gt_indicator * pred_indicator)
        # user_union[userIdx] = sum(np.fromiter(map(lambda x: (1 if x > 0 else 0), gt_indicator + pred_indicator), dtype=int))    
                  
        gt_length[userIdx] = np.sum(gt_indicator)
        user_length[userIdx] = np.sum(pred_indicator)

    recall = user_intersection / gt_length
    p = user_intersection / user_length

    f_measure = np.zeros(len(p))
    for idx in range(len(p)):
        if p[idx] > 0 or recall[idx] > 0:
            f_measure[idx] = 2 * recall[idx] * p[idx] / (recall[idx] + p[idx])

    max_f_measure = np.max(f_measure)
    f_measure = np.mean(f_measure)
    
    nnz_idx = np.nonzero(pred_indicator)
    nbNNZ = len(nnz_idx[0])
         
    summary_rate = float(nbNNZ) / float(len(pred_indicator))
       
    recall = np.mean(recall)
    p = np.mean(p)

    # spearman 
    # spearman_cor_indices = np.mean(user_spearman_indices)
    # max_spearman_cor_indices = np.max(user_spearman_indices)
    
    # spearman_indicator = np.mean(user_spearman_indicator)
    # max_spearman_indicator = np.max(user_spearman_indicator)

    print('spearman: ')
    print(user_spearman_dif_len)
    print('kendall: ')
    print(user_kendall_dif_len)

    mean_spearman_dif_len = np.mean(user_spearman_dif_len)
    max_spearman_dif_len = np.max(user_spearman_dif_len)

    if len(user_spearman_same_len) == 0:
        mean_spearman_same_len = 0
        max_spearman_same_len = 0
    else:
        mean_spearman_same_len = np.mean(user_spearman_same_len)
        max_spearman_same_len = np.max(user_spearman_same_len)

    # kendall
    # kendall_cor_indices = np.mean(user_kendall_indices)
    # max_kendall_cor_indices = np.max(user_kendall_indices)

    # kendall_indicator = np.mean(user_kendall_indicator)
    # max_kendall_indicator = np.max(user_kendall_indicator)

    mean_kendall_dif_len = np.mean(user_kendall_dif_len)
    max_kendall_dif_len = np.max(user_kendall_dif_len)

    if len(user_kendall_same_len) == 0:
        mean_kendall_same_len = 0
        max_kendall_same_len = 0
    else:
        mean_kendall_same_len = np.mean(user_kendall_same_len)
        max_kendall_same_len = np.max(user_kendall_same_len)

    # FClip
    # mean_FClip = np.mean(user_FClip)
    # max_FClip = np.max(user_FClip)
     
    return {
        'mean_f1': f_measure, 
        'max_f1': max_f_measure, 

        'mean_spearman_dif_len': mean_spearman_dif_len,
        'max_spearman_dif_len': max_spearman_dif_len,
        'mean_kendall_dif_len': mean_kendall_dif_len,
        'max_kendall_dif_len': max_kendall_dif_len,

        'mean_spearman_same_len': mean_spearman_same_len,
        'max_spearman_same_len': max_spearman_same_len,
        'mean_kendall_same_len': mean_kendall_same_len,
        'max_kendall_same_len': max_kendall_same_len

        # 'spear_indicator': spearman_indicator, 
        # 'max_spear_indicator': max_spearman_indicator, 
        # 'ken_indicator': kendall_indicator, 
        # 'max_ken_indicator': max_kendall_indicator, 

        # 'mean_FClip_dif_len': mean_FClip_dif_len, 
        # 'max_FClip_dif_len': max_FClip_dif_len,
        # 'mean_FClip_same_len': mean_FClip_same_len, 
        # 'max_FClip_same_len': max_FClip_same_len

        # 'summary_rate': summary_rate
    }, pred_indicator


def cal_metric_from_txt():
    metric = {}

    dataset_name = 'TVSum'
    path = './result/' + dataset_name + '/'

    metric_file = 'metric.txt'

    with open(path + metric_file, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            line = line.split(' ')

            if len(line) < 2:
                continue

            metric_name = line[-2]
            metric_value = float(line[-1])

            if metric_name not in metric.keys():
                metric[metric_name] = []
            
            metric[metric_name].append(metric_value)

    with open(path + metric_file, 'a') as f:
        for metric_name, v in metric.items():
            mean_v = np.mean(np.array(v, dtype=float))

            print(metric_name + " " + str(mean_v) + "\n")
            f.write(metric_name + " " + str(mean_v) + "\n")


# cal_metric_from_txt()