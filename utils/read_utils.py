import os
import torch
import numpy as np
import pandas as pd
import cv2
import math
import json
import h5py


def read_segmentations(dataset_name, video_name):
    segments_idx = None
    nFrames = None

    with h5py.File('/home/lab345/gw/dataset/SumMeandTVSum/eccv16_dataset_' + dataset_name + '_google_pool5.h5',"r") as f:
        # for key in f.keys():
            #print(f[key], key, f[key].name, f[key].value) # 因为这里有group对象它是没有value属性的,故会异常。另外字符串读出来是字节流，需要解码成字符串。
            # print(f[key], key, f[key].name) # f[key] means a dataset or a group object. f[key].value visits dataset' value,except group object.

        if dataset_name == 'SumMe':
            for key in f.keys():
                group = f[key]
                a = group["video_name"][()]
                if video_name == str(a.decode("utf-8")):
                    segments_idx = np.array(group["change_points"], dtype=int)[:, 0]
                    nFrames = np.array(group["change_points"], dtype=int)[-1][1] + 1
                    break
                    
        elif dataset_name == 'TVSum':
            video_index = read_TVSum_video_index(video_name)
            # print("video_index: ", video_index)
            group = f["video_" + str(video_index)]
            segments_idx = np.array(group["change_points"], dtype=int)[:, 0]
            nFrames = np.array(group["change_points"], dtype=int)[-1][1] + 1

    return segments_idx.tolist() + [nFrames]


def read_summary(dataset_name, video_name):
    user_summary = None

    with h5py.File('/home/lab345/gw/dataset/SumMeandTVSum/eccv16_dataset_' + dataset_name + '_google_pool5.h5',"r") as f:
        # for key in f.keys():
            #print(f[key], key, f[key].name, f[key].value) # 因为这里有group对象它是没有value属性的,故会异常。另外字符串读出来是字节流，需要解码成字符串。
            # print(f[key], key, f[key].name) # f[key] means a dataset or a group object. f[key].value visits dataset' value,except group object.

        if dataset_name == 'SumMe':
            for key in f.keys():
                group = f[key]
                a = group["video_name"][()]
                if video_name == str(a.decode("utf-8")):
                    user_summary = np.array(group["user_summary"], dtype=int)
                    
                    break
                    
        elif dataset_name == 'TVSum':
            video_index = read_TVSum_video_index(video_name)
            # print("video_index: ", video_index)
            group = f["video_" + str(video_index)]
            user_summary = np.array(group["user_summary"], dtype=int)
            
    return user_summary

# def get_anno_SumMe(video_name):
#     HOMEDATA='/mnt/hdd8T/gw/dataset/SumMeandTVSum/SumMe/GT/'
#     # Load GT file
#     gt_file = HOMEDATA + '/' + video_name.split('.')[0] + '.mat'
#     gt_data = scipy.io.loadmat(gt_file)
    
#     user_score = gt_data.get('user_score')
#     nFrames = user_score.shape[0]
#     nbOfUsers = user_score.shape[1]


def read_TVSum_video_index(video_name):
    df = pd.read_csv('/home/lab345/gw/dataset/SumMeandTVSum/TVSum/data/ydata-tvsum50-anno.tsv', header=None, sep = '\t', names=('video_name', 'x', 'gt'))

    video_index = 0
    for name in df['video_name']:
        if name == video_name:
            break
        else:
            video_index = video_index + 1
    
    return video_index // 20 + 1


def read_TVSum_anno(video_name, gt_sumy_rate):
    df = pd.read_csv('/home/lab345/gw/dataset/SumMeandTVSum/TVSum/data/ydata-tvsum50-anno.tsv', header=None, sep = '\t', names=('video_name', 'x', 'gt'))
    nFrames = 0

    video_index = 0
    for name in df['video_name']:
        if name == video_name:
            break
        else:
            video_index = video_index + 1

    scores = []
    for i in range(video_index, video_index + 20):
        df['gt'][i] = df['gt'][i].split(',')
        num = []
        for j in df['gt'][i]:
            num.append(int(j))

        nFrames = len(num)

        scores.append(np.array(num, dtype=int))

    # indicators = []

    # for score in scores:
    #     score_list = score.copy().tolist()
        
    #     score_list = list(set(score_list))

    #     score_list.sort()

    #     max_s = score_list[-1]

    #     max_s2 = score_list[-2]

    #     not_one_num = 0
    #     indicator = np.zeros(len(score), dtype=int)
    #     for i in range(len(score)):
    #         if score[i] == max_s or score[i] == max_s2:
    #             not_one_num = not_one_num + 1
    #             indicator[i] = 1
        
    #     indicators.append(indicator)
        
    return scores, nFrames


def read_Daily_Mail_anno(video_id):
    # 打开annotation 
    with open('/home/lab345/gw/dataset/Daily_Mail/annotation/train.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    print(data[video_id]['video_label'])


def read_blip2_caption(video_name, dataset_name):
    if "video_text_" + video_name.split(".")[0] + ".json" in os.listdir("captions/" + dataset_name + "/"):
        with open("captions/" + dataset_name + "/" + "video_text_" + video_name.split(".")[0] + ".json", 'r') as load_f:
            load_dict = json.load(load_f)
            return load_dict
    else:
        return {}


def read_llama2_sumy(video_name, dataset_name):
    if "sumy_" + video_name.split(".")[0] + ".json" in os.listdir("text_sumy/" + dataset_name + "/"):
        seg_text_sumy_list = []
        with open("text_sumy/" + dataset_name + "/" + "sumy_" + video_name.split(".")[0] + ".json", 'r') as load_f:
            load_dict = json.load(load_f)

            for seg_idx in range(1, 1000000000):
                if str(seg_idx) in load_dict.keys():
                    seg_text_sumy_list.append(load_dict[str(seg_idx)])
                else:
                    break
            
            video_level_sumy = ""
            if 'video_level_sumy' in load_dict.keys():
                video_level_sumy = load_dict['video_level_sumy']

            return seg_text_sumy_list, load_dict, video_level_sumy
    else:
        return [], {}, ''
        

def read_gt_caption(captions, video_name):
    gtscore = None

    with h5py.File('/home/lab345/gw/dataset/SumMeandTVSum/eccv16_dataset_' + dataset_name + '_google_pool5.h5',"r") as f:
        

        if dataset_name == 'SumMe':
            for key in f.keys():
                group = f[key]
                a = group["video_name"][()]
                if video_name == str(a.decode("utf-8")):
                    gtscore = np.array(group["gtscore"])
                    break
                    
        elif dataset_name == 'TVSum':
            video_index = read_TVSum_video_index(video_name)
            print("video_index: ", video_index)
            group = f["video_" + str(video_index)]
            gtscore = np.array(group["gtscore"])
            
    gtscore = torch.from_numpy(gtscore).to(torch.float32)

    gt_indicator = torch.topk(gtscore, k=len(gtscore)*0.15)[1]

    gt_caption = ""

    for index in gt_indicator:
        if index in captions.keys():
            gt_caption = gt_caption + captions[index]
    
    return gt_caption

    


seg = read_summary('TVSum', 'Se3oxnaPsz0')
print(sum(seg[1]))
