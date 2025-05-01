import time
# import string
import os
# import cv2
import math
# from PIL import Image

import torch
import numpy as np
from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    Blip2Processor,
    Blip2ForConditionalGeneration
    # T5ForConditionalGeneration,
    # PreTrainedTokenizer,
    # T5TokenizerFast as T5Tokenizer,
)

import clip

# from abc import ABC
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from torch.utils.data import Dataset, DataLoader
# import pytorch_lightning as pl
# from torch.optim import AdamW
# from pytorch_lightning.callbacks.progress import TQDMProgressBar
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# from rouge import Rouge

# from T5 import myModel_for_Train

from all_modules import blip2_utils, filter_caption, get_all_frames_score
from eval2 import evaluate_SumMe, evaluate_TVSum
from utils.read_utils import read_segmentations, read_gt_caption, read_llama2_sumy
from utils.write_utils import write_sumy_video, write_single_video_result, write_dataset_result, write_llama2_sumy
from utils.plot_utils import plot_imp
from utils.text_sumy_utils import seg_text_sumy_pocess
from utils.kts_utils import segmentation
from utils.clip_utils import frame_features_extractor


# setup device to use
device = torch.device("cuda:1") if torch.cuda.is_available() else "cpu"

# video
dataset_name = "TVSum"
video_path = "/home/lab345/gw/dataset/SumMeandTVSum/" + dataset_name + "/videos/"
HOMEDATA = '/home/lab345/gw/dataset/SumMeandTVSum/SumMe/GT/'

### hyper-para
# caption间隔
blip2_rate = 3
# 选帧时采样间隔
k = 1
result_rate = 0.145
# 视频分段相似度阈值
similarity_threshold = 0.85
# smooth
beta = 0.75
# 段分数与段内帧分数的权重
alpha = 1.0
#
seg_num_frac = 10

os.makedirs("result/" + dataset_name, exist_ok=True)
os.makedirs("result_videos/" + dataset_name, exist_ok=True)
os.makedirs("result_videos_text/" + dataset_name, exist_ok=True)
os.makedirs(dataset_name + "_frame_weights_with_sm", exist_ok=True)
os.makedirs("captions/" + dataset_name, exist_ok=True)
os.makedirs("text_sumy/" + dataset_name, exist_ok=True)

# metric
metric = {}
metric['mean_f1'] = []
metric['max_f1'] = []

metric['mean_spearman_dif_len'] = []
metric['max_spearman_dif_len'] = []
metric['mean_kendall_dif_len'] = []
metric['max_kendall_dif_len'] = []

metric['mean_spearman_same_len'] = []
metric['max_spearman_same_len'] = []
metric['mean_kendall_same_len'] = []
metric['max_kendall_same_len'] = []

# metric['spear_indicator'] = []
# metric['max_spear_indicator'] = []
# metric['ken_indicator'] = []
# metric['max_ken_indicator'] = []

# metric['FClip_dif_len'] = []
# metric['max_FClip_dif_len'] = []
# metric['FClip_same_len'] = []
# metric['max_FClip_same_len'] = []

# metric['summary_rate'] = []

# load blip2
# blip2_processor = Blip2Processor.from_pretrained("/mnt1t/whl/myblip2/blip2-opt-2.7b")
# blip2_model = Blip2ForConditionalGeneration.from_pretrained(
#         "/mnt1t/whl/myblip2/blip2-opt-2.7b", torch_dtype=torch.float32
# )

# blip2_model.to(device)


# load llama2
llama_model_name_or_path = "/home/lab345/gw/models/models--TheBloke--Llama-2-13B-chat-GPTQ/snapshots/ea078917a7e91c896787c73dba935f032ae658e9"
llama_model = AutoModelForCausalLM.from_pretrained(
    llama_model_name_or_path,
    device_map="auto",
    trust_remote_code=False,
    revision="main",
)
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name_or_path, use_fast=True)

# load clip
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)


# main
for video_name in os.listdir(video_path):
    if video_name.endswith('.mp4'):
        # if "result_" + video_name in os.listdir("result_videos/" + dataset_name + "/"):
        #     continue
        # if video_name.split('.')[0] in os.listdir("result_videos/" + dataset_name + "/"):
        #     continue

        print(video_name)
        
        # tmp_list = [ 'Bhxk-O1Y7Ho.mp4', 'kLxoNp-UchI.mp4', '3eYKfiOEJNs.mp4', 'oDXZc0tZe04.mp4', 'PJrm840pAUI.mp4']
        
        # 'Bike Polo.mp4', 'Car_railcrossing.mp4', 'Notre_Dame.mp4', 'Saving dolphines.mp4', 'Excavators river crossing.mp4', 
        # tmp_list = ['Uncut_Evening_Flight.mp4']

        # tmp_list = ['Bhxk-O1Y7Ho.mp4', 'z_6gVvQb2d0.mp4', 'Se3oxnaPsz0.mp4', 'oDXZc0tZe04.mp4', 'akI8YFjEmUw.mp4'] # TVSum worse
        # tmp_list = ['Hl-__g2gn_A.mp4', 'gzDbaEs1Rlg.mp4', 'XkqCExn6_Us.mp4', 'vdmoEJ5YbrQ.mp4', 'WG0MBPpPC6I.mp4']

        # 'PJrm840pAUI.mp4', 'oDXZc0tZe04.mp4', 'akI8YFjEmUw.mp4', 'b626MiF1ew4.mp4', 'byxOvuiIJV0.mp4',    # gt之间的相关性极低

        # 利用kts分段：'Bhxk-O1Y7Ho.mp4', 'z_6gVvQb2d0.mp4', 'oDXZc0tZe04.mp4', 'Se3oxnaPsz0.mp4'

        # tmp_list = ['37rzWOQsNIw.mp4', 'cjibtmSLxQ4.mp4', 'JKpqYvAdIsw.mp4', 'PJrm840pAUI.mp4', 'E11zDS9XGzg.mp4', 'b626MiF1ew4.mp4']
        # tmp_list = ['Bhxk-O1Y7Ho.mp4', 'z_6gVvQb2d0.mp4', 'oDXZc0tZe04.mp4', 'Se3oxnaPsz0.mp4', 'b626MiF1ew4.mp4', "_xMr-HKMfVA.mp4", "eQu1rNs0an0.mp4", "VuWGsYPqAX8.mp4", "RBCABdttQmI.mp4", "jcoYJXDG9sw.mp4", "kLxoNp-UchI.mp4", "byxOvuiIJV0.mp4"]
        # _xMr-HKMfVA.mp4, eQu1rNs0an0.mp4, VuWGsYPqAX8.mp4, RBCABdttQmI.mp4, jcoYJXDG9sw.mp4, kLxoNp-UchI.mp4, byxOvuiIJV0.mp4

        # use read_seg()
        # tmp_list = ['91IHQYk1IQM.mp4', 'xwqBXPGE9pQ.mp4', 'iVt07TCkFM0.mp4', 'J0nA4VgnoCo.mp4', '4wU_LUjG5Ic.mp4', 'xxdtq8mxegs.mp4', 'i3wAGJaaktw.mp4', '3eYKfiOEJNs.mp4', 'uGu_10sucQo.mp4']
        
        # tmp_list = ['oDXZc0tZe04.mp4', 'gzDbaEs1Rlg.mp4', 'PJrm840pAUI.mp4', 'Bhxk-O1Y7Ho.mp4']
        # tmp_list = ['akI8YFjEmUw.mp4', 'JKpqYvAdIsw.mp4']
        # if video_name not in tmp_list:
            # continue
        
        # max_iter = 18
        # m_list=[45, 50, 55, 60, 65, 70]
        # if video_name == 'akI8YFjEmUw.mp4' or video_name == 'JKpqYvAdIsw.mp4':
        #     max_iter = 3
        #     m_list=[12]

        if video_name != '91IHQYk1IQM.mp4':
            continue

        
        
        time_start_1 = time.time()
        # blip-2 captioning each k frame
        frames_features, video_text_dict, blip2_k, nFrames = blip2_utils(blip2_rate, similarity_threshold, video_path, video_name, dataset_name, None, None, clip_preprocess, clip_model, device)
        # video_text_dict, blip2_k = blip2_utils(blip2_rate, similarity_threshold, video_path, video_name, dataset_name, None, None, clip_preprocess, clip_model, device)
        
        # if dataset_name == "TVSum":
        #     segments_idx = TVSum_segmentation(video_name.split('.')[0])
        #     segments_idx = [0] + segments_idx + [nFrames]
        # elif dataset_name == "SumMe":
        #     segments_idx = segmentation(frames_features, nFrames) * (nFrames // 150)
        #     segments_idx = [0] + segments_idx.tolist() + [nFrames]

        # segments_idx = segmentation(similarities)


        # segments_idx = read_segmentations(dataset_name, video_name.split('.')[0])

        time_end_1 = time.time()
        print("运行时间：" + str(time_end_1 - time_start_1) + "秒")
        print("segments_idx: ")
        # print(segments_idx)

        print("video caotioning end.")

        # metric
        # video_metric = {}
        # video_metric['mean_f1'] = -1.0
        # video_metric['max_f1'] = -1.0

        # # video_metric['spearman_dif_len'] = []
        # # video_metric['max_spearman_dif_len'] = []
        # # video_metric['kendall_dif_len'] = []
        # # video_metric['max_kendall_dif_len'] = []

        # video_metric['mean_spearman_same_len'] = -1.0
        # video_metric['max_spearman_same_len'] = -1.0
        # video_metric['mean_kendall_same_len'] = -1.0
        # video_metric['max_kendall_same_len'] = -1.0

        # # video_metric['spear_indicator'] = []
        # # video_metric['max_spear_indicator'] = []
        # # video_metric['ken_indicator'] = []
        # # video_metric['max_ken_indicator'] = []

        # # video_metric['mean_FClip_dif_len'] = []
        # # video_metric['max_FClip_dif_len'] = []
        # # video_metric['mean_FClip_same_len'] = []
        # # video_metric['max_FClip_same_len'] = []

        # # video_metric['summary_rate'] = []

        # iteration = 0
        # max_metric_iter = 0
        iteration = 0
        max_spearman = -2.0
        # m_list = [15,20,25,30,35,40]
        
        # m_list = [8, 10, 12]
        
        ### Vlog clip ###
        # frames_features, sample_rate, nFrames = frame_features_extractor(video_path, video_name, device)
        
        all_segments_text_sumy_list = []
        video_text_sumy = ""
        seg_text_sumy_dict = {}
        with torch.no_grad():
            # 第一轮：分段文本摘要
            all_segments_text_sumy_list, seg_text_sumy_dict, video_text_sumy = read_llama2_sumy(video_name, dataset_name)
            print(all_segments_text_sumy_list)

            ### Vlog kts ###
            # segments_idx = segmentation(frames_features, 40) * sample_rate
            
            ### our kts ###
            segments_idx = segmentation(frames_features, 25) * (nFrames // 200)
            
            segments_idx = [0] + segments_idx.tolist() + [nFrames]
            print(segments_idx)

            if len(all_segments_text_sumy_list) == 0:
                all_segments_text_sumy = ""
                
                for seg_idx in range(1, len(segments_idx)):
                    # 段头和段尾的帧idx
                    seg_beg = segments_idx[seg_idx - 1]
                    seg_end = segments_idx[seg_idx] - 1

                    seg_caption = filter_caption(video_text_dict, video_path, video_name, seg_beg, seg_end, blip2_k, None, None, clip_preprocess, clip_model, device,)

                    # print("seg_caption: " + seg_caption)

                    if len(seg_caption) == 0:
                        continue

                    # llama promt
                    prompt_template=f'''[INST] <<SYS>>
                    Please summarize what happened in few sentences, based on the following temporal description of a scene. Do not include any unnecessary details or descriptions.
                    <</SYS>>
                    {seg_caption}[/INST]

                    '''
                    # get text sumy of each video clip using llama
                    input_ids = llama_tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
                    output = llama_model.generate(inputs=input_ids, temperature=0.9, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=768) # 768
                    seg_text_sumy = llama_tokenizer.decode(output[0])

                    seg_text_sumy = seg_text_sumy_pocess(seg_text_sumy)

                    # gpt-3.5
                    # promt = "Please summarize what happened in few sentences, based on the following temporal description of a scene."  # Do not include any unnecessary details or descriptions."
                    # seg_text_sumy = gpt_api(promt, seg_caption)

                    # 如果llama2得到sumy全是空格
                    # if len(seg_text_sumy) == 0:
                    #     print("llama2 sumy is empty, using T5")
                    #     # get text sumy of each video clip using T5
                    #     test_t5_model = myModel_for_Train()
                    #     test_t5_model.load_model("outputs/simple_T5", use_gpu=True)
                    #     seg_text_sumy = test_t5_model.predict("summarize: " + seg_caption)[0]


                    print("seg_" + str(seg_idx) + " summary: " + seg_text_sumy)

                    all_segments_text_sumy = all_segments_text_sumy + seg_text_sumy
                    all_segments_text_sumy_list.append(seg_text_sumy)

                    seg_text_sumy_dict[seg_idx] = seg_text_sumy

            if len(video_text_sumy) == 0:
                all_segments_text_sumy = ""
                for seg_text_sumy in all_segments_text_sumy_list:
                    all_segments_text_sumy = all_segments_text_sumy + seg_text_sumy
                # 第二轮：总体摘要
                prompt_template=f'''[INST] <<SYS>>
                Please summarize what happened in few sentences, based on the following temporal description of multiple scenes in a video. Do not include any unnecessary details or descriptions.
                <</SYS>>
                {all_segments_text_sumy}[/INST]

                '''
                # get text sumy of each video clip using llama
                input_ids = llama_tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
                output = llama_model.generate(inputs=input_ids, temperature=0.9, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=1536) # 1536
                video_text_sumy = llama_tokenizer.decode(output[0])

                video_text_sumy = seg_text_sumy_pocess(video_text_sumy)

                # gpt-3.5
                # promt = "Please summarize what happened in few sentences, based on the following temporal description of multiple scenes. Do not include any unnecessary details or descriptions."
                # video_text_sumy = gpt_api(promt, all_segments_text_sumy)

                # 如果llama2得到sumy全是空格
                # if len(video_text_sumy) == 0:
                #     print("llama2 sumy is empty, using T5")
                #     # get text sumy of each video clip using T5
                #     test_t5_model = myModel_for_Train()
                #     test_t5_model.load_model("outputs/simple_T5", use_gpu=True)
                #     video_text_sumy = test_t5_model.predict("summarize: " + all_segments_text_sumy)[0]

                print('video_text_sumy: ' + video_text_sumy)
                
                seg_text_sumy_dict['video_level_sumy'] = video_text_sumy

                
            # 计算frames_score
            all_frames_score = get_all_frames_score(k, beta, alpha, video_text_sumy, all_segments_text_sumy_list, video_path, video_name, segments_idx, clip_preprocess, clip_model, device)  
        
            # 绘制frames重要度曲线
            plot_imp(dataset_name, all_frames_score.clone().detach().numpy(), video_name.split('.')[0])
            
            # nFrames = segments_idx[-1]
            video_metric = {}
            pred_indicator = None
            if dataset_name == "SumMe":
                video_metric, pred_indicator = evaluate_SumMe(all_frames_score.clone().detach().numpy(), video_path, video_name.split('.')[0], clip_model, clip_preprocess, device, segments_idx)
        
            elif dataset_name == "TVSum":
                video_metric, pred_indicator = evaluate_TVSum(all_frames_score.clone().detach().numpy(), video_path, video_name.split('.')[0], clip_model, clip_preprocess, device, result_rate, segments_idx)
            
            # 写入视频
            res_frame_number = write_sumy_video(dataset_name, video_path, video_name, pred_indicator)
            # 记录指标
            for metric_name, v in video_metric.items():
                print(metric_name + ": " + str(v))
                metric[metric_name].append(v)
        
            
            write_single_video_result(video_metric, dataset_name, video_name, res_frame_number, '', iteration)

            write_llama2_sumy(seg_text_sumy_dict, video_name, dataset_name, iteration, segments_idx)
            
            
            # if iteration >= max_iter or video_metric['mean_spearman_dif_len'] > 0.25:
            #     break

# 记录数据集指标
write_dataset_result(metric, dataset_name)