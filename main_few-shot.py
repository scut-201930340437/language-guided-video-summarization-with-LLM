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
    AutoTokenizer
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

from T5 import myModel_for_Train

from all_modules import blip2_utils, filter_caption, get_all_frames_score, gpt_api
from eval2 import evaluate_SumMe, evaluate_TVSum
from utils.read_utils import read_segmentations, read_gt_caption
from utils.write_utils import write_sumy_video, write_single_video_result, write_dataset_result
from utils.plot_utils import plot_imp
from utils.text_sumy_utils import seg_text_sumy_pocess


# setup device to use
device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

# video
dataset_name = "TVSum"
video_path = "/mnt/hdd8T/gw/dataset/SumMeandTVSum/" + dataset_name + "/videos/"
HOMEDATA='/mnt/hdd8T/gw/dataset/SumMeandTVSum/SumMe/GT/'

### hyper-para
# caption间隔
blip2_rate = 3
# 选帧时采样间隔
k = 1
result_rate = 0.15
# 视频分段相似度阈值
similarity_threshold = 0.85
# smooth
beta = 0.6
# 段分数与段内帧分数的权重
alpha = 1.25


os.makedirs("result/" + dataset_name, exist_ok=True)
os.makedirs("result_videos/" + dataset_name, exist_ok=True)
os.makedirs("result_videos_text/" + dataset_name, exist_ok=True)
os.makedirs(dataset_name + "_frame_weights_with_sm", exist_ok=True)
os.makedirs("captions/" + dataset_name, exist_ok=True)

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
blip2_processor = AutoProcessor.from_pretrained(
    "/mnt/hdd8T/hh/code/llm/Models/hub/models--Salesforce--blip2-flan-t5-xl/snapshots/e5025a34e3e769e72e2aab7f7bfd00bc84d5fd77"
)
blip2_model = Blip2ForConditionalGeneration.from_pretrained(
    "/mnt/hdd8T/hh/code/llm/Models/hub/models--Salesforce--blip2-flan-t5-xl/snapshots/e5025a34e3e769e72e2aab7f7bfd00bc84d5fd77",
    torch_dtype=torch.float16,
)
blip2_model.to(device)
blip2_promt = "Question: What does this image show? Please give your answer in as much detail as possible. Answer:"

# load llama2
llama_model_name_or_path = "/mnt/hdd8T/hh/code/llm/Models/hub/models--TheBloke--Llama-2-13B-chat-GPTQ/snapshots/ea078917a7e91c896787c73dba935f032ae658e9"
llama_model = AutoModelForCausalLM.from_pretrained(
    llama_model_name_or_path,
    device_map="auto",
    trust_remote_code=False,
    revision="main",
)
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name_or_path, use_fast=True)

# load clip
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# few-shot
numbers = np.arange(0, 25)
random_choice = np.random.choice(numbers, size=5, replace=False)
video_name_list = []
for video_name in os.listdir(video_path):
    if video_name.endswith('.mp4'):
        video_name_list.append(video_name)

few_shot_videos_captions_list = []
few_shot_videos_sumy_list = []

task_description = "Please summarize what happened in few sentences, based on the temporal description of a video containing multiple scenes. Do not include any unnecessary details. I will give some samples and the format is (description)->(summary)."

# llama promt
prompt_template=f'''[INST] <<SYS>>
Please summarize what happened in few sentences, based on the temporal description of a video containing multiple scenes. Do not include any unnecessary details. I will give some samples and the format is (description)->(summary).
<</SYS>>
{""}[/INST]

'''

# get text sumy of each video clip using llama
input_ids = llama_tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = llama_model.generate(inputs=input_ids, temperature=0.9, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
output = llama_tokenizer.decode(output[0])

for video_idx in range(len(video_name_list)):
    if video_idx in random_choice:
        video_name = video_name_list[video_idx]
        # blip-2 captioning each k frame
        frames_features, similarities, video_text_dict, blip2_k, nFrames, fps = blip2_utils(blip2_rate, similarity_threshold, video_path, video_name, dataset_name, blip2_processor, blip2_model, clip_preprocess, clip_model, device)
        # 获取 gt sumy caption
        sumy_caption = read_sumy_caption(video_text_dict, video_name.split('.')[0])

        video_caption = ""

        for _, caption in video_text_dict:
            video_caption = video_caption + caption
        
        example = "(" + video_caption + ")->(" + sumy_caption + ")" 

        # llama promt
        prompt_template=f'''[INST] <<SYS>>
        <</SYS>>
        {example}[/INST]

        '''

        input_ids = llama_tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
        output = llama_model.generate(inputs=input_ids, temperature=0.9, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
        summary = llama_tokenizer.decode(output[0])


# main
for video_name in os.listdir(video_path):
    if video_name.endswith('.mp4'):
        # if "result_" + video_name in os.listdir("result_videos/" + dataset_name + "/"):
        #     continue
        # if video_name.split('.')[0] in os.listdir("result_videos/" + dataset_name + "/"):
        #     continue

        print(video_name)
        
        # tmp_list = [ 'Bhxk-O1Y7Ho.mp4', 'kLxoNp-UchI.mp4', '3eYKfiOEJNs.mp4', 'oDXZc0tZe04.mp4', 'PJrm840pAUI.mp4']
        
        # if video_name != 'kLxoNp-UchI.mp4':
        #     continue
        

        time_start_1 = time.time()
        # blip-2 captioning each k frame
        frames_features, similarities, video_text_dict, blip2_k, nFrames, fps = blip2_utils(blip2_rate, similarity_threshold, video_path, video_name, dataset_name, blip2_processor, blip2_model, clip_preprocess, clip_model, device)

        # if dataset_name == "TVSum":
        #     segments_idx = TVSum_segmentation(video_name.split('.')[0])
        #     segments_idx = [0] + segments_idx + [nFrames]
        # else:
        #     segments_idx = segmentation(frames_features) * (nFrames // 150)
        #     segments_idx = [0] + segments_idx.tolist() + [nFrames]

        segments_idx = read_segmentations(dataset_name, video_name.split('.')[0])

        # segments_idx = segmentation(similarities)

        time_end_1 = time.time()
        print("运行时间：" + str(time_end_1 - time_start_1) + "秒")
        print("segments_idx: ")
        print(segments_idx)
        # 清空GPU缓存
        # torch.cuda.empty_cache()
        # 释放blip2模型占用的GPU内存
        # del blip2_model
        # del blip2_processor
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

        all_segments_text_sumy = ""
        all_segments_text_sumy_list = []
        video_text_sumy = ""
        with torch.no_grad():
            # 第一轮：分段文本摘要
            for seg_idx in range(1, len(segments_idx)):
                # 段头和段尾的帧idx
                seg_beg = segments_idx[seg_idx - 1]
                seg_end = segments_idx[seg_idx] - 1

                seg_caption = filter_caption(video_text_dict, video_path, video_name, seg_beg, seg_end, blip2_k, blip2_processor, blip2_model, clip_preprocess, clip_model, device,)

                print("seg_caption: " + seg_caption)

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
                output = llama_model.generate(inputs=input_ids, temperature=0.9, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
                seg_text_sumy = llama_tokenizer.decode(output[0])

                seg_text_sumy = seg_text_sumy_pocess(seg_text_sumy)

                # gpt-3.5
                # promt = "Please summarize what happened in few sentences, based on the following temporal description of a scene."  # Do not include any unnecessary details or descriptions."
                # seg_text_sumy = gpt_api(promt, seg_caption)

                # 如果llama2得到sumy全是空格
                if len(seg_text_sumy) == 0:
                    print("llama2 sumy is empty, using T5")
                    # get text sumy of each video clip using T5
                    test_t5_model = myModel_for_Train()
                    test_t5_model.load_model("outputs/simple_T5", use_gpu=True)
                    seg_text_sumy = test_t5_model.predict("summarize: " + seg_caption)[0]


                print("seg_" + str(seg_idx) + " summary: " + seg_text_sumy)

                all_segments_text_sumy = all_segments_text_sumy + seg_text_sumy
                all_segments_text_sumy_list.append(seg_text_sumy)

                with open("result_videos_text/" + dataset_name + "/" + "video_sumy_text_" + video_name.split('.')[0] + ".txt", "a") as f:
                    f.write(seg_text_sumy + "\n\n")

            # 第二轮：总体摘要

            prompt_template=f'''[INST] <<SYS>>
            Please summarize what happened in few sentences, based on the following temporal description of multiple scenes. Do not include any unnecessary details or descriptions.
            <</SYS>>
            {all_segments_text_sumy}[/INST]

            '''
            # get text sumy of each video clip using llama
            input_ids = llama_tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
            output = llama_model.generate(inputs=input_ids, temperature=0.9, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
            video_text_sumy = llama_tokenizer.decode(output[0])

            video_text_sumy = seg_text_sumy_pocess(video_text_sumy)

            # gpt-3.5
            # promt = "Please summarize what happened in few sentences, based on the following temporal description of multiple scenes. Do not include any unnecessary details or descriptions."
            # video_text_sumy = gpt_api(promt, all_segments_text_sumy)

            # 如果llama2得到sumy全是空格
            if len(video_text_sumy) == 0:
                print("llama2 sumy is empty, using T5")
                # get text sumy of each video clip using T5
                test_t5_model = myModel_for_Train()
                test_t5_model.load_model("outputs/simple_T5", use_gpu=True)
                video_text_sumy = test_t5_model.predict("summarize: " + all_segments_text_sumy)[0]

            print('video_text_sumy: ' + video_text_sumy)
            with open("result_videos_text/" + dataset_name + "/" + "video_sumy_text_" + video_name.split('.')[0] + ".txt", "a") as f:
                    f.write(video_text_sumy + "\n\n\n")

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
            res_frame_number = write_sumy_video(dataset_name, video_path, video_name, pred_indicator, nFrames)
            # 记录指标
            for metric_name, v in video_metric.items():
                print(metric_name + ": " + str(v))
                metric[metric_name].append(v)
        
        write_single_video_result(video_metric, dataset_name, video_name, res_frame_number, '')
            

# 记录数据集指标
write_dataset_result(metric, dataset_name)