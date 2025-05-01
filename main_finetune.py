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
    TrainingArguments,
    get_linear_schedule_with_warmup
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

from u_modules import blip2_utils, filter_caption, get_all_frames_score, gpt_api
from eval_ import evaluate_SumMe, evaluate_TVSum
from utils import seg_text_sumy_pocess, write_single_video_result, write_dataset_result, write_sumy_video, l2_loss, get_anno_SumMe
from plot_frame_weights import plot_imp

from peft import LoraConfig, get_peft_model


# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# video
dataset_name = "SumMe"
video_path = "/mnt/hdd8T/gw/dataset/SumMeandTVSum/" + dataset_name + "/videos/"
HOMEDATA='/mnt/hdd8T/gw/dataset/SumMeandTVSum/SumMe/GT/'

output_dir = '/mnt/hdd8T/gw/dataset/models/llama_lora/'

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


os.makedirs("result/" + dataset_name, exist_ok=True)
os.makedirs("result_videos/" + dataset_name, exist_ok=True)
os.makedirs("result_videos_text/" + dataset_name, exist_ok=True)
os.makedirs(dataset_name + "_frame_weights_with_sm", exist_ok=True)


# metric['spear_indicator'] = []
# metric['max_spear_indicator'] = []
# metric['ken_indicator'] = []
# metric['max_ken_indicator'] = []

# metric['FClip_dif_len'] = []
# metric['max_FClip_dif_len'] = []
# metric['FClip_same_len'] = []
# metric['max_FClip_same_len'] = []

# metric['summary_rate'] = []

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

# LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=['q_proj', 'v_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type='SEQ_2_SEQ_LM'
)

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=3,
    optim='adamw_torch',
    learning_rate=10e-4,
    eval_steps=50,
    save_steps=100,
    logging_steps=20,
    evaluation_strategy='steps',
    group_by_length=False,
    max_steps=200,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    bf16=True,
    lr_scheduler_type='cosine',
    warmup_steps=100
)

llama_model.enable_input_require_grads()
llama_model = get_peft_model(llama_model, peft_config)
llama_model.print_trainable_parameters()
llama_model.config.use_cache = False

optimizer = torch.optim.Adam(llama_model.parameters(), lr=10e-3)
lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(20*5),
    )


loss_fn = l2_loss

video_name_list = []
for video_name in os.listdir(video_path):
    if video_name.endswith('.mp4'):
        video_name_list.append(video_name)

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


for epoch in range(5):
    video_selector = np.random.randint(0, 24, size=20)

    # fine-tune
    for video_idx in video_selector:
        video_name = video_name_list[video_idx]

        # if "result_" + video_name in os.listdir("result_videos/" + dataset_name + "/"):
        #     continue
        # if video_name.split('.')[0] in os.listdir("result_videos/" + dataset_name + "/"):
        #     continue

        print(video_name + "\n")
        # if video_name == 'Fire Domino.mp4':
        #     continue

        time_start_1 = time.time()
        # blip-2 captioning each k frame
        segments_idx, video_text_list, blip2_k = blip2_utils(blip2_rate, similarity_threshold, video_path, video_name, dataset_name, None, None, clip_preprocess, clip_model, device)

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
        
        # 第一轮：分段文本摘要
        for seg_idx in range(1, len(segments_idx)):
            # 段头和段尾的帧idx
            seg_beg = segments_idx[seg_idx - 1]
            seg_end = segments_idx[seg_idx] - 1

            seg_caption = filter_caption(video_text_list[seg_beg // blip2_k + (0 if seg_beg % blip2_k == 0 else 1): seg_end // blip2_k + 1], 
            video_path, video_name, seg_beg, seg_end, blip2_k, None, None, clip_preprocess, clip_model, device)

            if len(seg_caption) == 0:
                continue

            # # llama promt
            prompt_template=f'''[INST] <<SYS>>
            Please summarize what happened in few sentences, based on the following temporal description of a video clip. Do not include any unnecessary details or descriptions.
            <</SYS>>
            {seg_caption}[/INST]

            '''
            # print(seg_caption)
            # get text sumy of each video clip using llama
            input_ids = llama_tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
            output = llama_model.generate(inputs=input_ids, temperature=0.9, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
            seg_text_sumy = llama_tokenizer.decode(output[0])

            print(seg_text_sumy)

            seg_text_sumy = seg_text_sumy_pocess(seg_text_sumy)

            # 如果llama2得到sumy全是空格
            if len(seg_text_sumy) == 0:
                print("llama2 sumy is empty, using T5")
                # get text sumy of each video clip using T5
                test_t5_model = myModel_for_Train()
                test_t5_model.load_model("outputs/simple_T5", use_gpu=True)
                seg_text_sumy = test_t5_model.predict("summarize: " + seg_caption)[0]

            # promt = "Please summarize what happened in few sentences, based on the following temporal description of a video clip. Do not include any unnecessary details or descriptions."
            # seg_text_sumy = gpt_api(promt, seg_caption)

            print("seg_" + str(seg_idx) + " summary: " + seg_text_sumy)

            all_segments_text_sumy = all_segments_text_sumy + seg_text_sumy
            all_segments_text_sumy_list.append(seg_text_sumy)

            with open("result_videos_text/" + dataset_name + "/" + "video_sumy_text_" + video_name.split('.')[0] + ".txt", "a") as f:
                f.write(seg_text_sumy + "\n\n")

        # 第二轮：总体摘要
        prompt_template=f'''[INST] <<SYS>>
        Please summarize what happened in few sentences, based on the following temporal description of a video. Do not include any unnecessary details or descriptions.
        <</SYS>>
        {all_segments_text_sumy}[/INST]

        '''
        # get text sumy of each video clip using llama
        input_ids = llama_tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
        output = llama_model.generate(inputs=input_ids, temperature=0.9, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
        video_text_sumy = llama_tokenizer.decode(output[0])

        video_text_sumy = seg_text_sumy_pocess(video_text_sumy)

        # promt = "Please summarize what happened in few sentences, based on the following temporal description of a video. Do not include any unnecessary details or descriptions."
        # video_text_sumy = gpt_api(promt, all_segments_text_sumy)

        # 如果llama2得到sumy全是空格
        if len(video_text_sumy) == 0:
            print("llama2 sumy is empty, using T5")
            # get text sumy of each video clip using T5
            test_t5_model = myModel_for_Train()
            test_t5_model.load_model("outputs/simple_T5", use_gpu=True)
            video_text_sumy = test_t5_model.predict("summarize: " + all_segments_text_sumy)[0]

        # 计算frames_score
        all_frames_score = get_all_frames_score(k, beta, alpha, video_text_sumy, all_segments_text_sumy_list, video_path, video_name, segments_idx, clip_preprocess, clip_model, device)  

        # 计算 loss
        # get gt
        user_score = get_anno_SumMe(video_name)
        loss = loss_fn(user_score, all_frames_score)

        loss.backward()

        # Optimizer step
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # 绘制frames重要度曲线
        plot_imp(dataset_name, all_frames_score.clone().detach().numpy(), video_name.split('.')[0])
        
        nFrames = segments_idx[-1]
        # 选择top rate的帧
        res_frame_idx = torch.topk(all_frames_score, k=math.ceil(nFrames * result_rate))[1]
    
        # 写入视频
        res_frame_number, summary_selections = write_sumy_video(dataset_name, video_path, video_name, res_frame_idx, nFrames)

        # 计算指标
        if dataset_name == "SumMe":
            video_metric = evaluate_SumMe(all_frames_score.clone().detach().numpy(), summary_selections, video_path, video_name.split('.')[0], HOMEDATA, clip_model, clip_preprocess, device)
        elif dataset_name == "TVSum":
            video_metric = evaluate_TVSum(all_frames_score.clone().detach().numpy(), summary_selections, video_path, video_name.split('.')[0], clip_model, clip_preprocess, device, result_rate)

        # 记录指标
        for metric_name, v in video_metric.items():
            print(metric_name + ": " + str(v))
            # metric[metric_name].append(v)
        
        write_single_video_result(video_metric, dataset_name, video_name, res_frame_number, 'train')

    # eval
    llama_model.eval()
    with torch.no_grad():
        for video_idx in range(24):
            if video_idx not in video_selector:
                video_name = video_name_list[video_idx]
                print(video_name + "\n")

                time_start_1 = time.time()
                # blip-2 captioning each k frame
                segments_idx, video_text_list, blip2_k = blip2_utils(blip2_rate, similarity_threshold, video_path, video_name, dataset_name, None, None, clip_preprocess, clip_model, device)

                time_end_1 = time.time()
                print("运行时间：" + str(time_end_1 - time_start_1) + "s")
                print("segments_idx: ")
                print(segments_idx)
                # 清空GPU缓存
                # torch.cuda.empty_cache()
                # 释放blip2模型占用的GPU内存
                # del blip2_model
                # del blip2_processor
                print("video caotioning end.")

                all_segments_text_sumy = ""
                all_segments_text_sumy_list = []
                video_text_sumy = ""
                
                # 第一轮：分段文本摘要
                for seg_idx in range(1, len(segments_idx)):
                    # 段头和段尾的帧idx
                    seg_beg = segments_idx[seg_idx - 1]
                    seg_end = segments_idx[seg_idx] - 1

                    seg_caption = filter_caption(video_text_list[seg_beg // blip2_k + (0 if seg_beg % blip2_k == 0 else 1): seg_end // blip2_k + 1], 
                    video_path, video_name, seg_beg, seg_end, blip2_k, None, None, clip_preprocess, clip_model, device)

                    if len(seg_caption) == 0:
                        continue

                    # # llama promt
                    print(seg_caption)
                    prompt_template=f'''[INST] <<SYS>>
                    Please summarize what happened in few sentences, based on the following temporal description of a video clip. Do not include any unnecessary details or descriptions.
                    <</SYS>>
                    {seg_caption}[/INST]

                    '''
                    # get text sumy of each video clip using llama
                    input_ids = llama_tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
                    output = llama_model.generate(inputs=input_ids, temperature=0.9, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
                    seg_text_sumy = llama_tokenizer.decode(output[0])

                    seg_text_sumy = seg_text_sumy_pocess(seg_text_sumy)

                    # 如果llama2得到sumy全是空格
                    if len(seg_text_sumy) == 0:
                        print("llama2 sumy is empty, using T5")
                        # get text sumy of each video clip using T5
                        test_t5_model = myModel_for_Train()
                        test_t5_model.load_model("outputs/simple_T5", use_gpu=True)
                        seg_text_sumy = test_t5_model.predict("summarize: " + seg_caption)[0]

                    # promt = "Please summarize what happened in few sentences, based on the following temporal description of a video clip. Do not include any unnecessary details or descriptions."
                    # seg_text_sumy = gpt_api(promt, seg_caption)

                    print("seg_" + str(seg_idx) + " summary: " + seg_text_sumy)

                    all_segments_text_sumy = all_segments_text_sumy + seg_text_sumy
                    all_segments_text_sumy_list.append(seg_text_sumy)

                    with open("result_videos_text/" + dataset_name + "/" + "video_sumy_text_" + video_name.split('.')[0] + ".txt", "a") as f:
                        f.write(seg_text_sumy + "\n\n")

                # 第二轮：总体摘要
                prompt_template=f'''[INST] <<SYS>>
                Please summarize what happened in few sentences, based on the following temporal description of a video. Do not include any unnecessary details or descriptions.
                <</SYS>>
                {all_segments_text_sumy}[/INST]

                '''
                # get text sumy of each video clip using llama
                input_ids = llama_tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
                output = llama_model.generate(inputs=input_ids, temperature=0.9, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
                video_text_sumy = llama_tokenizer.decode(output[0])

                video_text_sumy = seg_text_sumy_pocess(video_text_sumy)

                # promt = "Please summarize what happened in few sentences, based on the following temporal description of a video. Do not include any unnecessary details or descriptions."
                # video_text_sumy = gpt_api(promt, all_segments_text_sumy)

                # 如果llama2得到sumy全是空格
                if len(video_text_sumy) == 0:
                    print("llama2 sumy is empty, using T5")
                    # get text sumy of each video clip using T5
                    test_t5_model = myModel_for_Train()
                    test_t5_model.load_model("outputs/simple_T5", use_gpu=True)
                    video_text_sumy = test_t5_model.predict("summarize: " + all_segments_text_sumy)[0]

                # 计算frames_score
                all_frames_score = get_all_frames_score(k, beta, alpha, video_text_sumy, all_segments_text_sumy_list, video_path, video_name, segments_idx, clip_preprocess, clip_model, device)  

                # 绘制frames重要度曲线
                plot_imp(dataset_name, all_frames_score.clone().detach().numpy(), video_name.split('.')[0])
                
                nFrames = segments_idx[-1]
                # 选择top rate的帧
                res_frame_idx = torch.topk(all_frames_score, k=math.ceil(nFrames * result_rate))[1]
            
                # 写入视频
                res_frame_number, summary_selections = write_sumy_video(dataset_name, video_path, video_name, res_frame_idx, nFrames)

                # 计算指标
                if dataset_name == "SumMe":
                    video_metric = evaluate_SumMe(all_frames_score.clone().detach().numpy(), summary_selections, video_path, video_name.split('.')[0], HOMEDATA, clip_model, clip_preprocess, device)
                elif dataset_name == "TVSum":
                    video_metric = evaluate_TVSum(all_frames_score.clone().detach().numpy(), summary_selections, video_path, video_name.split('.')[0], clip_model, clip_preprocess, device, result_rate)

                # 记录指标
                for metric_name, v in video_metric.items():
                    print(metric_name + ": " + str(v))
                    metric[metric_name].append(v)
                
                write_single_video_result(video_metric, dataset_name, video_name, res_frame_number, 'test')

# 记录数据集指标
write_dataset_result(metric, dataset_name)