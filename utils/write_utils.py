import os
import torch
import numpy as np
import cv2
import math
import json

def write_blip2_caption(video_text, video_name, dataset_name):
    with open("captions/" + dataset_name + "/" + "video_text_" + video_name.split(".")[0] + ".json", 'w') as write_f:
	    json.dump(video_text, write_f, indent=4, ensure_ascii=False)


def write_llama2_sumy(video_text_sumy, video_name, dataset_name, iteration, segments_idx):
    video_text_sumy['segments_idx'] = segments_idx
    with open("text_sumy/" + dataset_name + "/" + "sumy_" + video_name.split(".")[0]+ "_" + str(iteration) + ".json", 'w') as write_f:
	    json.dump(video_text_sumy, write_f, indent=4, ensure_ascii=False)


def write_single_video_result(video_metric, dataset_name, video_name, res_frame_number, mode, iteration):
    with open("result/"+ dataset_name + "/" + 'metric.txt', 'a') as f:
        f.write('iteration: ' + str(iteration) + "\n")
        for metric_name, v in video_metric.items():
            print(mode + "video " + video_name + " " + metric_name + ": " + str(v))
            f.write(mode + "video " + video_name + " " + metric_name + ": " + str(v) + "\n")

            # print("video " + video_name + " " + metric_name + "_max: " + str(np.max(v)))
            # f.write("video " + video_name + " " + metric_name + "_max: " + str(np.max(v)) + "\n")

        f.write(mode + "video " + video_name + " summary_length: " + str(res_frame_number) + "\n\n")


def write_dataset_result(metric, dataset_name):
    with open("result/"+ dataset_name + "/" + 'metric.txt', 'a') as f:
        for metric_name, v in metric.items():
            mean_v = np.mean(v)

            print(metric_name + ": " + str(mean_v) + "\n")
            f.write(metric_name + ": " + str(mean_v) + "\n")


def write_sumy_video(dataset_name, video_path, video_name, pred_indicator):
    capture = cv2.VideoCapture(os.path.join(video_path, video_name))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")  # 视频编解码器
    out = cv2.VideoWriter("result_videos/" + dataset_name + "/" + "result_" + video_name, fourcc, fps // 2, (frame_width, frame_height))

    frame_idx = 0
    res_frame_number = 0
    # summary_selections = np.zeros(nFrames, dtype=int)
    while True:
        ret, frame = capture.read()
        if ret and frame_idx < len(pred_indicator):
            if pred_indicator[frame_idx] == 1:
                out.write(frame)  # 写入帧
                res_frame_number = res_frame_number + 1
        else:
            break

        frame_idx = frame_idx + 1
    capture.release()
    out.release()

    print("result length: " + str(res_frame_number))

    return res_frame_number