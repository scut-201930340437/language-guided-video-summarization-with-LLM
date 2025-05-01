import os
import torch
import numpy as np
import pandas as pd
import cv2
import math
import json
import h5py


def seg_text_sumy_pocess(seg_text_sumy):
    # 去掉llama输出中前一部分的template [/INST]
    index = 0
    for _ in range(len(seg_text_sumy)):
        if seg_text_sumy[index] == "[" and seg_text_sumy[index + 1] == "/":
            break
        index = index + 1

    index = index + 9

    # print(seg_text_sumy + "\n")

    # 去除 </s>
    seg_text_sumy = seg_text_sumy[index:-4].replace("\n", " ")
    seg_text_sumy = seg_text_sumy.replace("  ", " ")

    # 进一步去除无关信息
    index = 0
    work = False
    for _ in range(len(seg_text_sumy)):
        if seg_text_sumy[index] == ":":
            work = True
            break
        index = index + 1
    if work:
        index = index + 2
        seg_text_sumy = seg_text_sumy[index:]

    # 去除前面的空格
    index = 0
    for _ in range(len(seg_text_sumy)):
        if seg_text_sumy[index] == " ":
            index = index + 1
        else:
            break

    seg_text_sumy = seg_text_sumy[index:]

    return seg_text_sumy