
import os
import cv2
import scipy
from PIL import Image
import torch
import random
import string
import json

import clip
from transformers import AutoProcessor, Blip2Processor, Blip2ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
# import openai
# import requests

from utils.write_utils import write_blip2_caption
from utils.read_utils import read_blip2_caption

def blip2_utils(blip2_rate, similarity_threshold, video_path, video_name, dataset_name, blip2_processor, blip2_model, clip_preprocess, clip_model, device):
    HOMEDATA = '/home/lab345/gw/dataset/SumMeandTVSum/SumMe/GT'

    capture = cv2.VideoCapture(os.path.join(video_path, video_name))

    fps = int(capture.get(cv2.CAP_PROP_FPS))
    blip2_k = int(fps // blip2_rate)

    capture.release()

    # each frame each text
    video_text_dict = {}
    video_text = read_blip2_caption(video_name, dataset_name)

    if len(video_text) > 0:
        for k, v in video_text.items():
            if k != "segments_idx":
                video_text_dict[int(k)] = v

    else:
        # load blip2
        # blip2_processor = AutoProcessor.from_pretrained(
        #     "/mnt/hdd8T/hh/code/llm/Models/hub/models--Salesforce--blip2-flan-t5-xl/snapshots/e5025a34e3e769e72e2aab7f7bfd00bc84d5fd77"
        # )
        # blip2_model = Blip2ForConditionalGeneration.from_pretrained(
        #     "/mnt/hdd8T/hh/code/llm/Models/hub/models--Salesforce--blip2-flan-t5-xl/snapshots/e5025a34e3e769e72e2aab7f7bfd00bc84d5fd77",
        #     torch_dtype=torch.float16,
        # )
        # blip2_model.to(device)
        # blip2_promt = "Question: What does this image show? Please give your answer in as much detail as possible. Answer:"
        # blip2_promt = "Describe the image in detail."

        # opencv capture
        capture = cv2.VideoCapture(os.path.join(video_path, video_name))

        frame_idx = 0
        with torch.no_grad():
            while True:
                ret, frame = capture.read()
                if ret:
                    if frame_idx % blip2_k == 0:
                        inputs = blip2_processor(images=frame, return_tensors="pt").to(device, torch.float16)
                        # 过滤长度超过100的caption
                        
                        generated_ids = blip2_model.generate(**inputs)
                        generated_text = blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                        if len(generated_text) > 90:
                            frame_idx = frame_idx + 1
                            continue


                        # 记录caption
                        print("frame_" + str(frame_idx) + ":" + generated_text)
                        video_text[str(frame_idx)] = generated_text + ". "
                        video_text_dict[frame_idx] = generated_text + ". "

                else:
                    break
                
                frame_idx = frame_idx + 1

        capture.release()

        # write video captions to json
        write_blip2_caption(video_text, video_name, dataset_name)

    # 选出相似度差距最大的k个位置
    # gap_indices = torch.topk(similarities.to(torch.float32), k=nFrames // 150, largest=False)[1]
    # gap_indices = gap_indices.to("cpu").tolist()
            
    # capture.release()

    # for index in gap_indices:
    #     segments_idx.append(index + 1)

    # segments_idx.append(int(nFrames))
    # segments_idx.sort()

    capture = cv2.VideoCapture(os.path.join(video_path, video_name))
    nFrames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if dataset_name == "SumMe":
        gt_file = HOMEDATA + '/' + video_name.split('.')[0]+'.mat'
        gt_data = scipy.io.loadmat(gt_file)
        if nFrames != gt_data.get('nFrames')[0][0]:
            nFrames = gt_data.get('nFrames')[0][0]
            print("SumMe: opencv get frames not equal to gt file.")

    similarities = torch.zeros(nFrames - 1).to(device)
    pre_frame_features = None
    frames_features = []
    frame_idx = 0
    with torch.no_grad():
        while(True):
            ret, frame = capture.read()
            if ret and frame_idx % (nFrames // 200) == 0:
                # 提取帧特征
                frame = clip_preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
                frame_features = clip_model.encode_image(frame)[0]

                frames_features.append(frame_features.to("cpu").numpy().ravel())
                # 计算当前帧与前一帧的特征的相似度
                if pre_frame_features is not None:
                    similarities[frame_idx - 1] = torch.cosine_similarity(frame_features, pre_frame_features, dim=0)

                pre_frame_features = frame_features.clone().detach()
            
            elif not ret:
                break

            frame_idx = frame_idx + 1
    
    capture.release()

    nFrames = frame_idx
    return frames_features, video_text_dict, blip2_k, nFrames
    # return video_text_dict, blip2_k


def filter_caption(video_text_dict, video_path, video_name, seg_beg, seg_end, blip2_k, blip2_processor, blip2_model, clip_preprocess, clip_model, device):
    seg_text_list = []
    for frame_index, frame_caption in video_text_dict.items():
        if frame_index >= seg_beg and frame_index <= seg_end:
            seg_text_list.append(frame_caption)

    # 如果该段没有caption
    with torch.no_grad():
        if len(seg_text_list) == 0:
            # load blip2
            # blip2_processor = AutoProcessor.from_pretrained(
            #     "/mnt/hdd8T/hh/code/llm/Models/hub/models--Salesforce--blip2-flan-t5-xl/snapshots/e5025a34e3e769e72e2aab7f7bfd00bc84d5fd77"
            # )
            # blip2_model = Blip2ForConditionalGeneration.from_pretrained(
            #     "/mnt/hdd8T/hh/code/llm/Models/hub/models--Salesforce--blip2-flan-t5-xl/snapshots/e5025a34e3e769e72e2aab7f7bfd00bc84d5fd77",
            #     torch_dtype=torch.float16,
            # )
            # blip2_model.to(device)
            # blip2_promt = "Question: What does this image show? Please give your answer in as much detail as possible. Answer:"
            
            blip2_k = 1
            capture_tmp = cv2.VideoCapture(os.path.join(video_path, video_name))
            seg_frame_idx = 0

            while True:
                if seg_frame_idx > seg_end:
                    break

                ret, frame = capture_tmp.read()
                if seg_frame_idx >= seg_beg and seg_frame_idx <= seg_end:
                    if ret:
                        inputs = blip2_processor(images=frame, return_tensors="pt").to(device, torch.float16)

                        generated_ids = blip2_model.generate(**inputs)
                        generated_text = blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                        
                        # 过滤长度超过100的caption
                        # if len(generated_text) > 100:
                        #     seg_frame_idx = seg_frame_idx + 1
                        #     continue

                        # 记录caption
                        seg_text_list.append(generated_text + ". ")
                    else:
                        break
                seg_frame_idx = seg_frame_idx + 1
            
            capture_tmp.release()
        
        # 找最适合帧的caption

        # 先提取所有句子的特征
        texts_features = []
        for caption in seg_text_list:
            caption = caption[:76]
            texts_features.append(clip_model.encode_text(clip.tokenize(caption).to(device))[0])

        capture = cv2.VideoCapture(os.path.join(video_path, video_name))
        frame_idx = 0
        seg_caption = ""
        chosen_text_idx = []
        while True:
            if frame_idx > seg_end:
                break

            ret, frame = capture.read()
            if ret and frame_idx >= seg_beg and frame_idx % blip2_k == 0:
                # 提取帧特征
                frame = clip_preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
                frame_features = clip_model.encode_image(frame)[0]

                # 找最匹配的句子
                idx = 0
                max_similarity = -2.0
                for i in range(len(texts_features)):
                    similarity = torch.cosine_similarity(texts_features[i], frame_features, dim=0)
                    
                    if similarity > max_similarity:
                        max_similarity = similarity
                        idx = i
                        
                # 如果多帧选到同一个caption，去重
                if idx not in chosen_text_idx:
                    chosen_text_idx.append(idx)
                    seg_caption = seg_caption + seg_text_list[idx]
                
            elif not ret:
                break

            frame_idx = frame_idx + 1

        capture.release()

        # seg_caption = ''
        # for caption in seg_text_list:
        #     seg_caption = seg_caption + caption

    return seg_caption


def get_all_frames_score(k, beta, alpha, video_text_sumy, all_segments_text_sumy_list, video_path, video_name, segments_idx, clip_preprocess, clip_model, device):
    # 各段文本摘要与总体摘要做匹配，计算各段的段score
    video_text_sumy_sentences = video_text_sumy.split(". ")
    video_text_sumy_sentences = [sentence.strip(string.punctuation) for sentence in video_text_sumy_sentences]
    # 提取整个视频的文本摘要的各个句子特征
    video_texts_sumy_features = []
    
    with torch.no_grad():
        for video_text_sumy_sentence in video_text_sumy_sentences:
            video_text_sumy_sentence = video_text_sumy_sentence[:76]
            video_texts_sumy_features.append(clip_model.encode_text(clip.tokenize(video_text_sumy_sentence).to(device))[0])

    
    capture = cv2.VideoCapture(os.path.join(video_path, video_name))
    frame_idx = 0
    res_frame_idx = []
    # 所有帧的重要度scores
    nFrames = segments_idx[-1]
    all_frames_score = torch.zeros(nFrames, dtype=torch.float32)
    for seg_idx in range(1, len(segments_idx)):
        # 段头和段尾的帧idx
        seg_beg = segments_idx[seg_idx - 1]
        seg_end = segments_idx[seg_idx] - 1

        if seg_idx > len(all_segments_text_sumy_list):
            continue

        seg_text_sumy = all_segments_text_sumy_list[seg_idx - 1]
        # 将视频片段的文本摘要分割成句子
        sentences = seg_text_sumy.split(". ")
        sentences = [sentence.strip(string.punctuation) for sentence in sentences]

        # 段分数
        seg_score = 0.0
        with torch.no_grad():
            for sentence in sentences:
                sentence = sentence[:76]
                sentence_features = clip_model.encode_text(clip.tokenize(sentence).to(device))[0]
                
                sentence_similarity = 0.0
                for video_text_sumy_features in video_texts_sumy_features:
                    sentence_similarity = sentence_similarity + max(0.0, torch.cosine_similarity(sentence_features, video_text_sumy_features, dim=0).to("cpu").item())
                
                seg_score = seg_score + sentence_similarity / len(video_texts_sumy_features)
        
        seg_score = seg_score / len(sentences)

        print("seg_" + str(seg_idx) + ": " + str(seg_score))
        
        # find the most cloest frames for each sentence using clip
        sample_frames = []
        while True:
            if frame_idx > seg_end:
                break

            ret, frame = capture.read()
            if (seg_end - seg_beg + 1) < k or frame_idx % k == 0:
                if ret:
                    sample_frames.append(frame)
                else:
                    break

            frame_idx = frame_idx + 1
        
        # print("sample frames num: " + str(len(sample_frames)) + "\n")

        seg_frames_features = []
        seg_texts_features = []
        # 提取该段的sumy caption的特征
        with torch.no_grad():
            for sentence in sentences:
                sentence = sentence[:76]
                seg_texts_features.append(clip_model.encode_text(clip.tokenize(sentence).to(device))[0])

        # 提取该段的帧的特征
        with torch.no_grad():
            for sample_frame in sample_frames:
                seg_frames_features.append(clip_model.encode_image(clip_preprocess(Image.fromarray(sample_frame)).unsqueeze(0).to(device))[0])

        # 计算每个句子的与该段所有帧的总相似度 再经过softmax得到权重
        seg_texts_weights = torch.zeros(len(seg_texts_features))
        with torch.no_grad():
            for i in range(len(seg_texts_features)):
                sum_similarity = 0.0
                for sample_frame_features in seg_frames_features:
                    sum_similarity = sum_similarity + max(0.0, torch.cosine_similarity(seg_texts_features[i], sample_frame_features, dim=0).to("cpu").item())

                seg_texts_weights[i] = sum_similarity

        seg_texts_weights = torch.Tensor(seg_texts_weights)

        # print('seg_text_weights_before_softmax')
        # print(seg_texts_weights)

        seg_texts_weights_sum = torch.sum(seg_texts_weights)
        seg_texts_weights = seg_texts_weights / seg_texts_weights_sum
        # seg_texts_weights = torch.nn.functional.softmax(seg_texts_weights, dim=0)

        # print('seg_text_weights')
        # print(seg_texts_weights)

        # 加权求每个帧的重要度：与该句子特征的相似度 * 该句子的权重
        seg_frames_score = torch.zeros(len(sample_frames))
        with torch.no_grad():
            for i in range(len(seg_frames_features)):
                frame_score = 0.0
                for j in range(len(seg_texts_features)):
                    frame_score = frame_score + seg_texts_weights[j] * max(0.0, torch.cosine_similarity(seg_frames_features[i], seg_texts_features[j], dim=0).to("cpu").item())
                
                seg_frames_score[i] = frame_score

        # print('frames_score_before_smo')
        # print(seg_frames_score)

        # frames_weights 平滑
        for i in range(1, len(seg_frames_score)):
            seg_frames_score[i] = beta * seg_frames_score[i] + (1.0 - beta) * seg_frames_score[i - 1]

        # print('frames_score_after_smo')
        # print(seg_frames_score)
        
        # align-with seg_score
        # seg_frames_score = seg_score * (seg_frames_score - torch.min(seg_frames_score)) / (torch.max(seg_frames_score) - torch.min(seg_frames_score) + 1e-6)
        
        # 加上段score
        seg_frames_score = seg_frames_score + (seg_score * alpha)
        # seg_frames_score = torch.zeros(len(seg_frames_score)) + (seg_score * alpha)

        # print('frames_score_add_seg_score')
        # print(seg_frames_score)
        
        # 归一化
        # seg_frames_score = (seg_frames_score - torch.min(seg_frames_score)) / (torch.max(seg_frames_score) - torch.min(seg_frames_score) + 1e-6)

        seg_frames_score = seg_frames_score.to(torch.float32)
        for i in range(len(seg_frames_score)):
            all_frames_score[i * k + seg_beg + (0 if seg_beg % k == 0 else (k - (seg_beg % k)))] = seg_frames_score[i]
        
    capture.release()

    return all_frames_score


# def gpt_api(promt, content):
#     # optional; defaults to `os.environ['OPENAI_API_KEY']`
#     openai.api_key = "sk-ygTXLFrak1bmEEWq68D84c919c2843788361Fa8b77E19735"
#     # openai.api_key = "sk-cSTFg2wZhWOBOANp0829061155F542178d8dCdAd93B52c0c"

#     # all client options can be configured just like the `OpenAI` instantiation counterpart
#     openai.base_url = "https://free.gpt.ge/v1/"
#     openai.default_headers = {"x-foo": "true"}

#     completion = openai.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {
#                 "role": "user",
#                 "content": promt + content,
#             },
#         ],
#     )
#     return completion.choices[0].message.content

# def happy_gpt():
#     # 这是你要发送的数据
#     data = {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "who are you?"
#             }
#         ],
#         "model": "gpt-3.5-turbo",
#         "temperature": 0.5,
#         "presence_penalty": 2
#     }
#     data =  json.dumps(data)
#     # 这是你要发送的头信息
#     headers = {
#         "content-type": "application/json",
#         "Authorization": "sk-cSTFg2wZhWOBOANp0829061155F542178d8dCdAd93B52c0c"
#     }

#     # 这是请求的URL
#     url = 'https://ngedlktfticp.cloud.sealos.io/v1/chat/completions'

#     # 发送POST请求
#     response = requests.post(url, data=data, headers=headers)

#     # 打印响应内容
#     print(response.text)

#     # 如果响应是JSON格式，可以这样获取
#     # 如果返回的内容不是json，这将引发一个异常
#     try:
#         response_json = response.json()
#     except ValueError as e:
#         print("Response isn't in JSON format.")
