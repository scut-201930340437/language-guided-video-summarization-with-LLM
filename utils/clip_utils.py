import os
import cv2
import pdb
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModelWithProjection

def frame_features_extractor(video_path, video_name, device):
    processor = CLIPProcessor.from_pretrained('/home/lab345/gw/models/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268')
    model = CLIPVisionModelWithProjection.from_pretrained('/home/lab345/gw/models/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268').to(device)

    cap = cv2.VideoCapture(os.path.join(video_path, video_name))
    fps = cap.get(cv2.CAP_PROP_FPS)

    
    sample_rate = int(fps) // 2


    clip_features = []
    frame_idx = 0
    print("Extract the clip feature.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = processor(images=image, return_tensors="pt").pixel_values
            inputs = inputs.to(device)
            
            with torch.no_grad():
                feat = model(inputs)['image_embeds']
                clip_features.append(feat.cpu().numpy())
        
        frame_idx = frame_idx + 1

    print("Finished.")

    clip_features = np.concatenate(clip_features, axis=0)
    
    cap.release()
        
    return clip_features, sample_rate, frame_idx