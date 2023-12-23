import os
import cv2
import glob
import torch
import json

from viclip import get_clip, frames2tensor, get_vid_feat

root = "/home/ai2lab/Desktop/DLCV-Fall-2023-Final-1-catchingstar/data/Charades_frame/frames_fps1"
json_path = "/home/ai2lab/Desktop/DLCV-Fall-2023-Final-1-catchingstar/data/star/STAR_train.json"
# json_path = "/home/ai2lab/Desktop/DLCV-Fall-2023-Final-1-catchingstar/data/star/STAR_val.json"
# json_path = "/home/ai2lab/Desktop/DLCV-Fall-2023-Final-1-catchingstar/data/star/STAR_test.json"
json_data = json.load(open(json_path, 'r'))
print("json_data:", len(json_data))
device = torch.device('cuda')

clip, tokenizer = get_clip()
clip = clip.to(device)

# feature_dict = {}
# feature_dict = torch.load("STAR_train_video_feature.pt")
# feature_dict = torch.load("STAR_train_val_video_feature.pt")
feature_dict = torch.load("STAR_video_feature.pt")


# check if all question_id has been processed
'''
for item in json_data:
    question_id = item['question_id']
    if question_id not in feature_dict:
        print(question_id)
exit()
'''

for cnt, item in enumerate(json_data):
    print(f"{cnt}/{len(json_data)}")
    question_id = item['question_id']
    start, end = round(item['start']), round(item['end'])
    # print(item['question_id'])
    # print(item['start'], item['end'])
    # print(start, end)

    video_path = os.path.join(root, item['video_id'])
    frame_paths = glob.glob(os.path.join(video_path, "*.jpg"))

    frames = []
    for i in range(start, end+1):
        frame_path = os.path.join(video_path, f"frame_{i:04d}.jpg")
        frame = cv2.imread(frame_path)
        if frame is not None:
            frames.append(frame)
    # print("frames:", len(frames))

    # get feature (1 * 768)
    frames_tensor = frames2tensor(frames, device=device) 
    # print("frames_tensor:", frames_tensor.shape) # (1, 8, 3, 224, 224)
    vid_feat = get_vid_feat(frames_tensor, clip)
    # print("vid_feat:", vid_feat.shape) # (1, 768)

    feature_dict[question_id] = vid_feat

# print(feature_dict)
torch.save(feature_dict, "STAR_video_feature.pt")
