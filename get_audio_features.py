import h5py
import os
import numpy as np
from tqdm import tqdm 

h = h5py.File('/ssd1/linjie/tvqa/tvr_feature_release/audio_feature/tvr_logmel_spectrugram_freq_80_time_125_clip_1.5.h5', "w")
resNet_feat = h5py.File("/ssd1/linjie/tvqa/tvr_feature_release/video_feature/tvr_resnet152_rgb_max_cl-1.5.h5", "r")
data_len = len(resNet_feat)
audio_feat_dir = "/ssd1/linjie/tvqa/audio_features_1.5/"
for key in tqdm(resNet_feat.keys()):
    curr_resnet_feat = np.array(resNet_feat.get(key), dtype="float32")
    tv_name = key.split("_")[0]
    if tv_name not in ["castle", "friends", "grey", "house", "met"]:
        tv_name = "bbt"
    curr_feat_path = os.path.join(audio_feat_dir, tv_name, key+".npz")
    resNet_feat_len = curr_resnet_feat.shape[0]
    curr_feat = np.load(curr_feat_path)[
        "audio_features"][:resNet_feat_len].astype("float32")
    h.create_dataset(key, data=curr_feat)
h.close()
resNet_feat.close()