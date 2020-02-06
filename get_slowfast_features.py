import h5py
import os
import numpy as np
from tqdm import tqdm 
FEAT_LEN=2304+2048
h = h5py.File('/ssd1/linjie/tvqa/tvqa/tvr_feature_release/video_feature/tvr_resnet152_rgb_max_slowfast_rgb400_cat-1.5.h5', "w")
resNet_feat = h5py.File("/ssd1/linjie/tvqa/tvqa/tvr_feature_release/video_feature/tvr_resnet152_rgb_max_cl-1.5.h5", "r")
data_len = len(resNet_feat)
slowfast_feat_dir = "/ssd1/linjie/tvqa/slowfast_features/"
for key in tqdm(resNet_feat.keys()):
    curr_resnet_feat = np.array(resNet_feat.get(key), dtype="float32")
    tv_name = key.split("_")[0]
    if tv_name not in ["castle", "friends", "grey", "house", "met"]:
        tv_name = "bbt"
    curr_feat_path = os.path.join(slowfast_feat_dir, tv_name, key+".npz")
    curr_feat = np.load(curr_feat_path)["features"].astype("float32")
    resNet_feat_len = curr_resnet_feat.shape[0]
    new_feat = np.concatenate(
        (curr_resnet_feat, curr_feat[:resNet_feat_len]), axis=1)
    h.create_dataset(key, data=new_feat)
h.close()
resNet_feat.close()