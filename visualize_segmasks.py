import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

seg_folder = '/home/nick/segmentations'
im_folder = '/home/nick/seg_imgs'

os.makedirs(im_folder, exist_ok=True)

for mask in tqdm(os.listdir(seg_folder)):
    mask_arr = np.load(os.path.join(seg_folder, mask))
    save_file = os.path.join(im_folder, mask.replace('.npy', '.png'))
    plt.imsave(save_file, mask_arr)

