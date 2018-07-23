
# coding: utf-8

# # Simple Dataset using PyTorch

# In[12]:

import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
import pandas as pd

import torch

from torch.utils import data


get_ipython().magic('matplotlib inline')


# ## PyTorch Dataset definition

# In[3]:

class TGSSaltDataset(data.Dataset):
    
    def __init__(self, root_path, file_list):
        self.root_path = root_path
        self.file_list = file_list
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))
        
        file_id = self.file_list[index]
        
        image_folder = os.path.join(self.root_path, "images")
        image_path = os.path.join(image_folder, file_id + ".png")
        
        mask_folder = os.path.join(self.root_path, "masks")
        mask_path = os.path.join(mask_folder, file_id + ".png")
        
        image = np.array(imageio.imread(image_path), dtype=np.uint8)
        mask = np.array(imageio.imread(mask_path), dtype=np.uint8)
        
        return image, mask


# In[7]:

data_folder = '/home/eee/ug/15084015/.kaggle/competitions/tgs-salt-identification-challenge/'
depths_df = pd.read_csv(data_folder + 'train.csv')

train_path = data_folder + "train/"
file_list = list(depths_df['id'].values)


# In[9]:

dataset = TGSSaltDataset(train_path, file_list)


# Let us visualize a few samples from the dataset.

# In[18]:

def plot2x2Array(image, mask):
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(image)
    axarr[1].imshow(mask, cmap='binary')
    axarr[0].grid()
    axarr[1].grid()
    axarr[0].set_title('Image')
    axarr[1].set_title('Mask')


# In[19]:

for i in range(5):
    image, mask = dataset[np.random.randint(0, len(dataset))]
    plot2x2Array(image, mask)


# In[22]:

np.unique(mask, return_counts=True)


# In[ ]:

get_ipython().system('jupyter nbconvert --to script script.ipynb')

