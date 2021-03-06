import torch.utils.data as data
import os
import glob
import PIL.Image as Image
import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt


class AI4MarsDataset(data.Dataset):
    """
    A dataset class for the AI4Mars MSL data. It includes the grayscale images, depth data, and semantic labels.

    Parameters:
    -----------
    folder_path: the path to the MSL directory of the AI4Mars dataset

    is_train: True/False whether the dataset should be for the training or testing data
    
    """
    def __init__(self, folder_path, is_train, image_size=256):
        super(AI4MarsDataset, self).__init__()
        self.image_size = image_size
        self.img_files = []
        self.depth_files = []
        self.label_files = []
        if is_train:
            all_label_files = glob.glob(os.path.join(folder_path,'labels','train','*.png'))
        else:
            all_label_files = glob.glob(os.path.join(folder_path,'labels','test','masked-gold-min1-100agree','*.png'))
        for label_path in all_label_files:
            if is_train: 
                base_name = os.path.basename(label_path)[:-4]
            else:
                base_name = os.path.basename(label_path)[:-11]
            depth_path = os.path.join(folder_path,'images','rng_256',base_name[0:13]+'RNG'+base_name[16:]+".tiff")
            image_path = os.path.join(folder_path,'images','edr',base_name+".JPG")
            if os.path.exists(depth_path) and os.path.exists(image_path):
                self.img_files.append(image_path)
                self.depth_files.append(depth_path)
                self.label_files.append(label_path)

    def __getitem__(self, index):
        image_size = self.image_size
        img_path = self.img_files[index]
        label_path = self.label_files[index]
        depth_path = self.depth_files[index]
        image = Image.open(img_path)
        depth = Image.open(depth_path)
        label = Image.open(label_path)
        image = image.resize((image_size,image_size),Image.ANTIALIAS)
        depth = depth.resize((image_size,image_size),Image.ANTIALIAS)
        label = label.resize((image_size,image_size),Image.NEAREST)
        image = np.asarray(image)
        depth = np.asarray(depth)
        label = np.asarray(label)
        image = torch.from_numpy(image).float()
        depth = torch.from_numpy(depth).float()
        label = torch.from_numpy(label).long()
        
        return image, depth, label

    def __len__(self):
        return len(self.img_files)

    def display_image(self, index):
        image_size = self.image_size
        img_path = self.img_files[index]
        depth_path = self.depth_files[index]
        image = Image.open(img_path)
        depth = Image.open(depth_path)
        image = image.resize((image_size,image_size),Image.ANTIALIAS)
        depth = depth.resize((image_size,image_size),Image.ANTIALIAS)
        image = np.ascontiguousarray(image, dtype=np.float32).reshape(image_size,image_size,1)
        depth = np.ascontiguousarray(depth, dtype=np.float32).reshape(image_size,image_size,1)
        color_raw = o3d.geometry.Image(image)
        depth_raw = o3d.geometry.Image(depth)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
        plt.subplot(1, 2, 1)
        plt.title('Grayscale Image')
        plt.imshow(rgbd_image.color)
        plt.subplot(1, 2, 2)
        plt.title('Depth Image')
        plt.imshow(rgbd_image.depth)
        plt.show()