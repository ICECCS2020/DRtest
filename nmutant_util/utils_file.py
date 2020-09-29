from scipy import ndimage
import os
from PIL import Image, ImageFilter
import numpy as np

class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)

def get_multiple_data(file_path_list):
    image_list = []
    real_labels = []
    predicted_labels = []
    image_files = []
    for file_path in file_path_list:
        for img_file in os.listdir(file_path):
            if img_file.endswith('.npy'):
                img_file_split = img_file.split('_')
                real_label = int(img_file_split[-3])
                predicted_label = int(img_file_split[-2])

                if real_label != predicted_label:  # only extract those successfully generated adversaries
                    real_labels.append(real_label)
                    predicted_labels.append(predicted_label)
                    current_img = ndimage.imread(file_path + os.sep + img_file)
                    image_list.append(current_img)
                    image_files.append(img_file)
    print('--- Total number of images: ', len(image_list))
    return image_list, image_files, real_labels, predicted_labels

def get_data_file(file_path):
    image_list = []
    real_labels = []
    predicted_labels = []
    image_files =[]
    for img_file in os.listdir(file_path):
        if img_file.endswith('.npy'):
            img_file_split = img_file.split('_')
            real_label = int(img_file_split[-2])
            predicted_label = int(img_file_split[-1].split('.')[-2])

            if real_label!=predicted_label: # only extract those successfully generated adversaries
                real_labels.append(real_label)
                predicted_labels.append(predicted_label)
                current_img = np.load(file_path + os.sep + img_file)
                #print(current_img.shape)
                image_list.append(current_img)
                image_files.append(img_file)
    print('--- Total number of images: ', len(image_list))
    return image_list, image_files, real_labels, predicted_labels


def get_data_file_with_Gaussian(dataset, file_path, radius):
    image_list = []
    real_labels = []
    predicted_labels = []
    image_files =[]
    for img_file in os.listdir(file_path):
        if img_file.endswith('.npy'):
            img_file_split = img_file.split('_')
            real_label = int(img_file_split[-2])
            predicted_label = int(img_file_split[-1].split('.')[-2])

            if real_label!=predicted_label: # only extract those successfully generated adversaries
                real_labels.append(real_label)
                predicted_labels.append(predicted_label)
                
                #current_img = ndimage.imread(file_path + os.sep + img_file)
                current_img = np.load(file_path + os.sep + img_file)*255
                if dataset=='mnist':
                    current_img=current_img.reshape(current_img.shape[0], current_img.shape[1])
                elif dataset=='cifar10':
                    current_img=current_img.reshape(current_img.shape[0], current_img.shape[1], current_img.shape[2])
                #print(current_img)
                #print('**********')
                #print(current_img.shape)        
                img=Image.fromarray(np.uint8(current_img))
                #print(current_img)
                img_Gua=img.filter(MyGaussianBlur(radius=int(radius)))
                current_img = np.asarray(img_Gua)
                
                image_list.append(current_img)
                image_files.append(img_file)
    print('--- Total number of images: ', len(image_list))
    return image_list, image_files, real_labels, predicted_labels


def get_data_file_with_Compression(file_path):
    image_list = []
    real_labels = []
    predicted_labels = []
    image_files =[]
    for img_file in os.listdir(file_path):
        if img_file.endswith('.png'):
            img_file_split = img_file.split('_')
            real_label = int(img_file_split[-2])
            predicted_label = int(img_file_split[-1].split('.')[-2])

            if real_label!=predicted_label: # only extract those successfully generated adversaries
                real_labels.append(real_label)
                predicted_labels.append(predicted_label)

                current_img = ndimage.imread(file_path + os.sep + img_file)

                img=Image.fromarray(np.uint8(current_img))
                #print(current_img)
                w,h=img.size
                img.resize((w/2, h/2))

                current_img = np.asarray(img)
                image_list.append(current_img)
                image_files.append(img_file)
    print('--- Total number of images: ', len(image_list))
    return image_list, image_files, real_labels, predicted_labels





