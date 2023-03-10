import cv2
import json
import torch
from torch.utils.data import Dataset
import os
import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from constants import DATASETS


class BivariateDataset:
    def __init__(self, path_to_annotations, name):
        # Check
        if name not in ['train', 'val','test']:
            print("Be sure that the name is all in lowercase OR the name you defined is not acceptable")
            sys.exit()
            
        self.name = name
        self.path = os.path.join(path_to_annotations, 'instances_'+self.name+'.json')
        with open(self.path, 'r') as f:
            all_json = json.load(f)
            
        # There is an assumption that there are two categories. Check!
        if len(all_json['categories']) != 2:
            print("Dataset is not bivariate")
            sys.exit()
        
        self.first_cat_name = all_json['categories'][0]['name'] 
        self.second_cat_name = all_json['categories'][1]['name']
        self.num_first_cat = 0
        self.num_second_cat = 0
        
        # Accumulate values
        for i in range(len(all_json['annotations'])):
            if all_json['annotations'][i]['category_id'] == 1:
                self.num_first_cat += 1
            elif all_json['annotations'][i]['category_id'] == 2:
                self.num_second_cat += 1
            else:
                print("Unknown category at annotation" + str(i)+ "!")
                sys.exit() 
                
        self.num_total = self.num_first_cat + self.num_second_cat
        self.img_num = len(all_json['images'])
                
    def visulize_pie_chart(self, title= "Bivariate Dataset Pie Chart"):
        vis = np.array([self.num_first_cat, self.num_second_cat])
        labels = [self.first_cat_name, self.second_cat_name]
        plt.pie(vis,labels=labels, autopct='%1.1f%%' )
        plt.title(title)
        plt.show()
        
        
class DatasetNumericalAnalysis:
    def __init__(self, path_to_annotations):
        self.path = path_to_annotations
        self.datasets = DATASETS        
        self.train_img_num = None
        self.test_img_num = None
        self.val_img_num = None        
        for dt in self.datasets:
            bivariate_obj = BivariateDataset(self.path, dt)
            if dt == DATASETS[0]:
                self.train_img_num = bivariate_obj.img_num
            elif dt == DATASETS[1]:
                self.val_img_num = bivariate_obj.img_num
            elif dt == DATASETS[2]:
                self.test_img_num = bivariate_obj.img_num
                
                
    def vis_compare_categories(self):
        nums = [self.train_img_num, self.val_img_num, self.test_img_num]
        vis = np.array(nums)
        labels = self.datasets
        plt.pie(vis, labels=labels, autopct='%1.1f%%' )
        plt.title("Overall Dataset Distribution")
        plt.show()      
    
    def vis_compare_subcategories(self):
        for dt in self.datasets:
            bivariate_obj = BivariateDataset(self.path, dt)
            bivariate_obj.visulize_pie_chart(dt)
            
    def check_img_sizes(self):
        # TO DO
        pass
            

