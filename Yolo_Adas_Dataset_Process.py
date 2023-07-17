#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 18:22:59 2023

@author: jnr_loganvo
"""

import glob
import os
import numpy as np
import cv2
import copy

class ADAS_Dataset:
    
    def __init__(self,
                 base_dir,
                 img_dir,
                 label_dir,
                 data_type="train",
                 seg_type="lane",
                 lane_bg=0,
                 save_to_png=True):
        
        self.base_dir = base_dir
        self.label_dir = label_dir
        self.data_type = data_type
        self.seg_type = seg_type
        self.img_dir = os.path.join(img_dir,"100k","train")
        self.img_path_list = glob.glob(os.path.join(self.img_dir,"*.jpg"))
        self.label_path_list = glob.glob(os.path.join(self.label_dir,self.seg_type,"masks",self.data_type,"*.png"))
        self.label_dri_path_list = glob.glob(os.path.join(self.label_dir,"drivable","masks",self.data_type,"*.png"))
        self.lane_bg = lane_bg
        self.lane_mapping = {0: 2,            1: lane_bg,     2: 1,           3: 3,           4: 5, 
                        5: lane_bg,     6: 2,           7: 3,           8: lane_bg,     9: lane_bg,
                        10: lane_bg,    11: lane_bg,    12: lane_bg,    13: lane_bg,    14: lane_bg,
                        15: lane_bg,    16: 2,          17: lane_bg,    18: 1,          19: 3,
                        20: 5,          21: lane_bg,    22: 2,          23: 3,          24: lane_bg,
                        25: lane_bg,    26: lane_bg,    27: lane_bg,    28: lane_bg,    29: lane_bg,
                        30: lane_bg,    31: lane_bg,    32: 4,          33: lane_bg,    34: lane_bg,  
                        35: lane_bg,    36: 5,          37: lane_bg,    38: 4,          39: lane_bg,
                        40: lane_bg,    41: lane_bg,    42: lane_bg,    43: lane_bg,    44: lane_bg,
                        45: lane_bg,    46: lane_bg,    47: lane_bg,    48: 4,          49: lane_bg,
                        50: lane_bg,    51: lane_bg,    52: 5,          53: lane_bg,    54: 4,
                        55: lane_bg,
                        255: lane_bg}
        self.merge_mapping = {'drive': {0: 1,   1: 2,   2: 0},
                              'lane':  {1: 3,   2: 4,    3: 5,    4: 6,     5: 7}}
        self.save_to_png = save_to_png
        self.dilation_kernel = 7
        self.lane_order= [4, 2, 3, 1, 5] 
    def map_labels(self, mask, map, bg_lb):
        """Update the segmentation mask labels by the mapping variable."""
        out = np.empty((mask.shape), dtype=mask.dtype)
        for k, v in map.items():
            if isinstance(v, str):
                out[mask == k] = bg_lb
            else:
                out[mask == k] = v
        return out
    
    def Get_Horizontal_Line_Images_Labels(self):
        import cv2
        import shutil
        c = 1
        for i in range(len(self.label_path_list)):
            #print(self.label_path_list[i])
            mask_lane = cv2.imread(self.label_path_list[i],cv2.IMREAD_GRAYSCALE)
            label_file,label_file_name = self.parse_path(self.label_path_list[i])
            #print(label_file)
            #print(label_file_name)
            img_file = label_file_name + '.jpg'
            #print(img_file)
            img_file_path = os.path.join(self.img_dir,img_file)
            #print(img_file_path)
            '''Step 1 : Map class (Re-assign the lane line labels)'''
            new_mask_lane = self.map_labels(mask_lane,self.lane_mapping,self.lane_bg)
            '''Step 2 : Get the mask that contain the horizontal line'''
            print(i)
            if len(new_mask_lane[new_mask_lane==4])>0:
                print("found horizontal line {} !".format(c))
                c+=1
                if self.save_to_png:
                    '''Save label mask to .png'''
                    save_label_dir = os.path.join(base_dir,"label_horizontal","horizontal_line","masks",self.data_type)
                    if not os.path.exists(save_label_dir):
                        os.makedirs(save_label_dir)
                    label_file,label_file_name = self.parse_path(self.label_path_list[i])
                    label_path = os.path.join(save_label_dir,label_file)
                    cv2.imwrite(label_path,mask_lane)
                    
                    '''Save label mask to .png'''
                    save_color_label_dir = os.path.join(base_dir,"label_horizontal","horizontal_line","colormaps",self.data_type)
                    if not os.path.exists(save_color_label_dir):
                        os.makedirs(save_color_label_dir)
                    label_file,label_file_name = self.parse_path(self.label_path_list[i])
                    colormap = os.path.join(self.label_dir,self.seg_type,"colormaps",self.data_type,label_file)
                    shutil.copy(colormap,save_color_label_dir)
                    
                    '''Save image to .jpg'''
                    save_img_dir = os.path.join(base_dir,"image_horizontal","100k",self.data_type)
                    if not os.path.exists(save_img_dir):
                        os.makedirs(save_img_dir)
                    #img_file,img_file_name = self.parse_path(self.label_path_list[i])
                    #new_img_file = img_file_name+'.jpg'
                    #img_path = os.path.join(save_img_dir,new_img_file)
                    shutil.copy(img_file_path,save_img_dir)
                    print("copy h-image successful")
    def parse_path(self,path):
        file = path.split("/")[-1]
        file_name = file.split(".")[0]
        return file, file_name
    
    
    def merge_segmentation_heads(self, drive, lane):
        """Merging drivable area and lane line labels into one segmentation label."""
        out = np.zeros_like(drive, dtype=drive.dtype)
        
        drive_map = self.merge_mapping['drive']
        for k, v in drive_map.items():
            out[drive == k] = v
        
        lane_map = self.merge_mapping['lane']
        for k, v in lane_map.items():
            out[lane == k] = v
        return out

    
    
    def split_map_and_dilate(self, mask):
        """Split mask to n-channels map where n is equal to the number of foreground classes that needs to be dilated."""
        kernel = np.ones((self.dilation_kernel, self.dilation_kernel), dtype=np.uint8)
        cls = np.unique(mask)
        if len(cls) >= 2:   # if there is foreground
            cls = cls[1:]   # ignore background
            if isinstance(cls, int):
                cls = [cls]
            
            layers = np.zeros((len(cls), mask.shape[0], mask.shape[1]), dtype=np.uint8)
            for i, lb in enumerate(cls):
                    layers[i][mask == lb] = 255     # trick to dilate #255
                    layers[i] = cv2.dilate(layers[i], kernel, iterations=1)
                    layers[i][np.where(layers[i] == 255)] = lb
            
            outputs = np.zeros_like(mask, dtype=np.uint8)
            for j in self.lane_order:   # set it by priority
                if j in cls:
                    outputs[np.where(layers == j)[1:]] = j
            return outputs
        else:
            return mask
    
    def line_enhancer(self, mask_drive, mask_line):  # TODO: Optimize further with more flexible options
        drive_tmp = copy.deepcopy(mask_drive)
        line_tmp = copy.deepcopy(mask_line)
        mask_out = np.zeros(drive_tmp.shape)

        drive_tmp[mask_drive == 1] = 0
        drive_tmp[mask_drive == 2] = 255
        line_tmp[mask_line == 1] = 255
        line_tmp[mask_line == 2] = 255
        line_tmp[mask_line == 3] = 255
        line_tmp[mask_line == 4] = 0
        line_tmp[mask_line == 5] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        drive_tmp = cv2.dilate(drive_tmp, kernel=kernel)
        mask_out = cv2.bitwise_and(drive_tmp, line_tmp)

        mask_out[mask_out == 0] = mask_line[mask_out == 0]
        mask_out[mask_out > 0] = mask_line[mask_out > 0]
        # print('Reached!')
        return mask_out
    
    def Get_Boundary_Map(self):
        for i in range(len(self.label_path_list)):
            
            mask_lane = cv2.imread(self.label_path_list[i],cv2.IMREAD_GRAYSCALE)
            label_file,label_file_name = self.parse_path(self.label_path_list[i])
            mask_dri = cv2.imread(os.path.join(self.label_dir,"drivable","masks","train",label_file),cv2.IMREAD_GRAYSCALE)
            mask_lane = self.map_labels(mask_lane,self.lane_mapping,self.lane_bg)
            #mask_lane = self.merge_segmentation_heads(self,self.lane_mapping)
            mask_lane = self.split_map_and_dilate(mask_lane)
            mask_lane = self.line_enhancer(mask_dri,mask_lane)
            
            bd_mask = self.gen_boundary(mask_lane)
            
            cv2.imshow("bd", bd_mask)
            cv2.waitKey(0)
            '''Save boundary mask to .png'''
            save_boundary_map_dir = os.path.join(base_dir,"label_boundary","line_boundary","maps",self.data_type)
            if not os.path.exists(save_boundary_map_dir):
                os.makedirs(save_boundary_map_dir)
            
            bd_file = label_file_name + '.png'
            lane_file = label_file_name + '.jpg'
            save_bd_file_path = os.path.join(save_boundary_map_dir,bd_file)
            save_l_file_path = os.path.join(save_boundary_map_dir,lane_file)
            cv2.imwrite(save_bd_file_path,bd_mask)
            cv2.imwrite(save_l_file_path,mask_lane)
            print(i)
            
    def gen_boundary(self, mask, dilation=True, kernel_size=3):
        """Generate boundary mask in case of using boundary for segmentation task."""
        bd = cv2.Canny(mask, 0.1, 0.7)
        if dilation:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            bd = (cv2.dilate(bd, kernel=kernel, iterations=1) > 50) * 1.0   #TODO: Draw and check boundary map
        return bd
            
if __name__=="__main__":
    base_dir = "/home/jnr_loganvo/Alister/datasets/YOLO_ADAS/bdd100k_data"
    img_dir = os.path.join(base_dir,"images")
    print(img_dir)
    label_dir = os.path.join(base_dir,"labels")
    print(label_dir)
    data_type="train"
    seg_type="lane"
    yolo_adas_data = ADAS_Dataset(base_dir,
                                  img_dir,
                                  label_dir,
                                  data_type,
                                  seg_type)
    #yolo_adas_data.Get_Horizontal_Line_Images_Labels()
    yolo_adas_data.Get_Boundary_Map()