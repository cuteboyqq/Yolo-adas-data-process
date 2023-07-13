#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 13:34:00 2023

@author: jnr_loganvo
"""

import glob
import os
import numpy as np
import cv2

lane_bg = 0
lane_mapping = {0: 2,            1: lane_bg,     2: 1,           3: 3,           4: 5, 
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
'''
Merge label:
0:background
1:main lane
2:left line
3:right line
'''
merge_mapping = {'drive': {0: 1,   1: 0,   2: 0},
                'lane':   {0: 0,   2: 2,   3: 3}}

def map_labels(mask, map, bg_lb):
    """Update the segmentation mask labels by the mapping variable."""
    out = np.empty((mask.shape), dtype=mask.dtype)
    for k, v in map.items():
        if isinstance(v, str):
            out[mask == k] = bg_lb
        else:
            out[mask == k] = v
    return out


def split_map_and_dilate(mask):
    dilation_kernel = 7
    """Split mask to n-channels map where n is equal to the number of foreground classes that needs to be dilated."""
    kernel = np.ones((dilation_kernel,dilation_kernel), dtype=np.uint8)
    cls = np.unique(mask)
    if len(cls) >= 2:   # if there is foreground
        cls = cls[1:]   # ignore background
        if isinstance(cls, int):
            cls = [cls]
        
        layers = np.zeros((len(cls), mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for i, lb in enumerate(cls):
                layers[i][mask == lb] = 255     # trick to dilate
                layers[i] = cv2.dilate(layers[i], kernel, iterations=1)
                layers[i][np.where(layers[i] == 255)] = lb
        
        outputs = np.zeros_like(mask, dtype=np.uint8)
        for j in [4, 2, 3, 1, 5]:   # set it by priority
            if j in cls:
                outputs[np.where(layers == j)[1:]] = j
        return outputs
    return mask


def merge_segmentation_heads(drive, lane):
        """Merging drivable area and lane line labels into one segmentation label."""
        
        out = np.zeros_like(drive, dtype=drive.dtype)
        
        drive_map = merge_mapping['drive']
        for k, v in drive_map.items():
            out[drive == k] = v
        
        lane_map = merge_mapping['lane']
        for k, v in lane_map.items():
            out[lane == k] = v
        return out
def parse_path(path):
    file = path.split("/")[-1]
    file_name = file.split(".")[0]
    return file,file_name

def parsing_imgs_and_labels(data_dir):
    '''Step 0 : Get lane mask list
                Get dri mask list
                '''
    label_lane_dir = os.path.join(data_dir,"labels","lane","masks","train")
    label_dri_dir = os.path.join(data_dir,"labels","drivable","masks","train")
    train_lane_mask_list = glob.glob(os.path.join(label_lane_dir,"*.png"))
    train_dri_mask_list = glob.glob(os.path.join(label_dri_dir,"*.png"))
    
    c = 1
    
    '''Step 0 : Read the lane/dri mask in the list'''
    for i in range(len(train_lane_mask_list)):
        print("{}:{}".format(i,train_lane_mask_list[i]))
        mask_lane = cv2.imread(train_lane_mask_list[i],cv2.IMREAD_GRAYSCALE)
        mask_dri = cv2.imread(train_dri_mask_list[i],cv2.IMREAD_GRAYSCALE)
        #print(mask_lane.shape)
        
        file,file_name = parse_path(train_lane_mask_list[i])
        #print(file)
        #print(file_name)
        
        '''Step 1 : Map class (Re-assign the lane line labels)'''
        map_lane_class = map_labels(mask_lane, lane_mapping, lane_bg)
        '''Step 2 : Dilate lane line'''
        split_lane_map = split_map_and_dilate(map_lane_class)
        #'''Step 3 : Merge class'''
        #merged_map = merge_segmentation_heads(mask_dri,split_lane_map)
        '''Step 3 : Generate New laen line label'''
        save_dri_4cls_color = "/home/jnr_loganvo/Alister/datasets/YOLO_ADAS/bdd100k_data/labels/main_road/color_masks/train"
        if not os.path.exists(os.path.join(save_dri_4cls_color,file)):
            new_lane_map = pre_processing_mainlane_line(split_lane_map,
                                                        mask_dri,
                                                        c,
                                                        save_color_map=True,
                                                        save_file=file)
        else:
            print("label exists ! Pass Step3 !!")
        #'''Step 4 : Merge class'''
        #merged_map = merge_segmentation_heads(mask_dri,new_lane_map)
        
        c+=1
        #mask = np.squeeze(mask)
        #new_mask = split_map_and_dilate(mask)
        #cv2.imshow('My Image', new_mask)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
import torch
def pre_processing_mainlane_line(mask_line, 
                                 mask_dri,
                                 count,
                                 save_color_map=False,
                                 save_file=None):
        #lane_bg = self.hyp.get('lane_bg')
        # re-label mask_dri and mask_line
        #mask_line_color = np.empty(mask_dri.shape)
        mask_line_color = np.empty((mask_dri.shape[0],mask_dri.shape[1],3))
       
        mask_out = np.zeros(mask_dri.shape)
        for i in range(mask_dri.shape[0]):
            for j in range(mask_dri.shape[1]):
                if(mask_dri[i][j]==0):
                    mask_out[i][j] = 1
                elif(mask_dri[i][j]==1):
                    mask_out[i][j] = 0
                elif(mask_dri[i][j]==2):
                    mask_out[i][j] = 0 

        SAVE_COLOR_MAP = save_color_map
        if SAVE_COLOR_MAP:
            
            save_line_dir = "./line"
            save_dri_dir = "./dri"
            save_dri_4cls_color = "/home/jnr_loganvo/Alister/datasets/YOLO_ADAS/bdd100k_data/labels/main_road/color_masks/train"
            save_dri_4cls =      "/home/jnr_loganvo/Alister/datasets/YOLO_ADAS/bdd100k_data/labels/main_road/masks/train"
            #if not os.path.exists(save_line_dir):
            #    os.makedirs(save_line_dir)
            #if not os.path.exists(save_dri_dir):
            #    os.makedirs(save_dri_dir)
            if not os.path.exists(save_dri_4cls_color):
                os.makedirs(save_dri_4cls_color)
            if not os.path.exists(save_dri_4cls):
                os.makedirs(save_dri_4cls)
            
            
            
            mask_out_color = np.empty((mask_dri.shape[0],mask_dri.shape[1],3))
            mask_out_png = np.zeros((mask_dri.shape[0],mask_dri.shape[1],1))
            for i in range(mask_dri.shape[0]):
                for j in range(mask_dri.shape[1]):
                    if mask_out[i][j] == 0:
                        mask_out_color[i][j] = 255
                        mask_out_png[i][j] = 0
                    elif mask_out[i][j] == 1:
                        mask_out_color[i][j] = (255,165,0)
                        mask_out_png[i][j] = 1
                    
                   
            # mask_dri_color = torch.empty(mask_dri.shape)
            # for i in range(mask_dri.shape[0]):
            #     for j in range(mask_dri.shape[1]):
            #         if(mask_dri[i][j]==0):
            #             mask_dri_color[i][j] = 100
            #         elif(mask_dri[i][j]==2):
            #             mask_dri_color[i][j] = 255
            #         # elif(mask_dri[i][j]==2):
            #         #     mask_dri_color[i][j] = 255
        
        for i in range(mask_dri.shape[0]): # h=360 (0,1,2,3,4....,359)
            find_main_drivable_point=False
            main_dri_x = 0
            main_dri_y = 0

            left_line_x_l = -1
            left_line_y_l = -1
            
            right_line_x_r = -1
            right_line_y_r = -1
            #print("mask_dri shape : {}".format(mask_dri.shape))
            #print("search y = {}".format(i))
            #print("=================================================================================")
            for j in range(mask_dri.shape[1]): # w
                #print("mask_dri [{}][{}] = {}".format(i,j,mask_dri[i][j]))
                if mask_dri[i][j] == 0 and find_main_drivable_point==False: #Main Drivable Area
                    find_main_drivable_point=True
                    #print("find_main_drivable_point ")
                    main_dri_x = j
                    main_dri_y = i
                    #print("find_main_drivable_point :mask[{}][{}]={}".format(i,j,mask_dri[main_dri_y][main_dri_x]))
                    find_left_line_pixel = False
                    find_right_line_pixel = False
                    left_line_x_r = -1
                    left_line_y_r = -1

                    right_line_x_l = -1
                    right_line_y_l = -1

                    line_label = 0
                    if find_main_drivable_point:
                        #======================Left side line===========================================================
                        for x in range(main_dri_x,0,-1): #search right boundary of line which is at left side of image
                            if mask_line[main_dri_y][x] != 0 and find_left_line_pixel==False: #Not Main Drivable Area
                                find_left_line_pixel=True
                                left_line_x_r = x
                                left_line_y_r = main_dri_y
                                line_label = mask_line[main_dri_y][x]
                                #print("left side line_r_pixel : mask[{}][{}]={}".format(left_line_y_r,left_line_x_r,mask_line[left_line_y_r][left_line_x_r]))
                        
                        for x in range(left_line_x_r,0,-1): #search left boundary of line which is at left side of image
                            if mask_line[main_dri_y][x] != line_label  and find_left_line_pixel==True:
                                find_left_line_pixel = False
                                left_line_x_l = x
                                left_line_y_l = main_dri_y
                                #print("left side line_l_pixel : mask[{}][{}]={}".format(left_line_y_l,left_line_x_l,mask_line[left_line_y_l][left_line_x_l]))
                                #print("=======================================================")
                        #======================Right side line===========================================================
                        for x in range(main_dri_x+50,mask_dri.shape[1]-1,1): #search left boundary of line which is at right side of image
                            if mask_line[main_dri_y][x] != 0 and find_right_line_pixel==False: #Not Main Drivable Area
                                find_right_line_pixel=True
                                right_line_x_l = x
                                right_line_y_l = main_dri_y
                                line_label = mask_line[main_dri_y][x]
                                #print("right side line_l_pixel : mask[{}][{}]={}".format(right_line_y_l,right_line_x_l,mask_line[right_line_y_l][right_line_x_l]))

                        for x in range(right_line_x_l,mask_dri.shape[1]-1,1): #search right boundary of line which is at right side of image
                            if mask_line[main_dri_y][x] != line_label  and find_right_line_pixel==True:
                                find_right_line_pixel = False
                                right_line_x_r = x
                                right_line_y_r = main_dri_y
                                #print("right side line_l_pixel : mask[{}][{}]={}".format(right_line_y_r,right_line_x_r,mask_line[right_line_y_r][right_line_x_r]))
                                #print("=======================================================")
            
                        #left_line_x_r = left_line_x_r+10 if left_line_x_r+10< mask_line.shape[1] else left_line_x_r
            if left_line_x_l > 0:
                for k in range(left_line_x_l):
                    #if mask_line[i][k] != 3:
                    mask_out[i][k]=lane_bg #Set the left side  of left line as background
                # for k in range(left_line_x_l):
                #     if mask_dri[i][k] != 0:
                #         mask_dri[i][k]=2 #Set the left side  of area as background
                for k in range(left_line_x_l,left_line_x_r,1):
                    if not mask_line[i][k]==4: #if not horizon line and not road curve
                        mask_out[i][k] = 2 #main left line
                    #print("find left line")
                
            if right_line_x_r > 0:
                for k in range(right_line_x_r,mask_dri.shape[1],1):
                    #if mask_line[i][k] != 3:
                    mask_out[i][k]= lane_bg #Set the right side  of right line as background
                
                # for k in range(right_line_x_r,mask_dri.shape[0],1):
                #     if mask_dri[i][k] != 0:
                #         mask_dri[i][k]= 2 #Set the right side  of area as background

                for k in range(right_line_x_l,right_line_x_r+1,1):
                    #print("find right line")
                    if not mask_line[i][k]==4: #if not horizon line and not road curve
                        mask_out[i][k] = 3 #main right line

            #=====================color map========================================================================================
            #mask_out_color = cv2.cvtColor(mask_out_color.numpy(),cv2.COLOR_GRAY2RGB)
            if SAVE_COLOR_MAP:
                if left_line_x_l > 0:
                    for k in range(0,left_line_x_l,1):
                        #if mask_line[i][k] != 3:
                        mask_out_color[i][k]=(255,255,255) #Set the left side  of left line as background
                        mask_out_png[i][k] = 0
                    # for k in range(left_line_x_l):
                    #     if mask_dri[i][k] == 1 or mask_dri[i][k] == 2 :
                    #         #print("mask_dri is not main dri")
                    #         mask_dri_color[i][k]=255 #Set the left side  of area as background
                    #for k in range(50,150+1,1):
                    for k in range(left_line_x_l,left_line_x_r,1):
                        #print("find left line")
                        if not mask_line[i][k]==4: #if not horizon line and not road curve
                            mask_out_color[i][k] = (255,0,0) #main left line
                            mask_out_png[i][k] = 2
                #else:
                #    print("left_line_x_l is 0")     
                    
                if right_line_x_r > 0:
                    for k in range(right_line_x_r,mask_dri.shape[1],1):
                        #if mask_line[i][k] != 3:
                        mask_out_color[i][k] = (255,255,255) #Set the right side  of right line as background
                        mask_out_png[i][k] = 0
                    # for k in range(right_line_x_r,mask_dri.shape[0],1):
                    #     if mask_dri[i][k] == 1 or mask_dri[i][k] == 2 :
                    #         #print("mask_dri is not main dri")
                    #         mask_dri_color[i][k]= 255 #Set the right side  of area as background

                    for k in range(right_line_x_l,right_line_x_r+1,1):
                        #print("find right line")
                        if not mask_line[i][k]==4: #if not horizon line and not road curve
                            mask_out_color[i][k] = (0,0,255) #main right line
                            mask_out_png[i][k] = 3
                #else:
                #    print("right_line_x_r is 0")           
            #print(x)
        #input()
        if SAVE_COLOR_MAP:
            
            #cv2.imwrite(save_line_dir+'/'+str(self.count)+'_line.png',mask_line_color.numpy())
            #cv2.imwrite(save_dri_dir+'/'+str(self.count)+'_drive.png',mask_dri_color.numpy())
            if not os.path.exists(os.path.join(save_dri_4cls_color,save_file)):
                cv2.imwrite(save_dri_4cls_color+'/'+save_file,mask_out_color)
            else:
                print("file {} exists! pass it".format(save_file))
            if not os.path.exists(os.path.join(save_dri_4cls,save_file)):
                cv2.imwrite(save_dri_4cls+'/'+save_file,mask_out_png)
            else:
                print("file {} exists! pass it".format(save_file))
        return mask_out
if __name__=="__main__":
    data_dir="/home/jnr_loganvo/Alister/datasets/YOLO_ADAS/bdd100k_data"
    parsing_imgs_and_labels(data_dir)