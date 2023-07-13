#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 16:27:36 2023

@author: jnr_loganvo
"""

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
50:main lane
150:left line
250:right line
'''

merge_mapping = {'drive': {0: 1,   1: 0,   2: 0},
                'lane':   {0: 0,   2: 2,   3: 3}}


'''
main : 255
alter : 255
background : 0
'''
drive_mapping = {0:255, 1:255 , 2:0}

dilate_line_mapping ={0:0,
                         1:255,
                         2:255,
                         3:255,
                         4:150,
                         5:200}


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
    dilation_kernel = 11
    """Split mask to n-channels map where n is equal to the number of foreground classes that needs to be dilated."""
    kernel = np.ones((dilation_kernel,dilation_kernel), dtype=np.uint8)
    
    #Resize image 
    mask = cv2.resize(mask,(640,360),interpolation=cv2.INTER_NEAREST)
    
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


def Get_Fit_label(mask_dri,split_lane_map,mask_line,drive_mapping,file):
    bg_lb = 255
    w=640
    h=360
    '''resize'''
    mask_dri = cv2.resize(mask_dri,(w,h),interpolation=cv2.INTER_NEAREST)
    mask_line = cv2.resize(mask_line,(w,h),interpolation=cv2.INTER_NEAREST)
    
    mask_out = np.zeros(mask_dri.shape)
    
    mask_dri = map_labels(mask_dri, drive_mapping, bg_lb)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_dri = cv2.erode(mask_dri, kernel=kernel)
    mask_dri = cv2.bitwise_not(mask_dri)
    
    mask_out = cv2.bitwise_and(mask_dri, mask_line)
    
    
    for i in range(mask_line.shape[0]):
        for j in range(mask_line.shape[1]):
            if mask_out[i][j]!=0:
                mask_out[i][j]=split_lane_map[i][j]
                print(mask_out[i][j])
    
    save_line = "./drive_binary"
    if not os.path.exists(save_line):
        os.makedirs(save_line)
    if not os.path.exists(os.path.join(save_line,file)):
        cv2.imwrite(save_line+'/'+file,mask_out)
    else:
        print("file {} exists! pass it".format(mask_out))
    
def parsing_imgs_and_labels(data_dir):
    '''Step 0 : Get lane mask list
                Get dri mask list
                '''
    label_lane_dir = os.path.join(data_dir,"labels","lane","masks","train")
    label_dri_dir = os.path.join(data_dir,"labels","drivable","masks","train")
    img_dir = os.path.join(data_dir,"images","100k","train")
    train_lane_mask_list = glob.glob(os.path.join(label_lane_dir,"*.png"))
    train_dri_mask_list = glob.glob(os.path.join(label_dri_dir,"*.png"))
    train_img_list = glob.glob(os.path.join(img_dir,"*.jpg"))
    c = 1
    
    '''Step 0 : Read the lane/dri mask in the list'''
    for i in range(len(train_lane_mask_list)):

        #if i < 38:
        #    continue

        print("{}:{}".format(i,train_lane_mask_list[i]))
        mask_lane = cv2.imread(train_lane_mask_list[i],cv2.IMREAD_GRAYSCALE)
        mask_dri = cv2.imread(train_dri_mask_list[i],cv2.IMREAD_GRAYSCALE)
        
        #print(mask_lane.shape)
        
        file,file_name = parse_path(train_lane_mask_list[i])
        img_path = os.path.join(img_dir,file_name+".jpg")
        img = cv2.imread(img_path)
        #print(file)
        #print(file_name)
        
        
        '''Step 1 : Map class (Re-assign the lane line labels)'''
        map_lane_class = map_labels(mask_lane, lane_mapping, lane_bg)
        '''Step 2 : Dilate lane line'''
        
        split_lane_map = split_map_and_dilate(map_lane_class)
        split_lane_map_ver2 = map_labels(split_lane_map,dilate_line_mapping,lane_bg)
        '''
        for i in range(split_lane_map.shape[0]):
            for j in range(split_lane_map.shape[1]):
                if split_lane_map[i][j]==0:
                    split_lane_map[i][j]=0
                elif split_lane_map[i][j]==1:
                    split_lane_map[i][j]=255
                elif split_lane_map[i][j]==2:
                    split_lane_map[i][j]=255
                elif split_lane_map[i][j]==3:
                    split_lane_map[i][j]=255
                elif split_lane_map[i][j]==4:
                    split_lane_map[i][j]=255
                elif split_lane_map[i][j]==5:
                    split_lane_map[i][j]=255
        '''
        save_line = "./line"
        if not os.path.exists(save_line):
            os.makedirs(save_line)
        if not os.path.exists(os.path.join(save_line,file)):
            cv2.imwrite(save_line+'/'+file,split_lane_map)
        else:
            print("file {} exists! pass it".format(split_lane_map))
            
        '''Step 3: Get fit line label'''
        Get_Fit_label(mask_dri,split_lane_map,split_lane_map_ver2,drive_mapping,file)
        #'''Step 3 : Merge class'''
        #merged_map = merge_segmentation_heads(mask_dri,split_lane_map)
        #'''Step 3 : Generate New laen line label'''
        #save_dri_4cls_color = "/home/ggson/Desktop/For_Andy/Result"
        #if not os.path.exists(os.path.join(save_dri_4cls_color,file)):
        #    new_lane_map = pre_processing_mainlane_line(img,
        #                                                mask_dri,
        #                                                c,
        #                                                save_color_map=True,
        #                                                save_file=file)
        #else:
        #    print("label exists ! Pass Step3 !!")
        #'''Step 4 : Merge class'''
        #merged_map = merge_segmentation_heads(mask_dri,new_lane_map)
        
        c+=1
        #mask = np.squeeze(mask)
        #new_mask = split_map_and_dilate(mask)
        #cv2.imshow('My Image', new_mask)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
import torch

def pre_processing_mainlane_line(img,
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
                    mask_out[i][j] = 50
                elif(mask_dri[i][j]==1):
                    mask_out[i][j] = 0
                elif(mask_dri[i][j]==2):
                    mask_out[i][j] = 0

        SAVE_COLOR_MAP = save_color_map
        if SAVE_COLOR_MAP:

            save_line_dir = "./line"
            save_dri_dir = "./dri"
            save_ori_img = "/home/jnr_loganvo/Alister/datasets/YOLO_ADAS/bdd100k_data/labels/main_road/images/train"
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
            if not os.path.exists(save_ori_img):
                os.makedirs(save_ori_img)

                
            print(mask_dri.shape)


            mask_out_color = np.zeros((mask_dri.shape[0],mask_dri.shape[1],3), dtype=np.uint8)
            mask_out_png = np.zeros((mask_dri.shape[0],mask_dri.shape[1],1))

            # find center line
            max_lane_width = 0
            x_start = 0
            x_end = 0
            y_start = mask_dri.shape[0]
            y_left_start = mask_dri.shape[0]
            y_right_start = mask_dri.shape[0]
            y_left_end = 0
            y_right_end = 0
            y_end = 0

            x_center_list = []
            for i in range(mask_dri.shape[0]):
                x_list = []
                for j in range(mask_dri.shape[1]):
                    if mask_out[i][j] == 0:
                        mask_out_color[i][j] = (0, 0, 0)
                        mask_out_png[i][j] = 0
                    elif mask_out[i][j] == 50:
                        mask_out_color[i][j] = (255,165,0)
                        mask_out_png[i][j] = 1
                        x_list.append(j)


                if len(x_list) == 0:
                    x_center_list.append(None)
                else:
                    x = (x_list[0] + x_list[-1]) // 2
                    x_center_list.append(x)
                    if i < y_start:
                        y_start = i
                        x_start = x

                    if i > y_end and len(x_list) > max_lane_width:
                        max_lane_width = len(x_list)
                        y_end = i
                        x_end = x

            x_mid = (x_start + x_end) // 2



            for i in range(mask_dri.shape[0]):
                for j in range(mask_dri.shape[1]):
                    if mask_out[i][j] == 50:

                        if j < x_mid and i < y_left_start:
                            y_left_start = i

                        if j > x_mid and i < y_right_start:
                            y_right_start = i

                        if j < x_mid and i > y_left_end:
                            y_left_end = i

                        if j > x_mid and i > y_right_end:
                            y_right_end = i

            result = np.zeros((mask_dri.shape[0],mask_dri.shape[1],3), dtype=np.uint8)
            #for i in range(mask_dri.shape[0]):
            #    for j in range(mask_dri.shape[1]):
            #        if mask_out[i][j] == 50:
            #            result[i][j] = (255,165,0)
                        
            #cv2.imshow('res', mask_out_color)
            #if Dirvable area is too large , ignore it
            
            #convert img to grey
            img_grey = cv2.cvtColor(mask_out_color,cv2.COLOR_BGR2GRAY)
            #set a thresh
            thresh = 100
            #get threshold image
            ret,thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
            #find contours
            contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            #create an empty image for contours
            img_contours = np.zeros(mask_out_png.shape, dtype=np.uint8)
            # draw the contours on the empty image
            cv2.drawContours(img_contours, contours, -1, 255, 1, cv2.LINE_8)



            #cv2.imshow('res contour', img_contours)
            print(img_contours.shape)




            print('y left start - end => {} - {}'.format(y_left_start, y_left_end))
            print('y right start - end => {} - {}'.format(y_right_start, y_right_end))
            if max_lane_width <= mask_dri.shape[1] * 0.70:
                # print(x_center_list)
                shift_top = 15
                shift_down = 30
                shift_mid = 60
                for i in range(mask_dri.shape[0]):
                    for j in range(0, mask_dri.shape[1]-2):
                        if y_left_start+shift_top<mask_dri.shape[0] and y_left_end-shift_down>0:
                            if j < x_mid and i > y_left_start+shift_top and i < y_left_end-shift_down and \
                                mask_out_color[i][j][0] == 0 and mask_out_color[i][j+1][0] == 255 and mask_out_color[i][j+2][0] == 255\
                                    and abs(j-x_mid)>shift_mid:
                                result[i][j] = (255, 0, 0)
                                mask_out[i][j] = 150 #left line
                            elif j > x_mid and i > y_right_start+shift_top and i < y_right_end-shift_down and \
                                mask_out_color[i][j][0] == 255 and mask_out_color[i][j+1][0] == 0 and mask_out_color[i][j+2][0] == 0\
                                    and abs(j-x_mid)>shift_mid:
                                result[i][j] = (0, 0, 255)
                                mask_out[i][j] = 250 #right line



            dilation_kernel = 19
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel, dilation_kernel))
            result = cv2.dilate(result, kernel)
            
            for i in range(mask_dri.shape[0]):
                for j in range(mask_dri.shape[1]):
                    if mask_out[i][j] == 50 and result[i][j][0] == 0 and result[i][j][1] == 0 and result[i][j][2] == 0:
                    #if mask_out[i][j] == 50:
                        result[i][j] = (255,165,0)
            
            mask_out = cv2.dilate(mask_out, kernel)
            
            #cv2.imshow('res contour (new)', result)
            if not os.path.exists(os.path.join(save_dri_4cls_color,save_file)):
                cv2.imwrite(save_dri_4cls_color+'/'+save_file,result)
            else:
                print("file {} exists! pass it".format(save_file))

            if not os.path.exists(os.path.join(save_dri_4cls,save_file)):
                cv2.imwrite(save_dri_4cls+'/'+save_file,mask_out)
            else:
                print("file {} exists! pass it".format(save_file))
                
            
            #if not os.path.exists(os.path.join(save_ori_img,save_file)):
            #    cv2.imwrite(save_ori_img+'/'+save_file,img)
            #else:
            #    print("file ori image {} exists! pass it".format(save_file))

            #cv2.waitKey(0)



        # if SAVE_COLOR_MAP:

        #     #cv2.imwrite(save_line_dir+'/'+str(self.count)+'_line.png',mask_line_color.numpy())
        #     #cv2.imwrite(save_dri_dir+'/'+str(self.count)+'_drive.png',mask_dri_color.numpy())
        #     if not os.path.exists(os.path.join(save_dri_4cls_color,save_file)):
        #         cv2.imwrite(save_dri_4cls_color+'/'+save_file,mask_out_color)
        #     else:
        #         print("file {} exists! pass it".format(save_file))
        #     if not os.path.exists(os.path.join(save_dri_4cls,save_file)):
        #         cv2.imwrite(save_dri_4cls+'/'+save_file,mask_out_png)
        #     else:
        #         print("file {} exists! pass it".format(save_file))
        # return mask_out


if __name__=="__main__":
    data_dir="/home/jnr_loganvo/Alister/datasets/YOLO_ADAS/bdd100k_data"
    parsing_imgs_and_labels(data_dir)