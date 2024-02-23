from camera import VideoAnalyzer as va
import math 
from threading import Event, Lock
from queue import Queue
import pandas as pd
import os,re

def to_excel(path:str, name_sheet:str, data:dict) -> None:
    '''
    Foramt for input data:
    data = {
        'header_1': [1, 2, 3, 4, 5],
        'header_2': ['Car A', 'Car B', 'Car C', 'Car D', 'Car E'],
        'header_3': [25000, 30000, 35000, 40000, 45000]
    }
    '''  
    df = pd.DataFrame(data)
    custom_header = list(data.keys())
    # create file
    with pd.ExcelWriter(path) as excel_writer:
        df.to_excel(excel_writer, sheet_name=name_sheet, header = custom_header, index=False)


def to_excel_append(path:str, name_sheet:str, data:dict) -> None:
    '''
    Foramt for input data:
    data = {
        'header_1': [1, 2, 3, 4, 5],
        'header_2': ['Car A', 'Car B', 'Car C', 'Car D', 'Car E'],
        'header_3': [25000, 30000, 35000, 40000, 45000]
    }
    '''  
    df = pd.DataFrame(data)
    custom_header = list(data.keys())
    # create file
    if os.path.exists(path):
        fcn_mode = "a"
    else:
        fcn_mode = "w"
    with pd.ExcelWriter(path, mode = fcn_mode, engine="openpyxl") as excel_writer:
        df.to_excel(excel_writer, sheet_name=name_sheet, header = custom_header, index=False)
        
        
def get_inner_counters(image, max_area, mode:str = 'circle', draw = False):
    # Check input argument 'mode'
    mode = mode if mode == 'circle' else 'square'
    img = None
    imgBlur = image
    imgBlur = cv.GaussianBlur(image,(5,5),1)
    imgCanny = cv.Canny(imgBlur,100,200)
    # cv.imshow('',cv.resize(imgCanny, (0,0),fx = 0.7, fy = 0.7))
    # cv.waitKey(1)
    kernel = np.ones((5,5))
    imgDial = cv.dilate(imgCanny,kernel,iterations=1)
    imgThre = cv.erode(imgDial,kernel,iterations=1)
    cnts = cv.findContours(imgThre,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    hierarchy = cnts[1][0]
    cnts = imutils.grab_contours(cnts)
    all_data = []
    
    for cnt, hier in zip(cnts, hierarchy):
        epsilon = 0.001*cv.arcLength(cnt,True)
        approx = cv.approxPolyDP(cnt,epsilon,True)
        area = cv.contourArea(cnt)
        if area > 300 and area < max_area:
            #x, y, w, h = cv.boundingRect(cnt)
            rect = cv.minAreaRect(cnt)
            box_float = cv.boxPoints(rect)
            box = np.int0(box_float)
            w = va.euk_dist(box_float[0],box_float[1])
            h = va.euk_dist(box_float[0],box_float[3])
            ratio= float(w)/h
            if ratio >= 0.89 and ratio <= 1.1 and mode == 'circle' and (hier[2] < 0 or hier[3] < 0):
            #if ratio >= 0.8 and ratio <= 1.2 and mode == 'circle':
                (x,y),radius = cv.minEnclosingCircle(cnt)
                all_data.append([area, cnt, [x,y,radius], [w,h],rect])
            
            elif mode == 'square':
                if len(approx) == 4:
                    all_data.append([area, cnt, [x, y, w, h], rect])   
                    
        #all_data = sorted(all_data,key = lambda x:x[0] ,reverse= True)      
        all_data = sorted(all_data,key = lambda x:(x[2][0],x[2][1]) ,reverse= True)      
    
    if draw and all_data:
        img = image.copy()
        for con in all_data :
            if mode == 'circle':
                cv.drawContours(img, [con[1]], -1,(0,0,255),1)
                center = (int(con[2][0]),int(con[2][1]))
                radius = int(con[2][2])
                cv.circle(img,center,radius,(0,255,0),1)
            else:
                cv.drawContours(img, [con[1]], -1,(0,0,255),1)
                cv.drawContours(img, [con[3]], -1,(0,0,255),1)
    return img, all_data


def vision_squence(image, H):
    min_area = 1e4
    max_area = 1e7
    obj_point = np.empty((1,1,2), dtype=np.float32)
    DEV = 1.2 # deviation from real value
    DEV_AREA = 10000
    img, all_data = va.get_contours(image, min_area, max_area, th_val = 120)
    area = all_data[0][0]
    # cv.imshow('a',img)
    # cv.waitKey(-1)
    #print(area)
    key = '' # index from 'obj_desriptor' structure
    for i in obj_descriptor:
        temp = cv.imread(PATH_template+'/'+i+'.png')
        img1, all_data1 = va.get_contours(temp, min_area, max_area, th_val = 120)
        #va.img_viewer(img1, 'Cut frame')
        #time.sleep(3)
        ret = cv.matchShapes(all_data1[0][1],all_data[0][1],1,0.0)
        #print('matchShapes ratio:',ret)
        if ret < 1e-2 and area > obj_descriptor[i]['ext_area'] - DEV_AREA and area < obj_descriptor[i]['ext_area'] + DEV_AREA:
            key = i
            break 
    print(key)
    if key:
        cnts = [sublist[1] for sublist in all_data]
        draw_img, ort_pose = va.moments_locator(image,cnts)
        
        obj_point[0,0,0] = ort_pose[0][0] #we have to remeber about ROI translation
        obj_point[0,0,1] = ort_pose[0][1]
        rob_pose = cv.perspectiveTransform(obj_point, H)
        in_rob_frame = [1000*rob_pose[0,0,0], 1000*rob_pose[0,0,1], ort_pose[0][2]]
        #va.img_viewer(img, 'Pose and rotation')
        
        w = va.euk_dist(all_data[0][5][0], all_data[0][5][1])/RATIO
        h = va.euk_dist(all_data[0][5][1], all_data[0][5][2])/RATIO
 
        tmp = [h,w]
        tmp.sort()
        #print(tmp)
        result_out = [False, False]
        val = obj_descriptor[key]['dim']

        result_out[0] = True if tmp[0] > val[0] - DEV and tmp[0] < val[0] + DEV else False
        result_out[1] = True if tmp[1] > val[1] - DEV and tmp[1] < val[1] + DEV else False       
                
        if obj_descriptor[key]['inner_shapes']:
            img1, data = get_inner_counters(image, all_data[0][0] + 10000, mode='circle', draw = True)
            
            #va.img_viewer(cv.resize(img1, (0,0),fx = 0.7, fy = 0.7), 'Inner coutures')
            #print(len(data))

            res_list = []
           
            # # average 
            # print(len(data))
            # for i in range(len(data)//2):
            #     res_list.append([sum(i)/2 for i in zip(data[2*i][2], data[2*i+1][2])])
            # print(res_list)
            # average 
            dim_one = [2*i[2][2]/RATIO for i in data]
            dim_one[-1] += 3.05
            
            # for i in range(len(data)//2):
            #     res_list.append([sum(i)/2 for i in zip(data[2*i][2], data[2*i+1][2])])
            # #print([i[2] for i in data])
            # dim_rect_all = [(sum(i)/2)/RATIO  for i in res_list]
            
            # in_dim = [2*i[2]/RATIO for i in res_list]
            # print(in_dim)
            # ext_dim = (w+h)/2
            #ext_dim, in_dim, ort_pose =0,0,0
            return in_rob_frame + dim_one
        
    else:
        return [0]
def init_excel_struct():
    excel_struct_m = {'X':[], 'Y':[], 'ort':[], 'c1':[], 'c2':[], 'c3':[], 'c4':[], 'c5':[]}
    return  excel_struct_m

def sort_key(x):
    ex_name = str(re.findall('img_[0-9]*_[0-9]*.png',x))
    re.findall('img_[0-9]*',ex_name)[0][4:]
    re.findall('[0-9]*.png',ex_name)[0][:-4]

if __name__ == '__main__':

    # Project path
    PATH = r'\path_22_08'
    PATH_template = r'\path_22_08\templates'
    JSON_NAME_mako = '/mako_new/mako_conf'
    JSON_NAME_usb = 'usb_conf'
    JSON_POSE = 'pose_all'
    # RATIO for messurment
    RATIO = 6.311

    # Read Mako camera calculated matix
    with open(PATH + '/'+JSON_NAME_mako+'.json','r') as config_file:
        data_mako =json.load(config_file)

    # Assigning the determined values of the camera parameters matrix and the distortion vector to variables in the configuration file
    mtx_mako = np.array(data_mako['cam_calibration']['mtx'])
    dist_mako = np.array(data_mako['cam_calibration']['dist'])
    H_mako = np.array(data_mako['pos_calibration']['T'])

    # Read USB camera calculated matix
    with open(PATH + '/'+JSON_NAME_usb+'.json','r') as config_file:
        data_usb =json.load(config_file)

    # Assigning the determined values of the camera parameters matrix and the distortion vector to variables in the configuration file
    mtx_usb = np.array(data_usb['cam_calibration']['mtx'])
    dist_usb = np.array(data_usb['cam_calibration']['dist'])
    H_usb = np.array(data_usb['pos_calibration']['T'])

    #Objects descriptor structure

    obj_descriptor = {'obj_0_tp':{'ext_area':139760.5,'dim':[69.999,75.7],'inner_shapes':True, 'shapes':[5,2.5], 'cnt_shapes':[2,5]},
                    'obj_1_tp':{'ext_area':73063.5, 'dim':[50,50],'inner_shapes':True, 'shapes':[5], 'cnt_shapes':[4]},
                    'obj_2_tp':{'ext_area':77396.0, 'dim':[45,55],'inner_shapes':True, 'shapes':[4,2], 'cnt_shapes':[3,1]},
                    'obj_3_tp':{'ext_area':57960.5, 'dim':[40,40],'inner_shapes':False, 'shapes':[], 'cnt_shapes':[]},
                    'obj_4_tp':{'ext_area':88279.5, 'dim':[55,60],'inner_shapes':True, 'shapes':[4,2], 'cnt_shapes':[2,5]},
                    'obj_5_tp':{'ext_area':288079.0, 'dim':[100,100],'inner_shapes':True, 'shapes':[10], 'cnt_shapes':[4]},}


    # min and max area for get_coutures() function 
    min_area_usb = 1e3
    max_area_usb = 1e5

    min_area = 1e3
    max_area = 1e7



    dict ={'circle_0':None, 'circle_1':None, 'circle_2':None,'circle_3':None, 'circle_4':None, 'X':None, 'Y':None, 'ort':None}

    excel_struct_m = init_excel_struct()
    
    excel_struct = {'mean_X':[],'std_X':[],'mean_std_X':[],'median_X':[],'abs_err_X':[],'rel_err_X':[],
                   'mean_Y':[],'std_Y':[],'mean_std_Y':[],'median_Y':[],'abs_err_Y':[],'rel_err_Y':[],
                   'mean_ort':[],'std_ort':[],'mean_std_ort':[],'median_ort':[],'abs_err_ort':[],'rel_err_ort':[],
                   'mean_c1':[],'std_c1':[],'mean_std_c1':[],'median_c1':[],'abs_err_c1':[],'rel_err_c1':[],
                   'mean_c2':[],'std_c2':[],'mean_std_c2':[],'median_c2':[],'abs_err_c2':[],'rel_err_c2':[],
                   'mean_c3':[],'std_c3':[],'mean_std_c3':[],'median_c3':[],'abs_err_c3':[],'rel_err_c3':[],
                   'mean_c4':[],'std_c4':[],'mean_std_c4':[],'median_c4':[],'abs_err_c4':[],'rel_err_c4':[],
                   'mean_c5':[],'std_c5':[],'mean_std_c5':[],'median_c5':[],'abs_err_c5':[],'rel_err_c5':[]
                   }
    KEYS = list(excel_struct.keys())

    
    images = glob.glob(PATH+'/archiev_img_07_09/off_big/*.png')
    images = sorted(images, key = lambda x: (int(re.findall('img_[0-9]*',str(re.findall('img_[0-9]*_[0-9]*.png',x)))[0][4:]),int(re.findall('[0-9]*.png*',str(re.findall('img_[0-9]*_[0-9]*.png',x)))[0][:-4])))

    buff_list = []
    mem_id = 0
    
    for ind, name in enumerate(images):
        ex_name = str(re.findall('img_[0-9]*_[0-9]*.png',name))
        print(ex_name)
        img_id = int(re.findall('img_[0-9]*',ex_name)[0][4:])
        re.findall('[0-9]*.png',ex_name)[0][:-4]
        image = cv.imread(name)
        image = cv.undistort(image, mtx_mako, dist_mako)
        dim = vision_squence(image, H_mako)
        #print (dim)
        if mem_id == img_id:
            buff_list.append(dim)
        if mem_id != img_id or ind+1 >= len(images):                   
            # [X, Y, ort, c1, c2, c3, c4, c_big]
            mean = np.sum(buff_list, axis = 0)/len(buff_list)
            std = np.std(buff_list, axis = 0, ddof=1)
            mean_std = std/sqrt(len(buff_list))
            median = np.median(buff_list, axis = 0)
            exact_val = [mean[0],mean[1],mean[2],10,10,10,10,100]
            abs_error = abs(mean - exact_val)
            rel_error = abs_error/exact_val
            
            key_offset = 0
            for i in zip(mean, std, mean_std, median, abs_error, rel_error):
                for j in range(6):
                    excel_struct[KEYS[j+key_offset]].append(i[j])
                key_offset += (j+1)
                
            # Saving raw data to separate sheet
            for one_dim in buff_list:
                for k, i in enumerate(excel_struct_m):
                    excel_struct_m[i].append(one_dim[k])
            to_excel_append(PATH+'/excel_export/off_big_rawData.xlsx', 'pose_{}'.format(mem_id), excel_struct_m)   
            
            excel_struct_m = init_excel_struct()  
             
            buff_list = []
            buff_list.append(dim)
            mem_id = img_id

            print(mean)

    to_excel(PATH+'/excel_export/off_big.xlsx','sheet1',excel_struct)
       
    
        
       
        

