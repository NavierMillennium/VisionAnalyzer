from threading import Thread, Lock, Event
import cv2 as cv
import imutils
import logging 
import numpy as np
import time 
from typing import Optional
import os
from math import *
import json, glob, math
from matplotlib import pyplot as plt


__author__ = 'Kamil Skop'
__copyright__ = 'Copyright 2022-2023, Kamil Skop'
__license__ = 'MIT'

class USBCamera (Thread):
    def __init__(self, id, width = 640, height = 480):
        super().__init__()
        self.logger = logging.getLogger('USBcamera')
        #Camera object initialisation
        camera = cv.VideoCapture(id, cv.CAP_DSHOW)
        camera.set(cv.CAP_PROP_FRAME_WIDTH, width)#640
        camera.set(cv.CAP_PROP_FRAME_HEIGHT, height)#480
        camera.set(cv.CAP_PROP_FPS, 30)
        time.sleep(0.5)
        if not camera.isOpened():
            self.logger.error('Error! Cannot open camera: {}'.format(id))
            self.close()
        self.camera = camera
        self._catched_frame = np.array([])
        self._frame_lock = Lock()
        self._is_programm_running = True
        # Start thread
        self.start()
        
    def run(self):
        while self._is_programm_running:
            #with self._frame_lock:
            ret, self._catched_frame = self.camera.read() 
            if not ret:
                raise Exception("Error: Frames not received!")
        self.camera.release()   
            
    def get_frame(self):
        with self._frame_lock:
            return self._catched_frame.copy()
        
    def close (self):
        self._is_programm_running = False
         
    
class VideoAnalyzer():
   
    win_synchro_lock = Lock()
    
    def __init__(self, project_path, cam):
        self.logger = logging.getLogger('VideoAnalyzer')
        self.cam = cam
        # Object data
        self.row_corner = 6
        self.col_corner = 9
        self._project_path = project_path
        self.qcd = cv.QRCodeDetector()
        self.bd = cv.barcode.BarcodeDetector()
    
    @staticmethod 
    def convert_to_grayscale(img):
        if len(img.shape) == 3:
            if img.shape[2] == 3:
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            else:
                gray = img
        else: 
            gray = img
            
        return gray
    
    # @staticmethod
    # def get_contours(image, draw_cnt = True):
    #     gray = VideoAnalyzer.convert_to_grayscale(image)
    #     blurred = cv.GaussianBlur(gray, (5, 5), 0)
    #     thresh = cv.threshold(blurred, 70, 255, cv.THRESH_BINARY)[1]
    #     thresh = cv.bitwise_not(thresh)
    #     cv.imshow('Thresh',thresh)
    #     all_cnts = []
    #     cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
    #         cv.CHAIN_APPROX_SIMPLE)
    #     cnts = imutils.grab_contours(cnts)
    #     for cnt in cnts:
    #         area = cv.contourArea(cnt)
    #         # We reject contours which area is too large or too small 
    #         if area < 1e5 or 1e7 < area:
    #             continue
    #         epsilon = 0.001*cv.arcLength(cnt,True)
    #         approx = cv.approxPolyDP(cnt,epsilon,True)
    #         bbox = cv.boundingRect(cnt)
    #         rect = cv.minAreaRect(cnt)
    #         box = cv.boxPoints(rect)
    #         box = np.int0(box)
            
    #         all_cnts.append([area, cnt, approx, rect, box, bbox])
    #     all_cnts = sorted(all_cnts,key = lambda x:x[1] ,reverse= True)      
    #     if draw_cnt:
    #         img = image.copy()
    #         for con in all_cnts :
    #             cv.drawContours(img,[con[1]],-1,(0,255,0),1)
    #             cv.drawContours(img,[con[4]],-1,(0,0,255),1)
    #             cv.drawContours(img,[approx], -1,(255,0,0),1)
    #     return img, all_cnts
        
    @staticmethod
    def get_contours(image, min_area, max_area, method = 0, draw_cnt = True, th_val = 70):
        img = None
        gray = VideoAnalyzer.convert_to_grayscale(image)
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        thresh = cv.threshold(blurred, th_val, 255, cv.THRESH_BINARY)[1]
        thresh = cv.bitwise_not(thresh)
        #cv.imshow('thresh', thresh)
        #cv.waitKey(-1)
        all_data = []
        if method == 0:
            mth = cv.RETR_EXTERNAL
        else:
            mth = cv.RETR_TREE
        cnts = cv.findContours(thresh.copy(), mth, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for cnt in cnts:
            area = cv.contourArea(cnt)
            #print(area)
            # We reject contours which area is too large or too small 
            #area < 1e5 or 1e7 < area: 
            if area < min_area or area > max_area:
                continue
            epsilon = 0.001*cv.arcLength(cnt,True)
            approx = cv.approxPolyDP(cnt,epsilon,True)
            bbox = cv.boundingRect(cnt)
            rect = cv.minAreaRect(cnt)
            box_float = cv.boxPoints(rect)
            box = np.int0(box_float)
            
            all_data.append([area, cnt, approx, rect, box, box_float, bbox])
        all_data = sorted(all_data,key = lambda x:x[0] ,reverse= True)      
        
        if draw_cnt:
            img = image.copy()
            for con in all_data :
                cv.drawContours(img,[con[1]],-1,(0,255,0),1)
                cv.drawContours(img,[con[4]],-1,(0,0,255),1)
                #cv.drawContours(img,[approx], -1,(255,0,0),1)
        
        return img, all_data
    
    @staticmethod
    def get_inner_counters(image, max_area, mode:str = 'circle', draw = False):
        # Check input argument 'mode'
        mode = mode if mode == 'circle' else 'square'
        img = None
        
        imgBlur = cv.GaussianBlur(image,(5,5),1)
        imgCanny = cv.Canny(imgBlur,100,200)
        kernel = np.ones((5,5))
        imgDial = cv.dilate(imgCanny,kernel,iterations=1)
        imgThre = cv.erode(imgDial,kernel,iterations=1)
        cnts = cv.findContours(imgThre,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        all_data = []
        
        for cnt in cnts:
            epsilon = 0.001*cv.arcLength(cnt,True)
            approx = cv.approxPolyDP(cnt,epsilon,True)
            area = cv.contourArea(cnt)
            if area > 300 and area < max_area:
                #x, y, w, h = cv.boundingRect(cnt)
                rect = cv.minAreaRect(cnt)
                box = cv.boxPoints(rect)
                box = np.int0(box)
                w = VideoAnalyzer.euk_dist(box[0],box[1])
                h = VideoAnalyzer.euk_dist(box[0],box[3])
                ratio= float(w)/h
                if ratio >= 0.89 and ratio <= 1.2 and mode == 'circle':
                    (x,y),radius = cv.minEnclosingCircle(cnt)
                    all_data.append([area, cnt, [x,y,radius], rect])
                
                elif mode == 'square':
                    if len(approx) == 4:
                        all_data.append([area, cnt, [x, y, w, h], rect])   
                        
            #all_data = sorted(all_data,key = lambda x:x[0] ,reverse= True)      
            all_data = sorted(all_data,key = lambda x:(x[2][0],x[2][0]) ,reverse= True)      
        
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
    
    @staticmethod
    def hough_line_transform(image, draw = True):
        gray = VideoAnalyzer.convert_to_grayscale(image)
        edges = cv.Canny(gray, 70, 130)
        edges = cv.dilate(edges, None, iterations=5)
        edges = cv.erode(edges, None, iterations=5)
        cv.imshow('edges',edges)
        lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
        if draw:
            img = image.copy()
            for line in lines:
                x1,y1,x2,y2 = line[0]
                cv.line(img,(x1,y1),(x2,y2),(0,255,0),1)
            
        return img, lines
    
    @staticmethod
    def hough_circle_transform(image, draw = True):
        gray = VideoAnalyzer.convert_to_grayscale(image)
        gray = cv.medianBlur(gray, 5)
        rows = gray.shape[0]
        circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,20,
        param1 = 50, param2 = 30, minRadius = 0, maxRadius = 0)
        circles = np.uint16(np.around(circles))
        if draw:
            img = image.copy()
            for i in circles[0,:]:
                # draw the outer circle
                cv.circle(img,(i[0],i[1]),i[2],(0,255,0),1)
                # draw the center of the circle
                cv.circle(img,(i[0],i[1]),2,(0,0,255),2)
        return img, circles[0,:]
    
    @staticmethod
    def cut_ROI(image, rec, padding:int = 0):
        x,y,w,h = rec
        #if x-padding > 0 and y - padding > 0 and y+h+padding < image.shape[1] and x+w+padding < image.shape[0] :
        roi = image[y-padding:y+h+padding, x-padding:x+w+padding]
        return roi
        #else:
        #    raise ValueError('Padding out of image size!')
    
    @staticmethod
    def ignore_not_area(image, frame ):
        image[:,:int(frame[0])] = 255
        image[:int(frame[1]),int(frame[0]):int(frame[0]+frame[2]):] = 255
        image[int(frame[1]+frame[3]):,int(frame[0]):int(frame[0]+frame[2]):] = 255
        image[:,int(frame[0]+frame[2]):] = 255
        return image
    
    @staticmethod
    def ROI_selector (image):
        frame = cv.selectROI(image)
        roi = image[int(frame[1]):int(frame[1]+frame[3]), int(frame[0]):int(frame[0]+frame[2])]
        return roi, frame

    @staticmethod
    def euk_dist(x1,x2):
        return sqrt((x2[0]-x1[0])**2 + (x2[1]-x1[1])**2)

    @staticmethod
    def moments_locator(image, cnts, draw =True):
        if cnts:
            mu = [None]*len(cnts)
            ort_pose = []
            
            for i in range(len(cnts)):
                mu[i] = cv.moments(cnts[i])
                m00 = mu[i]['m00'] + 1e-5
                m10 = mu[i]['m10']
                m01 = mu[i]['m01']
                m11 = mu[i]['m11']
                m02 = mu[i]['m02']
                m20 = mu[i]['m20']
                
                cX = m10 / m00 
                cY = m01 / m00 

                a = m20/m00 - cX**2
                b = 2*(m11/m00 - cX*cY)
                c = m02/m00 - cY**2
                
                if a-c == 0:
                    continue
                # Orientation (radians)
                theta = 1/2*atan(b/(a-c)) + (a<c)*pi/2
                
                ort_pose.append([cX, cY, theta])
                
                w = sqrt(8*(a+c-sqrt(b**2+(a-c)**2)))/2
                l = sqrt(8*(a+c+sqrt(b**2+(a-c)**2)))/2
                d = sqrt(l**2-w**2)
                
                # x1 = cX + d*cos(theta)
                # y1 = cY + d*sin(theta)
                # x2 = cX - d*cos(theta)
                # y2 = cY - d*sin(theta)
                
                #Draw Axis 
                def draw_axis(img, x ,y ,theta, colour = (255, 255, 255), thickness = 1):
                    # Create the arrow hooks
                    d = 50
                    arr = 5
                    x1 = cX + d*cos(theta)
                    y1 = cY + d*sin(theta)
                    
                    cv.line(img, (int(x), int(y)), (int(x1), int(y1)), colour, thickness, cv.LINE_AA)
                    p0 = x1 - arr * cos(theta + pi / 4)
                    p1 = y1 - arr * sin(theta + pi / 4)
                    cv.line(img, (int(p0), int(p1)), (int(x1), int(y1)), colour, thickness, cv.LINE_AA)
                    p0 = x1 - arr * cos(theta - pi / 4)
                    p1 = y1 - arr * sin(theta - pi / 4)
                    cv.line(img, (int(p0), int(p1)), (int(x1), int(y1)), colour, thickness, cv.LINE_AA)
                    
            
                if draw:
                    img = image.copy()
                    draw_axis (img, cX, cY, theta, colour = (0, 255, 0)) 
                    draw_axis (img, cX, cY, theta+pi/2, colour = (0, 0, 255)) 
                    # Draw the contour and center of the shape on the image
                    cv.drawContours(img, [cnts[i]], -1, (0, 255, 0), 2)
                    cv.ellipse(img, (int(cX),int(cY)) , (int(w),int(l)), int(np.rad2deg(theta)), 0, 360, (255, 255, 255) , 1)
                    cv.circle(img, (int(cX),int(cY)), 7, (255, 255, 255), -1)
                    cv.putText(img, "Centroid", (int(cX) - 20, int(cY) - 20),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                # Show the image     
            return img, ort_pose
        else:
            return None, None

    @staticmethod
    def ORB_features_detector(input_img, train_img, draw = True):
        MIN_MATCH_COUNT = 50
        img1 = VideoAnalyzer.convert_to_grayscale(train_img)
        img2 = VideoAnalyzer.convert_to_grayscale(input_img)
        img3 = None
        
        orb = cv.ORB_create()
        # compute the descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1,None)
        kp2, des2 = orb.detectAndCompute(img2,None)

        index_params = dict(algorithm=6,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=2)

        search_params = dict(checks = 50)
        flann = cv.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1,des2,k=2)
        good = []
        
        # store all the good matches as per Lowe's ratio test.
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)
        print('Good matches:',len(good))
        if draw and len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            h,w = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv.perspectiveTransform(pts,M)
            img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)

            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
            img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
            
        if len(good)>MIN_MATCH_COUNT:
            return True, img3
        else:
            return False, img3 

        
    def _new_parallel_branch(fn):
        def wrapper(*args, **kwargs):
            thread = Thread(target=fn, args=args, kwargs=kwargs)
            thread.start()
            return thread
        return wrapper
    
    @staticmethod
    def calibrate_camera(chess_dim:tuple, project_path, img_folder, json_name, square_size):
        
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # Matrixs that store the coordinates of the object's points and the coordinates of the pixels in the image
        objpoints = [] # three-dimensional coordinates of the object
        imgpoints = [] # two-dimensional coordinates of a point in the image

        images = glob.glob(project_path+'/'+img_folder+'/*.png')  # Creating a list of objects with the extension .png located in the cam_correction_photos directory
        # Entering the dimensions of the array
        heigth = chess_dim[0]
        width = chess_dim[1]
        # Preparation of coordinates of points on the calibration board
        objp = np.zeros((heigth*width, 3), np.float32)
        objp[:,:2] = np.mgrid[0:heigth, 0:width].T.reshape(-1, 2) * square_size

        # Loop performed for each photo found in the above directory (camera correction photos)

        for fname in images:
            img = cv.imread(fname)  
            if img.shape[2] == 3:
                gray = VideoAnalyzer.convert_to_grayscale(img)
            else: 
                gray = img
            ret, corners = cv.findChessboardCorners(gray, chess_dim, None)    

            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)   # A function that increases the accuracy of the coordinates of detected corners
                imgpoints.append(corners2)

        # A function that returns the internal parameters of the camera, the distortion vector as well as the rotation and translation vectors
        ret, mtx_old, dist_old, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        mean_error = 0      # Initialization of a variable holding the average reprojection error
        camConfigReprot = np.empty(shape=(0, 4))    # Initializing an array containing [image name, rotation vector, translation vector, reprojection error]

        # Saving rotation, translation and reprojection error vectors to an external file
        for rv, tv, fname, objpnt, imgpnt in zip(rvecs, tvecs, images, objpoints, imgpoints):
            # Usuwanie nazwy folderu ze ścieżki zdjęcia
            if '/' in fname:
                fname = fname.split('/')
                fname = fname[1]
            if '\\' in fname:
                fname = fname.split('\\')
                fname = fname[1]
            # 2D coordinate projection based on camera parameters matrix, distortion vector, rotation and translation vector
            imgpoints2, _ = cv.projectPoints(objpnt, rv, tv, mtx_old, dist_old)
            # Reprojection error calculation
            error = cv.norm(imgpnt, imgpoints2, cv.NORM_L2)/len(imgpoints2)
            # Adding information about rotation and translation vectors and reprojection error to the table
            camConfigReprot = np.append(camConfigReprot, [[fname, rv, tv, error]], axis=0)
            mean_error += error

        mean_error = mean_error / len(objpoints)    # Calculation of the average reprojection error

        # Saving information about rotation and translation vectors as well as reprojection error for each analyzed image to the table file
        with open(project_path+'/'+img_folder+'/raport.txt', "w") as f:
            for x in camConfigReprot:
                s = '\n\nfile: ' + x[0] + ',\nrotation vector:\n' + str(x[1]) + ',\ntranslation vector:\n' + str(x[2]) + ',\nreprojection error: ' + str(x[3])
                f.write(s)

        print('The average value of the reprojection error is: ', mean_error)

        # Recalculation of the distortion vector 
        # This time for images with a reprojection error less than the user-specified
        error = float(input('Enter the error value for which the camera will be recalibrated: '))

        # Matrix that store the coordinates of the object's points and the coordinates of the pixels in the image
        objpoints = [] # three-dimensional coordinates of the object
        imgpoints = [] # two-dimensional coordinates of a point in the image

        good_img = 0

        for i, fname in enumerate(images):
            img = cv.imread(fname)  
            if '/' in fname:
                fname = fname.split('/')
                fname = fname[1]
            if '\\' in fname:
                fname = fname.split('\\')
                fname = fname[1]
            if camConfigReprot[i-1][3] <= error:
                good_img += 1
                print(fname)
                print(i)
                cv.imwrite(project_path+'/'+img_folder+'/second_run/'+fname, img)
                
                if img.shape[2] == 3:
                    gray = VideoAnalyzer.convert_to_grayscale(img)
                else: 
                    gray = img
              
                ret, corners = cv.findChessboardCorners(gray, chess_dim, None)  

                if ret == True:
                    objpoints.append(objp)
                    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria) 
                    imgpoints.append(corners2)

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        mean_error = 0
        for rv, tv, fname, objpnt, imgpnt in zip(rvecs, tvecs, images, objpoints, imgpoints):
            imgpoints2, _ = cv.projectPoints(objpnt, rv, tv, mtx_old, dist_old)
            error = cv.norm(imgpnt, imgpoints2, cv.NORM_L2) / len(imgpoints2)
            mean_error += error

        mean_error = mean_error / len(objpoints)
        print("Rejected %d images from %d" % (len(images)-good_img, len(images)))
        print("New average reprojection error: ", mean_error)


        print("\nPrimary matrix of internal parameters:\n", mtx_old)
        print('New matrix of internal parameters:\n', mtx)

        print("\nThe original distortion vector:\n", dist_old)
        print("New distortion vector:\n", dist)

        with open(project_path + '/'+json_name+'.json','r') as config_file:
            config = json.load(config_file)

        # Assigning the determined values of the camera parameters matrix and the distortion vector to variables in the configuration file
        config['cam_calibration']['mtx'] = mtx.tolist()
        config['cam_calibration']['dist'] = dist.tolist()

        # Saving the above values to the configuration file
        with open(project_path+'/'+json_name+'.json', 'w') as config_file:
            json.dump(config, config_file, sort_keys=True, indent=4)

        # Remove distortions from calibration images and save them to a folder (camera_correction_photos/undistored_images)
        images = glob.glob(project_path+'/'+img_folder+'/*.png')
        
        for fname in images:
            img = cv.imread(fname)
            if '/' in fname:
                fname = fname.split('/')
                fname = fname[1]
            if '\\' in fname:
                fname = fname.split('\\')
                fname = fname[1]
            dst = cv.undistort(img, mtx, dist)
            cv.imwrite(project_path+'/'+img_folder+'/undistored_images/' + fname, dst)
            
    @staticmethod
    @_new_parallel_branch
    def img_viewer(img, name:str = 'Window', pose:tuple = (0, 0), scale:tuple = (1,1), timeout:int = 3):
        img = cv.resize(img, (0,0), fx = scale[0], fy = scale[1])
        cv.imshow(name, img)
        cv.moveWindow(name,pose[0], pose[1])
        
        # mutex for cv.waitKey() methd we want to avoid window crash
        with VideoAnalyzer.win_synchro_lock:
            cv.waitKey(1)
        time.sleep(timeout)
        cv.destroyWindow(name)
        
    @staticmethod  
    def qrcode_detect (self, img):
        '''
        Parameters:
        img - input image
        
        Return parameters:
           retval - True if a QR code is detected and False if none is detected
           decoded_info - tuple whose elements are strings stored in QR codes. If it can be detected but not decoded, it is an empty string ''.
           points -  numpy.ndarray representing the coordinates of the four corners of the detected QR Code.
           straight_qrcode - tuple whose elements are numpy.ndarray. The numpy.ndarray is a binary value of 0 and 255 representing the black and white of each cell of the QR code.
        '''
        retval, decoded_info, points, straight_qrcode = self.qcd.detectAndDecodeMulti(img)
        if retval:
            return decoded_info
        else:
            self.logger.warning('qrcode_detect:Warrning! QR code not detected')
            return None
        
    @staticmethod  
    def barcode_detect (self, img):
        '''
        Parameters:
        img - input image
        
        Return parameters:
           retval - True if a barcode is detected and False if none is detected
           decoded_info - tuple whose elements are strings stored in barcodes. If it can be detected but not decoded, it is an empty string ''
           decoded_type - tuple whose elements are numbers representing barcode types.
           points - numpy.ndarray representing the coordinates of the four corners of the detected QR Code.
        '''
        retval, decoded_info, decoded_type, points = self.bd.detectAndDecode(img)
        if retval:
            return decoded_info
        else:
            self.logger.warning('barcode_detect:Warrning! QR code not detected')
            return None
        
    @staticmethod
    def draw_frame(self, img, points:list, color:tuple = (0, 255, 0)):
        cv.polylines(img, points.astype(int), True, (0, 255, 0), 3)
    
    @staticmethod
    def draw_matching_frame(txt, img, start_point, end_point, frame_color = (0,255,0), txt_color = (255,255,255)):
        # operating on input image
        image = img.copy()
        FONT = cv.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 1.4
        THICKNESS = 2
        cv.rectangle(image, start_point, end_point, frame_color, 2)
        txt_size, _ = cv.getTextSize(txt, FONT, FONT_SCALE, THICKNESS)
        cv.rectangle(image, (start_point[0], start_point[1] - 5 - txt_size[1]), (start_point[0]+txt_size[0], start_point[1]), frame_color, -1)
        cv.putText(image, txt, (start_point[0], start_point[1] - 5), FONT, FONT_SCALE, txt_color, THICKNESS)
        return image
    
    @staticmethod
    def draw_msg_frame(txt, img, start_point, frame_color = (0,255,0), txt_color = (255,255,255), toColor = True):
        # operating on input image
        if toColor:
            if len(img.shape) == 3:
                image = img.copy()
            else: 
                image = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        else:
            image = img.copy()
            
        offset = 0
        FONT = cv.FONT_HERSHEY_SIMPLEX
        THICKNESS = 2
        FONT_SCALE = 0.7
        FRAME_CONST = 20
        FRAME_CONST = int(FRAME_CONST * FONT_SCALE)
        for line in txt.splitlines():
            txt_size, _ = cv.getTextSize(line, FONT, FONT_SCALE, THICKNESS)
            cv.rectangle(image, (start_point[0], start_point[1] + offset), (start_point[0]+txt_size[0], start_point[1] + FRAME_CONST + txt_size[1] + offset), frame_color, -1)
            cv.putText(image, line, (start_point[0], start_point[1] + txt_size[1] + offset +  int(FRAME_CONST/2)), FONT, FONT_SCALE, txt_color, THICKNESS)
            offset += txt_size[1] + FRAME_CONST
        return image        
   
    @property
    def project_path(self):
        return self._project_path
    
    @project_path.setter
    def project_path(self, x:str):
        if os.path.exists(x):
            self._project_path = x 
        else:
            self.logger.error('project_path: Indicated path:{} does not exist'.format(x))
   
    @_new_parallel_branch  
    def calib_image_saver(self, chess_dim:tuple, folder_name, json_name):
        self._init_project_structure(folder_name, json_name) 
        self.row_corner = chess_dim[0]
        self.col_corner = chess_dim[1]
       
        currentPhoto = 0 
        while True:
            img = self.cam.get_frame()
            cv.imshow('Camera Calibration', img) 
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):  
                cv.destroyAllWindows   
                break
            elif key == ord('s'):
                gray = VideoAnalyzer.convert_to_grayscale(img)
                ret, _ = cv.findChessboardCorners(gray, chess_dim, None) 
                cv.imshow('Gray', gray)   
                if ret:  
                    self.logger.debug('calib_image_saver:Saved image nr:{}'.format(currentPhoto)) 
                    name = self._project_path + '/'+folder_name+'/pict' + str(currentPhoto) + '.png'
                    cv.imwrite(name, img)     
                    currentPhoto += 1      
                else: 
                    self.logger.warning('calib_image_saver:Calibration table not found')
                    
    def _init_project_structure(self, calib_folder_name, json_name):
        """
        Prepare folder for calibration images and initialisation
        configuration 'json.' file 
        """
        if not os.path.exists(self.project_path+'/'+calib_folder_name):
                os.makedirs(self.project_path+'/'+calib_folder_name)
                os.makedirs(self.project_path+'/'+calib_folder_name+'/undistored_images')
                os.makedirs(self.project_path+'/'+calib_folder_name+'/second_run')
        else:
            # Catch exception
            self.logger.warning('_init_project_structure: Folder already exist!')
        
        self.data = {
            'video_source':{
                'name':'',
                'id':''
            },
            'chess_board':{
                'row':'',
                'col':''
            },
            'cam_calibration': {
                'mtx':'',
                'dist':''
            },
            'pos_calibration':{
                'T':'',
                'distRatio':'',
                'U_vector':''
            }
        }
        # Create --> config.json file
        try: 
            config_file = open(self.project_path+'/'+json_name+'.json','w') 
        except: 
            raise Exception('Error! Cannot create config file')  
        else:
            json.dump(self.data, config_file, sort_keys=True, indent=4)
    
    @_new_parallel_branch 
    def cam_rob_calibrate(self, cam, obj:UR_robot, chess_dim:tuple, json_name):
        def distance(pointA, pointB):
            dist = math.sqrt((pointA[0]-pointB[0])**2 + (pointA[1]-pointB[1])**2)
            return dist

        # Downloading data about the camera, the homography matrix and the order of the elements
        # from the configuration file, setting the element pointer to the first value
        # config, order, mtx, dist, T, distRatio, thresholdValue, objectHeight = cvis.configRead('config.json')
        with open(self.project_path + '/'+json_name+'.json') as config_file:
            data =json.load(config_file)
            
            mtx = np.array(data['cam_calibration']['mtx'])
            dist = np.array(data['cam_calibration']['dist'])
            
        print("Readed camera matrix:\n ", mtx)
        print("\nReaded vector of dispersion coefficients:\n", dist)

        # Initialization of arrays to store coordinates of points in the image and in robot space
        U = np.empty(shape=(4, 2))
        X = np.empty(shape=(4, 2))
        print("\n--------------------------------------------------------------------------------------------------")
        print("-----------------------------------Camera calibration with the robot--------------------------------------")
        print("----------------------------------------------------------------------------------------------------")
        print("Description of the calibration procedure:")
        print("1.Move the robot's gripper out of the camera's field of view")
        print("2.Place the calibration chart in the field of view of the camera")
        print("3.Press 'Enter'")
        print("4.Position the robot tip at the indicated points")

        while True:
            img = cam.get_frame()   
            img = cv.undistort(img, mtx, dist)  # Removal of image distortion
            cv.startWindowThread()
            cv.namedWindow("Homography Calculation")
            cv.imshow("Homography Calculation", img)      # Displaying the image preview window
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # Waiting for the user to press a key, the frame is displayed for 1ms
            # The logical operator AND makes only the first byte returned by the function valid
            # so it doesn't matter if the key was pressed with CapsLock on or not
            
            key = cv.waitKey(1) & 0xFF

            # Loop if key pressed is ENTER  
            if key == 13:
                gray = VideoAnalyzer.convert_to_grayscale(img)
                ret, corners = cv.findChessboardCorners(gray, chess_dim, None)   
                if ret:
                    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria) 
                    cv.imwrite(self.project_path+'/homography_calculation.png', img)
                    # Assigning the extreme points of the calibration chart to the matrix
                    # For this we use the 'objp' matrix written....
                    # ....below specifying the coordinates of the points on the calibration chart
                    # objp = np.zeros((heigth*width, 3), np.float32)
                    # objp[:,:2] = np.mgrid[0:heigth, 0:width].T.reshape(-1, 2)
                    
                    U[0] = [corners2[0,0,0],corners2[0,0,1]]
                    U[1] = [corners2[chess_dim[0]-1,0,0],corners2[chess_dim[0]-1,0,1]]
                    U[2] = [corners2[len(corners2)-chess_dim[0],0,0],corners2[len(corners2)-chess_dim[0],0,1]]
                    U[3] = [corners2[len(corners2)-1,0,0],corners2[len(corners2)-1,0,1]]
                    
                    # Drawing positions and numbers of individual calibration points on the image
                    for num, cnt in enumerate(U):
                        cv.circle(img, (int(cnt[0]), int(cnt[1])), 5, (0, 0, 255), 3)
                        cv.putText(img, str(num + 1), (int(cnt[0]) + 10, int(cnt[1]) - 10), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 255), 1)
                    cv.startWindowThread()
                    cv.namedWindow("Position Calibration")
                    cv.imshow('Position Calibration', img)
                    cv.waitKey(0)
                    cv.imwrite(self.project_path + '/homography_calculation_points.png', img)
                    #Loop assigning coordinates of points in the space of the robot
                    for num, point in enumerate(X):
                        x ='not'
                        while x:
                            x = input('Move robot to pose: {} and press ENTER')
                        pose = obj.terminal.get_xyz_pose()
                        # x = input('x:')
                        # y = input('y:')
                        point[0] = float(pose[0])
                        point[1] = float(pose[1])
                        # point[0] = float(x)
                        # point[1] = float(y)
                        self.logger.debug('cam_rob_calibrate:Readed poses:{}, {}'.format(pose[0], pose[1]))
                    T = cv.findHomography(U, X) # Homography matrix calculation camera -> robot
                    print(X) # Displaying the coordinate matrix of points in the space of the robot
                    distRatio = float(distance(X[0], X[2]) / distance(U[0], U[2]))  # Calculation of the length proportionality coefficient in the m/pixel unit
                    print("Macierz homografii:\n", T[0])
                # Saving the calculated values to the configuration file
                    with open(self.project_path + '/'+json_name+'.json', 'r') as config_file:
                        config = json.load(config_file)
                    config['pos_calibration']['T'] = T[0].tolist()
                    config['pos_calibration']['distRatio'] = distRatio
                    config['pos_calibration']['U_vector'] = U.tolist()
                    with open(self.project_path +'/'+json_name+'.json', 'w') as config_file:
                        json.dump(config, config_file, sort_keys=True, indent=4)
                    cv.destroyAllWindows()
                else:
                    print("No calibration table was detected in the camera area")
            # Quit the program if the key pressed is q or Esc
            elif key == ord('q') or key == 27:
                print("The calibration process has been interrupted")
                cv.destroyAllWindows()
                break
    
             

    

    
