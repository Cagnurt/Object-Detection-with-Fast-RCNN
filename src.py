import platform,socket,json,psutil,logging
from constants import *
import torch
import cv2
import numpy as np
import os
import glob as glob
import json
from config import (
    CLASSES, RESIZE_TO, TRAIN_DIR, VALID_DIR, BATCH_SIZE
)


def video_to_img(video_path, dest_path):
    # Reference: https://www.geeksforgeeks.org/extract-images-from-video-in-python/
    # Read the video from specified path
    cam = cv2.VideoCapture(video_path)
    dest_full_path = os.path.join(dest_path, 'imgs')
    try:
        # creating a folder named data
        if not os.path.exists(dest_full_path):
            os.makedirs(dest_full_path)

    # if not created then raise error
    except OSError:
        print ('Error: Creating directory of data')

    # frame
    currentframe = 0

    while(True):

        # reading from frame
        ret,frame = cam.read()

        if ret:
            # if video is still left continue creating images
            name = dest_full_path + '/'+str(currentframe).zfill(4) + '.jpg'
            print ('Creating...' + name)

            # writing the extracted images
            cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()  
    
def getSystemInfo():
    try:
        info={}
        info['platform']=platform.system()
        info['platform-release']=platform.release()
        info['platform-version']=platform.version()
        info['architecture']=platform.machine()
        info['hostname']=socket.gethostname()
        info['ip-address']=socket.gethostbyname(socket.gethostname())
        # info['mac-address']=':'.join(re.findall('..', '%012x' % uuid.getnode()))
        info['processor']=platform.processor()
        info['ram']=str(round(psutil.virtual_memory().total / (1024.0 **3)))+" GB"
        return json.dumps(info)
    except Exception as e:
        logging.exception(e)

def make_video_from_dataloader(video_name, dataloader ):
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = video_name + '.mp4'
    output_video_fps = VIDEO_FPS
    output_video_size = (IMAGE_HEIGHT, IMAGE_WIDTH)
    out = cv2.VideoWriter(output_video_path, fourcc, output_video_fps, output_video_size)

    # iterate over the images and annotations
    for images, annotations in dataloader:
        image = images.numpy().reshape((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL_NUMBER))

        # visualize the images wth annotations
        for annotation in annotations:
            bbox = annotation['bbox']
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            ## TO DO!!
            # cv2.putText(image, annotation['category_id'].toString(), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(image, 'elma', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (0, 0, 255), 1)

        # Add image to video
        out.write(image)

def checkBbox(idx):
    dir_path = "/home/cagnur/stroma/dataset/images/train/imgs"
    type = dir_path.split(os.path.sep)[-2]
    annotation_path = os.path.join(dir_path, "../../../annotations/instances_" + type + ".json")
    image_paths = glob.glob(f"{dir_path}/*.jpg")
    with open(annotation_path, 'r') as f:
        all_json = json.load(f)
        all_annotations = all_json['annotations']

    all_images = [image_path.split(os.path.sep)[-1] for image_path in image_paths]
    all_images = sorted(all_images)


    image_name = all_images[idx]
    image_path = os.path.join(dir_path, image_name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image_width = image.shape[1]
    image_height = image.shape[0]
    # image_resized = cv2.resize(image, (RESIZE_TO, RESIZE_TO))
    image /= 255.0
    annotations = []
    for annotation in all_annotations:
        if annotation['image_id'] == idx+1:
            annotations.append(annotation)
    boxes = []
    labels = []
    for ann in annotations:
        labels.append(ann['category_id'])
        for box_num, box in enumerate([ann['bbox']]):
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[0]) + int(box[2])
            ymax = int(box[1]) + int(box[3])
            width = int(box[2])
            height = int(box[3])


            # xmin_final = (xmin / image_width) * self.width
            # xmax_final = (xmax / image_width) * self.width
            # ymin_final = (ymin / image_height) * self.height
            # ymax_final = (ymax / image_height) * self.height

            # boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
            boxes.append([xmin, ymin, xmax, ymax])
        for box_num, box in enumerate(boxes):
            cv2.rectangle(image,
                      (box[0], box[1]),
                      (box[2], box[3]),
                      (0, 0, 255), 1)

            # cv2.rectangle(image,
            #               (box[0], box[1] + 12),
            #               (box[2], box[3] + 12),
            #               (0, 255, 0), 1)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()