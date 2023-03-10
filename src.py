import platform,socket,json,psutil,logging
from constants import *
import json
import cv2
import os


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

