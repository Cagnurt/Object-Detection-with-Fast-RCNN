from torch.utils.data import Dataset, DataLoader
import json
import cv2
import os


class BoltNutDataset(Dataset):
    def __init__(self, path1, path2, transform= None):
        self.path1 = path1  # path to img files
        self.path2 = path2  # path to COCO annotations file
        self.transform = transform

        # read COCO annotations file
        with open(self.path2, 'r') as f:
            all_json = json.load(f)
            self.annotations = all_json['annotations']

    def __len__(self):
        # return the number of frames in the video
        # cap = cv2.VideoCapture(self.path1)
        length = len(os.listdir(self.path1))
        # cap.release()
        return length

    def __getitem__(self, idx):
        # read the idx-th frame from the video
        # cap = cv2.VideoCapture(self.path1)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        bgr_img = cv2.imread(self.path1)
        # if not success:
        #     raise ValueError(f"Failed to read frame {idx}")
        # cap.release()
        # Change channel
        image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HLS)
        if self.transform:
            image = self.transform(image)

        # get annotations for the idx-th frame
        annotations = []
        for annotation in self.annotations:
            if annotation['image_id'] == idx:
                annotations.append(annotation)

        # check if there are any annotations
        if len(annotations) == 0:
            return image, []

        return image, annotations
