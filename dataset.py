# from torch.utils.data import Dataset, DataLoader
# import json
# import cv2
# import os
#
#
# class BoltNutDataset(Dataset):
#     def __init__(self, path1, path2, transform= None):
#         self.path1 = path1  # path to img files
#         self.path2 = path2  # path to COCO annotations file
#         self.transform = transform
#
#         # read COCO annotations file
#         with open(self.path2, 'r') as f:
#             all_json = json.load(f)
#             self.annotations = all_json['annotations']
#
#     def __len__(self):
#         # return the number of frames in the video
#         # cap = cv2.VideoCapture(self.path1)
#         length = len(os.listdir(self.path1))
#         # cap.release()
#         return length
#
#     def __getitem__(self, idx):
#         # read the idx-th frame from the video
#         # cap = cv2.VideoCapture(self.path1)
#         # cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#         bgr_img = cv2.imread(self.path1)
#         # if not success:
#         #     raise ValueError(f"Failed to read frame {idx}")
#         # cap.release()
#         # Change channel
#         image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HLS)
#         if self.transform:
#             image = self.transform(image)
#
#         # get annotations for the idx-th frame
#         annotations = []
#         for annotation in self.annotations:
#             if annotation['image_id'] == idx:
#                 annotations.append(annotation)
#
#         # check if there are any annotations
#         if len(annotations) == 0:
#             return image, []
#
#         return image, annotations

import torch
import cv2
import numpy as np
import os
import glob as glob
import json
from xml.etree import ElementTree as et

from config import (
    CLASSES, RESIZE_TO, TRAIN_DIR, VALID_DIR, BATCH_SIZE
)
from torch.utils.data import Dataset, DataLoader
from custom_utils import collate_fn, get_train_transform, get_valid_transform


# the dataset class
class CustomDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None):
        # TODO: Remove empty images assign as valid images
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes
        # read COCO annotations file
        type = self.dir_path.split(os.path.sep)[-2]
        annotation_path  = os.path.join(self.dir_path, "../../../annotations/instances_"+type+".json")
        with open(annotation_path, 'r') as f:
            all_json = json.load(f)
            self.annotations = all_json['annotations']

        # get all the image paths in sorted order
        self.image_paths = glob.glob(f"{self.dir_path}/*.jpg")
        # self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.image_paths]
        # self.all_images = sorted(self.all_images)
        temp = []
        for ann in self.annotations:
            valid_id = ann['image_id']
            valid_id_name = str(valid_id).zfill(4) + '.jpg'
            temp.append(valid_id_name)
        self.all_images = sorted(list(dict.fromkeys(temp)))

    def __getitem__(self, idx):
        # capture the image name and the full image path
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)
        image_id = image_name.split('.')[0]
        # read the image
        image = cv2.imread(image_path)
        # convert BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        # capture the annotations
        annotations = []
        for annotation in self.annotations:
            if annotation['image_id'] == int(image_id):
                annotations.append(annotation)

        boxes = []
        labels = []
        areas = []

        # get the height and width of the image
        image_width = image.shape[1]
        image_height = image.shape[0]

        # box coordinates from coco to pascal
        for ann in annotations:
            labels.append(ann['category_id'])
            for box_num, box in enumerate([ann['bbox']]):
                xmin = int(box[0])
                ymin = int(box[1])
                xmax = int(box[0])+int(box[2])
                ymax = int(box[1])+int(box[3])
                width = int(box[2])
                height = int(box[3])
                areas.append(width*height)

                # resize the bounding boxes according to the...
                # ... desired `width`, `height`
                xmin_final = (xmin / image_width) * self.width
                xmax_final = (xmax / image_width) * self.width
                ymin_final = (ymin / image_height) * self.height
                ymax_final = (ymax / image_height) * self.height

                boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

        # bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # area of the bounding boxes
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(areas, dtype=torch.int16)
        # no crowd instances
        iscrowd = torch.as_tensor((boxes.shape[0],), dtype=torch.int64)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        # apply the image transforms
        if self.transforms:
            sample = self.transforms(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        # print(target["boxes"].size())
        return image_resized, target

    def __len__(self):
        return len(self.all_images)


# prepare the final datasets and data loaders
def create_train_dataset():
    train_dataset = CustomDataset(TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform())
    return train_dataset


def create_valid_dataset():
    valid_dataset = CustomDataset(VALID_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform())
    return valid_dataset


def create_train_loader(train_dataset, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_loader


def create_valid_loader(valid_dataset, num_workers=0):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return valid_loader


# execute datasets.py using Python command from Terminal...
# ... to visualize sample images
# USAGE: python datasets.py
if __name__ == '__main__':
    # sanity check of the Dataset pipeline with sample visualization
    dataset = CustomDataset(
        TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES
    )
    print(f"Number of training images: {len(dataset)}")


    # function to visualize a single sample
    def visualize_sample(image, target):
        for box_num in range(len(target['boxes'])):
            box = target['boxes'][box_num]
            label = CLASSES[target['labels'][box_num]]
            cv2.rectangle(
                image,
                (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                (0, 255, 0), 2
            )
            cv2.putText(
                image, label, (int(box[0]), int(box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
        cv2.imshow('Image', image)
        cv2.waitKey(0)


    NUM_SAMPLES_TO_VISUALIZE = 5
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = dataset[i]
        visualize_sample(image, target)