# ML Challenge

# Problem Definition

Let’s make the problem **SMART**
**S**pecific: The model should be able to accurately track different types of nuts and bolts, even under varying lighting conditions, and keep count of them.
**M**easurable: Defining end-2-end pipeline with decreasing loss values over iterations for train and validation. (Because it is my first attempt to solve this problem, this success definition is meaningful. For further iteration, this part should be modified like that: maximizing accuracy subject to <0.5sec CPU inference time)
**A**ctionable: Using simple and familiar methods for you. (since it is an initial attempt)
**R**elevant: End2end runnable pipeline for customer 
**T**ime-bound: it should be implemented in ~10h

## Constraint(s) and Approach to the constraint(s)

**Time**. 

Firstly, I researched which model is used in the industry: Fast RCNN according to the reference. 

So, I searched whether there is any ready pipeline for Fast-RCNN. If there is, I can use in my implementation as a starting point:

[https://debuggercafe.com/a-simple-pipeline-to-train-pytorch-faster-rcnn-object-detection-model/](https://debuggercafe.com/a-simple-pipeline-to-train-pytorch-faster-rcnn-object-detection-model/)

This is my reference for the pipeline. 

**GPU**

Google Colab GPU can be used. In order to access the data from Colab, I need to update the dataset to Google Drive. 

# Pipeline

![Figure - 1: Stroma Challange Pipeline
](/readme_imgs/pipeline.png/pipeline.png)

Figure - 1: Stroma Challange Pipeline

# Train

Dataset structure:

```bash
├── dataset
│   ├── annotations
│   │   ├── instances_test.json
│   │   ├── instances_train.json
│   │   └── instances_val.json
│   └── images
│       ├── test
│       │   ├── imgs
│       │   └── test.mp4
│       ├── train
│       │   ├── imgs
│       │   └── train.mp4
│       └── val
│           ├── imgs
│           └── val.mp4
```

1. Open config.py
2. Change paths with paths to imgs folder
3. Open the terminal and write:

```bash
python train.py
```

# Implementation Challanges

- Reference code was for different dataset formats such as Different folder structures, different annotation storing, and different bounding box formats. Modify them for my dataset structure.
- Pytorch does not accept annotations if it is empty. Eliminate empty images from datasets.
- The object and corresponding bbox does not fit well. Because there is a one-directional flow throughout the image, I assigned annotation to the previous image. It fits perfectly now.
- My computer's GPU could not handle it well so I carry the dataset to my drive so that Colab can connect and use the data.
