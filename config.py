import torch
VIDEO_FPS = 30
DATASETS = ['train', 'val', 'test']
BATCH_SIZE = 2 # increase / decrease according to GPU memeory
RESIZE_TO = 210 # resize the image for training and transforms
NUM_EPOCHS = 10 # number of epochs to train for
NUM_WORKERS = 8
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(DEVICE)
TRAIN_DIR = '/home/cagnur/stroma/dataset/images/train/imgs'
VALID_DIR = '/home/cagnur/stroma/dataset/images/val/imgs'
# classes: 0 index is reserved for background
CLASSES = ['__background__', 'bolt','nut']
NUM_CLASSES = len(CLASSES)
# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = True
# location to save model and plots
OUT_DIR = 'outputs'
