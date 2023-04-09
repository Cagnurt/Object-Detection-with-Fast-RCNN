# import torch.nn.functional as F
#
# class Config:
#     def __init__(self,name):
#         self.model = None
#         self.optimizer = None
#         self.criterion = F.nll_loss
#         self.train_dataloader = None
#         self.valid_dataloader = None
#         self.test_dataloader = None
#         self.log_freq = 1000
#         self.epoch = 15
#         self.hidden_dim = 64
#         self.sweep = {  "name" : "Stroma Challange",
#                         "method" : "grid",
#                         "parameters" : {
#                             "learning_rate" : {
#                                 "values" : [0.01, 0.001]
#                             },
#                             "momentum" :{
#                                 "values" : [0.1, 0.5, 0.9]
#                             },
#                         }
#                       }


import torch
VIDEO_FPS = 30
DATASETS = ['train', 'val', 'test']
BATCH_SIZE = 2 # increase / decrease according to GPU memeory
RESIZE_TO = 400 # resize the image for training and transforms
NUM_EPOCHS = 10 # number of epochs to train for
NUM_WORKERS = 4
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(DEVICE)
TRAIN_DIR = '/home/cagnur/stroma/dataset/images/train/imgs'
# TRAIN_DIR = "/home/cagnur/stroma/Uno Cards.v2-raw.voc/train"
VALID_DIR = '/home/cagnur/stroma/dataset/images/val/imgs'
# VALID_DIR = "/home/cagnur/stroma/Uno Cards.v2-raw.voc/valid"
# classes: 0 index is reserved for background
# CLASSES = [
#     '__background__', '11', '9', '13', '10', '6', '7', '0', '5', '4', '2', '14',
#     '8', '12', '1', '3'
# ]
CLASSES = ['__background__', 'bolt','nut']
NUM_CLASSES = len(CLASSES)
# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = True
# location to save model and plots
OUT_DIR = 'outputs'
