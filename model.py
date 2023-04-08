from efficientnet_pytorch import EfficientNet
from constants import IMAGE_CHANNEL_NUMBER, IMAGE_HEIGHT
import torch.nn as nn
import torch.nn.functional as F
#
#
# class Net(nn.Module):
#     def __init__(self, hidden_dim):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, hidden_dim, 3, 1)
#         self.conv2 = nn.Conv2d(hidden_dim, 64, 3, 1)
#         self.dropout = nn.Dropout(0.25)
#         self.fc1 = nn.Linear(147456, 1024)
#         self.fc2 = nn.Linear(1024, 128)
#         self.fc3 = nn.Linear(128, 8)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.fc3(x)
#         output = F.log_softmax(x, dim=1)
#         return output
#
#
# class MLP(nn.Module):
#     def __init__(self, hidden_dim):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(100 * 100 * 1, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1024),
#             nn.ReLU(),
#             nn.Dropout(0.25),
#             nn.Linear(1024, 128),
#             nn.ReLU(),
#             nn.Linear(128, 8),
#             nn.LogSoftmax(dim=1)
#         )
#
#     def forward(self, x):
#         return self.layers(x)
#
#
# class ResNet(nn.Module):
#     def __init__(self, hidden_dim):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, hidden_dim, 3, 1, padding=1)
#         self.conv2 = nn.Conv2d(hidden_dim, 64, 3, 1, padding=1)
#         self.dropout = nn.Dropout(0.25)
#         self.fc1 = nn.Linear(147456, 1024)
#         self.fc2 = nn.Linear(1024, 128)
#         self.fc3 = nn.Linear(128, 8)
#
#     def forward(self, x):
#         x = self.conv2(F.relu(self.conv1(x)) + x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.fc3(x)
#         output = F.log_softmax(x, dim=1)
#         return output
#
#
# class Efficient(nn.Module):
#     def __init__(self, num_class):
#         super(Efficient, self).__init__()
#         self.resnet = EfficientNet.from_pretrained('efficientnet-b0', in_channels=IMAGE_CHANNEL_NUMBER,
#                                                    image_size=IMAGE_HEIGHT)
#         self.l1 = nn.Linear(1000, 256)
#         self.dropout = nn.Dropout(0.75)
#         self.l2 = nn.Linear(256, num_class)
#         self.relu = nn.ReLU()
#
#     def forward(self, input):
#         x = self.resnet(input)
#         x = x.view(x.size(0), -1)
#         x = self.dropout(self.relu(self.l1(x)))
#         x = self.l2(x)
#         output = F.log_softmax(x, dim=1)
#         return output


import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model(num_classes):
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model