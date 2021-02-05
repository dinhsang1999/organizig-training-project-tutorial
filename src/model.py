"""
Contain the definition of model architecture
"""
import torch, torchvision
import torch.nn as nn


##### BACKBONE MODELS ######
class Backbone(nn.Module):
    def __init__(self, num_classes, is_trained=True):
        super().__init__()
        # Load Resnet50 with pretrained ImageNet weights
        self.net = torchvision.models.resnet.resnet18(pretrained=is_trained)

        # replace the last layer with a new layer that have `num_classes` nodes, followed by Sigmoid function
        classifier_input_size = self.net.fc.in_features
        self.net.fc = nn.Sequential(
                        nn.Linear(classifier_input_size, num_classes),
                        nn.Sigmoid())

    def forward(self, images):
        return self.net(images)


##### RETINA MODELS ######
class RetinaModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = Backbone(num_classes, True) 
        # TODO: develop your retinal model

    def forward(self, images):
        return self.backbone(images)
