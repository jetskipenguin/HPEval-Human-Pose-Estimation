import torch.nn as nn
import torchvision.models as models


class DeepPose(nn.Module):
    """
    DeepPose model for human pose estimation.
    Uses a pre-trained ResNet as a backbone to extract features,
    and a fully connected layer to regress keypoint coordinates.
    """
    def __init__(self, num_keypoints=17):
        super(DeepPose, self).__init__()
        resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(2048, num_keypoints * 2)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x