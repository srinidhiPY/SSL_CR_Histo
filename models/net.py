import torch
from torch import nn
import torchvision.models as models

###############################'Pre-train classifier'########################


class Classifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super(Classifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(True),
            nn.Linear(128, num_classes))
        print(self.classifier)

    def forward(self, x):
        score = self.classifier(x)
        return score


################################## 'Pre-train model' ########################

class TripletNet(nn.Module):

    def __init__(self, model):
        super(TripletNet, self).__init__()

        # set the model
        if model == 'resnet18':
            model = models.resnet18(pretrained=False)
            model.fc = torch.nn.Sequential()
            self.model = model
            print(self.model)
            self.fc = nn.Sequential(nn.Linear(512*2, 512),
                                    nn.ReLU(True), nn.Linear(512, 256))

        elif model == 'resnet50':
            model = models.resnet50(pretrained=False)
            model.fc = torch.nn.Sequential()
            self.model = model
            print(self.model)
            self.fc = nn.Sequential(nn.Linear(2048*2, 1024),
                                    nn.ReLU(True), nn.Linear(1024, 512))

        else:
            raise NotImplementedError('not supported model type: {}'.format(model))

    def forward(self, i1, i2, i3):
        E1 = self.model(i1)
        E2 = self.model(i2)
        E3 = self.model(i3)

        # Pairwise concatenation of features
        E12 = torch.cat((E1, E2), dim=1)
        E23 = torch.cat((E2, E3), dim=1)
        E13 = torch.cat((E1, E3), dim=1)

        f12 = self.fc(E12)
        f23 = self.fc(E23)
        f13 = self.fc(E13)

        features = torch.cat((f12, f23, f13), dim=1)

        return features

################################## 'Fine-tune model' ##########################

class TripletNet_Finetune(nn.Module):

    def __init__(self, model):
        super(TripletNet_Finetune, self).__init__()

        # set the model
        if model == 'resnet18':
            model = models.resnet18(pretrained=False)
            model.fc = torch.nn.Sequential()
            self.model = model
            print(self.model)
            self.fc = nn.Sequential(nn.Linear(512*2, 512),
                                     nn.ReLU(True), nn.Linear(512, 256))
        else:
            raise NotImplementedError('not supported model type: {}'.format(model))

    def forward(self, i):

        E1 = self.model(i)
        E2 = self.model(i)
        E3 = self.model(i)

        # Pairwise concatenation of features
        E12 = torch.cat((E1, E2), dim=1)
        E23 = torch.cat((E2, E3), dim=1)
        E13 = torch.cat((E1, E3), dim=1)

        f12 = self.fc(E12)
        f23 = self.fc(E23)
        f13 = self.fc(E13)

        features = torch.cat((f12, f23, f13), dim=1)

        return features

################################## Fine-tune Classifier ######################

class FinetuneResNet(nn.Module):

    def __init__(self, num_classes):
        super(FinetuneResNet, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(256*3, num_classes))  # (256 * 3) - 3 way siamese model

    def forward(self, x):
        y = self.classifier(x)
        return y

###############################################################################
