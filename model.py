import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.models import resnet34
from torch.autograd import Variable

class FeatureExtractor:
    def __init__(self, device):
        self.model = resnet34(pretrained=True)
        self.device = device
        self.model.to(self.device)
        self.layer = self.model._modules.get('avgpool')
        self.model.eval()

        self.transforms = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),


    def extract_features(self, obs):
        tensor_obs = Variable(self.transforms(obs).unsqueeze(0)).to(self.device)
        feature_emb = torch.zeros(512).to(self.device)

        def copy(m,i,o):
            feature_emb.copy_(o.data.reshape(o.data.size(1)))

        h = self.layer.register_forward_hook(copy)
        self.model(tensor_obs)
        h.remove()

        return feature_emb

class Actor(nn.Module):
    def __init__(self, num_actions):
        super(Actor, self).__init__()

        self.output_size = num_actions

        self.fc1 = nn.Linear(512, 128) # input size comes from pretrained resnet output
        self.fc2 = nn.Linear(128, self.output_size)
        print("Actor model set up:")
        print(self)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)

        return x

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.output_size = 1

        self.fc1 = nn.Linear(512, 128) # input size comes from pretrained resnet output
        self.fc2 = nn.Linear(128, self.output_size)
        print("Critic model set up:")
        print(self)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        return x
