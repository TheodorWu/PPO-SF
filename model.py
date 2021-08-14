import torch
import torch.nn as nn
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
        feature_emb = torch.zeros(512)

        def copy(m,i,o):
            feature_emb.copy_(o.data.reshape(o.data.size(1)))

        h = self.layer.register_forward_hook(copy)
        self.model(tensor_obs)
        h.remove()

        return feature_emb

class Actor(nn.Module):
    pass

class Critic(nn.Module):
    pass
