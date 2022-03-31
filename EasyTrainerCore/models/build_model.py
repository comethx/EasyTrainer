from torch.hub import load_state_dict_from_url
import torch.nn as nn
from EasyTrainerCore.models import resnet101, densenet121, densenet169, resnet50, mobilenet_v2
from EasyTrainerCore.models import resnext101_32x8d_wsl, resnext101_32x16d_wsl, resnext101_32x32d_wsl, resnext101_32x48d_wsl
from EasyTrainerCore.models import EfficientNet
from EasyTrainerCore.UrlMap import model_urls
from collections import OrderedDict


def Resnet50(num_classes, test=False):
    model = resnet50()
    if not test:
        state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True, model_dir="/weights")
        model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


def Resnet101(num_classes, test=False):
    model = resnet101()
    if not test:
        state_dict = load_state_dict_from_url(model_urls['resnet101'], progress=True, model_dir="./weights")
        model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


def Resnext101_32x8d(num_classes, test=False):
    model = resnext101_32x8d_wsl()
    if not test:
        state_dict = load_state_dict_from_url(model_urls['resnext101_32x8d'], progress=True,
                                              model_dir="./weights")
        model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


def Resnext101_32x16d(num_classes, test=False):
    model = resnext101_32x16d_wsl()
    if not test:
        state_dict = load_state_dict_from_url(model_urls['resnext101_32x16d'], progress=True,
                                              model_dir="./weights")
        model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


def Resnext101_32x32d(num_classes, test=False):
    model = resnext101_32x32d_wsl()
    if not test:
        state_dict = load_state_dict_from_url(model_urls['resnext101_32x32d'], progress=True,
                                              model_dir="./weights")
        model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


def Resnext101_32x48d(num_classes, test=False):
    model = resnext101_32x48d_wsl()
    if not test:
        state_dict = load_state_dict_from_url(model_urls['resnext101_32x48d'],
                                              progress=True, model_dir="./weights")
        model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


def Densenet121(num_classes, test=False):
    model = densenet121()
    if not test:
        state_dict = load_state_dict_from_url(model_urls['densenet121'], progress=True, model_dir="./weights")

        new_state_dict = OrderedDict()

        for k, v in state_dict.items():

            if k.split('.')[0] == 'features' and (len(k.split('.'))) > 4:
                k = k.split('.')[0] + '.' + k.split('.')[1] + '.' + k.split('.')[2] + '.' + k.split('.')[-3] + \
                    k.split('.')[-2] + '.' + k.split('.')[-1]
            else:
                pass
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    fc_features = model.classifier.in_features
    model.classifier = nn.Linear(fc_features, num_classes)
    return model


def Densenet169(num_classes, test=False):
    model = densenet169()
    if not test:
        state_dict = load_state_dict_from_url(model_urls['densenet169'], progress=True, model_dir="./weights")
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            if k.split('.')[0] == 'features' and (len(k.split('.'))) > 4:
                k = k.split('.')[0] + '.' + k.split('.')[1] + '.' + k.split('.')[2] + '.' + k.split('.')[-3] + \
                    k.split('.')[-2] + '.' + k.split('.')[-1]
            else:
                pass
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    fc_features = model.classifier.in_features
    model.classifier = nn.Linear(fc_features, num_classes)
    return model


def Mobilenetv2(num_classes, test=False):
    model = mobilenet_v2()
    if not test:
        state_dict = load_state_dict_from_url(model_urls['mobilenetv2'], progress=True, model_dir="./weights")
        model.load_state_dict(state_dict)
    fc_features = model.classifier[1].in_features
    model.classifier = nn.Linear(fc_features, num_classes)
    return model


def Efficientnet(model_name, num_classes, test=False):
    '''
    model_name :'efficientnet-b0', 'efficientnet-b1-7'
    '''
    model = EfficientNet.from_name(model_name)
    if not test:
        state_dict = load_state_dict_from_url(model_urls[model_name], progress=True, model_dir="./weights")
        model.load_state_dict(state_dict)
    fc_features = model._fc.in_features
    model._fc = nn.Linear(fc_features, num_classes)
    return model
