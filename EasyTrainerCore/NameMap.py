from EasyTrainerCore.models import Resnet50, Resnet101, Resnext101_32x8d, Resnext101_32x16d, Densenet121, Densenet169, Mobilenetv2, \
    Efficientnet, Resnext101_32x32d, Resnext101_32x48d
MAPPING = {
    'resnext101_32x8d': Resnext101_32x8d,
    'resnext101_32x16d': Resnext101_32x16d,
    'resnext101_32x48d': Resnext101_32x48d,
    'resnext101_32x32d': Resnext101_32x32d,
    'resnet50': Resnet50,
    'resnet101': Resnet101,
    'densenet121': Densenet121,
    'densenet169': Densenet169,
    'mobilenetv2': Mobilenetv2,
    'efficientnet-b0': Efficientnet,
    'efficientnet-b1': Efficientnet,
    'efficientnet-b2': Efficientnet,
    'efficientnet-b3': Efficientnet,
    'efficientnet-b4': Efficientnet,
    'efficientnet-b5': Efficientnet,
    'efficientnet-b6': Efficientnet,
    'efficientnet-b7': Efficientnet,
    'efficientnet-b8': Efficientnet,
}



