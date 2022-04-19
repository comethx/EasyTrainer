import torch
from torchvision import transforms
from EasyTrainerCore.data.transform import Resize
import json
from PIL import Image

model_path = "data/weights"
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_test_transform(mean=mean, std=std, size=0):
    return transforms.Compose([
        Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


class EasyModel:
    model_data = None

    @staticmethod
    def __init__(path):
        EasyModel.model_data = EasyModel.load(path)

    @staticmethod
    def predict(img):
        if isinstance(img, str):
            img = Image.open(img)
        dic1 = open('EasyTrainerCore/data/label_to_name.json', 'r', encoding='utf-8')
        label_to_name = json.loads(dic1.read())
        dic1.close()
        img = img.convert('RGB')
        img = get_test_transform(size=256)(img).unsqueeze(0)
        with torch.no_grad():
            out = EasyModel.model_data(img)
        label = torch.argmax(out, dim=1).cpu().item()
        confidence = out[0][label].item()
        name = label_to_name[str(label)]
        return name, confidence

    @staticmethod
    def load(path):
        checkpoint = torch.load(path, map_location='cpu')
        model = checkpoint['model']
        model.load_state_dict(checkpoint['model_state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
        EasyModel.model_data = model
        print('<EasyTrainer> model has loaded.')
        return model
