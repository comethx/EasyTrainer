from EasyTrainerCore.Model import EasyModel

EasyModel.load('weights_path')
result, confidence = EasyModel.predict('image_path')
print(result, confidence)
