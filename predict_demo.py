from EasyTrainerCore.Model import EasyModel

EasyModel.load('weights_path')  # load weights
result, confidence = EasyModel.predict('image_path')  # predict image and get result and confidence
print(result, confidence)
