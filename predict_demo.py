from EasyTrainerCore.Model import EasyModel

EasyModel.load('weights/densenet169/epoch_10.pth')
result, confidence = EasyModel.predict('pictures/传单/baidu000005.jpg')
print(result, confidence)
