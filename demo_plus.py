from EasyTrainerCore import EasyTrain

if __name__ == "__main__":
    # after training, the EasyTrain.start() will return the latest model
    model = EasyTrain.start(

        train=True,  # True: train the model, False: test the model you choose to resume
        train_and_val_split=0.8,
        # ↑ train and validation split, if you decide to validate the model after training,
        # ↑ please don't change this value before and after retraining (default: 0.8)
        gpu_nums=1,  # 0: using cpu to train, 1: using gpu to train, more than 1: using multi-gpu to train (default: 0)
        model_name='densenet169',  # choose the model, you can choose from the list (default: efficientnet-b3)

        # 'resnext101_32x8d'
        # 'resnext101_32x16d',
        # 'resnext101_32x48d',
        # 'resnext101_32x32d',
        # 'resnet50',
        # 'resnet101',
        # 'densenet121',
        # 'densenet169',
        # 'mobilenetv2',
        # 'efficientnet-b0' ~ 'efficientnet-b8'

        froze_front_layers=True,  # To freeze the parameters of front layers (default: False)

        lr=1e-3,  # learning rate (default: 1e-2)
        lr_adjust_strategy="cosine",  # "cosine" or "step" (default: None)
        optimizer="Adam",  # SGD or Adam (default: Adam)
        loss_function="CrossEntropyLoss",
        # ↑ CrossEntropyLoss or FocalLoss or SoftmaxCrossEntropyLoss (default: CrossEntropyLoss)
        picture_size=256,  # the picture size of the model (default: 64)
        batch_size=200,  # batch size for training (default: 64)

        resume_epoch=10,  # resume training or testing from which epoch (default: 0)
        max_epoch=12,  # max epoch for training (default: 10)
        save_sequence=2  # save model every n epochs (default: 2)
    )
