from EasyTrainerCore import EasyTrain

if __name__ == "__main__":
    EasyTrain.start(

        gpu_nums=0,  # 0: using cpu to train, 1: using gpu to train, more than 1: using multi-gpu to train (default: 0)

        model_name="mobilenetv2",  # choose the model, you can choose from the list (default: mobilenetv2)

        # 'resnext101_32x8d'
        # 'resnext101_32x16d',
        # 'resnext101_32x48d',
        # 'resnext101_32x32d',
        # 'resnet50',
        # 'resnet101',
        # 'densenet121',
        # 'densenet169',
        # 'mobilenetv2',
        # 'efficientnet-b0'
        # 'efficientnet-b8'

        froze_front_layers=False,  # To freeze the parameters of front layers (default: False)

        lr=1e-1,  # learning rate
        lr_adjust_strategy="cosine",  # "cosine" or "step" (default: None)
        optimizer="SGD",  # SGD or Adam
        loss_function="CrossEntropyLoss",
        # ↑ CrossEntropyLoss/FocalLoss/SoftmaxCrossEntropyLoss (default: CrossEntropyLoss)
        train_and_val_split=0.8,  # train and val split ratio (default: 0.8)
        picture_size=256,  # picture size for train and val dataset (default: 256)

        batch_size=64,  # batch size for training (default: 64)
        resume_epoch=0,  # resume training from last_epoch (default: 0)
        max_epoch=3,  # max epoch for training (default: 10)
        save_sequence=2  # save model every n epochs (default: 2)

    )
