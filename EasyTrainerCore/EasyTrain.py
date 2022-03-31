import os
import torch

import torch.optim as optim

import EasyTrainerCore.data as data
import EasyTrainerCore.init_model as init_model

from EasyTrainerCore.utils import adjust_learning_rate_cosine, adjust_learning_rate_step, LabelSmoothingCrossEntropy
import torch.nn as nn
from EasyTrainerCore.Model import EasyModel


def start(model_name='efficientnet-b3',
          froze_front_layers=False,
          optimizer="Adam",
          batch_size=64,
          picture_size=64,
          loss_function="CrossEntropyLoss",
          lr=1e-2,
          lr_adjust_strategy=None,
          max_epoch=10,
          resume_epoch=0,
          save_sequence=5,
          gpu_nums=0,
          weight_decay=5e-4,
          momentum=0.9,
          train_and_val_split=0.8
          ):
    model, save_folder = init_model.load_model_and_save_dir(model_name, resume_epoch, gpu_nums, froze_front_layers,
                                                            train_and_val_split)

    train_dataloader, picture_num = data.get_train_dataloader_and_length(train_label_dir="EasyTrainerCore/data/train.txt",
                                                                         picture_size=picture_size,
                                                                         batch_size=batch_size)
    val_dataloader = data.get_val_dataloader(val_label_dir="EasyTrainerCore/data/val.txt",
                                             picture_size=picture_size,
                                             batch_size=int(batch_size * (1 - train_and_val_split)))

    if optimizer == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    elif optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    if loss_function == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    elif loss_function == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = LabelSmoothingCrossEntropy()

    epoch_size = picture_num // batch_size

    max_iter = max_epoch * epoch_size

    start_iter = resume_epoch * epoch_size

    epoch = resume_epoch

    warmup_epoch = int(max_epoch / 10) + 1
    warmup_steps = warmup_epoch * epoch_size
    global_step = 0

    stepvalues = (10 * epoch_size, 20 * epoch_size, 30 * epoch_size)
    step_index = 0

    base_lr = lr
    checkpoint_save_path = ""
    print('<EasyTrainer> started training')
    for iteration in range(start_iter, max_iter):

        global_step += 1

        if iteration % epoch_size == 0:
            batch_iterator = iter(train_dataloader)
            val_batch_iterator = iter(val_dataloader)
            epoch += 1
            model.train()
            # 保存模型
            if epoch % save_sequence == 0 and epoch > 0:
                checkpoint_save_path = os.path.join(save_folder, 'epoch_{}.pth'.format(epoch))
                checkpoint = {'model': model,
                              'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict(),
                              'epoch': epoch}
                torch.save(checkpoint, checkpoint_save_path)
                print("<EasyTrainer> saving checkpoint model at {}".format(checkpoint_save_path))

        if lr_adjust_strategy == "step":
            if iteration in stepvalues:
                step_index += 1
            lr = adjust_learning_rate_step(optimizer, base_lr, 0.1, epoch, step_index, iteration, epoch_size)
        if lr_adjust_strategy == "cosine":
            lr = adjust_learning_rate_cosine(optimizer, global_step=global_step,
                                             learning_rate_base=base_lr,
                                             total_steps=max_iter,
                                             warmup_steps=warmup_steps)

        images, labels = next(batch_iterator)

        if torch.cuda.is_available() and gpu_nums != 0:
            images, labels = images.cuda(), labels.cuda()
        out = model(images)

        loss = criterion(out, labels.long())

        optimizer.zero_grad()

        if froze_front_layers:
            loss.requires_grad_(True)
        loss.backward()

        optimizer.step()

        prediction = torch.max(out, 1)[1]

        train_correct = (prediction == labels).sum()

        train_acc = (train_correct.float()) / batch_size

        if ((iteration % epoch_size) + 1) / epoch_size >= 1:
            model.eval()

            val_images, val_labels = next(val_batch_iterator)

            if torch.cuda.is_available() and gpu_nums != 0:
                val_images, val_labels = val_images.cuda(), val_labels.cuda()

            val_out = model(val_images)
            val_loss = criterion(val_out, val_labels.long())

            val_prediction = torch.max(val_out, 1)[1]
            val_correct = (val_prediction == val_labels).sum()
            val_acc = (val_correct.float()) / batch_size
            print('<EasyTrainer> Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' +
                  repr(epoch_size) + ' || loss: %.6f||' % (loss.item()) + 'acc: %.3f ||' % (
                          train_acc * 100) + 'LR: %.8f' % lr + '||val_loss: %.6f||' % (
                      val_loss.item()) + 'val_acc: %.3f' % (
                          val_acc * 100))
        elif iteration % 10 == 0:
            print('<EasyTrainer> Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' +
                  repr(epoch_size) + ' || loss: %.6f||' % (loss.item()) + 'acc: %.3f ||' % (
                          train_acc * 100) + 'LR: %.8f' % lr)

    return EasyModel(checkpoint_save_path)
