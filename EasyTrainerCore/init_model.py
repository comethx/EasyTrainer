import os

import torch
import torch.nn as nn

import EasyTrainerCore.NameMap as NameMap
import EasyTrainerCore.preprocess as pre


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def load_model_and_save_dir(model_name, resume_epoch, gpu_nums, froze_front_layers, split):
    if resume_epoch == 0:
        num_classes = pre.init_txt(init=True, split=split)
    else:
        num_classes = pre.init_txt(init=False, split=split)

    save_folder = "weights/" + model_name
    os.makedirs(save_folder, exist_ok=True)
    if not resume_epoch:
        print('<EasyTrainer> Using: {}'.format(model_name))
        print('<EasyTrainer> Loading the weights')
        if model_name.startswith('adv-efficientnet') or model_name.startswith('efficientnet'):
            model = NameMap.MAPPING[model_name](num_classes=num_classes, model_name='efficientnet-b3')
        else:
            model = NameMap.MAPPING[model_name](num_classes=num_classes)
        if froze_front_layers:
            all_parameter_count = len(list(model.named_parameters()))
            current_parameter_id = 0
            print('<EasyTrainer> {} parameters will be frozen'.format(all_parameter_count / 4))
            for name, parameter in model.named_parameters():
                current_parameter_id += 1
                if current_parameter_id > (all_parameter_count / 4):
                    break
                parameter.requires_grad = False
    else:
        print('<EasyTrainer> Resume training from {}  epoch {}'.format(model_name, resume_epoch))
        model = load_checkpoint(os.path.join(save_folder, 'epoch_{}.pth'.format(resume_epoch)))

    if gpu_nums > 1 and torch.cuda.is_available():
        print('<EasyTrainer> Using multiple gpus to training')
        model = nn.DataParallel(model, device_ids=list(range(gpu_nums)))
        model.cuda()
    elif gpu_nums == 1 and torch.cuda.is_available():
        print('<EasyTrainer> Using single gpu to training')
        model.cuda()
    elif gpu_nums == 0:
        print('<EasyTrainer> Using cpu to training')
    print("<EasyTrainer> Initialize the model done")

    return model, save_folder
