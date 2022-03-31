import json


def init_label_to_name_json():
    dic = {}
    with open('EasyTrainerCore/data/train.txt', encoding='ansi') as file_object:
        for line in file_object:
            label = line.split(',')[1]
            name = line.split(',')[0].split('\\')[-2]
            dic[label.rstrip('\n')] = name

    f = open('EasyTrainerCore/data/label_to_name.json', 'w', encoding='utf8')
    f.writelines(json.dumps(dic, ensure_ascii=False))
