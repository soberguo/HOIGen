import os
import json
import xml.etree.ElementTree as ET
from hico_text_label import hico_unseen_index
from vcoco_list import vcoco_values, human_name
from hico_labels import all_classnames, human_name, object_name,human_seen_name,object_seen_name


def main(args):
    json_file_path = args.json_path

    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)


    if args.data == 'hoi_data':
        labels = all_classnames
        if args.zs:
            unseen_label = hico_unseen_index[args.zs_type]
        else:
            unseen_label = []
    elif args.data == 'human_data':
        labels = human_name
        if args.zs_type=='unseen_object':
            unseen_label = [item for item in human_name if item not in human_seen_name]
        else:
            unseen_label = []
    elif args.data == 'object_data':
        labels = object_name
        if args.zs_type=='unseen_object':
            unseen_label = [item for item in object_name if item not in object_seen_name]
        else:
            unseen_label = []

    xml_path = args.xml_path
    train_data = []
    test_data = []
    val_data = []


    aa = []
    i = 0
    for _, _, files in os.walk(xml_path):

        for file in files:
            i += 1

            xml_file = os.path.join(xml_path, file)


            tree = ET.parse(xml_file)
            tree_root = tree.getroot()
            filename = tree_root.find("filename").text
            filename2 = filename.split('\\')[-1]
            label = labels.index(tree_root.find("object").find("name").text)


            if label not in unseen_label:
                train_data.append([filename2, label, labels[label]])
                aa.append(label)
            else:
                test_data.append([filename2, label, labels[label]])
    data = {'train': train_data, 'val': val_data, 'test': test_data}


    if args.zs:
        output_json_file = f"./{args.dataset}_images/{args.data}/{args.data[:-5]}_split_data.json"
    else:
        output_json_file = f"./{args.dataset}_images/{args.data}/{args.data[:-5]}_split_data_{args.zs_type}.json"
    with open(output_json_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print("JSON finish")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', default='./hicodet/instances_train2015.json"', type=str)
    parser.add_argument('--dataset', default='hicodet_crop', type=str, choices=('vcoco_crop', 'hicodet_crop'))
    parser.add_argument('--xml_path', default='./hicodet_crop_images/hoi_data/annotations"', type=str)
    parser.add_argument('--data', default='hoi_data', type=str, choices=('hoi_data', 'human_data', 'object_data'))
    parser.add_argument('--zs', default=True, type=bool)
    parser.add_argument('--zs_type', type=str, default='rare_first',
                        choices=['rare_first', 'non_rare_first', 'unseen_verb', 'unseen_object', 'uc0', 'uc1', 'uc2',
                                 'uc3', 'uc4'])
    parser.add_argument('--backbone', default="ViT-B/16", type=str)

    args = parser.parse_args()
    print(args)
    main(args)






