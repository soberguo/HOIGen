import json
import os
import glob
from PIL import Image
import xml.etree.ElementTree as ET




def calculate_iou(rect1, rect2):

    x1, y1, x2, y2 = rect1

    x3, y3, x4, y4 = rect2


    x_intersection1 = max(x1, x3)
    y_intersection1 = max(y1, y3)
    x_intersection2 = min(x2, x4)
    y_intersection2 = min(y2, y4)


    width_intersection = max(0, x_intersection2 - x_intersection1)
    height_intersection = max(0, y_intersection2 - y_intersection1)


    area_intersection = width_intersection * height_intersection


    area_rect1 = (x2 - x1) * (y2 - y1)
    area_rect2 = (x4 - x3) * (y4 - y3)


    iou = area_intersection / (area_rect1 + area_rect2 - area_intersection)

    return iou

def filter_and_remove(rectangles, iou_threshold):
    filtered_rectangles = []
    for i in range(len(rectangles)):
        keep_this_rectangle = True
        for j in range(i+1, len(rectangles)):
            iou = calculate_iou(rectangles[i], rectangles[j])
            if iou >= iou_threshold:
                keep_this_rectangle = False
                break
        if keep_this_rectangle:
            filtered_rectangles.append(rectangles[i])

    return filtered_rectangles
def create_xml_from_coordinates(coordinates, category, filename):

    annotation = ET.Element("annotation")

    img_name=filename.split('\\')[-1][:-4]
    ET.SubElement(annotation, "filename").text = img_name+'.jpg'


    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(int(coordinates[2] - coordinates[0]))
    ET.SubElement(size, "height").text = str(int(coordinates[3] - coordinates[1]))
    ET.SubElement(size, "depth").text = "3"


    object = ET.SubElement(annotation, "object")
    ET.SubElement(object, "name").text = category
    bndbox = ET.SubElement(object, "bndbox")
    ET.SubElement(bndbox, "xmin").text = "0"
    ET.SubElement(bndbox, "ymin").text = "0"
    ET.SubElement(bndbox, "xmax").text = str(int(coordinates[2] - coordinates[0])  )
    ET.SubElement(bndbox, "ymax").text = str(int(coordinates[3] - coordinates[1]) )
    ET.SubElement(object, "difficult").text = '1'


    tree = ET.ElementTree(annotation)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'wb') as f:
        tree.write(f)

    print(f"XML file '{filename}' finish。")

def main(args):



    if args.dataset=='hicodet':
        from hico_label import object_name, all_classnames, human_name

        json_file_path ='./hicodet/instances_train2015.json'

        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)


        print(data.keys())

        print(data['empty'])
        empty_list = []
        for i in range(len(data['empty'])):
            empty_list.append(data['filenames'][data['empty'][i]])
        folder_path = './train2015_gt'
        img_path='./datasets/hico_20160224_det/images/train2015/'
        HOI_IDX_TO_VERB_IDX = [4, 17, 25, 30, 41, 52, 76, 87, 111, 57, 8, 36, 41, 43, 37, 62, 71, 75, 76, 87,
                               98, 110, 111, 57, 10, 26, 36, 65, 74, 112, 57, 4, 21, 25, 41, 43, 47, 75, 76, 77,
                               79, 87, 93, 105, 111, 57, 8, 20, 36, 41, 48, 58, 69, 57, 4, 17, 21, 25, 41, 52,
                               76, 87, 111, 113, 57, 4, 17, 21, 38, 41, 43, 52, 62, 76, 111, 57, 22, 26, 36,
                               39, 45, 65, 80, 111, 10, 57, 8, 36, 49, 87, 93, 57, 8, 49, 87, 57, 26, 34, 36,
                               39, 45, 46, 55, 65, 76, 110, 57, 12, 24, 86, 57, 8, 22, 26, 33, 36, 38, 39, 41,
                               45, 65, 78, 80, 98, 107, 110, 111, 10, 57, 26, 33, 36, 39, 43, 45, 52, 37, 65,
                               72, 76, 78, 98, 107, 110, 111, 57, 36, 41, 43, 37, 62, 71, 72, 76, 87, 98, 108,
                               110, 111, 57, 8, 31, 36, 39, 45, 92, 100, 102, 48, 57, 8, 36, 38, 57, 8, 26, 34,
                               36, 39, 45, 65, 76, 83, 110, 111, 57, 4, 21, 25, 52, 76, 87, 111, 57, 13, 75, 112,
                               57, 7, 15, 23, 36, 41, 64, 66, 89, 111, 57, 8, 36, 41, 58, 114, 57, 7, 8, 15, 23,
                               36, 41, 64, 66, 89, 57, 5, 8, 36, 84, 99, 104, 115, 57, 36, 114, 57, 26, 40,
                               112, 57, 12, 49, 87, 57, 41, 49, 87, 57, 8, 36, 58, 73, 57, 36, 96, 111, 48,
                               57, 15, 23, 36, 89, 96, 111, 57, 3, 8, 15, 23, 36, 51, 54, 67, 57, 8, 14, 15,
                               23, 36, 64, 89, 96, 111, 57, 8, 36, 73, 75, 101, 103, 57, 11, 36, 75, 82,
                               57, 8, 20, 36, 41, 69, 85, 89, 27, 111, 57, 7, 8, 23, 36, 54, 67, 89, 57, 26, 36, 38, 39,
                               45, 37, 65, 76, 110, 111, 112, 57, 39, 41, 58, 61, 57, 36, 50, 95, 48, 111, 57, 2, 9, 36,
                               90, 104, 57, 26, 45, 65, 76, 112, 57, 36, 59, 75, 57, 8, 36, 41, 57, 8, 14, 15, 23, 36,
                               54,
                               57, 8, 12, 36, 109, 57, 1, 8, 30, 36, 41, 47, 70, 57, 16, 36, 95, 111, 115, 48, 57, 36,
                               58,
                               73, 75, 109, 57, 12, 58, 59, 57, 13, 36, 75, 57, 7, 15, 23, 36, 41, 64, 66, 91, 111, 57,
                               12,
                               36, 41, 58, 75, 59, 57, 11, 63, 75, 57, 7, 8, 14, 15, 23, 36, 54, 67, 88, 89, 57, 12, 36,
                               56, 58,
                               57, 36, 68, 99, 57, 8, 14, 15, 23, 36, 54, 57, 16, 36, 58, 57, 12, 75, 111, 57, 8, 28,
                               32, 36,
                               43, 67, 76, 87, 93, 57, 0, 8, 36, 41, 43, 67, 75, 76, 93, 114, 57, 0, 8, 32, 36, 43, 76,
                               93, 114,
                               57, 36, 48, 111, 85, 57, 2, 8, 9, 19, 35, 36, 41, 44, 67, 81, 84, 90, 104, 57, 36, 94,
                               97, 57, 8,
                               18, 36, 39, 52, 58, 60, 67, 116, 57, 8, 18, 36, 41, 43, 49, 52, 76, 93, 87, 111, 57, 8,
                               36, 39, 45,
                               57, 8, 36, 41, 99, 57, 0, 15, 36, 41, 70, 105, 114, 57, 36, 59, 75, 57, 12, 29, 58, 75,
                               87, 93, 111,
                               57, 6, 36, 111, 57, 42, 75, 94, 97, 57, 17, 21, 41, 52, 75, 76, 87, 111, 57, 8, 36, 53,
                               58,
                               75, 82, 94, 57, 36, 54, 61, 57, 27, 36, 85, 106, 48, 111, 57, 26, 36, 65, 112, 57]
    elif args.dataset=='vcoco':
        from vcoco_list import vcoco_values,human_name
        json_file_path = './vcoco/instances_vcoco_train.json'

        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

        folder_path='./trainval_gt_vcoco'
        img_path='./datasets/v-coco/images/train2014/'
        empty_list=[]
        object_name=data['objects']
        all_classnames=[]
        for i in vcoco_values:
            all_classnames.append(i[0]+' '+i[1])

    json_files = glob.glob(os.path.join(folder_path, '*.json'))

    for json_file in json_files:
        if json_file+'.jpg' not in empty_list:
            with open(json_file, 'r') as file:
                json_data = json.load(file)
            json_file_= json_file.strip('\n').split('\\')

            img_name=json_file_[-1].split('.')[0]
            image = Image.open(img_path+img_name+'.jpg')
            len_label=len(json_data['labels'])
            json_box=json_data['boxes']
            json_object_name=json_data['labels'][int(len_label/2):]
            json_hoi_name = json_data['hois']

            boxes_hs=json_box[:int(len_label/2)]
            boxes_os=json_box[int(len_label/2):]

            boxes_o=filter_and_remove(boxes_os,0.5)
            boxes_h = filter_and_remove(boxes_hs, 0.5)




            if args.box_category=='object':

                for o_i in range(len(boxes_o)):
                    input_label=object_name[json_object_name[boxes_os.index(boxes_o[o_i])]]

                    output_path=os.path.join('{}_crop_images\\object_data\\annotations'.format(args.dataset),'{}_{}.xml'.format(img_name,o_i))
                    create_xml_from_coordinates(boxes_o[o_i],input_label,output_path)
                    cropped_img_o=image.crop((boxes_o[o_i][0],boxes_o[o_i][1],boxes_o[o_i][2],boxes_o[o_i][3]))
                    if not os.path.exists('./{}_crop_images/object/{}/'.format(args.dataset,json_object_name[boxes_os.index(boxes_o[o_i])])):

                        os.makedirs('./{}_crop_images/object/{}/'.format(args.dataset,json_object_name[boxes_os.index(boxes_o[o_i])]))
                    cropped_img_o.save('./{}_crop_images/object/{}/{}_{}.jpg'.format(args.dataset,json_object_name[boxes_os.index(boxes_o[o_i])],img_name,o_i))
            elif args.box_category=='human':

                for h_i in range(len(boxes_h)):
                    input_label = human_name[json_object_name[boxes_hs.index(boxes_h[h_i])]]
                    output_path = os.path.join('{}_crop_images\\human_data\\annotations'.format(args.dataset), '{}_{}.xml'.format(img_name, h_i))
                    create_xml_from_coordinates(boxes_h[h_i], input_label, output_path)
                    cropped_img_h = image.crop((boxes_h[h_i][0], boxes_h[h_i][1], boxes_h[h_i][2], boxes_h[h_i][3]))
                    if not os.path.exists('./{}_crop_images/human/{}/'.format(args.dataset,json_object_name[boxes_hs.index(boxes_h[h_i])])):

                        os.makedirs('./{}_crop_images/human/{}/'.format(args.dataset,json_object_name[boxes_hs.index(boxes_h[h_i])]))
                    cropped_img_h.save('./{}_crop_images/human/{}/{}_{}.jpg'.format(args.dataset,json_object_name[boxes_hs.index(boxes_h[h_i])], img_name, h_i))
            elif args.box_category=='hoi':

                boxes_pair = []
                for len_box in range(len(boxes_hs)):
                    boxes_pair.append([min(boxes_os[len_box][0], boxes_hs[len_box][0]), min(boxes_os[len_box][1], boxes_hs[len_box][1]),
                              max(boxes_os[len_box][2], boxes_hs[len_box][2]), max(boxes_os[len_box][3], boxes_hs[len_box][3])])

                for pair_i in range(len(boxes_pair)):
                    input_label = all_classnames[json_hoi_name[pair_i]]
                    output_path = os.path.join('{}_crop_images\\hoi_data\\annotations'.format(args.dataset), '{}_{}.xml'.format(img_name, pair_i))
                    create_xml_from_coordinates(boxes_pair[pair_i], input_label, output_path)
                    cropped_img=image.crop((boxes_pair[pair_i][0],boxes_pair[pair_i][1],boxes_pair[pair_i][2],boxes_pair[pair_i][3]))
                    if not os.path.exists('./{}_crop_images/hoi/{}/'.format(args.dataset,json_hoi_name[pair_i])):

                        os.makedirs('./{}_crop_images/hoi/{}/'.format(args.dataset,json_hoi_name[pair_i]))
                    cropped_img.save('./{}_crop_images/hoi/{}/{}_{}.jpg'.format(args.dataset,json_hoi_name[pair_i],img_name,pair_i))



            print(f'{img_name} is ok!')

    print('end')

    import shutil
    if args.box_category=='hoi':

        source_folder = "./{}_crop_images/hoi/".format(args.dataset)

        destination_folder = "./{}_crop_images/hoi_data/images/".format(args.dataset)
        if not os.path.exists(destination_folder):

            os.makedirs(destination_folder)

    elif args.box_category=='object':

        source_folder = "./{}_crop_images/object/".format(args.dataset)

        destination_folder = "./{}_crop_images/object_data/images/".format(args.dataset)
        if not os.path.exists(destination_folder):

            os.makedirs(destination_folder)

    elif args.box_category=='human':

        source_folder = "./{}_crop_images/human/".format(args.dataset)

        destination_folder = "./{}_crop_images/human_data/images/".format(args.dataset)  #
        if not os.path.exists(destination_folder):

            os.makedirs(destination_folder)


    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                source_file = os.path.join(root, file)

                shutil.copy(source_file, destination_folder)

    print("Image file copied to destination folder")
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser("crop images")
    parser.add_argument('--dataset', type=str, default='hicodet',choices=('hicodet', 'vcoco'))
    parser.add_argument('--box_category', type=str, default='hoi',choices=('hoi', 'object','human'))
    parser.add_argument('--cache-dir', type=str, default='./')

    args = parser.parse_args()

    print(args)
    main(args)
