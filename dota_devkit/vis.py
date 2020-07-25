# import math
import numpy as np
# import torchvision
import cv2
import os
import json
import shutil
import sys
import math

import xml.dom.minidom

sys.path.append('./')
from DOTA_devkit.dota_utils import wordname_15
from pbox_utils.coordinate_convert import quad2hboxwh_convert, hboxwh2quad_convert, \
    rbox2quad_convert, quad2rbox_convert

_GRAY = [218, 227, 218]
_GREEN = [18, 127, 15]
_WHITE = [255, 255, 255]
_RED = [0, 0, 255]
_origan= [0, 140, 255]



def get_class_string(class_index, score, dataset):
    # class_text = dataset.classes[class_index] if dataset is not None else \
    #     'id{:d}'.format(class_index)
    class_text = ''#''id{:d}'.format(class_index)
    return class_text + ' {:0.2f}'.format(score).lstrip('0')


def vis_bbox(img, bbox, bbox_color):
    """Visualizes a bounding box."""
    (x0, y0, w, h) = bbox
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    cv2.rectangle(img, (x0, y0), (x1, y1), bbox_color, thickness=2)

    return img


def vis_class(img, pos, class_str, bg_color):
    """Visualizes the class."""
    font_color = _WHITE
    font_scale = 0.7

    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    # Place text background.
    back_tl = x0, y0 - int(1.3 * txt_h)
    back_br = x0 + txt_w, y0
    cv2.rectangle(img, back_tl, back_br, bg_color, -1)
    # Show text.
    txt_tl = x0, y0 - int(0.3 * txt_h)
    cv2.putText(img, txt, txt_tl, font, font_scale, font_color, lineType=cv2.LINE_AA)

    return img


def vis_pbox_v1(img, bbox):
    """Visualizes a bounding box."""
    # (x0, y0, w, h, p_w, p_h) = bbox
    bbox = np.array(bbox)
    bbox = hboxwh2quad_convert(bbox.reshape(-1, 6), False)#
    bbox = np.reshape(bbox, (1, -1, 2)).astype(np.int32)
    # print(bbox)
    # x1, y1 = int(x0 + w), int(y0 + h)
    # x0, y0 = int(x0), int(y0)
    # cv2.rectangle(img, (x0, y0), (x1, y1), bbox_color, thickness=2)

    cv2.polylines(img, bbox.astype(np.int32), 1, _GREEN, 2)
    return img


def json_vis(anno_path, img_dir, category_list):
    with open(anno_path, 'r') as load_f:
        load_dict = json.load(load_f)
    # for key in load_dict.keys():
    #     print(key)
    # # annotations, info, categories, images
    # print('anno:', load_dict["annotations"][0])
    # # {'segmentation': [[320, 218, 701, 225, 668, 608, 311, 599]],
    # # 'bbox': [311, 218, 390, 390], 'area': 141126.0,
    # # 'image_id': 4, 'category_id': 2, 'id': 1, 'iscrowd': 0}
    #
    # print('img:', load_dict["images"][0])
    # # {'width': 1024, 'file_name': 'P3536__1__13860___11088.png', 'height': 1024, 'id': 1}

    anno_dict = load_dict["annotations"]
    img_index_dict = load_dict["images"]
    ins_color = _GREEN
    for img_ins in img_index_dict:
        index = img_ins["id"]
        if index ==1:
            continue
        img_name = img_ins["file_name"]
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)

        annos = [anno for anno in anno_dict if anno["image_id"]==index]
        # print('num:', len(annos))
        if len(annos)==0:
            continue
        for instance in annos:
            bbox = instance["bbox"]
            # img = vis_bbox(img, bbox[:4], ins_color)
            img = vis_pbox_v1(img, bbox, ins_color)

            class_index = instance["category_id"]
            # print('class_index:', class_index)
            txt = '{}'.format(category_list[class_index])  # .lstrip('0')wordname_18
            img = vis_class(img, (bbox[0], bbox[1] - 2), txt, ins_color)
        cv2.imwrite('./{}.png'.format(img_name[:-4]), img)
        break


def dots4ToRec8(poly):
    xmin, ymin, xmax, ymax = poly[0], poly[1], poly[2], poly[3],
    return [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]


def pred_vis(img_index_path, json_file_path, image_dir):
    img_index_dict = {}
    with open(img_index_path, 'r') as load_f:
        index_dict = json.load(load_f)["images"]
    for img_ins in index_dict:
        # print(img_ins)
        img_index_dict[img_ins["id"]] = img_ins["file_name"][:-4]

    file_list = os.listdir(image_dir)
    file_list.sort()
    file_name = file_list[8]
    # print(file_name)

    img_path = os.path.join(image_dir, file_name)
    img = cv2.imread(img_path)

    with open(json_file_path, 'r') as load_f:
        anno_list = json.load(load_f)
    for anno_ins in anno_list:
        img_name = img_index_dict[anno_ins["image_id"]]
        if img_name != file_name[:-4]:
            continue
        # print(anno_ins["category_id"])
        category = wordname_18[anno_ins["category_id"]]
        score = anno_ins["score"]
        bbox = anno_ins["bbox"]

        # rbox = dots4ToRec8([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
        print (bbox)
        # img = vis_bbox(img, [rbox[0], rbox[1], rbox[4]-rbox[0], rbox[5]-rbox[1]], _GREEN)
        img = vis_bbox(img, bbox, _GREEN)
        # break
    print(file_name[:-4])
    cv2.imwrite('./{}.png'.format(file_name[:-4]), img)


def dots4result_vis(pred_dir, img_dir, cat_list):
    objects = []

    for categroy in cat_list:
        if categroy =='__background__':
            continue
        file_path = os.path.join(pred_dir, categroy+'.txt')
        with open(file_path, 'r') as f:
            lines = f.readlines()
            splitlines = [x.strip().split(' ') for x in lines]
            for splitline in splitlines:
                # if len(splitline) < 9:
                #     # print(splitline)
                #     continue
                object_struct = {}
                object_struct['name'] = splitline[0]
                object_struct['bbox'] = [int(float(splitline[2])),
                                             int(float(splitline[3])),
                                             int(float(splitline[4])),
                                             int(float(splitline[5]))]
                objects.append(object_struct)

    file_list = os.listdir(img_dir)
    file_list.sort()
    file_name = 'P1178.png'
    print(file_name)

    img_path =os.path.join(img_dir, file_name)
    img = cv2.imread(img_path)
    for obj in objects:
        # print(obj["name"])
        if obj["name"]!= file_name[:-4]:
            continue
        bbox = obj["bbox"]
        # print([bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]])
        img = vis_bbox(img, [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]], _GREEN)
        # break
    cv2.imwrite('./{}.png'.format(file_name[:-4]), img)


def result_vis(pred_dir, img_dir, cat_list):
    objects = []

    for categroy in cat_list:
        if categroy == '__background__':
            continue
        file_path = os.path.join(pred_dir, categroy+'.txt')#'Task1_'+
        with open(file_path, 'r') as f:
            lines = f.readlines()
            splitlines = [x.strip().split(' ') for x in lines]
            num = 0
            for splitline in splitlines:
                if len(splitline) < 9:
                    continue
                num += 1
                object_struct = {}
                object_struct['name'] = splitline[0]
                object_struct['bbox'] = np.array(splitline[2:]).astype(float)
                object_struct['index'] = num
                object_struct['category'] = categroy
                objects.append(object_struct)

    file_list = os.listdir(img_dir)
    file_list.sort()
    # print(file_list)
    file_name = 'P0007.png'#file_list[1]#'P0760.png'#'P2802.png'#'P2163.png'#
    # for index, file_name in enumerate(file_list):
    #     print(file_name)

    img_path = os.path.join(img_dir, file_name)
    img = cv2.imread(img_path)
    # file_name = file_name.split('_')[0] + '__1__' + file_name.split('_')[1] + '___' + \
    #             file_name.split('_')[2] + '.png'
    print(file_name)
    # cv2.imwrite('./{}_ori.png'.format(file_name[:-4]), img)#_poly
    num = 0
    for obj in objects:
        if obj["name"] != file_name[:-4]:
            continue
        num += 1
        # print(obj["name"])
        bbox = obj["bbox"]
        # img = vis_bbox(img, bbox, _GREEN)
        box = np.reshape(bbox, (1, -1, 2)).astype(np.int32)
        # print(box)
        cv2.polylines(img, box.astype(np.int32), 1, _GREEN, 2)
        # cv2.circle(img, (int(bbox[0]), int(bbox[1])), 1, _RED, 1)
        img = vis_class(img, (int(bbox[0] + bbox[4])//2, int(bbox[1] + bbox[5])//2), str(obj['category']), _RED)
    cv2.imwrite('./{}_rbox.png'.format(file_name[:-4]), img)#_poly


def txt_vis(pred_dir, img_dir):
    objects = []

    label_list = os.listdir(pred_dir)
    print(len(label_list))
    for label_txt in label_list:
        file_path = os.path.join(pred_dir, label_txt)
        with open(file_path, 'r') as f:
            lines = f.readlines()
            splitlines = [x.strip().split(',') for x in lines]
            for splitline in splitlines:
                if len(splitline) < 3:
                    # print(splitline)
                    continue
                object_struct = {}
                object_struct['name'] = label_txt[:-4]
                object_struct['categroy'] = int(splitline[4])
                object_struct['bbox'] = np.array(splitline[:4]).astype(float)
                objects.append(object_struct)

    file_list = os.listdir(img_dir)
    file_list.sort()
    print(len(file_list))
    # file_name = file_list[0]#'P2181.png'#
    for index, file_name in enumerate(file_list):
        print(file_name)

        img_path =os.path.join(img_dir, file_name)
        img = cv2.imread(img_path)
        for obj in objects:
            # print(obj["name"])
            # print(obj["name"], file_name[:-4])
            if obj["name"] != file_name[:-4]:
                continue
            bbox = obj["bbox"]
            txt_ = mininame_3[obj["categroy"]]
            vis_bbox(img, [bbox[0], bbox[1], bbox[2] - bbox[1], bbox[3] - bbox[0]], _GREEN)
            img = vis_class(img, (bbox[0], bbox[1] - 2), txt_, _GREEN)
            # box = np.reshape(bbox, (1, -1, 2)).astype(np.int32)
            # print(box)
            # cv2.polylines(img, box.astype(np.int32), 1, _GREEN, 2)
            # cv2.circle(img, tuple(bbox[:2].astype(np.int32)), 1, _RED, 2)
            # cv2.circle(img, tuple(bbox[2:4].astype(np.int32)), 1, _WHITE, 2)
        # cv2.imwrite('./{}_0.png'.format(file_name[:-4]), img[:4000, :4000, :])
        # cv2.imwrite('./{}_1.png'.format(file_name[:-4]), img[4000:8000, :4000, :])
        # cv2.imwrite('./{}.png'.format(file_name[:-4]), img[:4000, :4000, :])
        cv2.imwrite('./{}.png'.format(file_name[:-4]), img)


def ori_label_vis(label_dir, img_dir):
    file_list = os.listdir(img_dir)
    file_list.sort()
    file_name = file_list[8][:-4]
    file_name = 'P0696__1__739___600'
    print(file_name)

    objects = []
    file_path = os.path.join(label_dir, file_name+'.txt')
    print(file_path)
    with open(file_path, 'r') as f:
        lines = f.readlines()
        splitlines = [x.strip().split(' ') for x in lines]
        for splitline in splitlines:
            if len(splitline) < 9:
                # print(splitline)
                continue
            object_struct = {}
            object_struct['category'] = splitline[8]
            object_struct['poly'] = np.array(splitline[:8]).astype(float)#.tolist()
            objects.append(object_struct)

    img_path =os.path.join(img_dir, file_name+'.png')
    img = cv2.imread(img_path)
    for obj in objects:
        # print(obj["name"])
        # if obj["category"]!= file_name[:-4]:
        #     continue
        bbox = obj["poly"]
        box = np.reshape(bbox, (1, -1, 2)).astype(np.int32)
        # print(box)
        cv2.polylines(img, box.astype(np.int32), 1, _GREEN, 3)
        cv2.circle(img, tuple(bbox[:2].astype(np.int32)), 1, _RED, 3)
    cv2.imwrite('./{}.png'.format(file_name), img)


def vis_polyes(img, rotated_pts, color):
    cv2.line(img, (int(rotated_pts[0,0]),int(rotated_pts[0,1])), (int(rotated_pts[1,0]),int(rotated_pts[1,1])), color,1)
    cv2.line(img, (int(rotated_pts[1,0]),int(rotated_pts[1,1])), (int(rotated_pts[2,0]),int(rotated_pts[2,1])), color,1)
    cv2.line(img, (int(rotated_pts[2,0]),int(rotated_pts[2,1])), (int(rotated_pts[3,0]),int(rotated_pts[3,1])), color,1)
    cv2.line(img, (int(rotated_pts[3,0]),int(rotated_pts[3,1])), (int(rotated_pts[0,0]),int(rotated_pts[0,1])), color,1)
    return img


if __name__ == '__main__':
    # img_dir = 'data/DOTA-v1/data_1024/trainval1024_1/images'
    # anno_dir = 'data/DOTA-v1/annotations'
    # anno_name = 'sub1024p1_train.json'
    # anno_path = os.path.join(anno_dir, anno_name)
    # json_vis(anno_path, img_dir, wordname_15)

    # anno_dir = r'result/faster_rcnn_C4_1x_split'#train_json'#cpt_Rbox_train_label'#error_label'#
    # imageset_file = r'data/DOTA-v1/data_1024_512/val/images'
    # result_vis(anno_dir ,imageset_file, wordname_15)

    # anno_dir = r'result/faster_rcnn_C4_1x_'#train_json'#cpt_Rbox_train_label'#error_label'#
    # imageset_file = r'data/DOTA-v1/data_1024_512/val/images'
    # result_vis(anno_dir ,imageset_file, wordname_15)

    anno_dir = r'result/before'  # val_json'#faster_rcnn_C4_ref'#error_label'#
    imageset_file = r'data/DOTA-v1/data_ori/val/images'  # 'data/DOTA-v1/data_ori/val/images'
    result_vis(anno_dir ,imageset_file, wordname_15)

    # img_dir = './data/DOTA-v1/data_ori/val/images'
    # file_name = 'P2214.png'
    # img_path = os.path.join(img_dir, file_name)
    # img = cv2.imread(img_path)
    # poly = np.array([244.0, 299.0, 286.0, 299.0, 286.0, 347.0, 244.0, 347.0]).reshape(-1, 2)
    # img = vis_polyes(img, poly, _GREEN)
    # poly = np.array([244, 333, 259, 299, 286, 311, 270, 347]).reshape(-1, 2)
    # img = vis_polyes(img, poly, _GREEN)
    # cv2.imwrite('./{}_test2.png'.format(file_name[:-4]), img)
    # # rbox = [153.0, 1867.0, 44.721359549995796, 22.360679774997898, 116.5650500949944]
    # # poly = [1062.0, 1886.0, 1062.0, 1826.0, 1120.0, 1826.0000000000002, 1120.0, 1886.0000000000002]
    # # rbox = get_rbox(poly)
    # # pbox = get_pbox(rbox)
    # # poly1 = pbox2poly(pbox)
    # poly1 = np.array([[1062.0, 1826.0], [1120.0, 1826.0], [1120.0, 1886.0000000000002], [1062.0, 1886.0000000000002]])
    # # poly2 = rbox2poly(rbox)
    # # print(poly2)
    # # get_pbox(rbox)

    # _GREEN = [18, 127, 15]
    # ins_color = _GREEN
    # img = np.zeros((800, 800, 3))
    # poly1 = np.array([[720., 217.00001051], [760., 148.00001526], [800., 181.863286], [760., 250.86328125]])
    # img = vis_polyes(img, poly1, ins_color)
    # poly1 = np.array([[720., 196.42735731], [760., 148.00001526], [800., 202.4359392], [760., 250.86328125]])
    # poly2 = np.array([720.0, 148.00001525878906, 80.0, 102.86326599121094])
    # img = vis_polyes(img, poly1, ins_color)
    # img = vis_bbox(img, poly2, ins_color)
    # cv2.imwrite('./{}_test2.png'.format(1), img)#file_name[:-4]