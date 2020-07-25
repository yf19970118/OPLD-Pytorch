import os
import numpy as np
import sys
import json
import pickle
sys.path.append('./')
import matplotlib
from .ResultMerge_multi_process import mergebypoly_multiprocess
from .dota_evaluation_task1 import do_eval
matplotlib.use('Agg')
wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
               'large-vehicle', 'ship', 'tennis-court','basketball-court', 'storage-tank',
               'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']


def voc_eval(quad_predict):
    txt_before_merge = os.path.join(quad_predict, '..', 'before_merge')
    quad_json = '/home/yangfan/Pet-dev/data/DOTA/dota_800_200/val/dota_800_200_val_merge.json'
    quad_json_result_to_txt(quad_json, quad_predict, txt_before_merge, wordname_15)
    txt_after_merge = os.path.join(quad_predict, '..', 'after_merge')
    mergebypoly_multiprocess(txt_before_merge, txt_after_merge)
    det_path = os.path.join(txt_before_merge, '{:s}.txt')
    anno_path = r'/home/yangfan/Pet-dev/data/DOTA-v1/data_ori/val/labelTxt-v1.0/{:s}.txt'
    imageset_file = r'home/yangfan/Pet-dev/data/DOTA-v1/data_ori/val/labelTxt-v1.0'
    do_eval(det_path, anno_path, imageset_file, wordname_15)


def rjson_result2txt(index_json, pred_file, txt_file_dir, categroy_list):
    # img_list = os.listdir(pred_file_dir)
    with open(index_json, 'r') as load_f:
        index_list = json.load(load_f)["images"]
    inex_dict = {}
    for index in index_list:
        # print(index["id"])
        inex_dict[index["id"]] = index["file_name"][:-4]

    with open(pred_file, 'r') as load_f:
        pred_list = json.load(load_f)  # ["images"]

    multi_mkdir(txt_file_dir)
    for category_index, category_name in enumerate(categroy_list):
        if category_name == '__background__':
            continue
        txt_file_path = os.path.join(txt_file_dir, category_name + '.txt')
        with open(txt_file_path, "w") as save_f:
            for pred_index in pred_list:
                if pred_index["category_id"] != category_index:
                    continue
                # print(file_index)
                score = pred_index["score"]
                # if score < 0.005:
                #     continue
                file_index = pred_index["image_id"]
                rbox = pred_index["bbox"]
                rbox[-1] = 180 - rbox[-1]
                rbox = rbox2quad_convert(np.array([rbox]), False).tolist()

                file_name = inex_dict[file_index].split('_')[0] + '__1__' + \
                            inex_dict[file_index].split('_')[1] + '___' + \
                            inex_dict[file_index].split('_')[2]
                line = '{} {} {}'.format(file_name, score, rbox)
                # line = '{} {} {}'.format(inex_dict[file_index], score, rbox)
                line = line.replace('[', '').replace(',', '').replace(']', '')
                save_f.writelines(line + '\n')
        save_f.close()
        # break


def rescale_quad_json_result_to_txt(index_json, pred_file, txt_file_dir, categroy_list, scale_factor):
    with open(index_json, 'r') as load_f:
        index_list = json.load(load_f)["images"]
    inex_dict = {}
    for index in index_list:
        inex_dict[index["id"]] = index["file_name"][:-4]

    with open(pred_file, 'r') as load_f:
        pred_list = json.load(load_f)

    print(len(pred_list))
    multi_mkdir(txt_file_dir)
    for category_index, category_name in enumerate(categroy_list):
        if category_name == '__background__':
            continue
        txt_file_path = os.path.join(txt_file_dir, category_name + '.txt')
        with open(txt_file_path, "w") as save_f:
            for pred_index in pred_list:
                if pred_index["category_id"] != category_index:
                    continue
                score = pred_index["score"]
                file_index = pred_index["image_id"]
                quad = pred_index["bbox"]
                quad = [point / scale_factor for point in quad]

                file_name = inex_dict[file_index].split('_')[0] + '__1__' + \
                            inex_dict[file_index].split('_')[1] + '___' + \
                            inex_dict[file_index].split('_')[2]
                line = '{} {} {}'.format(file_name, score, quad)
                line = line.replace('[', '').replace(',', '').replace(']', '')
                save_f.writelines(line + '\n')
        save_f.close()


def quad_json_result_to_txt(index_json, pred_file, txt_file_dir, categroy_list):
    with open(index_json, 'r') as load_f:
        index_list = json.load(load_f)["images"]
    inex_dict = {}
    for index in index_list:
        inex_dict[index["id"]] = index["file_name"][:-4]

    with open(pred_file, 'r') as load_f:
        pred_list = json.load(load_f)

    print(len(pred_list))
    multi_mkdir(txt_file_dir)
    for category_index, category_name in enumerate(categroy_list):
        if category_name == '__background__':
            continue
        txt_file_path = os.path.join(txt_file_dir, category_name + '.txt')
        with open(txt_file_path, "w") as save_f:
            for pred_index in pred_list:
                if pred_index["category_id"] != category_index:
                    continue
                score = pred_index["score"]
                file_index = pred_index["image_id"]
                quad = pred_index["bbox"][0:8]  # .tolist()

                file_name = inex_dict[file_index].split('_')[0] + '__1__' + \
                            inex_dict[file_index].split('_')[1] + '___' + \
                            inex_dict[file_index].split('_')[2]
                line = '{} {} {}'.format(file_name, score, quad)
                line = line.replace('[', '').replace(',', '').replace(']', '')
                save_f.writelines(line + '\n')
        save_f.close()


def bboxjson_result2txt(index_json, pred_file, txt_file_dir, categroy_list):
    with open(index_json, 'r') as load_f:
        index_list = json.load(load_f)["images"]
    inex_dict = {}
    for index in index_list:
        inex_dict[index["id"]] = index["file_name"][:-4]

    with open(pred_file, 'r') as load_f:
        pred_list = json.load(load_f)

    multi_mkdir(txt_file_dir)
    for category_index, category_name in enumerate(categroy_list):
        if category_name == '__background__':
            continue
        txt_file_path = os.path.join(txt_file_dir, 'Task1_' + category_name + '.txt')
        with open(txt_file_path, "w") as save_f:
            for pred_index in pred_list:
                if pred_index["category_id"] != category_index:
                    continue
                file_index = pred_index["image_id"]
                # print(file_index)
                bbox = pred_index["bbox"]
                ploy = dots4ToRec8([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                score = pred_index["score"]
                line = '{} {} {}'.format(inex_dict[file_index], score, ploy)
                line = line.replace('[', '').replace(',', '').replace(']', '').replace('(', '').replace(')', '')
                save_f.writelines(line + '\n')
        save_f.close()


# if __name__ == '__main__':
#     txt_file = os.path.join('./', 'result/before_split')
#
#     # val_json = 'data/DOTA-v1/dota_800_200/val/dota_800_200_val.json'
#     # pred_file = '/home/yangfan/Pet-dev/ckpts/rcnn/DOTA_rotated/e2e_rotated_faster_rcnn_R-101-C4-2FC_1x/test/bbox.json'
#     # json_result2txt(val_json, pred_file, txt_file, wordname_15)
#     # pkl_result2txt(val_json, pred_file, txt_file, wordname_15)
#     # result2category(pred_file, txt_file, wordname_15)
#     # json_result2txt(val_json, json_file, txt_file, wordname_15)
#
#     # quad_json = '/home/yangfan/Pet-dev/data/DOTA/dota_800_200/val/dota_800_200_val_merge.json'
#     # quad_json = 'data/DOTA-v1/dota_800_200/val/dota_800_200_val_quad_order_all.json'
#     # quad_json = '/home/yangfan/Pet-dev/data/DOTA/dota_1024_200/val/dota_1024_200_val_quad_order.json'
#     quad_json = 'data/DOTA-v1/dota_800_200/val/dota_800_200_val_quad_order.json'
#     quad_predict = '/home/yangfan/Pet-dev/ckpts/DOTA_rotated/five/e2e_hor_quad_grid_R-50-FPN-r2-1x/test/bbox.json'
#     quad_json_result_to_txt(quad_json, quad_predict, txt_file, wordname_15)
#     # scale_factor = 800/600
#     # rescale_quad_json_result_to_txt(quad_json, quad_predict, txt_file, wordname_15, scale_factor)
#
#     # hor_predict = ''
#     # box_json_result_to_txt(quad_json, hor_predict, txt_file, wordname_15)

