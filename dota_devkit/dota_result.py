import json
import os
from ResultMerge_multi_process import merge_poly_multiprocess

wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
               'tennis-court', 'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout',
               'harbor', 'swimming-pool', 'helicopter']


def get_dota_result(txt_path, pred_json_path, ann_path):
    off_quad_json_result_to_txt(ann_path, pred_json_path, txt_path, wordname_15)
    result_path = os.path.join(txt_path, 'result')
    merge_poly_multiprocess(txt_path, result_path)


def off_quad_json_result_to_txt(index_json, pred_file, txt_file_dir, categroy_list):
    with open(index_json, 'r') as load_f:
        index_list = json.load(load_f)["images"]
    inex_dict = {}
    for index in index_list:
        inex_dict[index["id"]] = index["file_name"][:-4]

    with open(pred_file, 'r') as load_f:
        pred_list = json.load(load_f)

    # multi_mkdir(txt_file_dir)
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
                quad = pred_index["bbox"][0:8]  # .tolist()        P0003__1__0___0.txt

                file_name = inex_dict[file_index].split('__')[0] + '__' + \
                            inex_dict[file_index].split('__')[1] + '__' + \
                            inex_dict[file_index].split('__')[2] + '___' + \
                            inex_dict[file_index].split('___')[-1].replace('.png', '')
                line = '{} {} {}'.format(file_name, score, quad)
                line = line.replace('[', '').replace(',', '').replace(']', '')
                save_f.writelines(line + '\n')
        save_f.close()
