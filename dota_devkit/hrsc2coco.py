import os
import math
import cv2
import json
import glob
import numpy as np
import xml.etree.ElementTree as ET

_classes = ("__background__", 'ship',)

data_list = './labels/trainval.txt'  # train, val or test txt, No suffix
img_path = './images/'  # image folder path
xml_path = './xml/'  # xml folder path
save_json_path = './trainval/HRSC2016_trianval_wo.json'  # name for save json
new_img_path = './trainval/images'


class Convert_xml_to_coco(object):
    def __init__(self):
        self.data_list = open(data_list, 'r').read().splitlines()

        self.images = []
        self.categories = []
        self.annotations = []

        self.label_map = {}
        for i in range(len(_classes)):
            self.label_map[_classes[i]] = i
        self.annID = 0

        self.transfer_process()
        self.save_json()

    def transfer_process(self):
        for i in range(1, len(_classes)):
            categories = {'supercategory': _classes[i], 'id': i,
                          'name': _classes[i]}

            self.categories.append(categories)

        print(self.categories)
        for num, data_name in enumerate(self.data_list):
            if num % 100 == 0 or num+1 == len(self.data_list):
                print('XML transfer process  {}/{}'.format(num+1, len(self.data_list)))

            # split index
            data_name = data_name.split('.')[0]
            # XML
            img = cv2.imread(img_path + data_name + '.bmp')
            tree = ET.parse(xml_path + data_name + '.xml')
            filename = data_name + '.png'
            height, width, _ = img.shape
            # cv2.imwrite(new_img_path + '/' + data_name + '.png', img)
            image = {'height': height, 'width': width, 'id': num, 'file_name': filename}
            self.images.append(image)

            objects = tree.find('HRSC_Objects')
            object = objects.findall('HRSC_Object')
            for ix, obj in enumerate(object):
                cx = float(obj.find('mbox_cx').text)
                cy = float(obj.find('mbox_cy').text)
                w = float(obj.find('mbox_w').text)
                h = float(obj.find('mbox_h').text)
                angle = float(obj.find('mbox_ang').text)
                head = (float(obj.find('header_x').text), float(obj.find('header_y').text))
                rbbox = ((cx, cy), (w, h), angle*180/math.pi)
                quad = cv2.boxPoints(rbbox)
                quad = np.array(quad, dtype=np.int).reshape(-1).tolist()
                # quad = self.chage_order(quad, head)
                quad.append(int(cx))
                quad.append(int(cy))
                try:
                    difficult = int(obj.find('difficult').text)
                except:
                    difficult = 0

                x1 = np.maximum(0.0, float(obj.find('box_xmin').text))
                y1 = np.maximum(0.0, float(obj.find('box_ymin').text))
                x2 = np.minimum(width - 1.0, float(obj.find('box_xmax').text))
                y2 = np.minimum(height - 1.0, float(obj.find('box_ymax').text))

                bbox = [x1, y1, x2, y2]  # [x,y,x,y]
                area = (x2 - x1 + 1) * (y2 - y1 + 1)

                # annotations
                annotation = {'segmentation': [quad], 'iscrowd': 0, 'area': area, 'image_id': num,
                              'bbox': bbox, 'difficult': difficult,
                              'category_id': 1, 'id': self.annID}
                self.annotations.append(annotation)
                self.annID += 1

    def save_json(self):
        data_coco = {'images': self.images, 'categories': self.categories, 'annotations': self.annotations}
        json.dump(data_coco, open(save_json_path, 'w'))

    def chage_order(self, poly, head):
        outpoly = []
        distances = [cal_line_length((poly[i * 2], poly[i * 2 + 1]), head) for i in range(len(poly)//2)]
        pos = np.array(distances).argsort()[0]
        for i in range(0, 4):
            outpoly.append(poly[((pos + i) % 4) * 2])
            outpoly.append(poly[((pos + i) % 4) * 2 + 1])
        return outpoly


def cal_line_length(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


if __name__ == '__main__':
    Convert_xml_to_coco()
