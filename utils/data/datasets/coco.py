import os
import cv2

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torchvision

from utils.data.structures.bounding_box import BoxList
from utils.data.structures.quad_bbox import QuadBoxes


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno, ann_types, filter_crowd=True):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    if filter_crowd:
        # if image only has crowd annotation, it should be filtered
        if 'iscrowd' in anno[0]:
            anno = [obj for obj in anno if obj["iscrowd"] == 0]
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    return True


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
            self, ann_file, root, remove_images_without_annotations, ann_types, transforms=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        self.ann_file = ann_file
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno, ann_types):
                    ids.append(img_id)
            self.ids = ids
        
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        category_ids = sorted(self.coco.getCatIds())
        self.json_category_id_to_contiguous_id = {v: i + 1 for i, v in enumerate(category_ids)}
        self.contiguous_category_id_to_json_id = {v: k for k, v in self.json_category_id_to_contiguous_id.items()}
        categories = [c['name'] for c in self.coco.loadCats(category_ids)]
        self.classes = ['__background__'] + categories
        self.ann_types = ann_types
        self._transforms = transforms

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        if len(anno) > 0:
            if 'iscrowd' in anno[0]:
                anno = [obj for obj in anno if obj["iscrowd"] == 0]

        if 'quad' in self.ann_types:
            quad = [obj["segmentation"][0] for obj in anno]
            quad = torch.as_tensor(quad)
            target = QuadBoxes(quad, img.size, mode="xyxy")
        else:
            boxes = [obj["bbox"] for obj in anno]
            boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
            target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        # target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

    def pull_image(self, index):
        """Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            img
        """
        img_id = self.id_to_img_map[index]

        path = self.coco.loadImgs(img_id)[0]['file_name']

        return cv2.imread(os.path.join(self.root, path), cv2.IMREAD_COLOR)
