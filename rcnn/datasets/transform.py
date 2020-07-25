import random
from torchvision.transforms import functional as F
from rcnn.core.config import cfg


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class QuadRandomRotate(object):
    def __init__(self, prob=0.5, theta=(0, 90, 180, 270)):
        self.prob = prob
        self.theta = theta

    def __call__(self, image, target):
        if random.random() < self.prob:
            theta = random.choice(self.theta)
            image = F.affine(image, -theta, (0, 0), 1, 0)
            target = target.rotate(theta)
        return image, target


class QuadChangeOrder(object):
    def __init__(self, categories):
        self.change_categories = categories
        
    def __call__(self, image, target):
        target = target.change_order(self.change_categories)
        return image, target


class QuadResize(object):
    def __init__(self, min_size, max_size, force_test_scale=[-1, -1]):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.force_test_scale = force_test_scale

    def __call__(self, image, target):
        if -1 not in self.force_test_scale:
            size = tuple(self.force_test_scale)
        else:
            size = random.choice(self.min_size)
        image = F.resize(image, (size, size))
        target = target.resize(image.size)
        return image, target


class QuadRandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target


class QuadRandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.vflip(image)
            target = target.transpose(1)
        return image, target


def build_transforms(is_train=True):
    if is_train:
        min_size = cfg.TRAIN.SCALES
        max_size = cfg.TRAIN.MAX_SIZE
        flip_prob = 0.5

        # for force resize
        force_test_scale = [-1, -1]
        change_categories = cfg.TRAIN.CHANGE_CATEGORIES

        to_bgr255 = cfg.TO_BGR255
        normalize_transform = Normalize(
            mean=cfg.PIXEL_MEANS, std=cfg.PIXEL_STDS, to_bgr255=to_bgr255
        )

        transform = Compose(
            [
                QuadResize(min_size, max_size, force_test_scale),
                QuadRandomRotate(flip_prob),
                QuadChangeOrder(change_categories),
                ToTensor(),
                normalize_transform,
            ]
        )
    else:
        transform = Compose(
            []
        )
    return transform

