import cv2
from collections import defaultdict

from utils.timer import Timer

_GRAY = [218, 227, 218]
_RED = [0, 0, 255]
_GREEN = [18, 127, 15]
_BULE = [255, 144, 30]
_WHITE = [255, 255, 255]
_BLACK = [0, 0, 0]
colors = [_RED, _GREEN, _BULE, _WHITE]


def get_class_string(class_index, score, dataset):
    class_text = dataset.classes[class_index] if dataset is not None else 'id{:d}'.format(class_index)
    return class_text + ' {:0.2f}'.format(score).lstrip('0')


def vis_quad(img, cfg_vis, quad, color=None):
    border_thick = cfg_vis.SHOW_QUAD_BOX.BORDER_THICK
    for j in range(4):
        str_point = (quad[j * 2], quad[j * 2 + 1])
        end_point = (quad[((j + 1) * 2) % len(quad)], quad[(((j + 1) * 2 + 1) % len(quad))])
        if color is not None:
            cv2.line(img, str_point, end_point, color, thickness=border_thick)
        else:
            cv2.line(img, str_point, end_point, _BULE, thickness=border_thick)
    cv2.circle(img, (quad[0], quad[1]), cfg_vis.SHOW_QUAD_BOX.CENTER_RADIUS, (0, 0, 255), -1)
    return img


def vis_point(img, cfg_vis, point, color):
    cv2.circle(img, (point[0], point[1]), cfg_vis.SHOW_QUAD_BOX.CENTER_RADIUS, color, -1)
    return img


def vis_class(img, cfg_vis, pos, class_str, bg_color):
    """Visualizes the class."""
    font_color = cfg_vis.SHOW_CLASS.COLOR
    font_scale = cfg_vis.SHOW_CLASS.FONT_SCALE

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


def vis_one_image_opencv(im, cfg_vis, boxes=None, classes=None, dataset=None):
    """Constructs a numpy array with the detections visualized."""
    timers = defaultdict(Timer)
    timers['bbox_prproc'].tic()

    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, -1]) < cfg_vis.VIS_TH:
        return im
    timers['bbox_prproc'].toc()

    for i in range(boxes.shape[0]):
        quad = boxes[i, :-1]
        score = boxes[i, -1]
        if score < cfg_vis.VIS_TH:
            continue

        if cfg_vis.SHOW_QUAD_BOX.ENABLED:
            timers['show_quad_box'].tic()
            if len(quad) == 8:
                im = vis_quad(im, cfg_vis, quad)
            elif len(quad) == 10:
                im = vis_quad(im, cfg_vis, quad[:8])
                center = quad[8:10]
                im = vis_point(im, cfg_vis, center, _GRAY)
            timers['show_quad_box'].toc()

        # show class (on by default)
        if cfg_vis.SHOW_CLASS.ENABLED:
            timers['show_class'].tic()
            class_str = get_class_string(classes[i], score, dataset)
            im = vis_class(im, cfg_vis, (quad[0], quad[1] - 2), class_str, _BLACK)
            timers['show_class'].toc()
    return im
