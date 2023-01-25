import time
import argparse

import cv2
import torch
import torchvision
import numpy as np
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import (
    check_img_size, scale_coords,
    xyxy2xywh, xywh2xyxy,
    non_max_suppression
)
from utils.torch_utils import select_device

DEBUG = False

if DEBUG:
    from utils.plots import colors, Annotator


def non_max_suppression_face(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=()):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 15  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    time_limit = 1.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 16), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 15), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 15] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 15:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, landmarks, cls)
        if multi_label:
            i, j = (x[:, 15:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 15, None], x[:, 5:15] ,j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 15:].max(1, keepdim=True)
            x = torch.cat((box, conf, x[:, 5:15], j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Batched NMS
        c = x[:, 15:16] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        # if i.shape[0] > max_det:  # limit detections
        #     i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def dynamic_resize(shape, stride=64):
    max_size = max(shape[0], shape[1])
    if max_size % stride != 0:
        max_size = (int(max_size / stride) + 1) * stride

    return max_size


def bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2] + boxA[0], boxB[2] + boxB[0])
    yB = min(boxA[3] + boxA[1], boxB[3] + boxB[1])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] + 1) * (boxA[3] + 1)
    boxBArea = (boxB[2] + 1) * (boxB[3] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# Only add after checking if the box does not overlap with another 
# and that it has a higer probability if it does
def add_new_box(box1, boxes, check=True):
    if check:
        dont_add_box = False
        to_delete = []

        for iter2, box2 in enumerate(boxes):
            # Over 90% overlap?
            if bbox_iou(box1, box2) > 0.9:
                # Which one has a higher probability?
                if box1[4] > box2[4]:
                    to_delete.append(iter2)
                else:
                    dont_add_box = True
                    break

        if dont_add_box:
            return False

        # Delete last indices first
        for item in sorted(to_delete, reverse=True):
            boxes.pop(item)

    boxes.append(box1)
    return True


class ObjectModel:
    def __init__(self, device_id, face_weights, objects_weights):
        self.device = select_device(device_id)
        self.face = self.attempt_load(face_weights)
        self.objects = []
        for objw in list(objects_weights):
            self.objects.append(self.attempt_load(objw))

    def attempt_load(self, weights):
        return self.upsample_hack(attempt_load(weights, map_location=self.device))

    def upsample_hack(self, model):
        if model is not None:
            for m in model.modules():
                if isinstance(m, torch.nn.Upsample):
                    m.recompute_scale_factor = None
        return model

def face_and_objects_detect(model, img0, imgsz=640, augment=False):
    stride = int(model.face.stride.max())  # model stride
    if imgsz <= 0:                    # original size
        imgsz = dynamic_resize(img0.shape)
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    img = letterbox(img0, imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(model.device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    boxes = []

    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]].to(model.device)  # normalization gain whwh
    h, w, c = img0.shape

    # Objects Inference

    for it, obj_model in enumerate(model.objects):
        # Get class names
        names = obj_model.module.names if hasattr(obj_model, 'module') else obj_model.names

        pred = obj_model(img, augment=augment)[0]
        # Apply NMS
        det = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.45, max_det=16)[0]

        if det is not None:
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for j, (*xyxy, conf, cls) in enumerate(det):
                xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1)
                xywh = xywh.data.cpu().numpy()

                x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
                y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
                x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
                y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
                conf = round(float(conf.cpu().numpy()), 3)

                # Get class name
                icls = int(cls)  # integer class
                name = names[icls]

                add_new_box([x1, y1, x2 - x1, y2 - y1, conf, name], boxes, check=(it > 0))

                if DEBUG:
                    Annotator(img0, pil=False).box_label(xyxy, label=f'{name} {conf:.2f}', color=colors(cls, True))

    # Face Inference

    pred = model.face(img, augment=augment)[0]
    # Apply NMS
    det = non_max_suppression_face(pred, conf_thres=0.8, iou_thres=0.5)[0]

    # gn = torch.tensor(img0.shape)[[1, 0, 1, 0]].to(model.device)  # normalization gain whwh

    if det is not None:
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        for j, (*xyxy, conf, cls) in enumerate(det):
            xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1)
            xywh = xywh.data.cpu().numpy()

            x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
            y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
            x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
            y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
            conf = round(float(det[j, 4].cpu().numpy()), 3)

            boxes.append([x1, y1, x2 - x1, y2 - y1, conf, "face"])

            if DEBUG:
                Annotator(img0, pil=False).box_label(xyxy, label=f'face {conf:.2f}', color=colors(1, True))

    return boxes


@torch.no_grad()
def infer_image(
        img_file,
        device='0',
        face_weights='weights/yolov5s-face.pt',
        objects_weights=[
            'weights/yolov5m.pt',
            'weights/yolov5s-pp.pt',
        ]
):

    model = ObjectModel(device, face_weights, objects_weights)

    if DEBUG:
        start = time.time()

    img = cv2.imread(img_file)
    boxes = face_and_objects_detect(model, img)

    if DEBUG:
        print('Completed in {:.1f}ms'.format((time.time() - start) * 1000))
        cv2.imwrite('test.jpg', img)

    return boxes
