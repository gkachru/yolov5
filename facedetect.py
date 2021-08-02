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

DEBUG = True

if DEBUG:
    from utils.plots import colors, plot_one_box


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


def face_and_objects_detect(device, face_model, objects_model, img0, imgsz=640, augment=False):
    stride = int(face_model.stride.max())  # model stride
    if imgsz <= 0:                    # original size
        imgsz = dynamic_resize(img0.shape)
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    img = letterbox(img0, imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    boxes = []

    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
    h, w, c = img0.shape

    if DEBUG:
        # Get class names
        names = objects_model.module.names if hasattr(objects_model, 'module') else objects_model.names

    # Objects Inference

    pred = objects_model(img, augment=augment)[0]
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

            boxes.append([x1, y1, x2 - x1, y2 - y1, conf, name])

            if DEBUG:
                plot_one_box(xyxy, img0, label=f'{name} {conf:.2f}', color=colors(cls, True))

    # Face Inference

    pred = face_model(img, augment=augment)[0]
    # Apply NMS
    det = non_max_suppression_face(pred, conf_thres=0.8, iou_thres=0.5)[0]

    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh

    if det is not None:
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        for j, (*xyxy, conf, cls) in enumerate(det):
            xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1)
            xywh = xywh.data.cpu().numpy()

            # class_num = det[j, 15].cpu().numpy()
            x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
            y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
            x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
            y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
            conf = round(float(det[j, 4].cpu().numpy()), 3)

            boxes.append([x1, y1, x2 - x1, y2 - y1, conf, "face"])

            if DEBUG:
                plot_one_box(xyxy, img0, label=f'face {conf:.2f}', color=colors(1, True))

    return boxes



@torch.no_grad()
def load_models_and_run(
        img_file,
        device='0',
        face_weights='weights/yolov5s-face.pt',
        objects_weights='weights/yolov5m.pt'
    ):
    img = cv2.imread(img_file)

    inf_device = select_device(device)
    face_model = attempt_load(face_weights, map_location=inf_device)
    objects_model = attempt_load(objects_weights, map_location=inf_device)

    if DEBUG:
        start = time.time()

    boxes = face_and_objects_detect(inf_device, face_model, objects_model, img)

    if DEBUG:
        print('Completed in {:.1f}ms'.format((time.time() - start) * 1000))
        cv2.imwrite('test.jpg', img)

    return boxes



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5s-face.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.8, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--image', type=str, help='image file to process')
    opt = parser.parse_args()
    print(opt)

    # Load model
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model

    print('Model Loaded')
    start = time.time()
    with torch.no_grad():
        img0 = cv2.imread(opt.image)  # BGR
        boxes = detect(model, img0)
        cv2.imwrite('test.jpg', img0)
        for box in boxes:
            print('%d %d %d %d (%.03f) => %s' % (box[0], box[1], box[2], box[3], box[4] if box[4] <= 1 else 1, box[5]) + '\n')

    print('Completed in {:.3f}'.format(time.time() - start))
