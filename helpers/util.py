
from typing import Any, Dict
import serve_settings as settings
import numpy as np
import boto3
import os
import torch 
import torch.nn as nn
import time
import torchvision
import cv2
import torch.nn as nn
import torch
import torch.nn.functional as F
import math

# DEV clients
s3_dev_client = boto3.client('s3',
                             aws_access_key_id=settings.DEV_ACCESS_KEY,
                             aws_secret_access_key=settings.DEV_SECRET_ACCESS_KEY)

# Staging clients
s3_staging_client = boto3.client('s3',
                                 aws_access_key_id=settings.STAGING_ACCESS_KEY,
                                 aws_secret_access_key=settings.STAGING_SECRET_ACCESS_KEY)

# production clients
s3_production_client = boto3.client('s3',
                                    aws_access_key_id=settings.PRODUCTION_ACCESS_KEY,
                                    aws_secret_access_key=settings.PRODUCTION_SECRET_ACCESS_KEY)

# clients and buckets switch case statements
select_S3_client = {
    "development": s3_dev_client,
    "staging": s3_staging_client,
    "production": s3_production_client
}

select_S3_bucket = {
    "development": settings.DEV_BUCKET,
    "staging": settings.STAGING_BUCKET,
    "production": settings.PRODUCTION_BUCKET
}


class RequestProcessor:
    def __init__(self, **kwargs):
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    def set(self, name, value: Any):
        self._fields[name] = value

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError(
                "Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def download_image(self, load_from="masked"):
        object_path = f"{self.assessment_folder}/{load_from}/{self.file_name}"
        local_path = os.path.join(
            settings.temp_folder_path, os.path.basename(object_path))
        select_S3_client[self.environment].download_file(
            select_S3_bucket[self.environment],
            object_path,
            local_path
        )
        self.set("image_path", local_path)

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "fields=[{}])".format(", ".join((f"{k}: {v}" for k,
                                   v in self._fields.items())))
        return s

    @staticmethod
    def process_request(json_data):
        data = {}
        for mandatory_field in settings.mandatory_fields:
            if mandatory_field in json_data.keys():
                data[mandatory_field] = json_data[mandatory_field]
            else:
                raise KeyError(f"{mandatory_field}")

        for optional_field, default_value in settings.optional_fields.items():
            if optional_field in json_data.keys():
                data[optional_field] = json_data[optional_field]
            else:
                data[optional_field] = default_value

        # request_body = RequestProcessor(**data)
        # request_body.download_image(load_from=settings.load_image_from)

        return data

    __repr__ = __str__


def dump_annotated_image(local_path, s3_path, env):
    select_S3_client[env].upload_file(
        local_path, select_S3_bucket[env], s3_path)
    os.remove(local_path)
    return "success"

def helper(model_pt_path,device,dnn=False, data=None, fp16=False):
    f=os.listdir(os.getcwd())
    if "models" not in f:
        import zipfile
        with zipfile.ZipFile("models_zip.zip","r") as zip_ref:
            zip_ref.extractall("models")
        with zipfile.ZipFile("utils_zip.zip","r") as zip_ref:
            zip_ref.extractall("utils")
    from models.common import DetectMultiBackend
    return DetectMultiBackend(model_pt_path, device=device, dnn=False, data=None, fp16=False)


# def xywh2xyxy(x):
#     # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
#     y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
#     y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
#     y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
#     y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
#     y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
#     return y

# def box_area(box):
#     # box = xyxy(4,n)
#     return (box[2] - box[0]) * (box[3] - box[1])


# def box_iou(box1, box2, eps=1e-7):
#     # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
#     """
#     Return intersection-over-union (Jaccard index) of boxes.
#     Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
#     Arguments:
#         box1 (Tensor[N, 4])
#         box2 (Tensor[M, 4])
#     Returns:
#         iou (Tensor[N, M]): the NxM matrix containing the pairwise
#             IoU values for every element in boxes1 and boxes2
#     """

#     # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
#     (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
#     inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

#     # IoU = inter / (area1 + area2 - inter)
#     return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter + eps)


# def non_max_suppression(
#     prediction,
#     conf_thres=0.25,
#     iou_thres=0.45,
#     classes=None,
#     agnostic=False,
#     multi_label=False,
#     labels=(),
#     max_det=300,
#     nm=0,  # number of masks
# ):
#     """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

#     Returns:
#         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
#     """

#     bs = prediction.shape[0]  # batch size
#     nc = prediction.shape[2] - nm - 5  # number of classes
#     xc = prediction[..., 4] > conf_thres  # candidates

#     # Checks
#     assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
#     assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

#     # Settings
#     # min_wh = 2  # (pixels) minimum box width and height
#     max_wh = 7680  # (pixels) maximum box width and height
#     max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
#     time_limit = 0.5 + 0.05 * bs  # seconds to quit after
#     redundant = True  # require redundant detections
#     multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
#     merge = False  # use merge-NMS

#     t = time.time()
#     mi = 5 + nc  # mask start index
#     output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
#     for xi, x in enumerate(prediction):  # image index, image inference
#         # Apply constraints
#         # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
#         x = x[xc[xi]]  # confidence

#         # Cat apriori labels if autolabelling
#         if labels and len(labels[xi]):
#             lb = labels[xi]
#             v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
#             v[:, :4] = lb[:, 1:5]  # box
#             v[:, 4] = 1.0  # conf
#             v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
#             x = torch.cat((x, v), 0)

#         # If none remain process next image
#         if not x.shape[0]:
#             continue

#         # Compute conf
#         x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

#         # Box/Mask
#         # center_x, center_y, width, height) to (x1, y1, x2, y2)
#         box = xywh2xyxy(x[:, :4])
#         mask = x[:, mi:]  # zero columns if no masks

#         # Detections matrix nx6 (xyxy, conf, cls)
#         if multi_label:
#             i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
#             x = torch.cat((box[i], x[i, 5 + j, None],
#                             j[:, None].float(), mask[i]), 1)
#         else:  # best class only
#             conf, j = x[:, 5:mi].max(1, keepdim=True)
#             x = torch.cat((box, conf, j.float(), mask), 1)[
#                 conf.view(-1) > conf_thres]

#         # Filter by class
#         if classes is not None:
#             x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

#         # Apply finite constraint
#         # if not torch.isfinite(x).all():
#         #     x = x[torch.isfinite(x).all(1)]

#         # Check shape
#         n = x.shape[0]  # number of boxes
#         if not n:  # no boxes
#             continue
#         elif n > max_nms:  # excess boxes
#             # sort by confidence
#             x = x[x[:, 4].argsort(descending=True)[:max_nms]]
#         else:
#             x = x[x[:, 4].argsort(descending=True)]  # sort by confidence

#         # Batched NMS
#         c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
#         # boxes (offset by class), scores
#         boxes, scores = x[:, :4] + c, x[:, 4]
#         i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
#         if i.shape[0] > max_det:  # limit detections
#             i = i[:max_det]
#         if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
#             # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
#             iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
#             weights = iou * scores[None]  # box weights
#             x[i, :4] = torch.mm(weights, x[:, :4]).float(
#             ) / weights.sum(1, keepdim=True)  # merged boxes
#             if redundant:
#                 i = i[iou.sum(1) > 1]  # require redundancy

#         output[xi] = x[i]
#         if (time.time() - t) > time_limit:
#             break  # time limit exceeded

#     return output

# def crop(masks, boxes):
#     """
#     "Crop" predicted masks by zeroing out everything not in the predicted bbox.
#     Vectorized by Chong (thanks Chong).

#     Args:
#         - masks should be a size [h, w, n] tensor of masks
#         - boxes should be a size [n, 4] tensor of bbox coords in relative point form
#     """

#     n, h, w = masks.shape
#     x1, y1, x2, y2 = torch.chunk(
#         boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
#     r = torch.arange(w, device=masks.device, dtype=x1.dtype)[
#         None, None, :]  # rows shape(1,w,1)
#     c = torch.arange(h, device=masks.device, dtype=x1.dtype)[
#         None, :, None]  # cols shape(h,1,1)

#     return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))



# def process_mask(protos, masks_in, bboxes, shape, upsample=False):
#     """
#     Crop before upsample.
#     proto_out: [mask_dim, mask_h, mask_w]
#     out_masks: [n, mask_dim], n is number of masks after nms
#     bboxes: [n, 4], n is number of masks after nms
#     shape:input_image_size, (h, w)

#     return: h, w, n
#     """

#     c, mh, mw = protos.shape  # CHW
#     ih, iw = shape
#     masks = (masks_in @ protos.float().view(c, -1)
#                 ).sigmoid().view(-1, mh, mw)  # CHW

#     downsampled_bboxes = bboxes.clone()
#     downsampled_bboxes[:, 0] *= mw / iw
#     downsampled_bboxes[:, 2] *= mw / iw
#     downsampled_bboxes[:, 3] *= mh / ih
#     downsampled_bboxes[:, 1] *= mh / ih

#     masks = crop(masks, downsampled_bboxes)  # CHW
#     try:
#         if upsample:
#             masks = F.interpolate(
#                 masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
#     except:
#         pass
#     return masks.gt_(0.5)

# def make_divisible(x, divisor):
#     # Returns nearest x divisible by divisor
#     if isinstance(divisor, torch.Tensor):
#         divisor = int(divisor.max())  # to int
#     return math.ceil(x / divisor) * divisor

# def check_img_size(imgsz, s=32, floor=0):
#     # Verify image size is a multiple of stride s in each dimension
#     if isinstance(imgsz, int):  # integer i.e. img_size=640
#         new_size = max(make_divisible(imgsz, int(s)), floor)
#     else:  # list i.e. img_size=[640, 480]
#         imgsz = list(imgsz)  # convert to list if tuple
#         new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    
#     return new_size


# def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
#     # Resize and pad image while meeting stride-multiple constraints
#     shape = im.shape[:2]  # current shape [height, width]
#     if isinstance(new_shape, int):
#         new_shape = (new_shape, new_shape)

#     # Scale ratio (new / old)
#     r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
#     # only scale down, do not scale up (for better val mAP)
#     if not scaleup:
#         r = min(r, 1.0)

#     # Compute padding
#     ratio = r, r  # width, height ratios
#     new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
#     dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
#         new_unpad[1]  # wh padding
#     if auto:  # minimum rectangle
#         dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
#     dw /= 2  # divide padding into 2 sides
#     dh /= 2

#     if shape[::-1] != new_unpad:  # resize
#         im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
#     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
#     im = cv2.copyMakeBorder(
#         im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
#     return im, ratio, (dw, dh)

# def clip_coords(boxes, shape):
#     # Clip bounding xyxy bounding boxes to image shape (height, width)
#     if isinstance(boxes, torch.Tensor):  # faster individually
#         boxes[:, 0].clamp_(0, shape[1])  # x1
#         boxes[:, 1].clamp_(0, shape[0])  # y1
#         boxes[:, 2].clamp_(0, shape[1])  # x2
#         boxes[:, 3].clamp_(0, shape[0])  # y2
#     else:  # np.array (faster grouped)
#         boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
#         boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])
        
# def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
#     # Rescale coords (xyxy) from img1_shape to img0_shape
#     if ratio_pad is None:  # calculate from img0_shape
#         # gain  = old / new
#         gain = min(img1_shape[0] / img0_shape[0],
#                     img1_shape[1] / img0_shape[1])
#         pad = (img1_shape[1] - img0_shape[1] * gain) / \
#             2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
#     else:
#         gain = ratio_pad[0][0]
#         pad = ratio_pad[1]

#     coords[:, [0, 2]] -= pad[0]  # x padding
#     coords[:, [1, 3]] -= pad[1]  # y padding
#     coords[:, :4] /= gain
#     clip_coords(coords, img0_shape)
#     return coords


# #pt, torch hub load
# class DetectMultiBackend(nn.Module):
#     # YOLOv5 MultiBackend class for python inference on various backends
#     def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False, fuse=True):
#         super().__init__()
#         pt=True
#         w = str(weights[0] if isinstance(weights, list) else weights)
#         fp16 &= True or False or False or False  # FP16
#         stride = 32  # default stride
#         if pt:  # PyTorch
#             # model = torch.hub.load('yashjain-99/yolov7_seg_', 'custom', w, device=device, trust_repo=True, force_reload=True)
#             model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
#             stride = max(int(model.stride), 32)  # model stride
#             names = model.module.names if hasattr(model, 'module') else model.names  # get class names
#             model.half() if fp16 else model.float()
#             self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
#             segmentation_model = True

#         # class names
#         if 'names' not in locals():
#             names = {i: f'class{i}' for i in range(999)}
#         self.__dict__.update(locals())  # assign all variables to self

#     def forward(self, im, augment=False, visualize=False, val=False):
#         # YOLOv5 MultiBackend inference
#         b, ch, h, w = im.shape  # batch, channel, height, width
#         if self.fp16 and im.dtype != torch.float16:
#             im = im.half()  # to FP16

#         if self.pt:  # PyTorch
#             y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)
#             if isinstance(y, tuple) and not self.segmentation_model:
#                 y = y[0]
#         if isinstance(y, np.ndarray):
#             y = torch.tensor(y, device='cpu')
#         return (y, []) if val else y