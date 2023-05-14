from ts.torch_handler.base_handler import BaseHandler
from pathlib import Path
import os
import sys
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from util import RequestProcessor
import cv2
import numpy as np
from structure import InstancesContainer
import serve_settings as settings
import time
import botocore
import zipfile

try:
    from utils.general import check_img_size, non_max_suppression, scale_coords
    from models.common import DetectMultiBackend
    from utils.segment.general import process_mask
    from utils.augmentations import letterbox
except Exception as e:
    pass


class car_parts_handler(BaseHandler):
    def __init__(self):
        self.model = None
        self.initialized = False
        self.map_location = None
        self.manifest = None
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.max_det = 1000
        self.device = None
        self.imgsz = None
        self.source = None
        self.preprocessed = None

        try:
            self.check_img_size = check_img_size
            self.non_max_suppression = non_max_suppression
            self.scale_coords = scale_coords
            self.DetectMultiBackend = DetectMultiBackend
            self.process_mask = process_mask
            self.letterbox = letterbox
        except Exception as e:
            self.check_img_size = None
            self.non_max_suppression = None
            self.scale_coords = None
            self.DetectMultiBackend = None
            self.process_mask = None
            self.letterbox = None

    def initialize(self, context):
        # torch serve part
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.map_location = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else self.map_location
        )
        self.manifest = context.manifest

        model_dir = properties.get("model_dir")

        # extract zip files
        zip_files = {
            "models": os.path.join(model_dir, "models_zip.zip"),
            "utils": os.path.join(model_dir, "utils_zip.zip")
        }
        for folder_name, zip_file in zip_files.items():
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(folder_name)

        model_pt_path = None
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            model_pt_path = os.path.join(model_dir, serialized_file)

        # model def file
        model_file = self.manifest["model"].get("modelFile", "")

        self.model = self._get_models(
            model_dir,
            model_file,
            model_pt_path
        )
        self.initialized = True

    def _get_models(self, model_dir, model_file, model_pt_path):
        model_def_path = os.path.join(model_dir, model_file)
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model.py file")

        # import packages
        from utils.general import check_img_size, non_max_suppression, scale_coords
        from models.common import DetectMultiBackend
        from utils.segment.general import process_mask
        from utils.augmentations import letterbox

        self.check_img_size = check_img_size
        self.non_max_suppression = non_max_suppression
        self.scale_coords = scale_coords
        self.DetectMultiBackend = DetectMultiBackend
        self.process_mask = process_mask
        self.letterbox = letterbox

        carparts = self.DetectMultiBackend(model_pt_path, self.device)
        return carparts

    def preprocess(self, request_data):

        source = request_data['file_path']
        stride, names, pt = self.model.stride, self.model.names, True
        imgsz = self.check_img_size(1088, s=stride)
        im0 = cv2.imread(source)
        self.source = im0
        im = self.letterbox(im0, imgsz, stride=stride, auto=pt)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        self.preprocessed = im
        self.imgsz = imgsz
        return im

    def inference(self, im):
        model = self.model
        with torch.inference_mode():
            pred, out = model(im, augment=False, visualize=False)
            proto = out[1]
            pred = pred.to('cpu')
            proto = proto.to('cpu')
            pred = self.non_max_suppression(
                pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det, nm=32)
        return pred, proto

    def postprocess(self, pred, proto):
        image_size = self.source.shape[:2]
        im = self.preprocessed
        im0 = self.source
        if pred[:,6:].shape[0] != 0:
            d = {}
            # scaling coordinates
            with torch.inference_mode():
                pred[:, :4] = self.scale_coords(
                    im.shape[2:], pred[:, :4], im0.shape).round()

            # processing and scaling  masks
            scaled_masks = self.process_mask(
                proto[0], pred[:, 6:], pred[:, :4], im0.shape[:2], upsample=True)
            scaled_masks = scaled_masks.numpy()
            scaled_masks = scaled_masks.astype('uint8')

            # removing invalid mask and adding masks to dictionary wrt to classes
            for i, mask in enumerate(scaled_masks):
                if len(np.unique(mask)) < 2:
                    continue
                pred_cls = int(pred[i][5])
                if pred_cls in d:
                    d[pred_cls]["masks"].append(mask)
                else:
                    d[pred_cls] = {"masks": [mask], "bbox": [], "scores": []}

            for *xyxy, conf, pred_cls in reversed(pred[:, :6]):
                pred_cls = int(pred_cls)
                if pred_cls in d:
                    d[pred_cls]["scores"].insert(0, conf)
                    d[pred_cls]["bbox"].insert(0, xyxy)

            class_id = []
            scores = []
            xyxy_cord = []
            final_mask = []
            for i in d:
                for j in range(len(d[i]["bbox"])):
                    class_id.append(i)
                    xyxy_cord.append(d[i]["bbox"][j])
                    final_mask.append(d[i]["masks"][j])
                    scores.append(d[i]["scores"][j])
        else:
            class_id=[]
            scores=[]
            xyxy_cord=[]
            final_mask=[]
        results = InstancesContainer.build_from_yolov7(
            image_size, final_mask, np.array(class_id), np.array(scores), np.array(xyxy_cord))
        return results

    def handle(self, data, context):
        start_time = time.time()

        self.context = context
        metrics = self.context.metrics

        try:
            # load the request and get the data in inbuilt data structure
            json_data = data[0]["body"] or data[0]["data"]
            request_data = RequestProcessor.process_request(json_data)

            # do the preprocessing and do the inference
            org_image = self.preprocess(request_data)
            pred, proto = self.inference(org_image)
            # do the postprocessing
            processed_data = self.postprocess(pred[0], proto)

            # export the results into json format
            output_json = processed_data.export_to_json(
                approximation_coeficient=settings.approx_coef)

            response = dict(
                status="success",
                prediction_data=output_json,
                classes=list(settings.class_mapping.values())
            )

        except KeyError as errormessage:
            response = {
                "status": "fail",
                "error_type": "Incomplete_Request",
                "trace": f"Expected {errormessage} into request body"
            }

        except botocore.exceptions.ClientError as e:
            response = {
                "status": "fail",
                "error_type": "Incorrect_path",
                "trace": str(e)
            }

        except Exception as e:
            response = {
                "status": "fail",
                "error_type": "Misc_Exception",
                "trace": str(e) + " please contact admin"
            }
            return [response]

        stop_time = time.time()
        metrics.add_time('HandlerTime', round(
            (stop_time - start_time) * 1000, 2), None, 'ms')

        return [response]
