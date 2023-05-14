from typing import Any, Dict, List, Tuple, Union
from imantics import Polygons, Mask
import numpy as np
import itertools
import torch
import copy
import cv2

class InstancesContainer:
    def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
        self._image_size = image_size
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self) -> Tuple[int, int]:
        return self._image_size

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        data_len = len(value)
        if len(self._fields):
            assert (
                    len(self) == data_len
            ), "Adding a field of length {} to a Instances of length {}".format(data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        return name in self._fields

    def sort(self, name: str, reverse=False, key=None):
        if key is None:
            key = lambda x: x._fields[name]
        return InstancesContainer.cat(sorted(self, key=key, reverse=reverse))

    def remove(self, name: str) -> None:
        del self._fields[name]

    def filter_by_class(self, class_id):
        key = lambda x: True if x.class_ids[0] == class_id else False
        class_instances = list(filter(key, self))
        if len(class_instances) > 0:
            class_instances = InstancesContainer.cat(class_instances)
        else:
            class_instances = InstancesContainer.build_empty(tuple(self.image_size))
        return class_instances

    def filter_by_confidence(self, thresh, method=">="):
        if method == ">=":
            key = lambda x: True if x.confidence_scores[0] >= thresh else False
        elif method == "<=":
            key = lambda x: True if x.confidence_scores[0] >= thresh else False
        elif method == "==":
            key = lambda x: True if x.confidence_scores[0] == thresh else False
        else:
            raise ValueError(f"invalid method {method} for filtering using confidence scores")

        confidence_instances = list(filter(key, self))
        if len(confidence_instances) > 0:
            confidence_instances = InstancesContainer.cat(confidence_instances)
        else:
            confidence_instances = InstancesContainer.build_empty(tuple(self.image_size))

        return confidence_instances

    def replace_class_ids(self, class_id_map):
        for key, value in class_id_map.items():
            self.class_ids[self.class_ids==key] = value
        return self

    def get(self, name: str) -> Any:
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        return self._fields

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "InstancesContainer":
        ret = InstancesContainer(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret

    def __getitem__(self, item: Union[int,list, slice]) -> "InstancesContainer":
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = InstancesContainer(self._image_size)
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            # use __len__ because len() has to be int and is not friendly to tracing
            return v.__len__()
        raise NotImplementedError("Empty Instances does not support __len__!")

    def export_to_mmdet(self, num_classes):
        _masks = []
        _bboxes = []
        for class_id in range(num_classes):
            class_data = self.filter_by_class(class_id=class_id)
            if len(class_data) > 0:
                conf_scores = np.expand_dims(class_data.confidence_scores, axis=-1)
                bboxes = np.append(class_data.bboxes, conf_scores, axis=-1)
                _bboxes.append(bboxes)
                _masks.append(list(class_data.masks))
            else:
                _bboxes.append(np.array([], dtype=np.float32).reshape(0, 5))
                _masks.append([])
        return _bboxes, _masks

    def export_to_json(self, approximation_coeficient=0.003):
        _data = {}
        if len(self) > 0:
            for instance in self:
                for key, value in instance._fields.items():
                    if key not in _data.keys():
                        _data[key] = _data.get(key, [])
                    if key == "confidence_scores":
                        _data[key].append(float(value[0]))
                    elif key == "class_ids":
                        _data[key].append(int(value[0]))
                    elif key == "bboxes":
                        _data[key].append(np.round(value[0]).astype(int).tolist())
                    elif key == "masks":
                        poly_points = np.squeeze(
                            InstancesContainer.mask_to_poly(
                                value[0],
                                approximation_coeficient=approximation_coeficient
                            )
                        )
                        _data[key].append(poly_points.tolist())
        else:
            _data["confidence_scores"] = []
            _data["class_ids"] = []
            _data["bboxes"] = []
            _data["masks"] = []

        _data["image_size"] = self.image_size.tolist()
        return _data

    def export_to_label_studio_json(self, class_mapping):
        formatted_json = []
        for idx, instance in enumerate(self):
            width = instance.image_size[0]
            height = instance.image_size[1]
            poly_approx = InstancesContainer.mask_to_poly(
                instance.masks[0],
                approximation_coeficient=0.009
            )
            poly_approx = np.squeeze(poly_approx)
            poly_approx[:, 0] = np.round((poly_approx[:, 0] / width) * 100)
            poly_approx[:, 1] = np.round((poly_approx[:, 1] / height) * 100)

            formatted_json.append(
                dict(
                    value=dict(
                        points=poly_approx.tolist(),
                        polygonlabels=[class_mapping[instance.class_ids.tolist()[0]]]
                    ),
                    original_width=int(width),
                    original_height=int(height),
                    image_rotation=0,
                    id=f"result{idx + 1}",
                    from_name="label",
                    to_name="image",
                    type="polygonlabels"
                )
            )
        return formatted_json

    def perform_nms(self, iou_thresh=0.1):
        conf_scores = np.expand_dims(self.confidence_scores, axis=-1)
        bboxes = np.append(self.bboxes, conf_scores, axis=-1)
        selected_ids = InstancesContainer.non_max_suppression(torch.tensor(bboxes), iou_thresh)
        return self[selected_ids]

    def annotate_image(self, image, class_mapping=None, approximation_coeficient=0.003):
        assert image.shape[:2] == tuple(self.image_size)[::-1]
        image_ = image.copy()
        for item in self:
            if class_mapping:
                class_name = class_mapping[item.class_ids[0]]
            else:
                class_name = None

            poly_points = InstancesContainer.mask_to_poly(
                item.masks[0],
                approximation_coeficient=approximation_coeficient
            )
            image_ = InstancesContainer.draw_instance(
                image_,
                poly_points,
                InstancesContainer.get_color_tuple(),
                bbox=item.bboxes[0],
                score=item.confidence_scores[0],
                class_name=class_name,
                transperency=0.8
            )
        return image_

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "image_width={}, ".format(self._image_size[0])
        s += "image_height={}, ".format(self._image_size[1])
        s += "fields=[{}])".format(", ".join((f"{k}: {v}" for k, v in self._fields.items())))
        return s

    @staticmethod
    def get_color_tuple():
        return tuple(np.random.choice(range(0, 256), size=3).astype(np.uint8))

    @staticmethod
    def mask_to_poly(mask, approximation_coeficient=0.009):
        poly_points = Mask(mask).polygons().points
        final_points=poly_points[0]
        final_points=poly_points[0]
        final_points_area=cv2.contourArea(final_points)
        for i in poly_points[1:]:
            if cv2.contourArea(i) > final_points_area:
                final_points=i
                final_points_area=cv2.contourArea(i)
        if approximation_coeficient:
            final_points = cv2.approxPolyDP(
                final_points, approximation_coeficient * cv2.arcLength(final_points, True), True)
        return final_points

    @staticmethod
    def poly_to_mask(polygon, image_size):
        segmentation_mask = np.zeros(image_size)
        segmentation_mask = cv2.drawContours(segmentation_mask, [np.array(polygon)], -1, 1, -1).astype(np.uint8)
        return segmentation_mask

    @staticmethod
    def get_bboxes(poly):
        poly = np.squeeze(poly)
        xmin = min(poly[:, 0].flatten().tolist())
        xmax = max(poly[:, 0].flatten().tolist())
        ymin = min(poly[:, 1].flatten().tolist())
        ymax = max(poly[:, 1].flatten().tolist())
        return [xmin, ymin, xmax, ymax]

    # TO-DO revamp to support individual instance
    @staticmethod
    def draw_instance(image, polygon, color, bbox=None, score=None, class_name=None, transperency=0.8, offset=10):
        if bbox is None:
            x1, y1, x2, y2 = InstancesContainer.get_bboxes(polygon)
            x1, y1, x2, y2 = x1 - offset, y1 - offset, x2 + offset, y2 + offset
        else:
            x1, y1, x2, y2 = bbox.astype(int)

        B, G, R = color

        blank = np.zeros(image.shape[:2])

        # create segmentation mask
        segmentation_mask = cv2.drawContours(blank.copy(), [polygon], -1, 1, -1).astype(np.uint8)

        # draw segmentation according to color
        car_part_color = np.stack(
            [segmentation_mask, segmentation_mask, segmentation_mask],
            axis=-1
        ).astype(np.uint8)

        car_part_color[:, :, 0] *= B
        car_part_color[:, :, 1] *= G
        car_part_color[:, :, 2] *= R

        image[segmentation_mask == 1] = cv2.addWeighted(
            image[segmentation_mask == 1],
            transperency,
            car_part_color[segmentation_mask == 1],
            1 - transperency,
            1.0
        )

        # draw border of contour and bbox
        border_mask = cv2.drawContours(blank.copy(), [polygon], -1, 1, 1, cv2.LINE_AA).astype(np.uint8)
        border_mask = cv2.rectangle(border_mask, (x1, y1), (x2, y2), 1, 1, cv2.LINE_AA)
        border_color = np.stack(
            [border_mask, border_mask, border_mask],
            axis=-1
        ).astype(np.uint8)

        border_color[:, :, 0] *= B
        border_color[:, :, 1] *= G
        border_color[:, :, 2] *= R

        image[border_mask == 1] = cv2.addWeighted(
            image[border_mask == 1], transperency - 0.4,
            border_color[border_mask == 1],
            1 - (transperency - 0.4), 1.0
        )

        # draw box around text and put text on
        if score:
            text = str(int(score * 100)) + "%"
            font_scale = 0.7
            if class_name:
                text = class_name + " " + text
                font_scale = 0.4

            x1_, y1_ =  (round(x1+font_scale), round(y1-(font_scale*26)))
            x2_, y2_ = (round(x1+ font_scale*len(text)*24) , round(y1+font_scale*2))
            image = cv2.rectangle(
                image, (x1_, y1_),
                (x2_, y2_), (0, 0, 0), -1
            )
            image = cv2.putText(
                image, text,
                (round(x1+font_scale*5), round(y1-font_scale*2)),
                cv2.FONT_HERSHEY_COMPLEX,
                font_scale, (255, 255, 255), 1, cv2.LINE_AA
            )

        return image

    @staticmethod
    def build_from_mmdet(result, image_size):
        _result = copy.deepcopy(result)
        _bboxes = []
        _conf_scores = []
        _class_ids = []
        _masks = []
        for class_id, (bboxes, masks) in enumerate(zip(*_result)):
            for bbox, mask in zip(bboxes, masks):
                _bboxes.append(bbox[:-1])
                _masks.append(mask)
                _conf_scores.append(bbox[-1])
                _class_ids.append(class_id)
        return InstancesContainer(
            image_size=np.array(list(image_size)),
            confidence_scores=np.array(_conf_scores),
            class_ids=np.array(_class_ids),
            bboxes=np.array(_bboxes),
            masks=np.array(_masks)
        )

    @staticmethod
    def build_from_detectron2(detectron2_inst):
        image_size = detectron2_inst.image_size
        _conf_scores = detectron2_inst.scores.numpy()
        _class_ids = detectron2_inst.pred_classes
        _bboxes = detectron2_inst.pred_boxes.tensor.numpy()
        _masks = detectron2_inst.pred_masks
        return InstancesContainer(
            image_size=np.array(list(image_size)[::-1]),
            confidence_scores=np.array(_conf_scores),
            class_ids=np.array(_class_ids),
            bboxes=np.array(_bboxes),
            masks=np.array(_masks)
        )
    @staticmethod
    def build_from_yolov7(image_size,masks,class_id,scores,xyxy_cord):
        image_size = image_size
        _conf_scores = scores
        _class_ids = class_id
        _bboxes = xyxy_cord
        _masks = masks
        return InstancesContainer(
            image_size=np.array(list(image_size)[::-1]),
            confidence_scores=np.array(_conf_scores),
            class_ids=np.array(_class_ids),
            bboxes=np.array(_bboxes),
            masks=np.array(_masks)
        )

    @staticmethod
    def build_from_json(json_data):
        _image_size = json_data["image_size"]
        cvt_mask = lambda mask: InstancesContainer.poly_to_mask(np.asarray(mask), _image_size[::-1])
        masks = list(map(cvt_mask, json_data["masks"]))
        return InstancesContainer(
            image_size=np.array(_image_size),
            confidence_scores=np.array(json_data["confidence_scores"]),
            class_ids=np.array(json_data["class_ids"]),
            bboxes=np.array(json_data["bboxes"]),
            masks=np.array(masks)
        )

    @staticmethod
    def build_empty(image_size):
        return InstancesContainer(
            image_size=np.array(list(image_size)),
            confidence_scores=np.array([]),
            class_ids=np.array([]),
            bboxes=np.array([], dtype=np.float32).reshape((0, 5)),
            masks=np.array([], dtype=np.float32)
        )

    @staticmethod
    def cat(instance_lists: List["InstancesContainer"]) -> "InstancesContainer":
        assert all(isinstance(i, InstancesContainer) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size
        if not isinstance(image_size, np.ndarray):  # could be a tensor in tracing
            for i in instance_lists[1:]:
                assert i.image_size == image_size
        ret = InstancesContainer(image_size)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, np.ndarray):
                values = np.concatenate(values, axis=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret

    @staticmethod
    def non_max_suppression(P: torch.tensor, thresh_iou: float):
        x1 = P[:, 0]
        y1 = P[:, 1]
        x2 = P[:, 2]
        y2 = P[:, 3]

        scores = P[:, 4]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()
        keep_idxes = []
        while len(order) > 0:
            idx = order[-1]
            keep_idxes.append(int(idx))
            order = order[:-1]
            if len(order) == 0:
                break
            xx1 = torch.index_select(x1, dim=0, index=order)
            xx2 = torch.index_select(x2, dim=0, index=order)
            yy1 = torch.index_select(y1, dim=0, index=order)
            yy2 = torch.index_select(y2, dim=0, index=order)

            xx1 = torch.max(xx1, x1[idx])
            yy1 = torch.max(yy1, y1[idx])
            xx2 = torch.min(xx2, x2[idx])
            yy2 = torch.min(yy2, y2[idx])

            w = xx2 - xx1
            h = yy2 - yy1

            w = torch.clamp(w, min=0.0)
            h = torch.clamp(h, min=0.0)

            inter = w * h
            rem_areas = torch.index_select(areas, dim=0, index=order)
            union = (rem_areas - inter) + areas[idx]

            IoU = inter / union
            mask = IoU < thresh_iou
            order = order[mask]

        return keep_idxes

    __repr__ = __str__