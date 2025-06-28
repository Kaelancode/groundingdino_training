from typing import Tuple, List

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from torchvision.ops import box_convert
from torchvision.ops import box_iou, generalized_box_iou ,sigmoid_focal_loss
import torch.nn.functional as F
import bisect

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.lora import add_lora_to_model

# ----------------------------------------------------------------------------------------------------------------------
# OLD API
# ----------------------------------------------------------------------------------------------------------------------


def focal_loss(logits, targets, alpha=0.25, gamma=2, eps=1e-7):
    logits = logits.clamp(min=-50, max=50)  # Clamp logits
    return sigmoid_focal_loss(logits,targets,reduction="mean")


def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."
    


# def load_model(model_config_path: str, model_checkpoint_path: str, model_lora_path: str, device: str = "cuda", use_lora: bool =False):
#     args = SLConfig.fromfile(model_config_path)
#     args.device = device
#     model = build_model(args)
#     checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
#     model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
#     if use_lora:
#         print(f"Adding Lora to model!")
#         model=add_lora_to_model(model)
#         print(f"Lora model is {model}")
#         # if "lora" in model_checkpoint_path:
#         #     lora_ckpt = torch.load(model_lora_path, map_location="cpu")
#         #     lora_ckpt_state_dict = clean_state_dict(lora_ckpt["model"]) if "model" in lora_ckpt else clean_state_dict(lora_ckpt)
#         #     new_lora_ckpt_state_dict = {}
#         #     for key, value in lora_ckpt_state_dict.items():
#         #         new_key = key.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight")
#         #         new_lora_ckpt_state_dict[new_key] = value
#         #     model.load_state_dict(new_lora_ckpt_state_dict, strict=False)
#         #     print("Lora weights loaded")
#     return model

def load_model(model_config_path: str, model_checkpoint_path: str, model_lora_path: str = None, device: str = "cuda", use_lora: bool = False):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)

    # Load base model weights
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint.get("model", checkpoint)), strict=False)

    if use_lora:
        print("Adding LoRA to model")
        model = add_lora_to_model(model)
        print("Lora Checkpoint path", model_lora_path)

        if model_lora_path and model_lora_path != "None":
            print("Loading LoRA adapter weights")
            lora_ckpt = torch.load(model_lora_path, map_location="cpu")
            lora_state_dict = clean_state_dict(lora_ckpt.get("model", lora_ckpt))

            # Handle key renaming for LoRA and modules_to_save
            new_lora_ckpt_state_dict = {}
            modules_to_save = []
            if hasattr(model, "peft_config") and "default" in model.peft_config:
                modules_to_save = model.peft_config["default"].modules_to_save

            for key, value in lora_state_dict.items():
                new_key = key.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight")
                for module_name in modules_to_save:
                    if module_name in key:
                        new_key = new_key.replace(".weight", ".original_module.weight").replace(".bias", ".original_module.bias")
                new_lora_ckpt_state_dict[new_key] = value

            # Verify keys
            missing_keys = [k for k in new_lora_ckpt_state_dict if k not in model.state_dict()]
            if missing_keys:
                print("Missing LoRA keys:")
                for k in missing_keys:
                    print(f" - {k}")
                raise RuntimeError("LoRA checkpoint has keys not found in model.")

            model.load_state_dict(new_lora_ckpt_state_dict, strict=False)
            print("LoRA weights loaded successfully.")
        else:
            print("WARNING: LoRA path not provided. LoRA layers will be randomly initialized.")
    
    return model



def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


def train_image(model,
                image_source,
                image: torch.Tensor,
                caption_objects: list,
                box_target: list,
                device: str = "cuda"):

    def get_object_positions(tokenized, caption_objects):
        positions_dict = {}
        for obj_name in caption_objects:
            obj_token = tokenizer(obj_name + ".")['input_ids']
            start_pos = next((i for i, _ in enumerate(tokenized['input_ids']) if 
                             tokenized['input_ids'][i:i+len(obj_token)-2] == obj_token[1:-1]), None)
            if start_pos is not None:
                positions_dict[obj_name] = [start_pos, start_pos + len(obj_token) - 2]
        return positions_dict

    # Tokenization and object position extraction
    tokenizer = model.tokenizer
    caption = preprocess_caption(caption=".".join(set(caption_objects)))
    #print(f"Caption is {caption}")
    tokenized = tokenizer(caption)
    object_positions = get_object_positions(tokenized, caption_objects)

    # Move model and input to the device
    model = model.to(device)
    image = image.to(device)

    outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"][0]
    boxes = outputs["pred_boxes"][0]

    # Bounding box losses
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h]).to(device)
    box_predicted = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
    box_target = torch.tensor(box_target).to(device)
    ious = generalized_box_iou(box_target, box_predicted)
    maxvals, maxidx = torch.max(ious, dim=1)
    selected_preds = box_predicted.gather(0, maxidx.unsqueeze(-1).repeat(1, box_predicted.size(1)))
    regression_loss = F.smooth_l1_loss(box_target, selected_preds)
    iou_loss = 1.0 - maxvals.mean()
    reg_loss = iou_loss + regression_loss

    # Logit losses
    selected_logits = logits.gather(0, maxidx.unsqueeze(-1).repeat(1, logits.size(1)))
    targets_logits_list = []
    for obj_name, logit in zip(caption_objects, selected_logits):
        target = torch.zeros_like(logit).to(device)
        start, end = object_positions[obj_name]
        target[start:end] = 1.0
        targets_logits_list.append(target)

   
    targets_logits = torch.stack(targets_logits_list, dim=0)
    cls_loss = focal_loss(selected_logits, targets_logits)
    #print(f"Output keys are {outputs.keys()}")
    print(f"Regression and Classification loss are {reg_loss} and {cls_loss}")

    # Total loss
    delta_factor=0.01
    total_loss = cls_loss + delta_factor*reg_loss  

    return total_loss


def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str]) -> np.ndarray:
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [
        f"{phrase} {logit:.2f}"
        for phrase, logit
        in zip(phrases, logits)
    ]

    box_annotator = sv.BoxAnnotator()
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame


# ----------------------------------------------------------------------------------------------------------------------
# NEW API
# ----------------------------------------------------------------------------------------------------------------------


class Model:

    def __init__(
        self,
        model_config_path: str,
        model_checkpoint_path: str,
        model_lora_path: str = None,
        device: str = "cuda"
    ):
        self.model = load_model(
            model_config_path=model_config_path,
            model_checkpoint_path=model_checkpoint_path,
            model_lora_path = model_lora_path,
            device=device
        ).to(device)
        self.device = device

    def predict_with_caption(
        self,
        image: np.ndarray,
        caption: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25
    ) -> Tuple[sv.Detections, List[str]]:
        """
        import cv2

        image = cv2.imread(IMAGE_PATH)

        model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
        detections, labels = model.predict_with_caption(
            image=image,
            caption=caption,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        import supervision as sv

        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections, labels=labels)
        """
        processed_image = Model.preprocess_image(image_bgr=image).to(self.device)
        boxes, logits, phrases = predict(
            model=self.model,
            image=processed_image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold, 
            device=self.device)
        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits)
        return detections, phrases

    def predict_with_classes(
        self,
        image: np.ndarray,
        classes: List[str],
        box_threshold: float,
        text_threshold: float
    ) -> sv.Detections:
        """
        import cv2

        image = cv2.imread(IMAGE_PATH)

        model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
        detections = model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )


        import supervision as sv

        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections)
        """
        caption = ". ".join(classes)
        processed_image = Model.preprocess_image(image_bgr=image).to(self.device)
        boxes, logits, phrases = predict(
            model=self.model,
            image=processed_image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device)
        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits)
        class_id = Model.phrases2classes(phrases=phrases, classes=classes)
        detections.class_id = class_id
        return detections

    @staticmethod
    def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_pillow = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        image_transformed, _ = transform(image_pillow, None)
        return image_transformed

    @staticmethod
    def post_process_result(
            source_h: int,
            source_w: int,
            boxes: torch.Tensor,
            logits: torch.Tensor
    ) -> sv.Detections:
        boxes = boxes * torch.Tensor([source_w, source_h, source_w, source_h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        confidence = logits.numpy()
        return sv.Detections(xyxy=xyxy, confidence=confidence)

    @staticmethod
    def phrases2classes(phrases: List[str], classes: List[str]) -> np.ndarray:
        class_ids = []
        for phrase in phrases:
            for class_ in classes:
                if class_ in phrase:
                    class_ids.append(classes.index(class_))
                    break
            else:
                class_ids.append(None)
        return np.array(class_ids)
