import torch
from typing import List, Optional, Any, cast
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Load once (typed and explicit)
_clip_model: CLIPModel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
_processor: CLIPProcessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

VISUAL_LABELS: List[str] = [
    "diagram",
    "anatomy illustration",
    "medical image",
    "graph",
    "chart",
    "table screenshot",
    "microscope image",
    "flowchart",
    "neural structure",
    "spinal cord",
]


def describe_image_with_clip(image_path: str) -> str:
    if not VISUAL_LABELS:
        raise ValueError("No visual labels provided")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cast to Any to avoid Pylance type-checker issues with .to(device)
    cast(Any, _clip_model).to(device)

    with Image.open(image_path) as img:
        image = img.convert("RGB")

    # CLIPProcessor can process both images and text together
    # cast processor to Any to avoid static-checker complaints about keyword args
    proc = cast(Any, _processor)
    inputs = proc(images=image, text=VISUAL_LABELS, return_tensors="pt", padding=True)

    pixel_values = inputs["pixel_values"].to(device)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        image_features = _clip_model.get_image_features(pixel_values=pixel_values)
        text_features = _clip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarities = image_features @ text_features.T

    # handle batch dimension (we expect a single image)
    if similarities.dim() == 2:
        best_idx = int(similarities[0].argmax().item())
    else:
        best_idx = int(similarities.argmax().item())

    return VISUAL_LABELS[best_idx]