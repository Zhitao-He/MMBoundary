from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Default CLIP model. Override via load_clip_model() parameter.
DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14-336"

_clip_model = None
_clip_processor = None


def load_clip_model(model_path=DEFAULT_CLIP_MODEL):
    """Load (or reload) the CLIP model and processor.

    Args:
        model_path: HuggingFace model name or local path.
                    Defaults to DEFAULT_CLIP_MODEL.
    """
    global _clip_model, _clip_processor
    _clip_model = CLIPModel.from_pretrained(model_path)
    _clip_processor = CLIPProcessor.from_pretrained(model_path)


def get_clip_score(image, sentence_text):
    """Compute the CLIP similarity score between an image and a text sentence.

    Args:
        image: A PIL Image object.
        sentence_text: The text to compare against the image.

    Returns:
        A float CLIP similarity score.
    """
    global _clip_model, _clip_processor
    if _clip_model is None or _clip_processor is None:
        load_clip_model()

    inputs = _clip_processor(text=[sentence_text], images=image, return_tensors="pt", padding=True)

    outputs = _clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    clip_score = logits_per_image.cpu().detach().numpy()[0][0]

    return clip_score
