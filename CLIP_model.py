from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch


class CLIPInference:
    """
    Wrapper for CLIP model inference.
    """

    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32")

    def infer(self, image: Image.Image, texts: list[str], top_k: int = 3) -> dict:
        """
        Run CLIP inference on an image against candidate texts.
        Returns top-k matches with probabilities.
        """
        inputs = self.processor(text=texts, images=image,
                                return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)[0]

        results = [
            {"text": text, "probability": round(prob.item(), 4)}
            for text, prob in zip(texts, probs)
        ]
        results = sorted(results, key=lambda x: x["probability"], reverse=True)[
            :top_k]
        return {"results": results}
