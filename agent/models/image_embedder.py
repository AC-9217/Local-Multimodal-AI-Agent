import torch
import open_clip
from PIL import Image
from agent.config import Config
from agent.utils.logging import setup_logger

logger = setup_logger(__name__)

class ImageEmbedder:
    def __init__(self, model_name: str = Config.IMAGE_MODEL_NAME, pretrained: str = Config.IMAGE_MODEL_PRETRAINED, device: str = Config.DEVICE):
        logger.info(f"Loading Image Embedder: {model_name} ({pretrained}) on {device}")
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def embed_images(self, images: list[Image.Image]):
        if not images:
            return []
        processed_images = [self.preprocess(img).unsqueeze(0) for img in images]
        image_input = torch.cat(processed_images).to(self.device)
        
        with torch.no_grad():
            # Use autocast if on cuda
            if self.device == "cuda":
                with torch.cuda.amp.autocast():
                    image_features = self.model.encode_image(image_input)
            else:
                image_features = self.model.encode_image(image_input)
                
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
        return image_features.cpu().numpy()

    def embed_text(self, text: list[str]):
        if not text:
            return []
        text_input = self.tokenizer(text).to(self.device)
        
        with torch.no_grad():
            if self.device == "cuda":
                with torch.cuda.amp.autocast():
                    text_features = self.model.encode_text(text_input)
            else:
                text_features = self.model.encode_text(text_input)
                
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        return text_features.cpu().numpy()
