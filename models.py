"""
neural network models are here.

define all the vision language model here

"""
from transformers import AutoTokenizer, AutoModel

class vlm_model():
    def __init__(self, image_encoder, text_encoder, tokenizer):
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

class cxrclip(vlm_model):
    def __init__(self, image_encoder, text_encoder, tokenizer):
        # TODO: follow the ct-clip to load the image encoder with pretrained weights
        # image_encoder = None
        super().__init__(image_encoder, AutoModel.from_pretrained(text_encoder), AutoTokenizer.from_pretrained(tokenizer))

class medclip(vlm_model):
    pass