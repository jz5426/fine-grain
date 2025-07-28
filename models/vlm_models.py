"""
VLM models

"""
from transformers import AutoTokenizer, AutoModel
from torch import nn
import torch
import warnings

from models.cxrclip import ResNet50
from models.mgca.models import BertEncoder
from models.mgca.models import ImageEncoder

class vlm_model():
    def __init__(self, image_encoder, image_projection_head, text_encoder, text_projection_head, tokenizer):
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.image_projection = image_projection_head
        self.text_projection = text_projection_head

        # freeze the model weights to prevent training
        self.freeze_weights(self.image_encoder)
        self.freeze_weights(self.image_projection)
        self.freeze_weights(self.text_encoder)
        self.freeze_weights(self.text_projection)

    def freeze_weights(self, model):
        assert model == self.image_encoder or model == self.text_encoder or model == self.tokenizer or model == self.image_projection or model == self.text_projection
        for param in model.parameters():
            param.requires_grad = False

class cxrclip_model(vlm_model):
    def __init__(self, image_encoder_path, text_encoder_path, tokenizer_path):

        image_encoder, image_projection_head, text_encoder, text_projection_head = self.load_cxrclip_encoder(image_encoder_path, text_encoder_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        super().__init__(image_encoder, image_projection_head, text_encoder, text_projection_head, tokenizer)

    def load_cxrclip_encoder(self, image_encoder_path, text_encoder_path):
        """handle only loading the cxr_clip based xray encoder and its (not ours) projection layer only -- need special handling of the dictionary keys like below"""
        warnings.filterwarnings('ignore')
        cxrclip_ckpt = torch.load(image_encoder_path, map_location="cpu")

        # load image encoder
        image_encoder = ResNet50()
        image_projection_head = LinearProjectionHead(2048, 512)
            
        # load text encoder
        text_encoder = AutoModel.from_pretrained(text_encoder_path)
        text_projection_head = LinearProjectionHead(768, 512)

        saved_state_dict = cxrclip_ckpt["model"]
        image_encoder_state_dict, image_projection_state_dict = {}, {}
        text_encoder_state_dict, text_projection_state_dict = {}, {}
        rest = {}
        for key in saved_state_dict.keys():
            if 'image_encoder.' in key:
                image_encoder_state_dict[key.replace("image_encoder.", "", 1)] = saved_state_dict[key]
            elif 'image_projection.' in key:
                image_projection_state_dict[key.replace("image_projection.", "", 1)] = saved_state_dict[key]
            # load the pretrained text encoder
            elif 'text_encoder.' in key:
                text_encoder_state_dict[key.replace("text_encoder.text_encoder.", "", 1)] = saved_state_dict[key]
            elif 'text_projection.' in key:
                text_projection_state_dict[key.replace("text_projection.", "", 1)] = saved_state_dict[key]
            else:
                rest[key] = saved_state_dict[key]
        print('non loaded keys ', rest)
        # NOTE: this sanity check if the model weights are loaded properly

        # image encoder
        missing_keys, unexpected_keys = image_encoder.load_state_dict(image_encoder_state_dict, strict=False)
        model_keys = set(image_encoder.state_dict().keys())
        ckpt_keys = set(image_encoder_state_dict.keys())
        loaded_keys = ckpt_keys.intersection(model_keys) - set(missing_keys)
        assert (len(image_encoder.state_dict().keys())) == len(loaded_keys)

        # text encoder
        missing_keys, unexpected_keys = text_encoder.load_state_dict(text_encoder_state_dict, strict=False)
        model_keys = set(text_encoder.state_dict().keys())
        ckpt_keys = set(text_encoder_state_dict.keys())
        loaded_keys = ckpt_keys.intersection(model_keys) - set(missing_keys)
        assert (len(text_encoder.state_dict().keys())) == len(loaded_keys)

        # image projection head
        missing_keys, unexpected_keys = image_projection_head.load_state_dict(image_projection_state_dict, strict=False)
        model_keys = set(image_projection_head.state_dict().keys())
        ckpt_keys = set(image_projection_state_dict.keys())
        loaded_keys = ckpt_keys.intersection(model_keys) - set(missing_keys)
        assert (len(image_projection_head.state_dict().keys())) == len(loaded_keys)
        
        # text projection head
        missing_keys, unexpected_keys = text_projection_head.load_state_dict(text_projection_state_dict, strict=False)
        model_keys = set(text_projection_head.state_dict().keys())
        ckpt_keys = set(text_projection_state_dict.keys())
        loaded_keys = ckpt_keys.intersection(model_keys) - set(missing_keys)
        assert (len(text_projection_head.state_dict().keys())) == len(loaded_keys)

        print(f'    finished loading the weights the weights from {image_encoder_path}')
        return image_encoder, image_projection_head, text_encoder, text_projection_head

class medclip_model(vlm_model):
    pass

class mgca_model(vlm_model):
    def __init__(self, image_encoder_path, text_encoder_path, tokenizer_path):
        image_encoder, image_projection_head, text_encoder, text_projection_head, tokenizer = self.load_mgca_encoder(image_encoder_path, text_encoder_path, tokenizer_path)
        super().__init__(image_encoder, image_projection_head, text_encoder, text_projection_head, tokenizer)

    def load_mgca_encoder(self, image_encoder_path, text_encoder_path, tokenizer_path):
        warnings.filterwarnings('ignore')
        mgca_ckpt = torch.load(image_encoder_path, map_location="cpu")
        saved_state_dict = mgca_ckpt['state_dict']

        # init encoders
        image_encoder = ImageEncoder(model_name='resnet_50', output_dim=128)
        text_encoder = BertEncoder(text_encoder_path, tokenizer_path, output_dim=128, freeze_bert=True)

        image_encoder_state_dict = {}
        text_encoder_state_dict = {}
        rest = {}
        for key in saved_state_dict.keys():
            if 'img_encoder_q' in key:
                image_encoder_state_dict[key.replace("img_encoder_q.", "", 1)] = saved_state_dict[key]
            elif 'text_encoder_q' in key:
                text_encoder_state_dict[key.replace("text_encoder_q.", "", 1)] = saved_state_dict[key]
            else:
                rest[key] = saved_state_dict[key]
        print('non loaded keys ', rest.keys())

        # image encoder
        missing_keys, unexpected_keys = image_encoder.load_state_dict(image_encoder_state_dict, strict=False)
        model_keys = set(image_encoder.state_dict().keys())
        ckpt_keys = set(image_encoder_state_dict.keys())
        loaded_keys = ckpt_keys.intersection(model_keys) - set(missing_keys)
        assert (len(image_encoder.state_dict().keys())) == len(loaded_keys)

        # text encoder
        missing_keys, unexpected_keys = text_encoder.load_state_dict(text_encoder_state_dict, strict=False)
        model_keys = set(text_encoder.state_dict().keys())
        ckpt_keys = set(text_encoder_state_dict.keys())
        loaded_keys = ckpt_keys.intersection(model_keys) - set(missing_keys)
        assert (len(text_encoder.state_dict().keys())) == len(loaded_keys)
        
        # image projection head
        image_projection_head = image_encoder.global_embed
        
        # text projection head
        text_projection_head = text_encoder.global_embed

        # tokenizer
        tokenizer = text_encoder.tokenizer

        return image_encoder, image_projection_head, text_encoder, text_projection_head, tokenizer


class LinearProjectionHead(nn.Module):
    # NOTE: this should be used instead.
    def __init__(self, embedding_dim, projection_dim):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)

    def forward(self, x):
        return self.projection(x)
