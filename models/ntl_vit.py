"""
Import ViT models from timm library and modify them to return intermediate features
timm lid: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
"""

import torch
import torch.nn as nn
import timm

class ViT(nn.Module):
    def __init__(self, model_name, pretrained=True, **kwargs):
        super(ViT, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, **kwargs)

        self.patch_embed = self.model.patch_embed
        self._pos_embed = self.model._pos_embed
        self.patch_drop = self.model.patch_drop
        self.norm_pre = self.model.norm_pre
        self.blocks = self.model.blocks
        self.norm = self.model.norm
        self.pool = self.model.pool
        self.fc_norm = self.model.fc_norm
        self.head = self.model.head
        
        self.forward_head = self.model.forward_head
        
        
    def forward(self, x, y=None):
        """
        Get final output and middle block features
        """
        if y == None:
            return self.model(x)
        else: 
            # get all layer features
            feature_x_f, feature_layer_x = self.model.forward_intermediate(x)
            feature_y_f, feature_layer_y = self.model.forward_intermediate(y)
            
            # get middle block features
            split = len(feature_layer_x) // 2
            feature_x = feature_layer_x[split]
            feature_y = feature_layer_y[split]
            
            # get final output
            out_x = self.model.forward_head(feature_x_f)
            out_y = self.model.forward_head(feature_y_f)
            
            return out_x, out_y, self.model.pool(feature_x), self.model.pool(feature_y)
    
    def forward_f(self, x):
        """
        Get final output and final block features
        """
        # get final block features
        feature_f = self.model.forward_features(x)
        
        # get final output
        out = self.model.forward_head(feature_f)
        
        return out, self.model.pool(feature_f)
    
    def forward_layer_f(self, x):
        """
        Get final output and all block features
        """
        # get all block features
        feature_f, layer_features = self.model.forward_intermediates(x)
        
        # Process layer features via pooling
        layer_features = [self.model.pool(layer) for layer in layer_features]
        
        # get final output
        out = self.forward_head(feature_f)
        
        return out, layer_features
    
    def init_weights(self, mode = ''):
        self.model.init_weights(mode)
        
        

def vit_tiny(pretrained=True, **kwargs):
    """
    patch_size=16, embed_dim=192, depth=12, num_heads=3
    """
    model = ViT('vit_tiny_patch16_224', pretrained, **kwargs)
    
    return model

def vit_base(pretrained=True, **kwargs):
    """
    patch_size=16, embed_dim=768, depth=12, num_heads=12
    """
    model = ViT('vit_base_patch16_224', pretrained, **kwargs)
    
    return model

def vit_large(pretrained=True, **kwargs):
    """
    patch_size=16, embed_dim=1024, depth=24, num_heads=16
    """
    model = ViT('vit_large_patch16_224', pretrained, **kwargs)
    
    return model

def vit_huge(pretrained=True, **kwargs):
    """
    patch_size=14, embed_dim=1280, depth=32, num_heads=16
    
    """
    model = ViT('vit_huge_patch14_224', pretrained, **kwargs)
    
    return model