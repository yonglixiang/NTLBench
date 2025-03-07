# Vision Transformer model implementation
# which is similar to the original implementation in the official repository:
# https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py
# But we use TransformerEncoder instead of a custom implementation of Transformer.

# YONGLI: NOT COMPLETED YET

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from einops.layers.torch import Rearrange

class VisionTransformer(nn.Module):
    def __init__(self,
                 init_weights=True,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 num_classes=1000,
                 embed_dim=768,
                 representation_size=None,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 dropout=0.1,
                 norm_first=True,
                 layer_norm_eps=1e-6):
        super().__init__()
        
        # Image Patching
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c')
        )
        
        # Pos Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=norm_first,
            layer_norm_eps=layer_norm_eps
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=depth)
        
        # MLP Head
        if representation_size is None:
            self.mlp_head = nn.Sequential(
                nn.Linear(embed_dim, num_classes)
            )
        else:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, representation_size),
                nn.Tanh(),
                nn.Linear(representation_size, num_classes)
            )
        
        # Initialize weights
        if init_weights:
            self._init_weights()
    
    def _init_weights(self):
        # Init pos embedding and cls token
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.zeros_(self.cls_token)
        
        # Init patch embedding
        if isinstance(self.patch_embed[0], nn.Conv2d):
            # Conv2d weights
            nn.init.trunc_normal_(self.patch_embed[0].weight, std=0.02)
            if self.patch_embed[0].bias is not None:
                nn.init.zeros_(self.patch_embed[0].bias)
        
        # Init mlp head
        if hasattr(self, 'mlp_head'):
            if isinstance(self.mlp_head, nn.Sequential):
                if len(self.mlp_head) == 1:
                    # Final classification head should be zero
                    nn.init.zeros_(self.mlp_head[0].weight)
                    nn.init.zeros_(self.mlp_head[0].bias)
                else:
                    # Pre-logits: linear + tanh case
                    if isinstance(self.mlp_head[1], nn.Linear):
                        nn.init.trunc_normal_(self.mlp_head[1].weight, std=0.02)
                        nn.init.zeros_(self.mlp_head[1].bias)
                    # Final classification layer
                    if isinstance(self.mlp_head[-1], nn.Linear):
                        nn.init.zeros_(self.mlp_head[-1].weight)
                        nn.init.zeros_(self.mlp_head[-1].bias)
    
    
    def forward(self, x):
        # Patch embedding
        B = x.shape[0]
        x = self.patch_embed(x)
        
        # Add CLS token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer encoder
        x = self.transformer(x)
        
        # Get CLS token output
        x = x[:, 0]
        
        # Apply MLP head
        x = self.mlp_head(x)
        
        return x

model_urls = {
    'vit_base_patch16_224': 'https://download.pytorch.org/models/vit_b_16-c867db91.pth',
    'vit_large_patch16_224': 'https://download.pytorch.org/models/vit_l_16-852ce7e3.pth',
    'vit_huge_patch14_224': 'https://download.pytorch.org/models/vit_h_14-6f678296.pth'
}

def vit_base(pretrained=False, **kwargs):
    """
    patch_size=16, embed_dim=768, depth=12, num_heads=12
    
    """
    if pretrained:
        kwargs['init_weights'] = False
    kwargs['patch_size'] = 16
    kwargs['embed_dim'] = 768
    kwargs['depth'] = 12
    kwargs['num_heads'] = 12
    
    model = VisionTransformer(**kwargs)
    model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['vit_base_patch16_224']), strict=False)
    
    return model

def vit_large(pretrained=False, **kwargs):
    """
    patch_size=16, embed_dim=1024, depth=24, num_heads=16
    
    """
    if pretrained:
        kwargs['init_weights'] = False
    kwargs['patch_size'] = 16
    kwargs['embed_dim'] = 1024
    kwargs['depth'] = 24
    kwargs['num_heads'] = 16
    
    model = VisionTransformer(**kwargs)
    model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['vit_large_patch16_224']), strict=False)
    
    return model

def vit_huge(pretrained=False, **kwargs):
    """
    patch_size=14, embed_dim=1280, depth=32, num_heads=16
    
    """
    if pretrained:
        kwargs['init_weights'] = False
    kwargs['patch_size'] = 14
    kwargs['embed_dim'] = 1280
    kwargs['depth'] = 32
    kwargs['num_heads'] = 16
    
    model = VisionTransformer(**kwargs)
    model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['vit_huge_patch14_224']), strict=False)
    
    return model