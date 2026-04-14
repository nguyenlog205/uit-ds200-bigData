import torch
import torch.nn as nn

class AST(nn.Module):
    def __init__(
        self,
        num_classes: int = 50,
        input_fdim: int = 128,
        input_tdim: int = 512,
        patch_size: int = 16,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12
    ):
        super().__init__()
        # Patch embedding layer
        self.patch_embed = nn.Conv2d(
            1, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        num_patches = (input_fdim // patch_size) * (input_tdim // patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, F, T) mel-spectrogram
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.transformer_encoder(x)
        return self.mlp_head(x[:, 0])