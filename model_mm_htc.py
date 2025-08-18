# model_mm_htc.py
import torch, torch.nn as nn, timm
from torch import nn
from einops import rearrange

class MMHTC(nn.Module):
    def __init__(self, d_fuse=384, nhead=6, nlayers=2):
        super().__init__()
        # encoders
        self.rgb = timm.create_model('convnext_tiny', pretrained=True, num_classes=0, global_pool='avg')  # 768-d
        self.fft = timm.create_model('resnet18', pretrained=False, in_chans=3, num_classes=0, global_pool='avg')  # 512-d
        self.proj = nn.Sequential(
            nn.Linear(768+512, 512), nn.ReLU(),
            nn.Linear(512, d_fuse)
        )
        enc_layer = nn.TransformerEncoderLayer(d_model=d_fuse, nhead=nhead, dim_feedforward=d_fuse*4,
                                               dropout=0.1, batch_first=True)
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head = nn.Sequential(nn.LayerNorm(d_fuse), nn.Linear(d_fuse, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128,1))
        # optional aux head
        self.aux = nn.Linear(d_fuse, 1)

    def forward(self, x_rgb, x_fft):  # x_*: [B,T,3,224,224]
        B,T = x_rgb.shape[:2]
        x_r = rearrange(x_rgb, 'b t c h w -> (b t) c h w')
        x_f = rearrange(x_fft, 'b t c h w -> (b t) c h w')
        fr = self.rgb(x_r)  # [(B*T),768]
        ff = self.fft(x_f)  # [(B*T),512]
        z = torch.cat([fr, ff], dim=1)
        z = self.proj(z)                     # [(B*T), d_fuse]
        z = rearrange(z, '(b t) d -> b t d', b=B, t=T)  # [B,T,d]
        zt = self.temporal(z)                # [B,T,d]
        vid_feat = zt.mean(dim=1)            # [B,d]
        logit = self.head(vid_feat)          # [B,1]
        aux_logits = self.aux(z).squeeze(-1) # [B,T]
        return logit, aux_logits
