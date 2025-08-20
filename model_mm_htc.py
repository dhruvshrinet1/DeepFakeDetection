# model_mm_htc.py
import torch, torch.nn as nn, timm
from einops import rearrange

class MMHTC(nn.Module):
    """
    Multimodal (RGB, FFT) with optional Temporal Transformer.
    Switch branches with use_rgb/use_fft/use_temporal.
    """
    def __init__(self, d_fuse=384, nhead=6, nlayers=2, use_rgb=True, use_fft=True, use_temporal=True):
        super().__init__()
        assert use_rgb or use_fft, "At least one branch (RGB or FFT) must be enabled."
        self.use_rgb = use_rgb
        self.use_fft = use_fft
        self.use_temporal = use_temporal

        # Encoders
        if self.use_rgb:
            self.rgb = timm.create_model('convnext_tiny', pretrained=True, num_classes=0, global_pool='avg')  # 768-d
            self.proj_rgb = nn.Linear(self.rgb.num_features, d_fuse)
        if self.use_fft:
            self.fft = timm.create_model('resnet18', pretrained=False, in_chans=3, num_classes=0, global_pool='avg')  # 512-d
            self.proj_fft = nn.Linear(512, d_fuse)

        # Temporal encoder (optional)
        if self.use_temporal:
            enc_layer = nn.TransformerEncoderLayer(d_model=d_fuse, nhead=nhead, dim_feedforward=d_fuse*4,
                                                   dropout=0.1, batch_first=True)
            self.temporal = nn.TransformerEncoder(enc_layer, num_layers=nlayers)

        # Heads
        self.head = nn.Sequential(nn.LayerNorm(d_fuse), nn.Linear(d_fuse, 128),
                                  nn.ReLU(), nn.Dropout(0.2), nn.Linear(128,1))
        self.aux = nn.Linear(d_fuse, 1)  # optional frame-level auxiliary head

    def forward(self, x_rgb, x_fft=None):   # x_*: [B,T,3,224,224]
        B,T = x_rgb.shape[:2]
        feats = []

        if self.use_rgb:
            xr = rearrange(x_rgb, 'b t c h w -> (b t) c h w')
            fr = self.rgb(xr)                    # [(B*T), 768]
            zr = self.proj_rgb(fr)               # [(B*T), d]
            feats.append(zr)

        if self.use_fft:
            assert x_fft is not None, "x_fft is required when use_fft=True"
            xf = rearrange(x_fft, 'b t c h w -> (b t) c h w')
            ff = self.fft(xf)                    # [(B*T), 512]
            zf = self.proj_fft(ff)               # [(B*T), d]
            feats.append(zf)

        # Fuse by (weighted) mean to keep scale stable
        if len(feats) == 1:
            z = feats[0]
        else:
            z = sum(feats) / len(feats)          # [(B*T), d]

        # To sequence
        z = rearrange(z, '(b t) d -> b t d', b=B, t=T)  # [B,T,d]

        # Temporal modeling or simple temporal mean
        if self.use_temporal:
            zt = self.temporal(z)                 # [B,T,d]
        else:
            zt = z

        vid_feat = zt.mean(dim=1)                 # [B,d]
        logit = self.head(vid_feat)               # [B,1]
        aux_logits = self.aux(z).squeeze(-1)      # [B,T]
        return logit, aux_logits