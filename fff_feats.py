# fft_feats.py
import torch, torch.nn.functional as F

def rgb_to_fft_amp(x):  # x: [B*T,3,H,W] in [-1,1] normalized
    x01 = (x*0.5+0.5).clamp(0,1)
    outs=[]
    for c in range(3):
        X = torch.fft.rfft2(x01[:,c], norm="ortho")         # [N,H,W//2+1]
        amp = torch.log1p(torch.abs(X))                     # [N,H,W//2+1]
        amp_full = F.interpolate(amp.unsqueeze(1), size=x.shape[-2:], mode="bilinear", align_corners=False)[:,0]
        outs.append(amp_full)
    return torch.stack(outs, dim=1)  # [N,3,H,W]
