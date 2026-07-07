"""3D volume encoder + text projection for contrastive text-conditioned localization."""
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import resnet10


class TextVolumeAligner(nn.Module):
    def __init__(self, text_dim=768, proj_dim=256):
        super().__init__()
        self.volume_encoder = resnet10(spatial_dims=3, n_input_channels=4, num_classes=proj_dim)
        self.text_proj = nn.Linear(text_dim, proj_dim)

    def encode_image(self, patches):
        return F.normalize(self.volume_encoder(patches), dim=-1)

    def encode_text(self, text_embeds):
        return F.normalize(self.text_proj(text_embeds), dim=-1)
