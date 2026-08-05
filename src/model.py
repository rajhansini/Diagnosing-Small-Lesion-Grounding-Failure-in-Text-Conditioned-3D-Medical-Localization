"""3D volume encoder + text projection for contrastive text-conditioned localization."""
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import resnet10


class TextVolumeAligner(nn.Module):
    """Projects 3D image patches and sentence embeddings into one shared, L2-normalized space.

    The image branch is a MONAI 3D ResNet-10 whose classification head is repurposed as a projection
    head (num_classes=proj_dim). The text branch is a single trainable linear layer over frozen
    PubMedBERT sentence embeddings. That asymmetry is deliberate and is what makes the model work at
    all: raw PubMedBERT embeddings for our four class descriptions sit at ~0.99 pairwise cosine
    similarity (BERT anisotropy), so the frozen encoder supplies almost no discriminative signal and
    this linear head has to do essentially all the separating work during contrastive training.

    Both branches L2-normalize their output, so a dot product between them is a cosine similarity --
    which is what the training loss and the sliding-window localizer both consume.

    Args:
        text_dim: dimensionality of the incoming frozen sentence embeddings (768 for PubMedBERT).
        proj_dim: dimensionality of the shared space both branches project into.
    """

    def __init__(self, text_dim=768, proj_dim=256):
        """Build the ResNet-10 volume encoder and the linear text projection head."""
        super().__init__()
        self.volume_encoder = resnet10(spatial_dims=3, n_input_channels=4, num_classes=proj_dim)
        self.text_proj = nn.Linear(text_dim, proj_dim)

    def encode_image(self, patches):
        """Embed a batch of 3D patches into the shared space.

        Args:
            patches: (B, 4, D, H, W) float tensor of multi-modal MRI patches.

        Returns:
            (B, proj_dim) L2-normalized tensor.
        """
        return F.normalize(self.volume_encoder(patches), dim=-1)

    def encode_text(self, text_embeds):
        """Project frozen sentence embeddings into the shared space.

        Args:
            text_embeds: (N, text_dim) float tensor of mean-pooled PubMedBERT embeddings.

        Returns:
            (N, proj_dim) L2-normalized tensor.
        """
        return F.normalize(self.text_proj(text_embeds), dim=-1)
