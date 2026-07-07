#!/bin/bash
#SBATCH --job-name=nlp_rq3_smoke
#SBATCH --partition=dev
#SBATCH --account=threedle_group
#SBATCH --gres=gpu:1
#SBATCH --time=00:08:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/net/projects/ranalab/rajhansini/nlp_project/logs/rq3_smoke_%j.out

source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects/ranalab/rajhansini/conda_envs/nlp_project
cd /net/projects/ranalab/rajhansini/nlp_project/src
python -c "
import torch
from model import TextVolumeAligner
from localize import sliding_window_heatmap
from text_encoder import REGION_ORDER
import numpy as np

device = 'cuda'
text_embeds = torch.load('/net/projects/ranalab/rajhansini/nlp_project/data/preprocessed/subregion_text_embeddings.pt', weights_only=True)
text_embeds_t = torch.stack([text_embeds[r] for r in REGION_ORDER])
model = TextVolumeAligner(text_dim=text_embeds_t.shape[1]).to(device)
model.load_state_dict(torch.load('/net/projects/ranalab/rajhansini/nlp_project/checkpoints/baseline_aligner.pt', map_location=device, weights_only=True))
model.eval()
with torch.no_grad():
    text_proj = model.encode_text(text_embeds_t.to(device))

data = np.load('/net/projects/ranalab/rajhansini/nlp_project/data/preprocessed/BraTS20_Training_225.npz')
image = torch.from_numpy(data['image']).float()
et_idx = REGION_ORDER.index('ET')

for ws, st in [(16, 8), (32, 16), (64, 32)]:
    hm = sliding_window_heatmap(model, image, text_proj[et_idx], window_size=ws, stride=st, model_input_size=32, device=device)
    print(f'window_size={ws} stride={st}: heatmap shape={tuple(hm.shape)}, min={hm.min():.3f}, max={hm.max():.3f}')
print('SMOKE_TEST_OK')
"
