"""Controlled spectrum encoder comparisons with a common fingerprint head."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class SpectrumDataset(Dataset):
    def __init__(self, directory, indexes=None):
        from pathlib import Path
        self.arrays = {key: np.load(Path(directory) / f'{key}.npy', mmap_mode='r')
                       for key in ['hist', 'loss', 'meta', 'peaks', 'mask', 'target']}
        self.indexes = np.arange(len(self.arrays['hist'])) if indexes is None else np.asarray(indexes)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        i = self.indexes[index]
        return {key: torch.from_numpy(np.array(value[i], copy=True)) for key, value in self.arrays.items()}


class FingerprintModel(nn.Module):
    def __init__(self, architecture, metadata_dim):
        super().__init__()
        self.architecture = architecture
        if architecture in ['mlp', 'enhanced', 'metadata']:
            size = 1250 + (metadata_dim if architecture != 'mlp' else 0)
            if architecture == 'enhanced':
                size += 1250
            self.encoder = nn.Sequential(nn.Linear(size, 512), nn.ReLU(), nn.Dropout(.1),
                                         nn.Linear(512, 256), nn.ReLU(), nn.Dropout(.1))
            self.head = nn.Linear(256, 2048)
        elif architecture in ['deepsets', 'transformer']:
            self.embedding = nn.Sequential(nn.Linear(19, 64), nn.ReLU(), nn.Linear(64, 128))
            if architecture == 'transformer':
                layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256,
                                                   dropout=.1, batch_first=True)
                self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
                size = 128 + metadata_dim
            else:
                size = 256 + metadata_dim
            self.head = nn.Sequential(nn.Linear(size, 256), nn.ReLU(), nn.Dropout(.1), nn.Linear(256, 2048))
        else:
            raise ValueError(f'Unknown architecture: {architecture}')

    def forward(self, batch):
        if self.architecture in ['mlp', 'enhanced', 'metadata']:
            pieces = [batch['hist']]
            if self.architecture == 'enhanced':
                pieces.append(batch['loss'])
            if self.architecture != 'mlp':
                pieces.append(batch['meta'])
            return self.head(self.encoder(torch.cat(pieces, dim=1)))
        mask = batch['mask'].bool()
        embedded = self.embedding(batch['peaks'])
        if self.architecture == 'transformer':
            embedded = self.encoder(embedded, src_key_padding_mask=~mask)
        pooled = (embedded * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        if self.architecture == 'deepsets':
            maximum = embedded.masked_fill(~mask[..., None], float('-inf')).max(1).values
            pooled = torch.cat([pooled, maximum], 1)
        return self.head(torch.cat([pooled, batch['meta']], 1))
