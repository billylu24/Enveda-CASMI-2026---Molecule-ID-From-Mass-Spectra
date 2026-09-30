"""Larger encoders with a shared 2048-bit fingerprint target."""
import torch
from torch import nn

from casmi_ml.models import FingerprintModel


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.block = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2),
                                   nn.GELU(), nn.Dropout(.1), nn.Linear(width * 2, width), nn.Dropout(.1))

    def forward(self, x):
        return x + self.block(x)


class ScaleModel(nn.Module):
    def __init__(self, architecture, metadata_dim):
        super().__init__()
        self.architecture = architecture
        self.metadata_dim = metadata_dim
        if architecture == 'metadata_control':
            self.model = FingerprintModel('metadata', metadata_dim)
        elif architecture in ['wide_metadata', 'wide_enhanced']:
            size = 1250 + metadata_dim + (1250 if architecture == 'wide_enhanced' else 0)
            self.encoder = nn.Sequential(nn.Linear(size, 768), nn.GELU(),
                                         *[ResidualBlock(768) for _ in range(3)], nn.LayerNorm(768))
            self.head = nn.Linear(768, 2048)
        elif architecture in ['peak_transformer', 'hybrid_transformer']:
            self.embedding = nn.Sequential(nn.Linear(19, 256), nn.GELU(), nn.Linear(256, 256))
            self.cls = nn.Parameter(torch.zeros(1, 1, 256))
            nn.init.normal_(self.cls, std=.02)
            layer = nn.TransformerEncoderLayer(256, 8, 1024, .1, activation='gelu',
                                               batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, 4, norm=nn.LayerNorm(256), enable_nested_tensor=False)
            size = 512 + metadata_dim
            if architecture == 'hybrid_transformer':
                self.histogram = nn.Sequential(nn.Linear(2500 + metadata_dim, 512), nn.GELU(),
                                               nn.Dropout(.1), nn.Linear(512, 256), nn.GELU())
                size += 256
            self.head = nn.Sequential(nn.Linear(size, 512), nn.GELU(), nn.Dropout(.1), nn.Linear(512, 2048))
        else:
            raise ValueError(architecture)

    @property
    def input_names(self):
        if self.architecture in ['metadata_control', 'wide_metadata']:
            return ['hist', 'meta']
        if self.architecture == 'wide_enhanced':
            return ['hist', 'loss', 'meta']
        if self.architecture == 'peak_transformer':
            return ['peaks', 'mask', 'meta']
        return ['hist', 'loss', 'peaks', 'mask', 'meta']

    def forward(self, batch):
        if self.architecture == 'metadata_control':
            return self.model(batch)
        if self.architecture.startswith('wide_'):
            pieces = [batch['hist'], batch['meta']]
            if self.architecture == 'wide_enhanced':
                pieces.append(batch['loss'])
            return self.head(self.encoder(torch.cat(pieces, -1)))
        mask = batch['mask'].bool()
        tokens = self.embedding(batch['peaks'])
        tokens = torch.cat([self.cls.expand(len(tokens), -1, -1), tokens], 1)
        valid = torch.cat([torch.ones((len(tokens), 1), device=tokens.device, dtype=torch.bool), mask], 1)
        encoded = self.encoder(tokens, src_key_padding_mask=~valid)
        mean = (encoded[:, 1:] * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        pieces = [encoded[:, 0], mean, batch['meta']]
        if self.architecture == 'hybrid_transformer':
            pieces.append(self.histogram(torch.cat([batch['hist'], batch['loss'], batch['meta']], -1)))
        return self.head(torch.cat(pieces, -1))
