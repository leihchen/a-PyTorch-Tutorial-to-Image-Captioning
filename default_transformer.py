
import pickle
from typing import List, Optional, Tuple, Dict

import numpy as np
import math
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import ticker

import torch
from torch.nn import Module, Linear, Softmax, ReLU, LayerNorm, ModuleList, Dropout, Embedding, CrossEntropyLoss
from torch.optim import Adam

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader


class PatchEmbedding(Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, E, P, P)
        x = x.flatten(2)  # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, device = torch.device("cpu"), dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model).to(device)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class ImageTransformer(Module):

    def __init__(self, 
                 len_vocab,
                 img_size=256, 
                 patch_size=7, 
                 in_chans=1, 
                 embed_dim=100, 
                 max_len=8, 
                 nhead=2, 
                 num_encoder_layers=3,
                 num_decoder_layers=3,
                 dim_feedforward=400,
                 dropout=0.1,
                 start_token=2631,
                 device=torch.device("cpu")
                ):
        super().__init__()

        self.device=device
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, embed_dim))
        # self.pos_embed = PositionalEncoding(embed_dim, device=device)
        
        self.trg_emb = nn.Embedding(len_vocab, embed_dim)
        self.trg_pos_emb = nn.Embedding(max_len, embed_dim)
        self.max_len = max_len

        self.transformer = torch.nn.Transformer(
            embed_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout
        )
        
        self.l = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, len_vocab)

        self.start_token = start_token

    def forward(self, images, captions):
        # embed images
        embed_imgs = self.patch_embed(images)
        embed_imgs = self.pos_embed + embed_imgs
        # embed_imgs = self.pos_embed(embed_imgs)
        # embed captions
        B, trg_seq_len = captions.shape
        trg_positions = (torch.arange(0, trg_seq_len).expand(B, trg_seq_len).to(self.device))
        embed_trg = self.trg_emb(captions) + self.trg_pos_emb(trg_positions)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_len).to(self.device)
        tgt_padding_mask = captions == 0
        # transformer
        y = self.transformer(
            embed_imgs.permute(1,0,2),  
            embed_trg.permute(1,0,2),  
            tgt_mask=trg_mask, 
            tgt_key_padding_mask = tgt_padding_mask
        ).permute(1,0,2) 
        # head
        return self.fc(self.l(y))

    def predict(self, images):
        self.eval()
        with torch.no_grad():
            images = images.to(self.device)
            B = images.shape[0]
            eos = torch.tensor([self.start_token], dtype=torch.long, device=self.device).expand(B, 1)
            trg_input = eos
            for _ in range(self.max_len):
                preds = self.forward(images, trg_input)
                preds = torch.log(torch.nn.functional.softmax(preds, dim=2))
                preds = torch.argmax(preds, axis=2)
                trg_input = torch.cat([eos, preds], 1)
            return preds
        
    def compute_loss_and_acc(self, batch):
        x, y = batch
        y_hat = self(x, y[:,:-1])
        trg_output = y[:,1:] 
        loss = F.cross_entropy(y_hat.permute(0,2,1), trg_output) 
        # I know this is not the best metric...
        acc = (torch.argmax(y_hat, axis=2) == trg_output).sum().item() / (trg_output.shape[0]*trg_output.shape[1])
        return loss, acc
    
    def training_step(self, batch, batch_idx):
        loss, acc = self.compute_loss_and_acc(batch)
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx} loss: {loss.item():.4f}, acc: {acc:.4f}')
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.compute_loss_and_acc(batch)
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx} val loss: {loss.item():.4f}, val acc: {acc:.4f}')

    def configure_optimizers(self, lr=0.0001):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer
