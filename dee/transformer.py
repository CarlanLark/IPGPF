# -*- coding: utf-8 -*-

import copy
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, List


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences."""
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # self.a_2 = nn.Parameter(torch.ones(features))
        # self.b_2 = nn.Parameter(torch.zeros(features))
        # fit for bert optimizer
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class Encoder(nn.Module):
    """"Core encoder is a stack of N layers"""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask, **kwargs):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask, **kwargs)
        return self.norm(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask, **kwargs):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask, **kwargs))
        return self.sublayer[1](x, self.feed_forward)



def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subseq_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subseq_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, **kwargs):
        """Implements Figure 2"""
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(device=x.device)
        return self.dropout(x)



# def make_transformer_encoder(num_layers, hidden_size, ff_size=2048, num_att_heads=8, dropout=0.1):
#     dcopy = copy.deepcopy
#     mh_att = MultiHeadedAttention(num_att_heads, hidden_size, dropout=dropout)
#     pos_ff = PositionwiseFeedForward(hidden_size, ff_size, dropout=dropout)

#     tranformer_encoder = Encoder(
#         EncoderLayer(hidden_size, dcopy(mh_att), dcopy(pos_ff), dropout=dropout),
#         num_layers
#     )

#     return tranformer_encoder


def make_transformer_encoder(num_layers, hidden_size, ff_size=2048, num_att_heads=8, dropout=0.1):
    dcopy = copy.deepcopy
    mh_att = MultiHeadedAttention(num_att_heads, hidden_size, dropout=dropout)
    pos_ff = PositionwiseFeedForward(hidden_size, ff_size, dropout=dropout)

    tranformer_encoder = Encoder(
        EncoderLayer(hidden_size, dcopy(mh_att), dcopy(pos_ff), dropout=dropout),
        num_layers
    )

    return tranformer_encoder

############

class ReferenceDecoderLayer(nn.Module):
  def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
    super().__init__()
    self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
    self.corpus_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
    self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
    # Implementation of Feedforward model
    self.linear1 = nn.Linear(d_model, dim_feedforward)
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(dim_feedforward, d_model)

    self.norm1 = nn.LayerNorm(d_model)
    self.norm_c = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.norm3 = nn.LayerNorm(d_model)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout_c = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)
    self.dropout3 = nn.Dropout(dropout)

    self.activation = F.relu
    self.normalize_before = normalize_before

  def forward(self, tgt, memory, corpus, 
          tgt_mask: Optional[Tensor] = None,
          memory_mask: Optional[Tensor] = None,
          corpus_mask: Optional[Tensor] = None,
          tgt_key_padding_mask: Optional[Tensor] = None,
          memory_key_padding_mask: Optional[Tensor] = None,
          corpus_key_padding_mask: Optional[Tensor] = None):
    q = k = tgt
    tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                          key_padding_mask=tgt_key_padding_mask)[0]
    tgt = tgt + self.dropout1(tgt2)
    tgt = self.norm1(tgt)
    tgt2 = self.corpus_attn(tgt, corpus, corpus, attn_mask=corpus_mask,
                                key_padding_mask=corpus_key_padding_mask)[0]
    tgt = tgt + self.dropout_c(tgt2)
    tgt = self.norm_c(tgt)
    tgt2 = self.multihead_attn(query=tgt,
                                key=memory,
                                value=memory, attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask)[0]
    tgt = tgt + self.dropout2(tgt2)
    tgt = self.norm2(tgt)
    tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    tgt = tgt + self.dropout3(tgt2)
    tgt = self.norm3(tgt)
    return tgt

    
class ReferenceDecoder(nn.Module):
  def __init__(self, d_model, nhead = 8, dim_feedforward = 2048, dropout = 0.1, num_layers = 6):
    super().__init__()
    self.num_layers = num_layers
    self.d_model = d_model
    self.layers = nn.ModuleList([ReferenceDecoderLayer(d_model=self.d_model, nhead=nhead, dim_feedforward=4*self.d_model, dropout=dropout) for _ in range(self.num_layers)])

  def forward(self, tgt, memory, corpus, 
          tgt_mask: Optional[Tensor] = None,
          memory_mask: Optional[Tensor] = None,
          corpus_mask: Optional[Tensor] = None,
          tgt_key_padding_mask: Optional[Tensor] = None,
          memory_key_padding_mask: Optional[Tensor] = None, 
          corpus_key_padding_mask: Optional[Tensor] = None):
    dec_output = tgt
    for layer in self.layers:
      dec_output = layer(dec_output, memory, corpus, 
                tgt_mask=tgt_mask, 
                memory_mask=memory_mask, 
                corpus_mask = corpus_mask, 
                tgt_key_padding_mask=tgt_key_padding_mask, 
                memory_key_padding_mask=memory_key_padding_mask, 
                corpus_key_padding_mask = corpus_key_padding_mask)
    return dec_output



def make_transformer_decoder(num_layers, hidden_size, ff_size=2048, num_att_heads=8, dropout=0.1):
    decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_att_heads, dim_feedforward = ff_size, dropout = dropout)
    transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    return transformer_decoder

def make_reference_decoder(num_layers, hidden_size, ff_size=2048, num_att_heads=8, dropout=0.1):
    reference_decoder = ReferenceDecoder(d_model=hidden_size, nhead=num_att_heads, dim_feedforward = ff_size, dropout = dropout)

    return reference_decoder


class PointerNetwork(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_model, bias=False)
        self.w2 = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, 1, bias=False)
    
    def forward(self, tgt, memory):
        memory = self.w1(memory).unsqueeze(0)
        tgt = self.w2(tgt).unsqueeze(1)
        pred_logits = self.v(torch.tanh(memory + tgt)).squeeze(-1) 
        return pred_logits