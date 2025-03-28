from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.forebacklearning import ForeBackLearning
from .att_model import pack_wrapper, AttModel
from modules.utils import compute_clip_loss, compute_clip_loss2


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def diff_attention(query, key, value,lambda_full, mask=None, dropout=None):
    #query,key,value (B,diff_num_head,N,2d)
    d_k = query.size(-1) #2d
    diff_d_k = d_k // 2 #d
    B, diff_num_head, N, d_k = value.size()
    # print(B,diff_num_head,N,d_k)
    query = query.reshape(B,2*diff_num_head,-1,d_k//2) #(B,2*diff_num_head,N,d)
    key = key.reshape(B,2*diff_num_head,-1,d_k//2) #(B,2*diff_num_head,N,d)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k//2) #(B,2*diff_num_head,N,N)
    #mask: (B,1,2*Ns)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        # scores = scores.masked_fill(mask == 0, 0)
    p_attn = F.softmax(scores, dim=-1)
    p_attn = p_attn.reshape(B,diff_num_head,2,-1,N)
    p_attn = p_attn[:,:,0] - lambda_full* p_attn[:,:,1] #(B,diff_num_head,N,N)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def attention(query, key, value, mask=None, dropout=None):
    
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        # scores = scores.masked_fill(mask == 0, 0)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


#         c = copy.deepcopy
#         attn = MultiHeadedAttention(self.num_heads, self.d_model)
#         ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
#         position = PositionalEncoding(self.d_model, self.dropout)
#         rm = RelationalMemory(num_slots=self.rm_num_slots, d_model=self.rm_d_model, num_heads=self.rm_num_heads)
#         model = Transformer(
#             Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
#             Decoder(
#                 DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout, self.rm_num_slots, self.rm_d_model),
#                 self.num_layers),
#             lambda x: x,
#             nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)),
#             rm, fbl=self.fbl)
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, rm, fbl=False):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.rm = rm
        self.fbl = fbl
    #att_feats	(B, 2*Ns, d_model)
        #seq	(B, seq_len - 1) or None
        #att_masks	(B, 1, 2*Ns)
        #seq_mask	(B, 1, seq_len - 1) or None
        #src_embed: lambda x:x
        #tgt_embed: nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position))
    #out, fore_rep_encoded, target_embed, align_attns = self.model(att_feats, seq, att_masks, seq_mask)
    def forward(self, src, tgt, src_mask, tgt_mask, mode='train'):
        #src: att_feats (B, 2*Ns, d_model)
        #tgt: seq (B, seq_len - 1) 
        #src_mask: att_masks (B, 1, 2*Ns)
        #tgt_mask: seq_mask (B, 1, seq_len - 1) or None
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask, mode)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, hidden_states, src_mask, tgt, tgt_mask, mode='sample'):
        memory = self.rm.init_memory(hidden_states.size(0)).to(hidden_states)
        target_emb = self.tgt_embed(tgt)
        memory = self.rm(target_emb, memory)
        if self.fbl:
            hidden_states, src_mask = hidden_states[:, 1:], src_mask[:, :, 1:]
        if mode == 'train':
            out, align_attns = self.decoder(target_emb, hidden_states, src_mask, tgt_mask, memory)
            return out, hidden_states[:, 0], target_emb, align_attns
        return self.decoder(target_emb, hidden_states, src_mask, tgt_mask, memory)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

#          c = copy.deepcopy
#         attn = MultiHeadedAttention(self.num_heads, self.d_model)
#         ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
#         position = PositionalEncoding(self.d_model, self.dropout)
#         rm = RelationalMemory(num_slots=self.rm_num_slots, d_model=self.rm_d_model, num_heads=self.rm_num_heads)
#         model = Transformer(
#             Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
#             Decoder(
#                 DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout, self.rm_num_slots, self.rm_d_model),
#                 self.num_layers),
#             lambda x: x,
#             nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)),
#             rm, fbl=self.fbl)
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory):
        attns = []
        for layer in self.layers:
            x, attn = layer(x, hidden_states, src_mask, tgt_mask, memory)
            attns.append(attn)
        return self.norm(x), attns


class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout, rm_num_slots, rm_d_model):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ConditionalSublayerConnection(d_model, dropout, rm_num_slots, rm_d_model), 3)

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory):
        m = hidden_states
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask), memory)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask), memory)
        return self.sublayer[2](x, self.feed_forward, memory), self.src_attn.attn


class ConditionalSublayerConnection(nn.Module):
    def __init__(self, d_model, dropout, rm_num_slots, rm_d_model):
        super(ConditionalSublayerConnection, self).__init__()
        self.norm = ConditionalLayerNorm(d_model, rm_num_slots, rm_d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, memory):
        return x + self.dropout(sublayer(self.norm(x, memory)))


class ConditionalLayerNorm(nn.Module):
    def __init__(self, d_model, rm_num_slots, rm_d_model, eps=1e-6):
        super(ConditionalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.rm_d_model = rm_d_model
        self.rm_num_slots = rm_num_slots
        self.eps = eps

        self.mlp_gamma = nn.Sequential(nn.Linear(rm_num_slots * rm_d_model, d_model),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(rm_d_model, rm_d_model))

        self.mlp_beta = nn.Sequential(nn.Linear(rm_num_slots * rm_d_model, d_model),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(d_model, d_model))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x, memory):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        delta_gamma = self.mlp_gamma(memory)
        delta_beta = self.mlp_beta(memory)
        gamma_hat = self.gamma.clone()
        beta_hat = self.beta.clone()
        gamma_hat = torch.stack([gamma_hat] * x.size(0), dim=0)
        gamma_hat = torch.stack([gamma_hat] * x.size(1), dim=1)
        beta_hat = torch.stack([beta_hat] * x.size(0), dim=0)
        beta_hat = torch.stack([beta_hat] * x.size(1), dim=1)
        gamma_hat += delta_gamma
        beta_hat += delta_beta
        return gamma_hat * (x - mean) / (std + self.eps) + beta_hat


class DiffMultiHeadedAttention(nn.Module): # MultiHeadedAttention(self.num_heads, self.d_model)
    #head_dim stay the same for Q,K and 2*head_dim for V while diff_num_head = 1/2 * num_head 
    def __init__(self, h, d_model, dropout=0.1):
        super(DiffMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.diff_num_head = h // 2 #h
        self.diff_d_k = d_model // self.diff_num_head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.lambda_init = 0.8
        self.lambda_q1 = nn.Parameter(torch.zeros(self.diff_d_k//2, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.diff_d_k//2, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.diff_d_k//2, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.diff_d_k//2, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.diff_num_head, self.diff_d_k).transpose(1, 2)   #Linear projection before attention?
             for l, x in zip(self.linears, (query, key, value))]  #Apply the first 3 linear projection for query key value
        #query,key,value (B,diff_num_head,N,2d)
        # print("lambda:",self.lambda_k1.shape)
        lambda1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float())
        lambda2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float())
        lambda_full = lambda1 - lambda2 + self.lambda_init

        x, self.attn = diff_attention(query, key, value, mask=mask, dropout=self.dropout,lambda_full=lambda_full)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.diff_num_head * self.diff_d_k)
        return self.linears[-1](x)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        print('In standard attention')

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # print("In multi-headed attention")
        # print("query:",query.shape)
        # print("mask:",mask is None)

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
    

class PositionwiseFeedForward(nn.Module):
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
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model) #(max_len,d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float() #(max_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #(1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class RelationalMemory(nn.Module):

    def __init__(self, num_slots, d_model, num_heads=1):
        super(RelationalMemory, self).__init__()
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.d_model = d_model

        self.attn = DiffMultiHeadedAttention(num_heads, d_model)
        self.mlp = nn.Sequential(nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU(),
                                 nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU())

        self.W = nn.Linear(self.d_model, self.d_model * 2)
        self.U = nn.Linear(self.d_model, self.d_model * 2)

    def init_memory(self, batch_size):
        memory = torch.stack([torch.eye(self.num_slots)] * batch_size)
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((batch_size, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]

        return memory

    def forward_step(self, input, memory):
        memory = memory.reshape(-1, self.num_slots, self.d_model)
        q = memory
        k = torch.cat([memory, input.unsqueeze(1)], 1)
        v = torch.cat([memory, input.unsqueeze(1)], 1)
        next_memory = memory + self.attn(q, k, v)
        next_memory = next_memory + self.mlp(next_memory)

        gates = self.W(input.unsqueeze(1)) + self.U(torch.tanh(memory))
        gates = torch.split(gates, split_size_or_sections=self.d_model, dim=2)
        input_gate, forget_gate = gates
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)

        next_memory = input_gate * torch.tanh(next_memory) + forget_gate * memory
        next_memory = next_memory.reshape(-1, self.num_slots * self.d_model)

        return next_memory

    def forward(self, inputs, memory):
        outputs = []
        for i in range(inputs.shape[1]):
            memory = self.forward_step(inputs[:, i], memory)
            outputs.append(memory)
        outputs = torch.stack(outputs, dim=1)

        return outputs


class EncoderDecoder(AttModel):

    def make_model(self, tgt_vocab):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        rm = RelationalMemory(num_slots=self.rm_num_slots, d_model=self.rm_d_model, num_heads=self.rm_num_heads)
        model = Transformer(
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            Decoder(
                DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout, self.rm_num_slots, self.rm_d_model),
                self.num_layers),
            lambda x: x,
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)),
            rm, fbl=self.fbl)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer):
        super(EncoderDecoder, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.rm_num_slots = args.rm_num_slots
        self.rm_num_heads = args.rm_num_heads
        self.rm_d_model = args.rm_d_model
        self.fbl = args.fbl
        self.clip = args.clip

        if self.clip:
            self.proj_image_clip = nn.Linear(self.d_model, self.d_model)
            self.proj_text_clip = nn.Linear(self.d_model, self.d_model)
            # nn.init.xavier_uniform_(self.proj_image_clip)
            # nn.init.xavier_uniform_(self.proj_text_clip)
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.loss_clip = nn.CrossEntropyLoss()

        tgt_vocab = self.vocab_size + 1

        self.model = self.make_model(tgt_vocab)
        self.logit = nn.Linear(args.d_model, tgt_vocab)
        # self.cls_logit = nn.Linear(args.d_model, 14)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, att_masks)

        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        #att_feats: patch_feats : (B,2*Ns,feat_size) V_s
        #seq: report_ids
        #att_mask is None
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2) 
        # (B, 1, 2*Ns), filled with 1s

        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        #fc_feat: gbl_feats : (B,feat_size) v_g
        #att_feats: patch_feats : (B,2*Ns,feat_size) V_s
        #seq: report_ids
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        #att_feats	(B, 2*Ns, d_model)
        #seq	(B, seq_len - 1) or None
        #att_masks	(B, 1, 2*Ns)
        #seq_mask	(B, 1, seq_len - 1) or None
        out, fore_rep_encoded, target_embed, align_attns = self.model(att_feats, seq, att_masks, seq_mask)
        if self.clip:
            avg_img_feats = torch.mean(att_feats, dim=1)
            clip_mask = (seq.data > 0)
            clip_mask[:, 0] += True
            avg_text_feats = torch.stack([torch.mean(out[i, clip_mask[i]], dim=0) for i in range(out.shape[0])], dim=0)
            img_feats_clip, text_feats_clip = self.proj_image_clip(avg_img_feats), self.proj_text_clip(avg_text_feats)

            img_feats_clip = img_feats_clip / img_feats_clip.norm(dim=-1, keepdim=True)
            text_feats_clip = text_feats_clip / text_feats_clip.norm(dim=-1, keepdim=True)
            #clip_loss = compute_clip_loss(img_feats_clip, text_feats_clip)
            clip_loss = compute_clip_loss2(img_feats_clip, text_feats_clip,self.logit_scale, self.loss_clip)

        else:
            clip_loss = None

        outputs = F.log_softmax(self.logit(out), dim=-1)
        # cls_logits = self.cls_logit(torch.mean(out, dim=1))
        return outputs, fore_rep_encoded, target_embed, align_attns, clip_loss


    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out, attns = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device))
        return out[:, -1], [ys.unsqueeze(0)], attns

# class ProjectionHead(nn.Module):
#     def __init__(
#             self,
#             embedding_dim,
#             projection_dim=CFG.projection_dim,
#             dropout=CFG.dropout
#     ):
#         super().__init__()
#         self.projection = nn.Linear(embedding_dim, projection_dim)
#         self.gelu = nn.GELU()
#         self.fc = nn.Linear(projection_dim, projection_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(projection_dim)
#
#     def forward(self, x):
#         projected = self.projection(x)
#         x = self.gelu(projected)
#         x = self.fc(x)
#         x = self.dropout(x)
#         x = x + projected
#         x = self.layer_norm(x)
#         return x
