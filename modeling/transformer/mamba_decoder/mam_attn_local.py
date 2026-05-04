from einops import rearrange
from copy import deepcopy
from torch import nn
import torch
import numpy as np
import torch.nn.functional


import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_3tuple, trunc_normal_

from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from mamba_ssm import Mamba
from .config import config as cf

class ContiguousGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.contiguous()

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RelativePositionBias(nn.Module):
    def __init__(self, seq_len, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2 * seq_len - 1, num_heads)  # Shape: (2L-1, num_heads)
        )
        # Precompute the relative position index
        coords = torch.arange(seq_len)
        relative_coords = coords[None, :] - coords[:, None]  # Shape: (L, L)
        relative_coords += seq_len - 1  # Shift to make all indices non-negative
        self.register_buffer("relative_position_index", relative_coords)
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x):
        # Gather relative position bias based on the precomputed index
        relative_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        return x + relative_bias.view(self.relative_position_index.size(0), 
                                  self.relative_position_index.size(1), -1).permute(2,0,1)  # Shape: (num_heads, L, L)
    
class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, seq_len, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  
        self.num_heads = num_heads
        #assert dim%num_heads==0, f"num_heads {num_heads} must be divisible by dim {dim}"
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.seq_len = seq_len

        '''# define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads)) 

        # get pair-wise relative position index for each token inside the window
        coords_s = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_s, coords_h, coords_w]))  
        coords_flatten = torch.flatten(coords, 1)  
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() 
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= 3 * self.window_size[1] - 1
        relative_coords[:, :, 1] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1) 
        self.register_buffer("relative_position_index", relative_position_index)

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)'''

        self.relative_position_bias = RelativePositionBias(seq_len, num_heads)

        self.softmax = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(0.3)

    def forward(self, q,k,v, mask=None,pos_embed=None):
        #x: B, D, L
        #because the input q,k,v have shape (B, D, L) now, need to transpose first
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        B_, L, _ = q.shape
        
        q = q.contiguous().view(B_, L, self.num_heads, -1).transpose(1, 2)
        k = k.contiguous().view(B_, L, self.num_heads, -1).transpose(1, 2)
        v = v.contiguous().view(B_, L, self.num_heads, -1).transpose(1, 2)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous())

        attn = self.relative_position_bias(attn)
        
        attn = self.softmax(attn)
        attn = self.drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, L, -1).contiguous()

        return x

    
class HymbaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        HymbaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # print("++++++ ", hidden_states.shape)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # print(hidden_states.shape, self.weight.shape)
        return self.weight * hidden_states.to(input_dtype)

class BiMambaBlock(nn.Module):
    def __init__(self, d_model, n_state, factor=128):
        super(BiMambaBlock, self).__init__()
        self.d_model = d_model
        
        self.mamba1 = Mamba(d_model, n_state)
        self.mamba2 = Mamba(d_model, n_state)
        if d_model<=factor:
            factor = factor/4
        #print("---------------- ", factor, d_model)
        # Norm and feed-forward network layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        try:
            self.feed_forward = nn.Sequential(
                nn.Linear(d_model, int(d_model//factor)), #------------------------------------------consider------------------------
                nn.GELU(),
                nn.Linear(int(d_model//factor), d_model) #--------------------------------------consider---------------------------------
            )
        except:
            print("========------- ", d_model, factor, d_model//factor)

    def forward(self, x):
        x = x.transpose(1,2) #(B, L, D)
        # Forward Mamba
        x_norm = self.norm1(x)
        mamba_out_forward = self.mamba1(x_norm)

        # Backward Mamba
        x_flip = torch.flip(x_norm, dims=[1])  # Flip Sequence
        mamba_out_backward = self.mamba2(x_flip)
        mamba_out_backward = torch.flip(mamba_out_backward, dims=[1])  # Flip back

        # Combining forward and backward
        mamba_out = mamba_out_forward + mamba_out_backward

        mamba_out = self.norm2(mamba_out)
        ff_out = self.feed_forward(mamba_out)

        return ff_out


class SimHymbaSwin(nn.Module):
    def __init__(self, embed_dim_att, num_attention_heads, window_size, shift_size, \
                 mamba_states, seq_len, hidden_size = 256, value_dim = 96, rms_norm_eps=1e-6, ith_layer = 0, \
                    qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., view=None, factor=128):
        super().__init__()
        '''ith_layer: order of the simhymbaswin layer (start from 0 to "total number of simhymbaswin layers")
        '''
        
        self.embed_dim_att = embed_dim_att
        
        # self.reduced_dim = hidden_size//reduce_dim_rate
        # self.reduce_dims = nn.Linear(hidden_size, self.reduced_dim)
        self.hidden_size = hidden_size
        self.value_dim = value_dim

        self.intermediate_mamba_size = hidden_size*cf.intermediate_mamba_size_expand_rate
        self.input_norm = HymbaRMSNorm(hidden_size, rms_norm_eps)
        self.proj_to_latent = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//cf.reduce_rate_in_swin),
            nn.Linear(hidden_size//cf.reduce_rate_in_swin, self.intermediate_mamba_size + \
                                        (embed_dim_att*2+value_dim), bias=cf.mamba_proj_bias) #output dim = intermediate mamba size + qkv attention size (here, v size is not embed_dim_att, because it must be output the same dimensions with ssm_state of mamba (hidden_size) so that they can be add at the end before the average)
        )
        # self.att =  MultiHeadAttention(embed_dim=embed_dim_att, num_heads=num_attention_heads, max_seq_len=sequence_len)
        
        # dim=cf.f_maps_dim[ith_layer]
        # self.input_resolution=cf.swin_input_resolutions[ith_layer]
        # num_heads=cf.swin_nheads[ith_layer]
        self.window_size=window_size
        self.shift_size=shift_size
        self.ith_layer = ith_layer

        self.att = WindowAttention(
            hidden_size, seq_len= seq_len, window_size=to_3tuple(self.window_size), num_heads=num_attention_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.ssm = BiMambaBlock(d_model=self.intermediate_mamba_size, n_state=mamba_states, factor=factor)

        self.pre_avg_layernorm1 = HymbaRMSNorm(self.intermediate_mamba_size, eps=rms_norm_eps)
        self.pre_avg_layernorm2 = HymbaRMSNorm(hidden_size, eps=rms_norm_eps)

        self.proj_out_att = nn.Sequential(
            nn.Linear(value_dim, 4),
            nn.Linear(4, hidden_size)
        )
        # self.increase_att_dims = nn.Linear(embed_dim_att, self.reduced_dim)
        # self.increase_dims = nn.Linear(hidden_size, hidden_size)

        self.view = view
        

    def forward(self, x, mask_matrix = None):
        x = self.input_norm(x)
        x = self.proj_to_latent(x).transpose(1,2) # (B, L, D)

        query_states, key_states, value_states, hidden_states = torch.split(x, [self.embed_dim_att, self.embed_dim_att, self.value_dim, self.intermediate_mamba_size], dim=1)

        att_out = self.att(query_states, key_states, value_states)
        att_out = self.proj_out_att(att_out)

        ssm_out = self.ssm(hidden_states)
        out = ssm_out + att_out

        return out

class SimHymbaSwinDec(nn.Module): #is equal to HymbaDecoderLayer in Hymba
    def __init__(self, embed_dim_att, embed_dim_att1, num_attention_heads, window_size, shift_size, \
                 mamba_states, seq_len, hidden_size = 256, hidden_size1 = 256, value_dim = 96, rms_norm_eps=1e-6, ith_layer = 0, \
                    qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., view=None, factor=128):
        super().__init__()
        '''ith_layer: order of the simhymbaswin layer (start from 0 to "total number of simhymbaswin layers")
        '''
        
        self.embed_dim_att = embed_dim_att
        self.embed_dim_att1 = embed_dim_att1
        # self.reduced_dim = hidden_size//reduce_dim_rate
        # self.reduce_dims = nn.Linear(hidden_size, self.reduced_dim)
        self.hidden_size = hidden_size
        self.value_dim = value_dim

        self.intermediate_mamba_size = hidden_size*cf.intermediate_mamba_size_expand_rate
        self.input_norm = HymbaRMSNorm(hidden_size, rms_norm_eps)
        self.input_norm1 = HymbaRMSNorm(hidden_size1, rms_norm_eps)
        self.proj_to_latent = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//cf.reduce_rate_in_swin),
            nn.Linear(hidden_size//cf.reduce_rate_in_swin, self.intermediate_mamba_size + \
                                        embed_dim_att, bias=cf.mamba_proj_bias) #output dim = intermediate mamba size + qkv attention size (here, v size is not embed_dim_att, because it must be output the same dimensions with ssm_state of mamba (hidden_size) so that they can be add at the end before the average)
        )

        self.proj_to_latent_kv = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//cf.reduce_rate_in_swin),
            nn.Linear(hidden_size//cf.reduce_rate_in_swin,
                                        embed_dim_att + value_dim) #output dim = kv attention size (here, v size is not embed_dim_att, because it must be output the same dimensions with ssm_state of mamba (hidden_size) so that they can be add at the end before the average)
        )
        self.proj_to_latent_kv1 = nn.Sequential(
            nn.Linear(hidden_size1, hidden_size1//cf.reduce_rate_in_swin),
            nn.Linear(hidden_size1//cf.reduce_rate_in_swin,
                                        embed_dim_att1 + value_dim) #output dim = kv attention size (here, v size is not embed_dim_att, because it must be output the same dimensions with ssm_state of mamba (hidden_size) so that they can be add at the end before the average)
        )
        # self.att =  MultiHeadAttention(embed_dim=embed_dim_att, num_heads=num_attention_heads, max_seq_len=sequence_len)
        
        # dim=cf.f_maps_dim[ith_layer]
        # self.input_resolution=cf.swin_input_resolutions[ith_layer]
        # num_heads=cf.swin_nheads[ith_layer]
        self.window_size=window_size
        self.shift_size=shift_size
        self.ith_layer = ith_layer

        self.att = WindowAttention(
            hidden_size, seq_len= seq_len, window_size=to_3tuple(self.window_size), num_heads=num_attention_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.ssm = BiMambaBlock(d_model=self.intermediate_mamba_size, n_state=mamba_states, factor=factor)
        self.ssm1 = BiMambaBlock(d_model=self.intermediate_mamba_size, n_state=mamba_states, factor=factor)
        self.pre_avg_layernorm1 = HymbaRMSNorm(self.intermediate_mamba_size, eps=rms_norm_eps)
        self.pre_avg_layernorm2 = HymbaRMSNorm(hidden_size, eps=rms_norm_eps)

        # self.increase_att_dims = nn.Linear(embed_dim_att, self.reduced_dim)
        # self.increase_dims = nn.Linear(hidden_size, hidden_size)

        self.view = view

        self.proj_out_att = nn.Sequential(
            nn.Linear(value_dim, 4),
            nn.Linear(4, hidden_size)
        )
        

    def forward(self, x, x_skip, mask_matrix = None):
        print("self.hidden_size", x.shape)
        x = self.input_norm(torch.cat([x, x_skip], dim=-1))
        # x_skip = self.input_norm1(x_skip)
        x = self.proj_to_latent(x).transpose(1,2) # (B, L, D)

        print("self.embed_dim_att, self.intermediate_mamba_size",self.embed_dim_att, self.intermediate_mamba_size)
        query_states, hidden_states = torch.split(x, [self.embed_dim_att, self.intermediate_mamba_size], dim=1)
        # x_skip = self.proj_to_latent_kv1(x_skip)

        # key_states, value_states = torch.split(x_skip, [self.embed_dim_att1, self.value_dim], dim=-1)

        # print("hidden_states shape:", hidden_states.shape)
        # print("query_states shape:", query_states.shape)
        # print("key_states shape:", key_states.shape)
        # print("value_states shape:", value_states.shape)
        # A_expanded = A.expand(-1, B.size(1), -1, -1)
        # key_states = key_states.repeat(1, 1, 3)
        # att_out = self.att(query_states, key_states, value_states)
        # att_out = self.proj_out_att(att_out)

        ssm_out = self.ssm(hidden_states)
        # att_out = self.ssm1(value_states.transpose(1,2))
        # out = ssm_out + att_out
        out = ssm_out

        return out

if __name__ == "__main__":
    batch_size = 2
    seq_len = 576
    embed_dim = 256
    num_heads = 4

    # model = SimHymbaSwin(embed_dim_att=embed_dim, num_attention_heads=num_heads, window_size=4, shift_size=2, \
    #              mamba_states=16, seq_len=seq_len, hidden_size=256, value_dim=96, ith_layer=0).cuda()
    model = SimHymbaSwinDec(embed_dim_att=embed_dim, num_attention_heads=num_heads, window_size=4, shift_size=2,
                 mamba_states=16, seq_len=seq_len, hidden_size=256, value_dim=96, ith_layer=0).cuda()
    input_tensor = torch.randn(batch_size, seq_len, 256).cuda()  # (B, L, D)
    output = model(input_tensor, input_tensor)  # (B, L, D)
    print(output.shape)  # Should be (B, L, D)  