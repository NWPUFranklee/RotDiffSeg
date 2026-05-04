import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import sys
sys.path.append("hymavi/network_architecture/hymavi_amos")

from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from mamba_ssm import Mamba
from config import config as cf
from timm.models.layers import DropPath, to_3tuple, trunc_normal_

class PatchPositionalEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        """
        Initialize patch-based positional embedding.
        
        Args:
            num_patches (int): Number of patches (sequence length).
            embed_dim (int): Embedding dimension.
        """
        super(PatchPositionalEmbedding, self).__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        # Learned positional embeddings for 0 to num_patches-1
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches, embed_dim))  # Shape: (1, num_patches, embed_dim)
        nn.init.normal_(self.position_embeddings, mean=0.0, std=0.02)  # Initialize embeddings

    def forward(self, input_tensor):
        """
        Add positional embeddings to the input tensor.

        Args:
            input_tensor (Tensor): Input tensor of shape (batch_size, num_patches, embed_dim).

        Returns:
            Tensor: Input tensor with positional embeddings added.
        """
        batch_size, seq_len, embed_dim = input_tensor.shape

        # Ensure the sequence length matches the number of patches
        if seq_len != self.num_patches:
            raise ValueError(f"Sequence length {seq_len} does not match num_patches {self.num_patches}.")

        # Add positional embeddings
        return input_tensor + self.position_embeddings
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len):
        """
        Multi-Head Attention with Relative Positional Encoding.

        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            max_seq_len (int): Maximum sequence length for relative positional bias.
        """
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Relative positional embeddings
        # self.rel_pos_bias = nn.Parameter(torch.randn(max_seq_len, max_seq_len, num_heads))  # (seq_len, seq_len, num_heads)
        # Relative positional embeddings using the earlier logic
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2 * max_seq_len - 1, num_heads)  # Shape: (2L-1, num_heads)
        )

        # Precompute the relative position index
        coords = torch.arange(max_seq_len)
        relative_coords = coords[None, :] - coords[:, None]  # Shape: (L, L)
        relative_coords += max_seq_len - 1  # Shift to make all indices non-negative
        self.register_buffer("relative_position_index", relative_coords)

        # Initialize bias table
        trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Output projection
        # self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(0.3)

    def forward(self, q, k, v):

        #because the input q,k,v have shape (B, D, L) now, need to transpose first
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)


        batch_size, seq_len, _ = q.shape

        # Reshape q, k, v to (batch_size, num_heads, seq_len, head_dim)
        q = q.contiguous().view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.contiguous().view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.contiguous().view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Add relative positional bias
        relative_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_bias = relative_bias.view(self.relative_position_index.size(0), 
                                  self.relative_position_index.size(1), -1).permute(2,0,1)
        attn_scores = attn_scores + relative_bias.unsqueeze(0)

        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1) 
        attn_weights = self.drop(attn_weights)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, v) 
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)  # Combine heads

        return attn_output

class HymbaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        HymbaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        # self.hdz = hidden_size

    def forward(self, hidden_states):

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class BiMambaBlock(nn.Module):
    def __init__(self, d_model, n_state):
        super(BiMambaBlock, self).__init__()
        self.d_model = d_model
        
        self.mamba1 = Mamba(d_model, n_state)
        self.mamba2 = Mamba(d_model, n_state)

        # Norm and feed-forward network layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model//128), #------------------------------------------consider------------------------
            nn.GELU(),
            nn.Linear(d_model//128, d_model) #--------------------------------------consider---------------------------------
        )

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



class SimHymba(nn.Module): #is equal to HymbaDecoderLayer in Hymba
    def __init__(self, embed_dim_att=None, num_attention_heads=None, sequence_len=None, mamba_states=16, hidden_size = 256, value_dim = 96, rms_norm_eps=1e-6, is_start_layer = False, is_att = True, factor=None):
        super().__init__()
        
        if factor==None:
            factor = 64

        self.embed_dim_att = embed_dim_att
        
        # self.reduced_dim = hidden_size//reduce_dim_rate
        # self.reduce_dims = nn.Linear(hidden_size, self.reduced_dim)
        self.hidden_size = hidden_size

        self.intermediate_mamba_size = hidden_size*cf.intermediate_mamba_size_expand_rate
        self.input_norm = HymbaRMSNorm(hidden_size, rms_norm_eps)
        self.value_dim = value_dim

        self.is_att = is_att

        if is_att:
            if not is_start_layer:
                self.proj_to_latent = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size//factor),
                    nn.Linear(hidden_size//factor, self.intermediate_mamba_size + \
                                                (embed_dim_att*2+value_dim), bias=cf.mamba_proj_bias) #output dim = intermediate mamba size + qkv attention size (here, v size is not embed_dim_att, because it must be output the same dimensions with ssm_state of mamba (hidden_size) so that they can be add at the end before the average)
                )
            else:
                self.proj_to_latent = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size//factor),
                    nn.Linear(hidden_size//factor, self.intermediate_mamba_size + \
                                            embed_dim_att, bias=cf.mamba_proj_bias) #output dim = intermediate mamba size + qkv attention size (here, v size is not embed_dim_att, because it must be output the same dimensions with ssm_state of mamba (hidden_size) so that they can be add at the end before the average)
                )

                self.proj_to_latent_kv = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size//factor),
                    nn.Linear(hidden_size//factor,
                                                embed_dim_att + value_dim) #output dim = kv attention size (here, v size is not embed_dim_att, because it must be output the same dimensions with ssm_state of mamba (hidden_size) so that they can be add at the end before the average)
                )

            self.att =  MultiHeadAttention(embed_dim=embed_dim_att, num_heads=num_attention_heads, max_seq_len=sequence_len)
            self.pre_avg_layernorm2 = HymbaRMSNorm(hidden_size, eps=rms_norm_eps)

            self.proj_out_att = nn.Sequential(
                nn.Linear(value_dim, 4),
                nn.Linear(4, hidden_size)
            )
        else:
            self.proj_to_latent = nn.Sequential(
                nn.Linear(hidden_size, hidden_size//factor),
                nn.Linear(hidden_size//factor, self.intermediate_mamba_size, bias=cf.mamba_proj_bias) #output dim = intermediate mamba size + qkv attention size (here, v size is not embed_dim_att, because it must be output the same dimensions with ssm_state of mamba (hidden_size) so that they can be add at the end before the average)
            )
            

        self.ssm = BiMambaBlock(d_model=self.intermediate_mamba_size, n_state=mamba_states)

        self.pre_avg_layernorm1 = HymbaRMSNorm(self.intermediate_mamba_size, eps=rms_norm_eps)
        

        # self.increase_att_dims = nn.Linear(embed_dim_att, self.reduced_dim)
        # self.increase_dims = nn.Linear(hidden_size, hidden_size)

        # self.num_att_heads = num_attention_heads
        # self.seq_len = sequence_len
        
        

    def forward(self, x, x_skip=None):
        if x_skip==None:
            x = self.input_norm(x)
            x = self.proj_to_latent(x).transpose(1,2) # (B, L, D)

            if self.is_att:
                query_states, key_states, value_states, hidden_states = torch.split(x, [self.embed_dim_att, self.embed_dim_att, self.value_dim, self.intermediate_mamba_size], dim=1)
                att_out = self.att(query_states, key_states, value_states)
                att_out = self.proj_out_att(att_out)
            else:
                hidden_states = x

            ssm_out = self.ssm(hidden_states)

            if self.is_att:
                out = ssm_out + att_out
            else:
                out = ssm_out

        else:
            x = self.input_norm(x)
            x = self.proj_to_latent(x).transpose(1,2) # (B, L, D)

            if self.is_att:
                query_states, hidden_states = torch.split(x, [self.embed_dim_att, self.intermediate_mamba_size], dim=1) 
                x_skip = self.proj_to_latent_kv(x_skip)
                key_states, value_states = torch.split(x_skip, [self.embed_dim_att, self.value_dim], dim=-1)
                
                att_out = self.att(query_states, key_states, value_states)
                att_out = self.proj_out_att(att_out)
            else:
                hidden_states = x

            ssm_out = self.ssm(hidden_states)

            if self.is_att:
                out = ssm_out + att_out
            else:
                out = ssm_out

        return out

if __name__ == "__main__":
    # Example usage
    batch_size = 2
    seq_len = 64
    embed_dim = 64
    num_heads = 4

    model = SimHymba(embed_dim_att=embed_dim, num_attention_heads=num_heads, sequence_len=seq_len, mamba_states=16, hidden_size=256, is_start_layer=True).cuda()
    input_tensor = torch.randn(batch_size, seq_len, 256).cuda()  # (B, L, D)
    output = model(input_tensor)
    print(output.shape)  # Should be (B, L, D)