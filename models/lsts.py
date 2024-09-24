import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import gaussian_blur
from torch.utils import checkpoint as checkpoint
from timm.layers import DropPath, trunc_normal_
from entmax import entmax15
from einops import rearrange
from functools import reduce
from operator import mul
from scipy import signal
import math


def get_activation_module(activation):
    if isinstance(activation, str):
        if activation == "relu" or activation == 'reglu':
            return nn.ReLU()
        elif activation == "gelu" or activation == 'geglu':
            return nn.GELU()
        elif activation == "swish" or activation == 'swiglu':
            return nn.SiLU()
        else:
            raise ValueError(f"activation={activation} is not supported.")
    elif callable(activation):
        return activation()
    else:
        raise ValueError(f"activation={activation} is not supported.")


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, post_norm=None):
        super().__init__()
        self.scale = dim ** -0.5
        self.max_t, self.max_h, self.max_w = max_seq_len
        
        self.emb_t = nn.Embedding(self.max_t, dim)
        self.emb_h = nn.Embedding(self.max_h, dim)
        self.emb_w = nn.Embedding(self.max_w, dim)
        
        self.post_norm = post_norm(dim) if callable(post_norm) else None

    def forward(self, x):
        t, h, w = x.shape[-3:]

        pos_t = torch.arange(t, device=x.device)
        pos_h = torch.arange(h, device=x.device)
        pos_w = torch.arange(w, device=x.device)
        
        pos_emb_t = self.emb_t(pos_t)
        pos_emb_h = self.emb_h(pos_h)
        pos_emb_w = self.emb_w(pos_w)
        
        pos_emb_t = rearrange(pos_emb_t, 't d -> d t 1 1') * self.scale
        pos_emb_h = rearrange(pos_emb_h, 'h d -> d 1 h 1') * self.scale
        pos_emb_w = rearrange(pos_emb_w, 'w d -> d 1 1 w') * self.scale
        
        x = x + pos_emb_t + pos_emb_h + pos_emb_w
        
        if self.post_norm is not None:
            x = rearrange(x, 'b c t h w -> b t h w c')
            x = self.post_norm(x)
            x = rearrange(x, 'b t h w c -> b c t h w')
        
        return x


class GLU(nn.Module):
    def __init__(self, dim_in, dim_out, activation):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim = -1)
        return x * self.act(gate)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer='swish', drop=0.) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if 'glu' in act_layer:
            self.fc1 = GLU(in_features, hidden_features, get_activation_module(act_layer))
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(in_features, hidden_features),
                get_activation_module(act_layer)
            )
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(1,4,4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x


class PatchShift(nn.Module):
    """Temporal Patch Shift
    W. Xiang, C. Li, B. Wang, X. Wei, X.-S. Hua, and L. Zhang, “Spatiotemporal Self-attention Modeling with Temporal Patch Shift for Action Recognition,” in ECCV, 2022.
    """    
    def __init__(self, n_div=8, inv=False, ratio=1, pattern='C'):
        super(PatchShift, self).__init__()
        self.fold_div = n_div
        self.inv = inv
        self.ratio = ratio
        self.pattern = pattern

    def forward(self, x, batch_size, frame_len):
        x = self.shift(x, fold_div=self.fold_div, inv=self.inv, ratio=self.ratio, batch_size=batch_size,frame_len=frame_len, pattern=self.pattern)
        return x

    @staticmethod
    def shift(x, fold_div=3, inv=False, ratio=0.5, batch_size=8, frame_len=8, pattern='C'):
        B, num_heads, N, c = x.size()
        fold = int(num_heads * ratio)
        feat = x
        N_sqrt = int(math.sqrt(N))
        feat = feat.view(batch_size, frame_len, -1, num_heads, N_sqrt, N_sqrt, c)
        out = feat.clone()
        multiplier = 1
        stride = 1
        if inv:
            multiplier = -1

        if pattern == 'A':
            out[:, : ,:, :fold, 0::2, 1::2,:] = torch.roll(feat[:, :, :, :fold,0::2,1::2,:], shifts=multiplier*stride, dims=1)  
            out[:, : ,:, :fold, 1::2, 0::2,:] = torch.roll(feat[:, :, :, :fold,1::2,0::2,:], shifts=-1*multiplier*stride, dims=1)  
        
        elif pattern == 'B':
            out[:, : ,:, :fold, 0::2, 1::2,:] = torch.roll(feat[:, :, :, :fold,0::2,1::2,:], shifts=multiplier*stride, dims=1)  
            out[:, : ,:, :fold, 1::2, 0::2,:] = torch.roll(feat[:, :, :, :fold,1::2,0::2,:], shifts=-1*multiplier*stride, dims=1) 
            out[:, : ,:, :fold, 1::2, 1::2,:] = torch.roll(feat[:, :, :, :fold,1::2,1::2,:], shifts=2*multiplier*stride, dims=1) 
        
        elif pattern == 'C':
            ## Pattern C
            out[:, : ,:, :fold, 0::3, 0::3,:] = torch.roll(feat[:, :, :, :fold,0::3,0::3,:], shifts=-4*multiplier*stride, dims=1)  
            out[:, : ,:, :fold, 0::3, 1::3,:] = torch.roll(feat[:, :, :, :fold,0::3,1::3,:], shifts=multiplier*stride, dims=1)  
            out[:, : ,:, :fold, 1::3, 0::3,:] = torch.roll(feat[:, :, :, :fold,1::3,0::3,:], shifts=-multiplier*stride, dims=1) 
            out[:, : ,:, :fold, 0::3, 2::3,:] = torch.roll(feat[:, :, :, :fold,0::3,2::3,:], shifts=2*multiplier*stride, dims=1)
            out[:, : ,:, :fold, 2::3, 0::3,:] = torch.roll(feat[:, :, :, :fold,2::3,0::3,:], shifts=-2*multiplier*stride, dims=1)
            out[:, : ,:, :fold, 1::3, 2::3,:] = torch.roll(feat[:, :, :, :fold,1::3,2::3,:], shifts=3*multiplier*stride, dims=1) 
            out[:, : ,:, :fold, 2::3, 1::3,:] = torch.roll(feat[:, :, :, :fold,2::3,1::3,:], shifts=-3*multiplier*stride, dims=1)
            out[:, : ,:, :fold, 2::3, 2::3,:] = torch.roll(feat[:, :, :, :fold,2::3,2::3,:], shifts=4*multiplier*stride, dims=1) 
        
        elif pattern == 'D':
            out[:, : ,:, :fold, 0::4, 1::4,:] = torch.roll(feat[:, :, :, :fold,0::4,1::4,:], shifts=multiplier*stride, dims=1)  
            out[:, : ,:, :fold, 0::4, 2::4,:] = torch.roll(feat[:, :, :, :fold,0::4,2::4,:], shifts=2*multiplier*stride, dims=1) 
            out[:, : ,:, :fold, 0::4, 3::4,:] = torch.roll(feat[:, :, :, :fold,0::4,3::4,:], shifts=3*multiplier*stride, dims=1)
            out[:, : ,:, :fold, 1::4, 0::4,:] = torch.roll(feat[:, :, :, :fold,1::4,0::4,:], shifts=-1*multiplier*stride, dims=1)  
            out[:, : ,:, :fold, 1::4, 1::4,:] = torch.roll(feat[:, :, :, :fold,1::4,1::4,:], shifts=-7*multiplier*stride, dims=1)  
            out[:, : ,:, :fold, 1::4, 2::4,:] = torch.roll(feat[:, :, :, :fold,1::4,2::4,:], shifts=4*multiplier*stride, dims=1) 
            out[:, : ,:, :fold, 1::4, 3::4,:] = torch.roll(feat[:, :, :, :fold,1::4,3::4,:], shifts=5*multiplier*stride, dims=1)
            out[:, : ,:, :fold, 2::4, 0::4,:] = torch.roll(feat[:, :, :, :fold,2::4,0::4,:], shifts=2*multiplier*stride, dims=1)  
            out[:, : ,:, :fold, 2::4, 1::4,:] = torch.roll(feat[:, :, :, :fold,2::4,1::4,:], shifts=-4*multiplier*stride, dims=1)  
            out[:, : ,:, :fold, 2::4, 2::4,:] = torch.roll(feat[:, :, :, :fold,2::4,2::4,:], shifts=7*multiplier*stride, dims=1) 
            out[:, : ,:, :fold, 2::4, 3::4,:] = torch.roll(feat[:, :, :, :fold,2::4,3::4,:], shifts=6*multiplier*stride, dims=1)
            out[:, : ,:, :fold, 3::4, 0::4,:] = torch.roll(feat[:, :, :, :fold,3::4,0::4,:], shifts=-3*multiplier*stride, dims=1)  
            out[:, : ,:, :fold, 3::4, 1::4,:] = torch.roll(feat[:, :, :, :fold,3::4,1::4,:], shifts=-5*multiplier*stride, dims=1)  
            out[:, : ,:, :fold, 3::4, 2::4,:] = torch.roll(feat[:, :, :, :fold,3::4,2::4,:], shifts=-6*multiplier*stride, dims=1) 
            out[:, : ,:, :fold, 3::4, 3::4,:] = torch.roll(feat[:, :, :, :fold,3::4,3::4,:], shifts=8*multiplier*stride, dims=1)
        
        else:
            raise ValueError(f'Unknown pattern type {pattern}')
        
        out = out.view(B,num_heads, N,c)
        return out


class TemporalShift(nn.Module):
    """Temporal Channel Shift
    J. Lin, C. Gan, and S. Han, “TSM: Temporal Shift Module for Efficient Video Understanding,” in ICCV, 2019.
    """    
    def __init__(self, n_div=8):
        super(TemporalShift, self).__init__()
        self.fold_div = n_div

    def forward(self, x, batch_size, frame_len):
        x = self.shift(x, fold_div=self.fold_div, batch_size=batch_size, frame_len=frame_len)
        return x

    @staticmethod
    def shift(x, fold_div=8, batch_size=8, frame_len=8):
        B, num_heads, N, c = x.size()
        fold = c // fold_div
        feat = x
        feat = feat.view(batch_size, frame_len,-1, num_heads, N, c)
        out = feat.clone()

        out[:, 1: ,:, :, :, :fold] = feat[:, :-1, :, :, :, :fold]  # shift left
        out[:, :-1 ,:, :, :, fold:2*fold] = feat[:, 1:, :, :, :, fold:2*fold]  # shift right 

        out = out.view(B, num_heads,N,c)

        return out
    

class PeriodicShift(nn.Module):
    """Periodic Channel Shift proposed in this study.
    Here, we implement PCS via F.conv1d for better efficiency.

    Args:
        dim (int): Number of channels
        num_heads (int): Number of attention heads in MSA
        n_div (int): Number of folds (F)
        t_shift (int): Temporal deviation of the dynamic attention window (T_D)
        t_span (int): Temporal span of the dynamic attention window (T_W)
    """    
    def __init__(self, dim, num_heads, n_div=8, t_shift=20, t_span=11):
        super(PeriodicShift, self).__init__()
        
        c = dim // num_heads
        
        self.dim = dim
        self.fold_div = n_div
        self.fold = c // n_div
        self.num_heads = num_heads
        self.t_shift = t_shift
        self.t_span = t_span
        self.t_cut = self.t_shift+1-self.t_span

        assert self.t_span <= self.t_shift + 1, 't_span should be smaller than t_shift + 1'
        
        self.forward_pooling = nn.Linear(dim, 1)
        self.backward_pooling = nn.Linear(dim, 1)
        self.forward_window_gen = nn.Linear(dim, t_span*n_div)
        self.backward_window_gen = nn.Linear(dim, t_span*n_div)
    
    
    def shift(self, x, batch_size, num_heads, N, direction):
        
        if direction == 'forward':
            pooling = self.forward_pooling
            window_gen = self.forward_window_gen
            padding = (self.t_shift, 0)
        elif direction == 'backward':
            pooling = self.backward_pooling
            window_gen = self.backward_window_gen
            padding = (0, self.t_shift)
        else:
            raise ValueError(f'Unknown direction {direction}')
        
        k = x.shape[2]
        feat = rearrange(x, 'b t k h n (c f) -> (k n h c) (b f) t', f=self.fold_div)
        feat = F.pad(feat, padding)
        
        # Dynamic attention window generation
        x_ = rearrange(x, 'b t k h n c -> b (k n t) (h c)')
        attn = pooling(x_)
        pooled = (entmax15(attn, dim=1) * x_).sum(dim=1, keepdim=True)
        window = window_gen(pooled)
        window = rearrange(window, 'b 1 (T f) -> (b f) 1 T', T=self.t_span)
        window = entmax15(window, dim=-1)
        
        # Information extraction
        feat = F.conv1d(feat, window, groups=batch_size*self.fold_div)
        
        # Information aggregation
        feat = rearrange(feat, '(k n h c) (b f) t -> b t k h n c f', b=batch_size, k=k, n=N, h=num_heads, f=self.fold_div).mean(dim=-1)
        feat = feat[:, :-self.t_cut] if direction == 'forward' else feat[:, self.t_cut:]
        
        return feat

    def forward(self, x, batch_size, frame_len):
        
        B, num_heads, N, c = x.size()
        feat = x
        feat = feat.view(batch_size, frame_len, -1, num_heads, N, c)
        out = feat.clone()
                
        forward_feat = self.shift(feat, batch_size, num_heads, N, direction='forward')
        backward_feat = self.shift(feat, batch_size, num_heads, N, direction='backward')
        
        # Temporal shift
        out[..., :self.fold] = forward_feat
        out[..., self.fold:2*self.fold] = backward_feat
        
        out = out.view(B, num_heads, N, c)

        return out


class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0., shift=False, shift_type='tps', pattern='C', t_shift=20, t_span=11):

        super().__init__()
        self.dim = dim
        ## for bayershift
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        self.shift = shift
        self.shift_type = shift_type
        self.pattern = pattern
        self.attn_drop = attn_drop

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.shift and self.shift_type == 'tps':
            self.shift_op = PatchShift(self.num_heads, False, 1, pattern)
            self.shift_op_back = PatchShift(self.num_heads, True, 1, pattern)
        elif self.shift and self.shift_type == 'tcs':
            self.shift_op = TemporalShift(8) 
        elif self.shift and self.shift_type == 'pcs':
            self.shift_op = PeriodicShift(dim, num_heads, n_div=8, t_shift=t_shift, t_span=t_span)

    def forward(self, x, batch_size=8, frame_len=8):
        B_, N, C = x.shape
                
        if self.shift:
            x = x.view(B_, N, self.num_heads, C//self.num_heads).permute(0,2,1,3)

            x = self.shift_op(x, batch_size, frame_len)
            x = x.permute(0,2,1,3).reshape(B_, N, C)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop if self.training else 0.)

        if self.shift and self.shift_type=='tps':
             x = self.shift_op_back(attn, batch_size, frame_len).transpose(1, 2).reshape(B_, N, C)
        else:
            x = attn.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size=(1,5,5), shift_size=(0,0,0),
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer='swish', norm_layer=nn.LayerNorm, use_checkpoint=False, shift=False, shift_type='tps', pattern='C', t_shift=20, t_span=11):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint=use_checkpoint
        self.shift = shift
        self.shift_type = shift_type
        self.pattern = pattern

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            shift=self.shift, shift_type=self.shift_type, pattern=self.pattern, t_shift=t_shift, t_span=t_span)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, batch_size=B, frame_len=D)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size+(C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 >0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x)
        else:
            x = self.forward_part1(x)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class LSTSBasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 act_layer,
                 depth,
                 num_heads,
                 window_size=(1,5,5),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 shift_type='tps',
                 t_shift=20,
                 t_span=11
                 ):
        
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.shift_type = shift_type
        
        # Short-term blocks use TPS
        self.short_term_blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                act_layer=act_layer,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0,0,0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                shift = True,
                shift_type = shift_type,
            )
            for i in range(depth // 2)])
        
        # Long-term blocks use PCS
        self.long_term_blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                act_layer=act_layer,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0,0,0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                shift = True,
                shift_type = 'pcs',
                t_shift=t_shift,
                t_span=t_span
            )
            for i in range(depth // 2)])
        
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        x = rearrange(x, 'b c d h w -> b d h w c')
        for short_term_block, long_term_block in zip(self.short_term_blocks, self.long_term_blocks):
            x = short_term_block(x)
            x = long_term_block(x)
        
        x = x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x


class LSTS(nn.Module):
    
    def __init__(
        self,
        patch_size=(1,16,16),
        input_resolution=128,
        depth=12,
        embed_dim=192,
        act_layer='swish',
        num_heads=8,
        window_size=(1,4,4),
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        patch_norm=False,
        post_pos_norm=True,
        use_checkpoint=False,
        t_shift=20,
        t_span=11,
        M=5,
        tau=0.5,
        **kwargs,
    ):
        super().__init__()
        
        self.input_resolution = input_resolution
        self.num_layers = depth
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.M = M
        self.tau = tau
        
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=9, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
        self.pos_embed = AbsolutePositionalEmbedding(
            embed_dim,
            max_seq_len=(1800, math.ceil(input_resolution/patch_size[1]), math.ceil(input_resolution/patch_size[2])),
            post_norm=norm_layer if post_pos_norm else None,
        )
                
        self.pos_drop = nn.Dropout(p=drop_rate)

        drop_path_rate = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.layers = LSTSBasicLayer(
            dim=embed_dim,
            act_layer=act_layer,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path_rate,
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            shift_type='tps',
            t_shift=t_shift,
            t_span=t_span,
        )
        
        self.norm = norm_layer(embed_dim)
        
        self.out_pooling = nn.Linear(embed_dim, 1)
                
        self.out_fc = MLP(embed_dim, embed_dim, 1, act_layer)
        
        self.bn = nn.BatchNorm3d(9)
        self.register_buffer('rot', torch.tensor([[0, 1, -1], [-2, 1, 1]], dtype=torch.float))
        
        self.shift_probs = signal.windows.exponential(M, center=0, tau=tau, sym=False)
        self.shift_probs = self.shift_probs / self.shift_probs.sum()
        self.mask_base = None
        
        self.apply(_init_weights)
    
    def preprocess(self, x):
        
        x = rearrange(x, 'n d c h w -> n c d h w')

        N, C, D, H, W = x.shape
        
        # MPOS
        x_temp = x / x.mean(dim=2, keepdim=True)
        x_temp = rearrange(x_temp, 'n c d h w -> n h w c d')
        x_proj = torch.matmul(self.rot, x_temp)
        x_proj = rearrange(x_proj, 'n h w c d -> (n d) c h w')
        
        x_mpos = []
        for i in range(3):
            x_filt = gaussian_blur(x_proj, kernel_size=int(self.input_resolution / (2**(i+2))-1))
            x_filt = rearrange(x_filt, '(n d) c h w -> n c d h w', n=N)
            s0 = x_filt[:, :1]
            s1 = x_filt[:, 1:]
            mpos = s0 + (s0.std(dim=2, keepdim=True) / s1.std(dim=2, keepdim=True)) * s1
            mpos = (mpos - mpos.mean(dim=2, keepdim=True)) / mpos.std(dim=2, keepdim=True)
            x_mpos.append(mpos)
        
        # TSAug
        if self.training:
            if self.mask_base is None:
                self.mask_base = torch.ones(N, 1, D, 1, 1, device=x.device)
            for s, p in enumerate(self.shift_probs):
                if s == 0:
                    continue
                mask = torch.bernoulli(self.mask_base * p).bool()
                x = torch.where(mask, torch.roll(x, dims=2, shifts=s), x)
        
        # Raw
        x_norm = (x - 0.5) * 2

        # NDF
        x_diff = x.clone()
        x_diff[:, :, :-1] = x[:, :, 1:] - x[:, :, :-1]
        x_diff[:, :, :-1] = x_diff[:, :, :-1] / (x[:, :, 1:] + x[:, :, :-1])
        torch.nan_to_num_(x_diff, nan=0., posinf=0., neginf=0.)
        x_diff[:, :, :-1] = x_diff[:, :, :-1] / x_diff[:, :, :-1].std(dim=(1, 2, 3, 4), keepdim=True)
        x_diff[:, :, -1:].fill_(0.)
        
        x = torch.cat([x_norm, x_diff] + x_mpos, dim=1)
        
        torch.nan_to_num_(x, nan=0., posinf=0., neginf=0.)
        
        x = self.bn(x)
        
        return x
    
    def forward(self, x):
        """Forward function."""
        
        x = self.preprocess(x)
                
        x = self.patch_embed(x)
                                
        x = self.pos_drop(self.pos_embed(x))
        
        x = self.layers(x)

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n d (h w) c')
        
        attn_score = self.out_pooling(x)
        attn_score = entmax15(attn_score, dim=2)
        x = (x * attn_score).sum(dim=2)
        
        x = self.out_fc(x)
        x = (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True)
        
        return x
    
    @torch.no_grad()
    def predict(self, x):
        return self(x)


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
        