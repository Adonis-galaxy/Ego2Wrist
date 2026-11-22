"""Wan2.2-Fun-5B-InP as a head -> (left, right) wrist-view generator.

Layout ("latent panel"): each view is VAE-encoded on its own (so the causal VAE never mixes views),
and the three latents are concatenated along latent WIDTH as [head | left | right]. The DiT then
sees an ordinary video latent; its 3D RoPE places the views side by side at the same time index,
so no position-encoding change is needed.

Conditioning uses the Fun-InP input slots exactly as the base model was trained to read them:
  y = [mask (4 ch) | reference latent (48 ch)]
with mask = 1 and reference = the head latent on the head columns, mask = 0 and reference = the
latent of a mid-grey video on the wrist columns. The head columns of the noisy input are the clean
head latent and their per-token timestep is 0; the loss covers only the wrist columns.
"""
import os
import sys

import torch

_THIRD_PARTY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party")
if _THIRD_PARTY not in sys.path:                    # vendored Wan2.2 model code (third_party/training)
    sys.path.insert(0, _THIRD_PARTY)

from training.wan.models import AutoencoderKLWan3_8, WanT5EncoderModel, WanTransformer3DModel  # noqa: E402
from training.wan.utils.lora_utils import create_network  # noqa: E402

from e2w.paths import BASE_MODEL as BASE  # noqa: E402

SHIFT = 12.0  # scheduler shift of the 5B-InP pipeline, used for training and sampling
N_VIEWS = 3   # [head | left | right]
PATCH = (1, 2, 2)

VAE_KW = dict(vae_type="AutoencoderKLWan3_8", vae_subpath="Wan2.2_VAE.pth",
              temporal_compression_ratio=4, spatial_compression_ratio=16)
TF_KW = dict(transformer_subpath="./", transformer_combination_type="single",
             dict_mapping={"in_dim": "in_channels", "dim": "hidden_size"})
T5_KW = dict(text_encoder_subpath="models_t5_umt5-xxl-enc-bf16.pth", tokenizer_subpath="google/umt5-xxl",
             text_length=512, vocab=256384, dim=4096, dim_attn=4096, dim_ffn=10240, num_heads=64,
             num_layers=24, num_buckets=32, shared_pos=False, dropout=0.0)


def load_transformer(dtype=torch.bfloat16):
    return WanTransformer3DModel.from_pretrained(
        os.path.join(BASE, TF_KW["transformer_subpath"]), transformer_additional_kwargs=TF_KW, torch_dtype=dtype)


def load_vae():
    return AutoencoderKLWan3_8.from_pretrained(os.path.join(BASE, VAE_KW["vae_subpath"]), additional_kwargs=VAE_KW).eval()


def load_text_encoder(dtype=torch.bfloat16):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(BASE, T5_KW["tokenizer_subpath"]))
    enc = WanT5EncoderModel.from_pretrained(os.path.join(BASE, T5_KW["text_encoder_subpath"]),
                                            additional_kwargs=T5_KW, torch_dtype=dtype).eval()
    return tok, enc


def add_lora(transformer, rank=64, alpha=64):
    net = create_network(1.0, rank, alpha, None, transformer, neuron_dropout=None, skip_name=None)
    net.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
    return net


@torch.no_grad()
def encode_views(vae, views):
    """views [B, V, 3, F, H, W] in [-1, 1] -> panel latent [B, C, f, h, V*w]."""
    lats = []
    for v in range(views.shape[1]):
        lats.append(torch.cat([vae.encode(views[b:b + 1, v])[0].sample() for b in range(views.shape[0])]))
    return torch.cat(lats, dim=-1)


@torch.no_grad()
def decode_views(vae, panel, n_views=N_VIEWS):
    """panel latent [B, C, f, h, V*w] -> list of V pixel videos [B, 3, F, H, W] in [-1, 1]."""
    w = panel.shape[-1] // n_views
    return [vae.decode(panel[..., v * w:(v + 1) * w]).sample.clamp(-1, 1) for v in range(n_views)]


@torch.no_grad()
def grey_latent(vae, n_frames, height, width, device, dtype):
    z = torch.zeros(1, 3, n_frames, height, width, device=device, dtype=dtype)
    return vae.encode(z)[0].sample()  # [1, C, f, h, w]


def build_condition(z_panel, z_grey, drop_head=None):
    """Fun-InP y = [mask | reference] for a [head | L | R] panel. drop_head: bool tensor [B] (head
    replaced by grey and masked out -> the unconditional branch used for head guidance / the no-head control).
    Returns (y, head latent as it enters the head columns)."""
    B, C, f, h, W = z_panel.shape
    w = W // N_VIEWS
    head = z_panel[..., :w]
    mask = torch.zeros(B, 4, f, h, W, device=z_panel.device, dtype=z_panel.dtype)
    mask[..., :w] = 1.0
    if drop_head is not None and drop_head.any():
        head = torch.where(drop_head.view(B, 1, 1, 1, 1), z_grey.expand_as(head), head)
        mask[drop_head, :, :, :, :w] = 0.0
    ref = torch.cat([head] + [z_grey.expand(B, -1, -1, -1, -1)] * (N_VIEWS - 1), dim=-1)
    return torch.cat([mask, ref], dim=1), head


def token_timesteps(t, f, h, W, head_zero=True):
    """Per-token timesteps [B, f*hp*Wp] (frame-major, then row, then column, as the DiT flattens)."""
    B = t.shape[0]
    hp, Wp = h // PATCH[1], W // PATCH[2]
    tt = t.view(B, 1, 1, 1).expand(B, f, hp, Wp).clone()
    if head_zero:
        tt[..., :Wp // N_VIEWS] = 0
    return tt.reshape(B, -1)


def seq_len_of(f, h, W):
    return f * (h // PATCH[1]) * (W // PATCH[2])


def shift_sigma(u, shift=SHIFT):
    return shift * u / (1 + (shift - 1) * u)
