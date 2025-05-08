"""
Module for Rotary Positional Embeddings (RoPE) in Transformer Models.
"""

import torch
from typing import Tuple


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency components for the rotary positional embeddings.

    Arguments:
        - dim (int): The dimension of the embedding.
        - end (int): The length of the sequence for which frequencies are computed.
        - theta (float): A scaling factor for the frequencies, default is 10000.0.

    Returns:
        - freqs_cis (torch.Tensor): A tensor of complex numbers representing the precomputed frequencies.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape the frequency components for broadcasting with the input tensor.

    Arguments:
        - freqs_cis (torch.Tensor): The frequency components tensor.
        - x (torch.Tensor): The input tensor to which the frequencies will be applied.

    Returns:
        - torch.Tensor: The reshaped frequency components tensor.
    """
    # Reference: https://github.com/meta-llama/llama3/blob/main/llama/model.py#L61
    ndim = x.ndim
    shape = [d if i in (0, 2, ndim - 1) else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # TODO: Check if any other positional embedding is relevant to this use case
    """
    Apply rotary positional embeddings to the query and key tensors.

    Arguments:
        - xq (torch.Tensor): The query tensor.
        - xk (torch.Tensor): The key tensor.
        - freqs_cis (torch.Tensor): The precomputed frequency components.

    Returns:
        - Tuple[torch.Tensor, torch.Tensor]: The transformed query and key tensors.

    Note:
        For more information on rotary positional embeddings, refer to the blog post:
        https://spaces.ac.cn/archives/8265/ or paper https://arxiv.org/abs/2104.09864
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)