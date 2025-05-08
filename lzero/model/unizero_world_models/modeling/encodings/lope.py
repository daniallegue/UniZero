"""
Module for LoPE (Learned Positional Encoding) in Transformer Models.
"""
from typing import Tuple

import torch
import torch.nn as nn

class LoPE(nn.Module):
    """
    Learnable Positional Encoding (LoPE) module.

    Args:
        num_heads (int): Number of attention heads.
        max_distance (int): Maximum relative position distance (both positive and negative).
        max_seq_len (int): Maximum sequence length.
    """

    def __init__(self, num_heads : int, head_dim : int, max_seq_len : int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        self.query_pos = nn.Parameter(torch.randn(max_seq_len, num_heads, max_seq_len, head_dim))
        self.key_pos = nn.Parameter(torch.randn(max_seq_len, num_heads, max_seq_len, head_dim))

    def forward(self, q : torch.Tensor, k : torch.Tensor, seq_len : int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q (Tensor): Query tensor of shape [B, num_heads, T, head_dim]
            k (Tensor): Key tensor of shape [B, num_heads, T, head_dim]
            seq_len (int): Actual sequence length T

        Returns:
            Tuple[Tensor, Tensor]: q and k with positional encodings added
        """

        q_pos = self.query_pos[:seq_len].unsqueeze(0).transpose(1, 2)
        k_pos = self.key_pos[:seq_len].unsqueeze(0).transpose(1, 2)

        q = q + q_pos
        k = k + k_pos
        return q, k
