"""
Module for Relative Position Bias (RPB) in Transformer Models (Positional Encoding).
"""

import torch
import torch.nn as nn

class RelativePositionBias(nn.Module):
    def __init__(self, num_heads: int, max_distance: int):
        """
        Implements Shaw-style learnable relative position bias.

        Args:
            num_heads (int): Number of attention heads.
            max_distance (int): Maximum relative position distance (both positive and negative).
        """
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(2 * max_distance - 1, num_heads)

    def forward(self, q_len: int, k_len: int) -> torch.Tensor:
        """
        Compute relative position bias matrix to add to attention scores.

        Args:
            q_len (int): Length of the query sequence.
            k_len (int): Length of the key sequence.

        Returns:
            Tensor of shape [1, num_heads, q_len, k_len]
        """
        # Create [q_len, k_len] matrix of relative positions
        context_pos = torch.arange(q_len, device=self.relative_attention_bias.weight.device)[:, None]
        memory_pos = torch.arange(k_len, device=self.relative_attention_bias.weight.device)[None, :]
        relative_pos = memory_pos - context_pos  # [q_len, k_len]
        relative_pos = relative_pos.clamp(-self.max_distance + 1, self.max_distance - 1)
        relative_pos += self.max_distance - 1  # Shift to index range [0, 2*max-1]

        # Look up bias embeddings: [q_len, k_len, num_heads]
        bias = self.relative_attention_bias(relative_pos)  # [q_len, k_len, num_heads]
        return bias.permute(2, 0, 1).unsqueeze(0)  # [1, num_heads, q_len, k_len]