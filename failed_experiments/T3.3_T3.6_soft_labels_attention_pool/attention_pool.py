"""
Query attention pooling (T3.6, Experiment_Design.md Tier 3) -- replaces global-average
pooling as the bridge between the backbone's spatial feature map and the country/cell/offset
heads.

Motivation (literature.md, Clark et al. 2023, "GeoDecoder"): the current model
(`hierarchical_geo_classifier.py`) feeds every head the SAME globally-average-pooled vector.
GAP collapses the whole spatial feature map (e.g. a 7x7 grid of tokens at 224px) into one
vector before any head sees it, forcing country-level cues (terrain, vegetation, light) and
fine within-country cues (signage text, architectural style, plate design) to share one
representation, diluted further by irrelevant background tokens (sky, road) that GAP weighs
identically to the discriminative ones.

This module ports the paper's core mechanism -- learned query vectors that cross-attend over
the backbone's un-pooled spatial tokens -- at a scale that fits our 5M-param budget, not
their Swin-Transformer, 4.72M-image scale:
  - **1 shared cross-attention stage**: both queries (country, cell/offset) attend jointly
    over the same projected spatial tokens with shared weights -- "hierarchy independent"
    layer in the paper's terms, general visual features extracted once.
  - **2 specialized FFN branches**: after the shared stage, each query gets its own small
    feed-forward network -- "hierarchy dependent" stage, where a query becomes specifically
    "the country representation" vs. "the cell/offset representation," not just an attention
    output.
  - The paper uses 6 shared + 2 specialized transformer layers; this uses 1 shared
    cross-attention layer + 2 specialized FFNs, sized down to fit alongside a <=5M-param
    backbone instead of a separate large transformer decoder.

Offset shares the cell query's output (not a third query) -- both need the same fine local
visual cues (exact signage/architecture detail), and a third query would cost params without
an evidenced reason to expect it helps; add one only if error analysis specifically motivates
it (this project's standing rule, Experiment_Design.md's guiding principle).

Param-budget note (see also `src/protocol/param_check.py`): a naive multi-head attention
block operating at a backbone's native feature width (e.g. EfficientNet-B0's 1280) would cost
`4 * 1280**2` ~= 6.5M params in the Q/K/V/O projections alone -- over the whole model's budget
by itself. The 1x1 conv projection down to `attn_dim` (default 128) before attention is what
keeps this affordable; verified via `check_compliance_hierarchical_model`, not assumed.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QueryAttentionPool(nn.Module):
    def __init__(
        self,
        in_channels: int,
        attn_dim: int = 128,
        n_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert attn_dim % n_heads == 0, f"attn_dim={attn_dim} must divide by n_heads={n_heads}"
        self.attn_dim = attn_dim

        # Shared token projection: backbone channels -> attention dim. This is what keeps the
        # attention block cheap (see module docstring) -- attends in a reduced space, not the
        # backbone's native (much wider) feature width.
        self.token_proj = nn.Conv2d(in_channels, attn_dim, kernel_size=1)

        # Two learned queries: index 0 = country, index 1 = cell/offset. Small init (0.02 std)
        # matches standard transformer query/positional-embedding initialization practice.
        self.queries = nn.Parameter(torch.randn(2, attn_dim) * 0.02)

        # Shared ("hierarchy independent") stage: both queries cross-attend jointly over the
        # same projected tokens, one shared FFN.
        self.shared_attn = nn.MultiheadAttention(
            attn_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.shared_norm1 = nn.LayerNorm(attn_dim)
        self.shared_ffn = nn.Sequential(
            nn.Linear(attn_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, attn_dim)
        )
        self.shared_norm2 = nn.LayerNorm(attn_dim)

        # Specialized ("hierarchy dependent") stage: separate small FFN per query, so each
        # query's representation diverges to serve its own head rather than staying identical
        # copies of the same shared-stage output.
        self.country_ffn = nn.Sequential(
            nn.Linear(attn_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, attn_dim)
        )
        self.country_norm = nn.LayerNorm(attn_dim)
        self.cell_ffn = nn.Sequential(
            nn.Linear(attn_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, attn_dim)
        )
        self.cell_norm = nn.LayerNorm(attn_dim)

    def forward(self, feature_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`feature_map`: (B, in_channels, H, W) -- a backbone's pre-pool spatial output
        (`backbone.forward_features(images)`, NOT the already-pooled `backbone(images)`).
        Returns `(country_repr, cell_repr)`, each (B, attn_dim)."""
        batch_size = feature_map.shape[0]
        tokens = self.token_proj(feature_map).flatten(2).transpose(1, 2)  # (B, H*W, attn_dim)

        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 2, attn_dim)
        attn_out, _ = self.shared_attn(queries, tokens, tokens)
        shared = self.shared_norm1(queries + attn_out)
        shared = self.shared_norm2(shared + self.shared_ffn(shared))

        country_q, cell_q = shared[:, 0], shared[:, 1]  # each (B, attn_dim)
        country_repr = self.country_norm(country_q + self.country_ffn(country_q))
        cell_repr = self.cell_norm(cell_q + self.cell_ffn(cell_q))
        return country_repr, cell_repr
