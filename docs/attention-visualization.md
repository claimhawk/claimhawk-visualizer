# Attention Visualization Methodology

## Overview

The LoRA Attention Visualizer captures and displays two types of attention patterns from Qwen3-VL:

1. **Edge Detection** (Cyan) - Vision encoder self-attention
2. **Decision Focus** (Red) - LM decoder attention to vision tokens (WIP)

## Edge Detection Heatmap

### Source
Vision encoder self-attention from the **last 2 layers** (layers 25 and 26 of 27) of the Qwen3-VL vision transformer.

### Method

1. **Hook Registration**: Forward hooks are registered on `model.visual.blocks[25].attn` and `model.visual.blocks[26].attn`

2. **Attention Computation**: During the forward pass, we intercept the hidden states and compute attention:
   ```python
   # Get QKV projection
   qkv = module.qkv(hidden)  # (batch, seq, 3*heads*head_dim)

   # Reshape to separate Q, K, V
   qkv = qkv.view(batch, seq_len, 3, num_heads, head_dim)
   q, k = qkv[:, :, 0], qkv[:, :, 1]

   # Compute attention scores
   scale = head_dim ** -0.5
   attn = softmax(Q @ K^T / sqrt(d))  # (batch, heads, seq, seq)
   ```

3. **Saliency Extraction**: For each patch position, we sum how much attention it receives from all other patches:
   ```python
   # Average over batch and heads, then sum columns
   attn_received = attn.mean(dim=(0, 1)).sum(dim=0)  # (num_patches,)
   ```
   This gives higher values to patches that other patches attend to strongly (edges, boundaries).

4. **Aggregation**: Results from both layers are averaged and normalized to [0, 1].

### Why It Shows Edges

Vision transformer self-attention naturally highlights boundaries because:
- Patches at edges have distinct features from their neighbors
- Other patches attend to these distinctive features to understand spatial relationships
- High attention received = visually salient (usually edges/boundaries)

### Grid Mapping

The attention values are mapped to a spatial grid matching Qwen3-VL's vision encoding:
- Grid dimensions from `image_grid_thw` (e.g., 98×174 for a typical image)
- Each cell in the heatmap corresponds to one vision patch token

## Decision Focus Heatmap (WIP)

### Goal
Show where the model "looked" in the image when generating click coordinates.

### Method (Planned)
1. Hook the last LM decoder layer's self-attention
2. After generation, do a forward pass with the full sequence
3. Extract attention from coordinate tokens (e.g., "107", "243") to vision tokens
4. Average these attention patterns to show decision-relevant regions

### Current Status
Finding the correct path to LM decoder layers in the Qwen3-VL + PeftModel structure.

## Color Scheme

- **Cyan** (Edge Detection): `rgba(0, 200-255, 255, alpha)` where alpha scales with value
- **Red/Yellow** (Decision Focus): `rgba(255, 0-180, 0, alpha)` where alpha scales with value
- **Green** (Click Target): Crosshair showing predicted coordinates
- **Dark Blue** (Crosshair): `#1e3a5f` for coordinate marker

## Version History

- v2.8.x: Working edge detection with Q*K attention from vision encoder
- Previous attempts: Feature magnitude (uniform), gradient saliency (diffuse)
