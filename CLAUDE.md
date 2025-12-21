# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## ⚡ FIRST: Read system.md

**Before doing anything else**, read and execute the startup actions in [`system.md`](./system.md).

That file contains:
- **REQUIRED** background processes you must start immediately
- Git workflow rules (fetch & rebase, never pull)
- Inter-agent communication protocol
- Your role and identity as an autonomous agent

**Do not skip this step.**

---

## Project Overview

LoRA Attention Visualizer - an interactive web UI for visualizing attention patterns in Qwen3-VL LoRA adapters. Helps debug, understand, and present what the adapters learn.

## Architecture

```
visualizer/
├── frontend/        # Next.js web UI
│   └── src/
│       ├── app/           # Next.js app router
│       └── components/    # React components
├── backend/         # FastAPI proxy server
│   └── app.py             # API endpoints
└── modal/           # Modal GPU inference
    └── attention.py       # Attention extraction
```

## Commands

### Frontend

```bash
cd frontend
npm install
npm run dev          # Start dev server on :3000
```

### Backend

```bash
cd backend
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
uvicorn app:app --reload  # Start API on :8000
```

### Modal

```bash
cd modal
modal deploy attention.py  # Deploy to Modal
modal run attention.py     # Test locally
```

## Key Technical Details

### Attention Extraction

- Uses `attn_implementation="eager"` (NOT flash_attention_2) to enable `output_attentions=True`
- Vision tokens are identified by `<|vision_start|>` and `<|vision_end|>` markers
- Attention shape: `(batch, num_heads, seq_len, seq_len)`
- 2×2 spatial pooling in Qwen makes direct pixel mapping approximate

### API Flow

1. Frontend uploads image + query to FastAPI backend
2. Backend calls Modal function with base64 image
3. Modal runs inference with attention capture on A100
4. Attention weights returned and visualized as heatmap overlay

## Code Standards

- TypeScript for frontend
- Python 3.11+ for backend/modal
- Type hints required
