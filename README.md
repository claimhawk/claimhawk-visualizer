# LoRA Adapter Visualizer

Interactive web UI for visualizing attention patterns in Qwen3-VL LoRA adapters.

## Features

- **Attention Heatmaps**: Visualize which parts of the image the model attends to
- **LoRA Comparison**: Compare attention patterns between base model and different adapters
- **Layer Explorer**: Inspect attention at different transformer layers and heads

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Next.js UI    │────▶│   FastAPI       │────▶│   Modal GPU     │
│   (Frontend)    │     │   (Backend)     │     │   (Inference)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Quick Start

### Backend (FastAPI)

```bash
cd backend
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
uvicorn app:app --reload
```

### Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

## API Endpoints

- `POST /api/analyze` - Run inference with attention extraction
- `GET /api/adapters` - List available LoRA adapters
- `GET /api/layers` - Get layer/head metadata

## Notes

- Uses `attn_implementation="eager"` (not flash_attention_2) to enable attention output
- Attention maps are upsampled to match original image resolution
- GPU inference runs on Modal for heavy lifting

---

<div align="center">

### 🚀 We're Hiring

**ClaimHawk** builds computer-use agents that automate real work using vision-language models.

If you have a passion for machine learning (and some real background) and want to see the path to **100x developer** — we have open intern positions.

**No resumes.** Just shoot an email with your qualifications and passions to:

📧 **hello@claimhawk.app**

</div>
