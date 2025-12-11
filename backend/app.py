"""FastAPI backend for LoRA Attention Visualizer."""

import base64
from io import BytesIO
from pathlib import Path
from typing import Annotated

import modal
import yaml
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="LoRA Attention Visualizer API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modal function references
extract_attention = modal.Function.from_name(
    "lora-attention-visualizer", "extract_attention"
)

# Load adapters from config
ADAPTERS_CONFIG = Path(__file__).parent.parent.parent.parent / "config" / "adapters.yaml"


def load_adapters_from_config() -> list[dict]:
    """Load adapter list from adapters.yaml (experts only)."""
    if not ADAPTERS_CONFIG.exists():
        return []

    with open(ADAPTERS_CONFIG) as f:
        config = yaml.safe_load(f)

    adapters = []

    # Add experts only (no router for visualizer)
    if "experts" in config:
        for name, expert in config["experts"].items():
            adapters.append({
                "name": name,
                "description": expert.get("description", ""),
                "type": "expert",
                "label": expert.get("label"),
            })

    return adapters


class AnalyzeRequest(BaseModel):
    """Request body for analyze endpoint."""

    query: str = "What action should I take?"
    adapter_name: str | None = None
    selected_layers: list[int] | None = None
    selected_heads: list[int] | None = None


class AnalyzeResponse(BaseModel):
    """Response from analyze endpoint."""

    output_text: str
    attention_data: list[dict]
    num_layers: int
    num_heads: int
    tokens: list[str]
    vision_token_range: list[int | None]
    image_size: list[int]
    adapter_name: str | None
    coordinates: list[int] | None = None
    generated_tokens: list[str] = []
    coordinate_token_indices: list[int] = []
    vision_grid: list[int] | None = None


class AdapterInfo(BaseModel):
    """LoRA adapter info."""

    name: str
    path: str


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/api/adapters")
async def get_adapters():
    """List available LoRA adapters from config."""
    return load_adapters_from_config()


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    image: Annotated[UploadFile, File()],
    query: Annotated[str, Form()] = "What action should I take?",
    adapter_name: Annotated[str | None, Form()] = None,
    selected_layers: Annotated[str | None, Form()] = None,
    selected_heads: Annotated[str | None, Form()] = None,
):
    """
    Analyze an image with attention extraction.

    Upload an image and get back attention heatmaps showing
    which parts of the image the model focuses on.
    """
    # Read and encode image
    image_bytes = await image.read()
    image_b64 = base64.b64encode(image_bytes).decode()

    # Parse layer/head selections
    layers = None
    if selected_layers:
        layers = [int(x) for x in selected_layers.split(",")]

    heads = None
    if selected_heads:
        heads = [int(x) for x in selected_heads.split(",")]

    # Call Modal function
    result = extract_attention.remote(
        image_base64=image_b64,
        query=query,
        adapter_name=adapter_name if adapter_name else None,
        selected_layers=layers,
        selected_heads=heads,
    )

    return AnalyzeResponse(**result)


@app.post("/api/compare")
async def compare_adapters(
    image: Annotated[UploadFile, File()],
    query: Annotated[str, Form()] = "What action should I take?",
    adapter_names: Annotated[str, Form()] = "",
):
    """
    Compare attention patterns across multiple adapters.

    Returns attention data for base model and each specified adapter.
    """
    image_bytes = await image.read()
    image_b64 = base64.b64encode(image_bytes).decode()

    adapters_to_compare = [""]  # Empty string = base model
    if adapter_names:
        adapters_to_compare.extend(adapter_names.split(","))

    results = {}

    # Run analysis for each adapter (could parallelize with Modal)
    for adapter in adapters_to_compare:
        adapter_name = adapter if adapter else None
        result = extract_attention.remote(
            image_base64=image_b64,
            query=query,
            adapter_name=adapter_name,
            selected_layers=[0, 10, 20, -1],  # Sample layers for comparison
        )
        key = adapter if adapter else "base"
        results[key] = result

    return results


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
