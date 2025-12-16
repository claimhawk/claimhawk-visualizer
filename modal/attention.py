"""Modal function for extracting attention/saliency from Qwen3-VL during generation.

Uses gradient-based saliency instead of attention weights to avoid OOM issues
with eager attention on large vision token sets.
"""

from __future__ import annotations

import re
from typing import Any

import modal

app = modal.App("lora-attention-visualizer")

# Version to force container refresh
__version__ = "4.1.0"

# Volumes
lora_volume = modal.Volume.from_name("moe-inference", create_if_missing=False)
model_cache = modal.Volume.from_name("claimhawk-model-cache", create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers>=4.57.0",
        "accelerate>=0.26.0",
        "peft>=0.14.0",
        "qwen-vl-utils",
        "Pillow>=10.0.0",
    )
    .run_commands(f"echo 'Version: {__version__}'")  # Force rebuild on version change
)

# System prompt
SYSTEM_PROMPT = """Use a mouse and keyboard to interact with a computer, and take screenshots.
* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is 1000x1000.
* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
* If you tried clicking on a program or link but it failed to load even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element.

# Tools

You may call one or more functions to assist with the user query.

<tools>
{
  "name": "computer_use",
  "description": "Perform computer actions",
  "parameters": {
    "type": "object",
    "properties": {
      "action": {
        "type": "string",
        "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "triple_click", "scroll", "hscroll", "wait", "terminate", "answer", "ocr"]
      },
      "coordinate": {
        "type": "array",
        "items": {"type": "integer"},
        "description": "X and Y coordinates in 1000x1000 normalized space"
      },
      "text": {
        "type": "string",
        "description": "Text content for ocr action"
      }
    },
    "required": ["action"]
  }
}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Response format:
1) Action: a short imperative describing what to do in the UI.
2) One or more <tool_call>...</tool_call> blocks.

Rules:
- Be brief: one sentence for Action.
- Do not output anything else outside those parts."""


def parse_coordinates(output_text: str) -> tuple[int, int] | None:
    """Extract coordinates from model output like [213, 613]."""
    patterns = [
        r'"coordinate"\s*:\s*\[(\d+),\s*(\d+)\]',
        r'\[(\d+),\s*(\d+)\]',
    ]
    for pattern in patterns:
        match = re.search(pattern, output_text)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            if 0 <= x <= 1000 and 0 <= y <= 1000:
                return x, y
    return None


@app.cls(
    image=image,
    gpu="H100",
    timeout=600,
    scaledown_window=300,
    volumes={
        "/volume": lora_volume,
        "/models": model_cache,
    },
)
class AttentionServer:
    """Warm GPU server for attention visualization."""

    @modal.enter()
    def load_models(self) -> None:
        """Load base model on startup."""
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.device = torch.device("cuda")
        base_model = "Qwen/Qwen3-VL-8B-Instruct"

        print(f"Loading model: {base_model}")
        self.processor = AutoProcessor.from_pretrained(base_model)
        self.base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.base_model.eval()
        print("Model loaded!")

        # Track loaded adapters
        self.loaded_adapters: dict[str, Any] = {}

    def _get_model_with_adapter(self, adapter_name: str | None):
        """Get model with adapter loaded."""
        import torch
        from peft import PeftModel

        if adapter_name is None:
            return self.base_model

        if adapter_name not in self.loaded_adapters:
            adapter_path = f"/volume/loras/{adapter_name}/adapter"
            print(f"Loading adapter: {adapter_path}")
            model = PeftModel.from_pretrained(
                self.base_model,
                adapter_path,
                torch_dtype=torch.bfloat16,
            )
            model.eval()
            self.loaded_adapters[adapter_name] = model

        return self.loaded_adapters[adapter_name]

    @modal.method()
    def extract_attention(
        self,
        image_base64: str,
        query: str,
        adapter_name: str | None = None,
        num_groups: int = 6,
    ) -> dict:
        """Run inference and extract vision transformer attention."""
        import base64
        from io import BytesIO

        import numpy as np
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        print(f"=== Attention Visualizer v{__version__} ===")

        # Decode image
        image_data = base64.b64decode(image_base64)
        pil_image = Image.open(BytesIO(image_data)).convert("RGB")
        original_size = pil_image.size

        # Get model
        model = self._get_model_with_adapter(adapter_name)

        # Prepare input
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": query},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        input_ids = inputs["input_ids"]
        input_length = input_ids.shape[1]
        tokens = self.processor.tokenizer.convert_ids_to_tokens(input_ids[0])

        # Find vision token range
        vision_start, vision_end = None, None
        for i, token in enumerate(tokens):
            if "<|vision_start|>" in token:
                vision_start = i
            if "<|vision_end|>" in token:
                vision_end = i
                break

        print(f"Vision tokens: {vision_start} to {vision_end}")
        num_vision_tokens = (vision_end - vision_start) if (vision_start and vision_end) else 0

        # Get vision grid dimensions
        image_grid_thw = inputs.get("image_grid_thw")
        vision_grid = None
        if image_grid_thw is not None:
            thw = image_grid_thw[0].tolist()
            vision_grid = [int(thw[1]), int(thw[2])]
            print(f"Vision grid: {vision_grid}")

        # Step 1: Capture vision encoder attention (edge detection) during generation
        attention_data = []
        vision_attn_maps = []
        vision_hooks = []

        if num_vision_tokens > 0:
            # ============================================================
            # EDGE DETECTION HOOK - DO NOT MODIFY
            # This hook captures vision encoder self-attention which shows
            # patch-to-patch relationships (edge/boundary detection).
            # It computes Q*K attention from the QKV projection and sums
            # attention received by each patch to create a saliency map.
            # Result: Cyan overlay showing visual boundaries/edges.
            # ============================================================
            def make_vision_hook(layer_idx):
                def hook(module, args, output):
                    try:
                        # Get hidden states from args
                        hidden = args[0] if args else None
                        if hidden is None:
                            return

                        # Compute QKV projection
                        qkv = module.qkv(hidden)

                        # Handle variable dimensions (2D when batch=1)
                        if qkv.dim() == 2:
                            qkv = qkv.unsqueeze(0)

                        batch, seq_len, hidden_dim = qkv.shape
                        num_heads = getattr(module, 'num_heads', 16)
                        head_dim = hidden_dim // (3 * num_heads)

                        # Reshape: (batch, seq, 3, heads, head_dim)
                        qkv = qkv.view(batch, seq_len, 3, num_heads, head_dim)
                        q, k = qkv[:, :, 0], qkv[:, :, 1]

                        # Transpose to (batch, heads, seq, head_dim)
                        q = q.permute(0, 2, 1, 3)
                        k = k.permute(0, 2, 1, 3)

                        # Compute attention: Q @ K^T / sqrt(d)
                        scale = head_dim ** -0.5
                        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
                        attn = torch.softmax(attn, dim=-1)

                        # Sum attention received by each patch (column sum)
                        attn_received = attn.mean(dim=(0, 1)).sum(dim=0).detach().cpu()
                        vision_attn_maps.append((layer_idx, attn_received))
                        print(f"Vision hook layer {layer_idx}: {len(attn_received)} patches")
                    except Exception as e:
                        print(f"Vision hook error layer {layer_idx}: {e}")
                        import traceback
                        traceback.print_exc()
                return hook
            # ============================================================

            # Handle both base model and PeftModel
            if hasattr(model, 'base_model'):
                vision_blocks = model.base_model.model.visual.blocks
            else:
                vision_blocks = model.visual.blocks
            n_layers = len(vision_blocks)
            # Hook ALL layers for real-time filtering in frontend
            hook_indices = list(range(n_layers))
            print(f"Hooking all {n_layers} vision layers for real-time frontend filtering")
            for idx in hook_indices:
                if idx >= 0 and hasattr(vision_blocks[idx], 'attn'):
                    h = vision_blocks[idx].attn.register_forward_hook(make_vision_hook(idx))
                    vision_hooks.append(h)

        # Hook LM decoder during generate() to capture decision attention
        lm_hooks = []
        lm_decision_attn = []  # Will capture attention during actual generation

        def make_lm_decision_hook():
            """Capture LM attention to vision tokens during generation."""
            call_count = [0]
            def hook(module, args, kwargs, output):
                call_count[0] += 1
                # Only capture every 10th call to avoid too much data
                if call_count[0] % 10 != 1:
                    return
                try:
                    if kwargs and 'hidden_states' in kwargs:
                        hidden = kwargs['hidden_states']
                    elif args and len(args) > 0:
                        hidden = args[0]
                    else:
                        return

                    batch, seq_len, hidden_dim = hidden.shape
                    if vision_start is None or vision_end is None:
                        return
                    if seq_len <= vision_end:
                        return  # Not enough tokens yet

                    q = module.q_proj(hidden)
                    k = module.k_proj(hidden)

                    num_q_heads = q.shape[-1] // 128
                    num_kv_heads = k.shape[-1] // 128
                    head_dim = 128

                    q = q.view(batch, seq_len, num_q_heads, head_dim).transpose(1, 2)
                    k = k.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)

                    if num_kv_heads != num_q_heads:
                        k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1)

                    scale = head_dim ** -0.5
                    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
                    attn = torch.softmax(attn, dim=-1)

                    # Get attention from LAST token to vision tokens
                    last_to_vision = attn[0, :, -1, vision_start:vision_end].mean(dim=0).detach().cpu()
                    lm_decision_attn.append(last_to_vision)
                except Exception as e:
                    pass  # Silent fail during generation
            return hook

        # Find and hook last LM layer for decision attention
        try:
            if hasattr(model, 'base_model'):
                base = model.base_model.model
                if hasattr(base, 'model') and hasattr(base.model, 'language_model'):
                    lm_layers = base.model.language_model.layers
                    if lm_layers:
                        h = lm_layers[-1].self_attn.register_forward_hook(make_lm_decision_hook(), with_kwargs=True)
                        lm_hooks.append(h)
                        print(f"Hooked LM layer for decision attention during generate()")
        except Exception as e:
            print(f"Could not hook LM for decision: {e}")

        print("Generating output...")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        for h in vision_hooks:
            h.remove()
        for h in lm_hooks:
            h.remove()
        print(f"Captured {len(vision_attn_maps)} vision attention maps")
        print(f"Captured {len(lm_decision_attn)} LM decision attention samples")

        generated_ids = output_ids[0][input_length:]
        output_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        generated_tokens = [self.processor.tokenizer.decode([t]) for t in generated_ids]
        print(f"Generated {len(generated_tokens)} tokens: {output_text[:100]}...")

        # Parse coordinates
        coordinates = parse_coordinates(output_text)
        print(f"Coordinates: {coordinates}")

        # Step 2: Capture LM attention to vision tokens
        # Do a forward pass with hooks on LM decoder to see what it attends to
        lm_attn_to_vision = []
        lm_debug = []  # Debug info for LM layer discovery
        lm_debug.append(f"vision_start={vision_start}, vision_end={vision_end}")

        if vision_start is not None and vision_end is not None:
            lm_debug.append("Entered LM block")
            # Find coordinate token positions
            coord_positions = [i for i, tok in enumerate(generated_tokens) if any(c.isdigit() for c in tok)]
            if not coord_positions:
                coord_positions = list(range(min(5, len(generated_tokens))))
            print(f"Coord token positions: {coord_positions[:5]}")

            def make_lm_hook(layer_idx):
                def hook(module, args, kwargs, output):
                    try:
                        lm_debug.append(f"LM hook called! layer={layer_idx}")
                        lm_debug.append(f"args len: {len(args)}, kwargs keys: {list(kwargs.keys()) if kwargs else 'None'}")

                        # Try to get hidden states from args or kwargs
                        if args and len(args) > 0:
                            hidden = args[0]
                        elif kwargs and 'hidden_states' in kwargs:
                            hidden = kwargs['hidden_states']
                        else:
                            lm_debug.append(f"Cannot find hidden states")
                            return

                        batch, seq_len, hidden_dim = hidden.shape
                        lm_debug.append(f"hidden shape: {hidden.shape}")
                        lm_debug.append(f"input_length={input_length}, coord_positions={coord_positions[:3]}")
                        lm_debug.append(f"vision_start={vision_start}, vision_end={vision_end}")

                        # Get Q and K
                        q = module.q_proj(hidden)
                        k = module.k_proj(hidden)

                        # Handle GQA - K/V may have fewer heads than Q
                        num_q_heads = q.shape[-1] // 128  # head_dim is typically 128
                        num_kv_heads = k.shape[-1] // 128
                        head_dim = 128
                        lm_debug.append(f"num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}, q.shape={q.shape}, k.shape={k.shape}")

                        # Reshape Q and K
                        q = q.view(batch, seq_len, num_q_heads, head_dim).transpose(1, 2)
                        k = k.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)

                        # Expand K to match Q heads for attention computation
                        if num_kv_heads != num_q_heads:
                            k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1)

                        # Compute attention
                        scale = head_dim ** -0.5
                        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
                        attn = torch.softmax(attn, dim=-1)
                        lm_debug.append(f"attn shape: {attn.shape}")

                        # Extract attention from coord tokens to vision tokens
                        for pos in coord_positions[:3]:
                            seq_pos = input_length + pos
                            lm_debug.append(f"pos={pos}, seq_pos={seq_pos}, seq_len={seq_len}")
                            if seq_pos < seq_len:
                                # Attention from this token to vision tokens
                                attn_slice = attn[0, :, seq_pos, vision_start:vision_end]  # (heads, vision_tokens)
                                attn_avg = attn_slice.mean(dim=0).detach().cpu()  # (vision_tokens,)
                                lm_attn_to_vision.append(attn_avg)
                                lm_debug.append(f"Captured slice, len={len(attn_avg)}")
                            else:
                                lm_debug.append(f"seq_pos {seq_pos} >= seq_len {seq_len}, skipping")
                    except Exception as e:
                        lm_debug.append(f"LM hook error: {e}")
                        import traceback
                        lm_debug.append(traceback.format_exc())
                return hook

            # ============================================================
            # DECISION FOCUS - Skip second forward pass approach
            # The model doesn't "look" at the target during coord generation.
            # Decision already happened during generate().
            # Would need to hook inside generate() loop for real decision focus.
            # ============================================================
            lm_layers = None
            hooks = []
            try:
                print(f"Model type: {type(model).__name__}")
                print(f"Model has base_model: {hasattr(model, 'base_model')}")

                if hasattr(model, 'base_model'):
                    # PeftModel wrapping Qwen3VLForConditionalGeneration
                    base = model.base_model.model
                    lm_debug.append(f"base type: {type(base).__name__}")
                    lm_debug.append(f"base has model: {hasattr(base, 'model')}")

                    if hasattr(base, 'model'):
                        m = base.model
                        children = [n for n, _ in list(m.named_children())[:10]]
                        lm_debug.append(f"base.model type: {type(m).__name__}")
                        lm_debug.append(f"base.model children: {children}")

                        # Check for language_model
                        if hasattr(m, 'language_model'):
                            lm = m.language_model
                            lm_debug.append(f"language_model type: {type(lm).__name__}")
                            lm_debug.append(f"language_model has layers: {hasattr(lm, 'layers')}")
                            if hasattr(lm, 'layers'):
                                lm_layers = lm.layers
                                lm_debug.append(f"FOUND LM layers: {len(lm_layers)} layers")
                        else:
                            lm_debug.append("No language_model attribute")
                            # Maybe layers are directly on base.model
                            if hasattr(m, 'layers'):
                                lm_layers = m.layers
                                lm_debug.append(f"FOUND LM layers at base.model.layers: {len(lm_layers)} layers")
                else:
                    # Base model without PEFT: Qwen3VLForConditionalGeneration
                    lm_debug.append(f"model has model: {hasattr(model, 'model')}")
                    if hasattr(model, 'model'):
                        m = model.model
                        children = [n for n, _ in list(m.named_children())[:10]]
                        lm_debug.append(f"model.model type: {type(m).__name__}")
                        lm_debug.append(f"model.model children: {children}")

                        if hasattr(m, 'language_model'):
                            lm = m.language_model
                            if hasattr(lm, 'layers'):
                                lm_layers = lm.layers
                                lm_debug.append(f"FOUND LM layers: {len(lm_layers)} layers")
                        elif hasattr(m, 'layers'):
                            lm_layers = m.layers
                            lm_debug.append(f"FOUND LM layers at model.model.layers: {len(lm_layers)} layers")
            except Exception as e:
                lm_debug.append(f"Error: {e}")
                import traceback
                traceback.print_exc()

            if lm_layers is not None:
                last_layer = lm_layers[-1]
                lm_debug.append(f"last_layer type: {type(last_layer).__name__}")
                lm_debug.append(f"has self_attn: {hasattr(last_layer, 'self_attn')}")
                if hasattr(last_layer, 'self_attn'):
                    h = last_layer.self_attn.register_forward_hook(make_lm_hook(len(lm_layers) - 1), with_kwargs=True)
                    hooks.append(h)
                    lm_debug.append(f"Hooked last LM layer ({len(lm_layers)} total)")
                else:
                    # Try to find attention module with different name
                    lm_debug.append(f"last_layer children: {[n for n, _ in last_layer.named_children()]}")
            else:
                lm_debug.append("LM layers is None - skipping")

            # Forward pass with full sequence
            lm_debug.append("Running forward pass...")
            try:
                with torch.no_grad():
                    _ = model(
                        input_ids=output_ids,
                        pixel_values=inputs.get("pixel_values"),
                        image_grid_thw=inputs.get("image_grid_thw"),
                        return_dict=True,
                    )
                lm_debug.append("Forward pass complete")
            except Exception as e:
                lm_debug.append(f"Forward pass error: {e}")

            for h in hooks:
                h.remove()
            lm_debug.append(f"Captured {len(lm_attn_to_vision)} LM attention slices")

        # Process vision encoder attention - output each layer separately
        if vision_attn_maps:
            print(f"Processing {len(vision_attn_maps)} vision encoder maps")
            for layer_idx, saliency in sorted(vision_attn_maps, key=lambda x: x[0]):
                arr = saliency.float().numpy() if hasattr(saliency, 'numpy') else np.array(saliency)
                if arr.ndim == 0:
                    print(f"Skipping scalar saliency from layer {layer_idx}")
                    continue
                # Normalize
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                attention_data.append({
                    "layer": layer_idx,
                    "head": -1,
                    "attention": arr.tolist(),
                    "type": f"layer_{layer_idx}",
                })
                print(f"Layer {layer_idx}: {len(arr)} patches, std={arr.std():.4f}")

        # Process LM decision attention (captured during generate())
        lm_debug.append(f"lm_decision_attn has {len(lm_decision_attn)} samples")
        if lm_decision_attn:
            try:
                lm_debug.append("Processing LM decision attention from generate()...")
                avg_lm = torch.stack(lm_decision_attn).mean(dim=0).float().numpy()
                avg_lm = (avg_lm - avg_lm.min()) / (avg_lm.max() - avg_lm.min() + 1e-8)
                lm_debug.append(f"LM decision attention: {len(avg_lm)} tokens, std={avg_lm.std():.4f}")

                attention_data.append({
                    "layer": 1,
                    "head": -1,
                    "attention": avg_lm.tolist(),
                    "type": "decision_focus",
                })
                lm_debug.append("Added decision_focus to attention_data")
            except Exception as e:
                lm_debug.append(f"Error processing LM decision attention: {e}")

        torch.cuda.empty_cache()

        # Fallback if no attention captured
        if not attention_data and vision_grid:
            print("Using uniform fallback")
            uniform = np.ones(vision_grid[0] * vision_grid[1]) * 0.5
            attention_data.append({
                "layer": 0,
                "head": -1,
                "attention": uniform.tolist(),
            })

        # Ensure all values are JSON serializable
        def to_json_safe(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, list):
                return [to_json_safe(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: to_json_safe(v) for k, v in obj.items()}
            return obj

        result = {
            "version": __version__,
            "output_text": output_text,
            "attention_data": attention_data,
            "num_layers": 28,
            "num_heads": 28,
            "tokens": tokens,
            "vision_token_range": [vision_start, vision_end],
            "image_size": list(original_size),
            "adapter_name": adapter_name,
            "coordinates": list(coordinates) if coordinates else None,
            "generated_tokens": generated_tokens,
            "coordinate_token_indices": coord_positions if 'coord_positions' in dir() else [],
            "vision_grid": vision_grid,
            "lm_debug": lm_debug,
        }
        return to_json_safe(result)


# Web endpoint
web_image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi[standard]")


@app.function(image=web_image)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def web():
    """ASGI web app for attention visualization."""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    fastapi_app = FastAPI(title="Attention Visualizer API")
    server = AttentionServer()

    @fastapi_app.post("/infer")
    async def infer(data: dict) -> JSONResponse:
        result = server.extract_attention.remote(
            image_base64=data.get("image_b64", ""),
            query=data.get("prompt", ""),
            adapter_name=data.get("adapter"),
        )
        return JSONResponse(content=result)

    @fastapi_app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(content={"status": "healthy"})

    return fastapi_app


# Keep the old function interface for backwards compatibility
@app.function(
    image=image,
    gpu="H100",
    timeout=600,
    volumes={"/volume": lora_volume, "/models": model_cache},
)
def extract_attention(
    image_base64: str,
    query: str,
    adapter_name: str | None = None,
    selected_layers: list[int] | None = None,
    selected_heads: list[int] | None = None,
) -> dict:
    """Legacy function interface - delegates to class method."""
    server = AttentionServer()
    return server.extract_attention.remote(
        image_base64=image_base64,
        query=query,
        adapter_name=adapter_name,
    )


@app.function(image=image, volumes={"/volume": lora_volume})
def list_adapters() -> list[dict]:
    """List available LoRA adapters."""
    import os

    adapters_dir = "/volume/loras"
    if not os.path.exists(adapters_dir):
        return []

    adapters = []
    for name in os.listdir(adapters_dir):
        adapter_path = os.path.join(adapters_dir, name, "adapter")
        if os.path.isdir(adapter_path):
            config_path = os.path.join(adapter_path, "adapter_config.json")
            if os.path.exists(config_path):
                adapters.append({"name": name, "path": adapter_path})

    return adapters


@app.local_entrypoint()
def main():
    """Test the inference."""
    import base64

    with open("test_image.png", "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    server = AttentionServer()
    result = server.extract_attention.remote(
        image_base64=image_b64,
        query="click Oct 7",
        adapter_name="calendar",
    )

    print(f"Output: {result['output_text']}")
    print(f"Coordinates: {result['coordinates']}")
    print(f"Attention data: {len(result['attention_data'])} entries")
