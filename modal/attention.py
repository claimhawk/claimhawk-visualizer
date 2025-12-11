"""Modal function for extracting attention weights from Qwen3-VL during generation.

v3 - Fixed coordinate parsing and aggregated attention.
"""

from __future__ import annotations

import modal
import re
from typing import Optional

app = modal.App("lora-attention-visualizer")

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
)

# System prompt from projects/system-prompt/SYSTEM_PROMPT.txt
SYSTEM_PROMPT = """Use a mouse and keyboard to interact with a computer, and take screenshots.
* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.
* The screen's resolution is 1000x1000.
* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
* If you tried clicking on a program or link but it failed to load even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
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

# Action descriptions

* `key`: Press keys in order, release in reverse.
* `type`: Type a string of text.
* `mouse_move`: Move the cursor to (x, y).
* `left_click`: Left click at (x, y).
* `left_click_drag`: Click and drag from current to (x, y).
* `right_click`: Right click at (x, y).
* `middle_click`: Middle click at (x, y).
* `double_click`: Double-click at (x, y).
* `triple_click`: Triple-click at (x, y).
* `scroll`: Scroll the mouse wheel.
* `hscroll`: Horizontal scroll.
* `wait`: Wait N seconds.
* `terminate`: End the task with a status.
* `answer`: Answer a question.
* `ocr`: Return extracted text from the image.

# Response format

Response format for every step:
1) Action: a short imperative describing what to do in the UI.
2) One or more <tool_call>...</tool_call> blocks, one per line, each containing only the JSON.

Rules:
- Output exactly in the order: Action, <tool_call>(s).
- Be brief: one sentence for Action.
- Multiple tool calls can be output, one per line.
- Do not output anything else outside those parts.
- If finishing, use action=terminate in the tool call."""


def parse_coordinates(output_text: str) -> tuple[int, int] | None:
    """Extract coordinates from model output like [213, 613]."""
    # Try various formats the model might output
    patterns = [
        r'"coordinate"\s*:\s*\[(\d+),\s*(\d+)\]',  # "coordinate": [x, y]
        r'\[(\d+),\s*(\d+)\]',  # Just [x, y] anywhere
    ]
    for pattern in patterns:
        match = re.search(pattern, output_text)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            # Sanity check - coordinates should be in 0-1000 range
            if 0 <= x <= 1000 and 0 <= y <= 1000:
                return x, y
    return None


@app.function(
    image=image,
    gpu="H100",
    timeout=600,
    volumes={
        "/volume": lora_volume,
        "/models": model_cache,
    },
)
def extract_attention(
    image_base64: str,
    query: str,
    adapter_name: str | None = None,
    base_model: str = "Qwen/Qwen3-VL-8B-Instruct",
    selected_layers: list[int] | None = None,
    selected_heads: list[int] | None = None,
) -> dict:
    """
    Run inference with Qwen3-VL and capture attention during generation.

    Captures attention from coordinate tokens back to vision tokens to show
    what the model looked at when deciding where to click.
    """
    import base64
    import json
    import torch
    from io import BytesIO
    from PIL import Image
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info

    # Decode image
    image_data = base64.b64decode(image_base64)
    pil_image = Image.open(BytesIO(image_data)).convert("RGB")
    original_size = pil_image.size

    # Load model with eager attention (required for output_attentions)
    print(f"Loading model from HuggingFace: {base_model}")
    processor = AutoProcessor.from_pretrained(base_model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    # Apply LoRA adapter if specified
    if adapter_name:
        adapter_path = f"/volume/loras/{adapter_name}/adapter"
        print(f"Loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            torch_dtype=torch.bfloat16,
        )

    model.eval()

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

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    input_ids = inputs["input_ids"]
    input_length = input_ids.shape[1]
    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids[0])

    # Find vision token range in input
    vision_start = None
    vision_end = None
    for i, token in enumerate(tokens):
        if "<|vision_start|>" in token:
            vision_start = i
        if "<|vision_end|>" in token:
            vision_end = i
            break

    print(f"Vision tokens: {vision_start} to {vision_end}")

    # Generate with attention capture using a custom generation loop
    # We need to capture attention at each generation step
    generated_ids = input_ids.clone()
    all_step_attentions = []  # Store attention from each generation step
    generated_tokens = []

    max_new_tokens = 256
    eos_token_id = processor.tokenizer.eos_token_id

    with torch.no_grad():
        for step in range(max_new_tokens):
            # Forward pass with attention
            outputs = model(
                input_ids=generated_ids,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                output_attentions=True,
                return_dict=True,
            )

            # Get next token
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Decode token for logging
            token_str = processor.tokenizer.decode(next_token[0])
            generated_tokens.append(token_str)

            # Store attention from this token to vision tokens
            # attentions is tuple of (num_layers,) tensors of shape (batch, heads, seq, seq)
            if outputs.attentions and vision_start is not None and vision_end is not None:
                # Get attention from the last token (just generated) to vision tokens
                # We'll store attention from multiple layers
                step_attention = {}
                num_layers = len(outputs.attentions)

                # Sample layers: early, middle, late
                layers_to_capture = selected_layers or [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]

                for layer_idx in layers_to_capture:
                    if 0 <= layer_idx < num_layers:
                        layer_attn = outputs.attentions[layer_idx][0]  # (heads, seq, seq)
                        # Attention from last token to vision tokens, averaged across heads
                        attn_to_vision = layer_attn[:, -1, vision_start:vision_end].mean(dim=0)
                        step_attention[layer_idx] = attn_to_vision.float().cpu().numpy().tolist()

                all_step_attentions.append({
                    "step": step,
                    "token": token_str,
                    "attention": step_attention,
                })

            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Check for EOS
            if next_token.item() == eos_token_id:
                break

    # Decode full output
    output_ids = generated_ids[0][input_length:]
    output_text = processor.decode(output_ids, skip_special_tokens=True)

    # Parse coordinates from output
    coordinates = parse_coordinates(output_text)
    print(f"Output: {output_text}")
    print(f"Parsed coordinates: {coordinates}")

    # Find coordinate tokens and aggregate their attention
    # Look for tokens that are part of coordinate numbers
    coord_token_indices = []

    # Find tokens that look like they're part of coordinates (digits in the output)
    for i, step_data in enumerate(all_step_attentions):
        token = step_data["token"]
        # Check if this token contains digits (likely part of coordinate)
        if any(c.isdigit() for c in token):
            coord_token_indices.append(i)

    print(f"Found {len(coord_token_indices)} coordinate-related tokens")
    print(f"Coordinate tokens: {[all_step_attentions[i]['token'] for i in coord_token_indices]}")

    # Aggregate attention from coordinate tokens across ALL layers into single heatmap
    import numpy as np
    aggregated_attention = None

    if coord_token_indices and all_step_attentions:
        layers = list(all_step_attentions[0]["attention"].keys())
        num_vision_tokens = len(all_step_attentions[0]["attention"][layers[0]]) if layers else 0

        all_attentions = []
        for step_idx in coord_token_indices:
            for layer_idx in layers:
                if layer_idx in all_step_attentions[step_idx]["attention"]:
                    all_attentions.append(all_step_attentions[step_idx]["attention"][layer_idx])

        if all_attentions:
            # Average across all coordinate tokens AND all layers
            aggregated_attention = np.mean(all_attentions, axis=0)
            print(f"Aggregated attention: min={aggregated_attention.min():.6f}, max={aggregated_attention.max():.6f}, std={aggregated_attention.std():.6f}")

    # Format attention data for frontend - single aggregated entry
    attention_data = []
    if aggregated_attention is not None:
        attention_data.append({
            "layer": 0,  # Dummy layer index
            "head": -1,  # Averaged
            "attention": aggregated_attention.tolist(),
        })

    # Get model info
    num_layers = len(outputs.attentions) if outputs.attentions else 0
    num_heads = outputs.attentions[0].shape[1] if outputs.attentions else 0

    # Get vision grid dimensions from image_grid_thw
    # Shape is (batch, 3) where 3 = (temporal, height, width) for each image
    image_grid_thw = inputs.get("image_grid_thw")
    vision_grid = None
    if image_grid_thw is not None:
        # For single image: [1, H, W] where H and W are grid dimensions
        thw = image_grid_thw[0].tolist()  # [t, h, w]
        vision_grid = [int(thw[1]), int(thw[2])]  # [height, width]
        print(f"Vision grid from image_grid_thw: {vision_grid} (h x w)")

    return {
        "output_text": output_text,
        "attention_data": attention_data,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "tokens": tokens,
        "vision_token_range": [vision_start, vision_end],
        "image_size": list(original_size),
        "adapter_name": adapter_name,
        "coordinates": list(coordinates) if coordinates else None,
        "generated_tokens": generated_tokens,
        "coordinate_token_indices": coord_token_indices,
        "vision_grid": vision_grid,
    }


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

    result = extract_attention.remote(
        image_base64=image_b64,
        query="click Oct 7",
        adapter_name="calendar",
    )

    print(f"Output: {result['output_text']}")
    print(f"Coordinates: {result['coordinates']}")
    print(f"Adapter: {result['adapter_name']}")
