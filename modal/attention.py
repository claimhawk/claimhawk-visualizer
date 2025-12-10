"""Modal function for extracting attention weights from Qwen3-VL."""

import modal

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


@app.function(
    image=image,
    gpu="A100",
    timeout=300,
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
    Run inference with Qwen3-VL and optional LoRA adapter.

    Uses flash_attention_2 for speed. Attention visualization disabled for now.
    """
    import base64
    import os
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

    # Use cached model path
    # Model stored as "Qwen--Qwen3-VL-8B-Instruct" (/ replaced with --)
    model_cache_path = "/models/models/Qwen--Qwen3-VL-8B-Instruct"

    # Check if cached, otherwise use HF
    if os.path.exists(model_cache_path):
        print(f"Loading model from cache: {model_cache_path}")
        model_path = model_cache_path
    else:
        print(f"Cache not found, using HF: {base_model}")
        model_path = base_model

    # Load processor and model
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",  # Fast!
    )

    # Apply LoRA adapter if specified
    if adapter_name:
        # Adapters at /volume/loras/{name}/adapter (from moe-inference volume)
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
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": query},
            ],
        }
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
    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids[0])

    # Run inference (no attention output with flash_attention_2)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    # Decode output
    generated_ids = outputs[0][len(input_ids[0]):]
    output_text = processor.decode(generated_ids, skip_special_tokens=True)

    # Find vision token range
    vision_start = None
    vision_end = None
    for i, token in enumerate(tokens):
        if "<|vision_start|>" in token:
            vision_start = i
        if "<|vision_end|>" in token:
            vision_end = i
            break

    return {
        "output_text": output_text,
        "attention_data": [],  # Empty for now with flash_attention_2
        "num_layers": 0,
        "num_heads": 0,
        "tokens": tokens,
        "vision_token_range": [vision_start, vision_end],
        "image_size": list(original_size),
        "adapter_name": adapter_name,
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
    print(f"Adapter: {result['adapter_name']}")
