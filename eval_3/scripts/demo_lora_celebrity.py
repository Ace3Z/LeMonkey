#!/usr/bin/env python
"""
Interactive Dash demo for the LoRA-fine-tuned SmolVLM2 celebrity recognition model.

Upload a photo, the model predicts who it is. Shows both the LoRA-adapted output
and (optionally) the base model output side-by-side for comparison.

Run (on AWS with port 6008 forwarded):
    python eval_3/scripts/demo_lora_celebrity.py \
        --adapter $DATA_ROOT/lora_celeb_v0/checkpoint-7000 \
        --host 0.0.0.0 --port 6008
"""
from __future__ import annotations

import argparse
import base64
import io
import time
from pathlib import Path

import dash
import torch
from dash import Input, Output, State, dcc, html
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DEFAULT_PROMPT = "Who is shown in this photo?"


def pick_device_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def load_model(adapter_path: Path):
    device, dtype = pick_device_dtype()
    print(f"[load] base={MODEL_ID}  adapter={adapter_path}")
    print(f"[load] device={device}  dtype={dtype}")
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model = model.to(device).eval()

    print(f"[load] done in {time.time() - t0:.1f}s")
    return processor, model, device, dtype


def parse_uploaded_image(contents: str | None) -> Image.Image | None:
    if not contents:
        return None
    _, b64 = contents.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def generate(processor, model, img, prompt, max_new_tokens, device, dtype):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=text, images=[img],
        return_tensors="pt", do_image_splitting=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if "pixel_values" in inputs and inputs["pixel_values"].dtype.is_floating_point:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = out[0, inputs["input_ids"].shape[-1]:]
    decoded = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    elapsed = time.time() - t0
    return decoded.split("\n")[0].strip(), elapsed, len(new_tokens)


def build_app(processor, model, device, dtype, adapter_path: str):
    app = dash.Dash(__name__)
    app.title = "SmolVLM2 LoRA Celebrity Demo"
    app.layout = html.Div([
        html.H1("SmolVLM2 + LoRA Celebrity Recognition"),
        html.Div([
            html.Code(f"adapter: {adapter_path}"),
            html.Span(f"  ·  device: {device}  ·  dtype: {dtype}",
                      style={"color": "#666"}),
        ], style={"marginBottom": "16px", "fontSize": "13px"}),

        dcc.Upload(
            id="upload",
            children=html.Div(["Drag & drop or ", html.A("select image")]),
            style={
                "width": "100%", "height": "80px", "lineHeight": "80px",
                "borderWidth": "1px", "borderStyle": "dashed",
                "borderRadius": "5px", "textAlign": "center",
                "marginBottom": "12px",
            },
            multiple=False,
        ),
        html.Div(id="image-preview", style={"marginBottom": "12px"}),

        dcc.Textarea(
            id="prompt",
            value=DEFAULT_PROMPT,
            style={"width": "100%", "height": "50px", "fontSize": "14px"},
        ),
        html.Div([
            html.Button("Ask (LoRA)", id="ask-btn", n_clicks=0,
                        style={"padding": "8px 16px", "fontWeight": "bold"}),
            html.Button("Compare with base", id="compare-btn", n_clicks=0,
                        style={"padding": "8px 16px", "marginLeft": "8px"}),
            dcc.Slider(
                id="max-tokens", min=16, max=128, step=16, value=32,
                marks={16: "16", 32: "32", 64: "64", 128: "128"},
            ),
        ], style={"display": "grid", "gridTemplateColumns": "140px 160px 1fr",
                  "gap": "12px", "alignItems": "center",
                  "marginTop": "8px", "marginBottom": "12px"}),

        dcc.Loading(html.Div(id="results", style={"marginTop": "12px"}), type="default"),
    ], style={"maxWidth": "800px", "margin": "32px auto",
              "fontFamily": "system-ui, sans-serif", "padding": "0 16px"})

    @app.callback(Output("image-preview", "children"), Input("upload", "contents"))
    def _preview(contents):
        if not contents:
            return ""
        return html.Img(src=contents,
                        style={"maxWidth": "400px", "maxHeight": "400px",
                               "borderRadius": "4px"})

    @app.callback(
        Output("results", "children"),
        Input("ask-btn", "n_clicks"),
        Input("compare-btn", "n_clicks"),
        State("upload", "contents"),
        State("prompt", "value"),
        State("max-tokens", "value"),
        prevent_initial_call=True,
    )
    def _ask(n_lora, n_compare, contents, prompt, max_tokens):
        img = parse_uploaded_image(contents)
        if img is None:
            return html.Pre("Upload an image first.",
                            style={"color": "#c00", "padding": "12px"})
        if not prompt or not prompt.strip():
            prompt = DEFAULT_PROMPT

        ctx = dash.callback_context
        triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
        do_compare = "compare" in triggered

        # LoRA inference
        text_lora, t_lora, n_tok = generate(
            processor, model, img, prompt.strip(), int(max_tokens), device, dtype
        )
        result_blocks = [
            html.Div([
                html.H3("LoRA adapter", style={"margin": "0 0 4px 0", "color": "#2e7d32"}),
                html.Pre(text_lora, style={
                    "background": "#e8f5e9", "padding": "12px",
                    "borderRadius": "4px", "fontSize": "15px",
                }),
                html.Span(f"{t_lora:.1f}s · {n_tok} tokens",
                          style={"fontSize": "12px", "color": "#666"}),
            ])
        ]

        if do_compare:
            # Base model (disable adapter)
            model.disable_adapter_layers()
            text_base, t_base, n_tok_b = generate(
                processor, model, img, prompt.strip(), int(max_tokens), device, dtype
            )
            model.enable_adapter_layers()
            result_blocks.append(
                html.Div([
                    html.H3("Base model (no LoRA)",
                            style={"margin": "16px 0 4px 0", "color": "#c62828"}),
                    html.Pre(text_base, style={
                        "background": "#ffebee", "padding": "12px",
                        "borderRadius": "4px", "fontSize": "15px",
                    }),
                    html.Span(f"{t_base:.1f}s · {n_tok_b} tokens",
                              style={"fontSize": "12px", "color": "#666"}),
                ])
            )

        return html.Div(result_blocks)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True,
                        help="Path to LoRA adapter (e.g. checkpoint-7000/)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6008)
    args = parser.parse_args()

    processor, model, device, dtype = load_model(args.adapter)
    app = build_app(processor, model, device, dtype, str(args.adapter))
    print(f"[serve] http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
