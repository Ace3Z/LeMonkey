#!/usr/bin/env python
"""
SmolVLM celebrity-recognition baseline — Dash app.

Loads the SmolVLM2-500M VLM (the same vision-language backbone that SmolVLA's
frozen VLM uses), exposes a localhost UI to upload an image + type a prompt,
and prints the model's answer.

Used to sanity-check the out-of-the-box VLM on Eval 3's celebrity-recognition
sub-task before any fine-tuning.

Run:
    conda activate lerobot
    python eval_3/scripts/smolvlm_celebrity_baseline.py
    # then open http://127.0.0.1:8050
"""
from __future__ import annotations

import argparse
import base64
import io
import time

import dash
import torch
from dash import Input, Output, State, dcc, html
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DEFAULT_PROMPT = "Who is this celebrity? Answer with the person's name and a brief justification."


def pick_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def load_model(model_id: str):
    device, dtype = pick_device()
    print(f"[load] {model_id} on {device} ({dtype})")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=dtype
    ).to(device).eval()
    print(f"[load] done in {time.time() - t0:.1f}s")
    return processor, model, device, dtype


def parse_uploaded_image(contents: str | None) -> Image.Image | None:
    if not contents:
        return None
    _, b64 = contents.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def build_app(processor, model, device: str, dtype: torch.dtype, model_id: str):
    app = dash.Dash(__name__)
    app.title = "SmolVLM celebrity baseline"
    app.layout = html.Div(
        [
            html.H1("SmolVLM celebrity baseline"),
            html.Div(
                [
                    html.Span("model: ", style={"color": "#666"}),
                    html.Code(model_id),
                    html.Span(f"  ·  device: {device}  ·  dtype: {dtype}",
                              style={"color": "#666"}),
                ],
                style={"marginBottom": "16px", "fontSize": "13px"},
            ),
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
                style={"width": "100%", "height": "60px", "fontSize": "14px"},
            ),
            html.Div(
                [
                    html.Button("Ask", id="ask-btn", n_clicks=0,
                                style={"padding": "8px 16px"}),
                    dcc.Slider(
                        id="max-tokens", min=32, max=512, step=32, value=200,
                        marks={32: "32", 128: "128", 256: "256", 512: "512"},
                    ),
                ],
                style={"display": "grid", "gridTemplateColumns": "120px 1fr",
                       "gap": "16px", "alignItems": "center",
                       "marginTop": "8px", "marginBottom": "12px"},
            ),
            dcc.Loading(
                html.Pre(
                    id="response",
                    style={
                        "whiteSpace": "pre-wrap", "background": "#f5f5f5",
                        "padding": "12px", "borderRadius": "4px",
                        "minHeight": "80px", "fontSize": "14px",
                    },
                ),
                type="default",
            ),
        ],
        style={"maxWidth": "780px", "margin": "32px auto",
               "fontFamily": "system-ui, sans-serif", "padding": "0 16px"},
    )

    @app.callback(Output("image-preview", "children"), Input("upload", "contents"))
    def _preview(contents):
        if not contents:
            return ""
        return html.Img(
            src=contents,
            style={"maxWidth": "400px", "maxHeight": "400px",
                   "borderRadius": "4px"},
        )

    @app.callback(
        Output("response", "children"),
        Input("ask-btn", "n_clicks"),
        State("upload", "contents"),
        State("prompt", "value"),
        State("max-tokens", "value"),
        prevent_initial_call=True,
    )
    def _ask(n_clicks, contents, prompt, max_new_tokens):
        img = parse_uploaded_image(contents)
        if img is None:
            return "Upload an image first."
        if not prompt or not prompt.strip():
            prompt = DEFAULT_PROMPT

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt.strip()},
            ],
        }]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if "pixel_values" in inputs and inputs["pixel_values"].dtype.is_floating_point:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
            )
        gen = out[:, inputs["input_ids"].shape[-1]:]
        text = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
        elapsed = time.time() - t0
        return f"{text}\n\n— generated in {elapsed:.1f}s ({gen.shape[-1]} tokens)"

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    processor, model, device, dtype = load_model(args.model_id)
    app = build_app(processor, model, device, dtype, args.model_id)
    print(f"[serve] http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
