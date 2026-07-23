"""
TOONIFY — HuggingFace Spaces Inference API
Runs Stable Diffusion img2img for real GPU cartoon generation.

Deploy to HuggingFace Spaces with T4 GPU hardware.
The Render backend calls this via gradio_client when HF_SPACES_URL is set.
"""
import os
import logging
import torch
import gradio as gr
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style presets — must match backend/pipeline.py STYLES dict
# ---------------------------------------------------------------------------
STYLES: dict[str, dict] = {
    "disney": {
        "label": "✨ Modern Disney",
        "prompt": "portrait, modern disney animation style, 3d digital artwork, expressive character, vibrant colors, smooth shading, cinematic lighting, masterpiece",
    },
    "anime": {
        "label": "⛩️ Anime",
        "prompt": "portrait, anime style, studio quality, detailed eyes, clean lines, cel shaded, makoto shinkai aesthetic, masterpiece",
    },
    "arcane": {
        "label": "🔮 Arcane Painterly",
        "prompt": "portrait, arcane style, painterly brushstrokes, dramatic side lighting, high contrast, masterpiece portrait, 8k",
    },
    "claymation": {
        "label": "🧱 3D Claymation",
        "prompt": "portrait, claymation artwork, tactile clay texture, plasticine, stop motion aesthetic, handmade character",
    },
    "ghibli": {
        "label": "🌿 Studio Ghibli",
        "prompt": "portrait, studio ghibli style, painted, whimsical, soft colors, hand-drawn, miyazaki art",
    },
    "comic": {
        "label": "💥 Comic Book",
        "prompt": "portrait, comic book illustration style, bold black ink outlines, halftone patterns, vibrant colors, dynamic pop art, Marvel comic style",
    },
    "pixar": {
        "label": "✨ Pixar 3D",
        "prompt": "portrait, pixar 3d animation style, cute character, warm lighting, subsurface scattering, detailed 3d render, artstation",
    },
    "vangogh": {
        "label": "🌻 Van Gogh",
        "prompt": "portrait, van gogh painting style, swirling brushstrokes, thick impasto texture, vibrant yellows and blues, post-impressionist, starry night aesthetic, museum quality oil painting",
    },
    "simpsons": {
        "label": "🟡 Simpsons",
        "prompt": "portrait, simpsons cartoon style, yellow skin tone, bold thick black outlines, flat solid colors, matt groening art style, springfield cartoon, 2d animation",
    },
    "cyberpunk": {
        "label": "🌆 Cyberpunk",
        "prompt": "portrait, cyberpunk art style, neon lights, dark futuristic city, glowing cyan and magenta, rain reflections, blade runner aesthetic, holographic, highly detailed, 4k digital art",
    },
    "watercolor": {
        "label": "🎨 Watercolor",
        "prompt": "portrait, watercolor painting, soft flowing colors, wet on wet technique, paper texture, delicate color washes, loose brushwork, fine art illustration",
    },
    "sketch": {
        "label": "✏️ Pencil Sketch",
        "prompt": "portrait, detailed pencil sketch, graphite drawing, crosshatching technique, fine line art, black and white, charcoal shading, hand-drawn, traditional art",
    },
}

NEGATIVE_PROMPT = (
    "ugly, blurry, low quality, deformed, disfigured, extra limbs, "
    "watermark, text, nsfw, realistic photo, photorealistic, grain, noise"
)

MODEL_ID = os.getenv("MODEL_ID", "nitrosocke/mo-di-diffusion")

# ---------------------------------------------------------------------------
# Lazy pipeline loader
# ---------------------------------------------------------------------------
_pipe = None

def _get_pipe():
    global _pipe
    if _pipe is not None:
        return _pipe

    logger.info(f"Loading model: {MODEL_ID}")
    from diffusers import StableDiffusionImg2ImgPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    _pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    if device == "cuda":
        try:
            _pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            _pipe.enable_attention_slicing()

    logger.info(f"Pipeline loaded on {device}.")
    return _pipe


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------
def cartoonize(
    image: Image.Image,
    style_key: str,
    strength: float,
    guidance_scale: float,
    num_steps: int,
) -> Image.Image:
    """Run SD img2img and return cartoonized image."""
    if image is None:
        raise gr.Error("Please provide an input image.")

    style = STYLES.get(style_key)
    if style is None:
        raise gr.Error(f"Unknown style '{style_key}'. Valid: {list(STYLES.keys())}")

    pipe = _get_pipe()
    img = image.convert("RGB").resize((512, 512), Image.LANCZOS)

    result = pipe(
        prompt=style["prompt"],
        negative_prompt=NEGATIVE_PROMPT,
        image=img,
        strength=float(strength),
        guidance_scale=float(guidance_scale),
        num_inference_steps=int(num_steps),
    ).images[0]

    return result


# ---------------------------------------------------------------------------
# Gradio UI + API
# ---------------------------------------------------------------------------
STYLE_CHOICES = list(STYLES.keys())

with gr.Blocks(
    title="TOONIFY Inference API",
    theme=gr.themes.Base(primary_hue="purple"),
    css=".gradio-container { max-width: 860px !important; }",
) as demo:
    gr.Markdown(
        "# 🎨 TOONIFY — Stable Diffusion Inference API\n"
        "Real GPU cartoon generation. Connected to [toonify-studio.vercel.app](https://toonify-studio.vercel.app)."
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Input Photo", height=300)
            style_key = gr.Dropdown(
                choices=STYLE_CHOICES,
                value="disney",
                label="Art Style",
            )
            with gr.Row():
                strength = gr.Slider(
                    minimum=0.4, maximum=0.9, value=0.70, step=0.01,
                    label="Style Strength",
                )
                guidance_scale = gr.Slider(
                    minimum=4.0, maximum=12.0, value=7.5, step=0.5,
                    label="Guidance Scale",
                )
            num_steps = gr.Slider(
                minimum=10, maximum=50, value=30, step=5,
                label="Inference Steps",
            )
            btn = gr.Button("🎨 Cartoonize", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_image = gr.Image(type="pil", label="Cartoon Result", height=300)

    btn.click(
        fn=cartoonize,
        inputs=[input_image, style_key, strength, guidance_scale, num_steps],
        outputs=output_image,
        api_name="cartoonize",   # exposes /api/predict & /run/cartoonize
    )

    gr.Examples(
        examples=[
            [None, "disney", 0.70, 7.5, 30],
        ],
        inputs=[input_image, style_key, strength, guidance_scale, num_steps],
        label="Example Settings",
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
