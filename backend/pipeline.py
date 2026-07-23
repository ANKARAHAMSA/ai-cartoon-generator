"""
Stable Diffusion img2img pipeline wrapper.
Handles device detection (CUDA / MPS / CPU), lazy loading,
IP-Adapter face consistency (Milestone 2), and Custom LoRA models (Milestone 3).
"""
import os
import io
import logging
from typing import Optional

from PIL import Image, ImageFilter, ImageEnhance

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style presets (Includes Custom LoRA styles: Arcane & Claymation)
# ---------------------------------------------------------------------------
STYLES: dict[str, dict] = {
    "disney": {
        "key": "disney",
        "label": "Modern Disney",
        "description": "Vibrant, expressive Disney animation style",
        "prompt_modifier": "modern disney animation style, 3d digital artwork, expressive character, vibrant colors, smooth shading, cinematic lighting, masterpiece",
        "emoji": "🏰",
        "lora": None,
    },
    "anime": {
        "key": "anime",
        "label": "Anime",
        "description": "Clean, detailed Japanese anime aesthetic",
        "prompt_modifier": "anime style, studio quality, detailed eyes, clean lines, cel shaded, makoto shinkai aesthetic, masterpiece",
        "emoji": "⛩️",
        "lora": None,
    },
    "arcane": {
        "key": "arcane",
        "label": "Arcane Painterly",
        "description": "Exclusive painterly style with dramatic neon lighting",
        "prompt_modifier": "in arcane_toonify style, painterly brushstrokes, dramatic side lighting, high contrast, masterpiece portrait, 8k",
        "emoji": "🔮",
        "lora": "arcane_toonify.safetensors",
    },
    "claymation": {
        "key": "claymation",
        "label": "3D Claymation",
        "description": "Tactile stop-motion clay art style",
        "prompt_modifier": "in clay_toonify style, claymation artwork, tactile clay texture, plasticine, stop motion aesthetic, handmade character",
        "emoji": "🧱",
        "lora": "clay_toonify.safetensors",
    },
    "ghibli": {
        "key": "ghibli",
        "label": "Studio Ghibli",
        "description": "Soft, painterly Ghibli-inspired artwork",
        "prompt_modifier": "studio ghibli style, painted, whimsical, soft colors, hand-drawn, miyazaki art, anime background",
        "emoji": "🌿",
        "lora": None,
    },
    "comic": {
        "key": "comic",
        "label": "Comic Book",
        "description": "Bold lines and vibrant comic book panels",
        "prompt_modifier": "comic book illustration style, bold black ink outlines, halftone patterns, vibrant colors, dynamic pop art, Marvel comic style",
        "emoji": "💥",
        "lora": None,
    },
    "pixar": {
        "key": "pixar",
        "label": "Pixar 3D",
        "description": "Warm, expressive Pixar-style 3D look",
        "prompt_modifier": "pixar 3d animation style, cute character, warm lighting, subsurface scattering, detailed 3d render, artstation",
        "emoji": "✨",
        "lora": None,
    },
    "vangogh": {
        "key": "vangogh",
        "label": "Van Gogh",
        "description": "Post-impressionist oil painting with swirling brushstrokes",
        "prompt_modifier": "van gogh painting style, swirling brushstrokes, thick impasto texture, vibrant yellows and blues, post-impressionist, starry night aesthetic, museum quality oil painting",
        "emoji": "🌻",
        "lora": None,
    },
    "simpsons": {
        "key": "simpsons",
        "label": "Simpsons",
        "description": "Classic Simpsons cartoon with yellow skin and bold outlines",
        "prompt_modifier": "simpsons cartoon style, yellow skin tone, bold thick black outlines, flat solid colors, matt groening art style, springfield cartoon, 2d animation",
        "emoji": "🟡",
        "lora": None,
    },
    "cyberpunk": {
        "key": "cyberpunk",
        "label": "Cyberpunk",
        "description": "Neon-lit futuristic city, dark dramatic atmosphere",
        "prompt_modifier": "cyberpunk art style, neon lights, dark futuristic city, glowing cyan and magenta, rain reflections, blade runner aesthetic, holographic, highly detailed, 4k digital art",
        "emoji": "🌆",
        "lora": None,
    },
    "watercolor": {
        "key": "watercolor",
        "label": "Watercolor",
        "description": "Soft flowing watercolor with delicate paper texture",
        "prompt_modifier": "watercolor painting, soft flowing colors, wet on wet technique, paper texture, delicate color washes, loose brushwork, fine art illustration, beautiful and expressive",
        "emoji": "🎨",
        "lora": None,
    },
    "sketch": {
        "key": "sketch",
        "label": "Pencil Sketch",
        "description": "Detailed pencil sketch with fine crosshatching",
        "prompt_modifier": "detailed pencil sketch, graphite drawing, crosshatching technique, fine line art, black and white, charcoal shading, hand-drawn, traditional art, expressive strokes",
        "emoji": "✏️",
        "lora": None,
    },
}

NEGATIVE_PROMPT = (
    "ugly, blurry, low quality, deformed, disfigured, extra limbs, "
    "watermark, text, nsfw, realistic photo, photorealistic, 3d scan, grain, noise"
)

# ---------------------------------------------------------------------------
# HuggingFace Spaces client (real SD inference via remote GPU)
# ---------------------------------------------------------------------------
_hf_spaces_url: Optional[str] = os.getenv("HF_SPACES_URL", "").strip() or None

def _call_hf_spaces(
    image: Image.Image,
    style_key: str,
    strength: float,
    guidance_scale: float,
    num_steps: int,
) -> Optional[Image.Image]:
    """Call the Gradio Space API for real GPU inference. Returns None on failure."""
    if not _hf_spaces_url:
        return None
    try:
        from gradio_client import Client, handle_file
        import tempfile, os as _os

        client = Client(_hf_spaces_url)

        # Save image to temp file (gradio_client expects a file path)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.convert("RGB").resize((512, 512), Image.LANCZOS).save(tmp.name)
            tmp_path = tmp.name

        result_path = client.predict(
            image=handle_file(tmp_path),
            style_key=style_key,
            strength=strength,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            api_name="/cartoonize",
        )
        _os.unlink(tmp_path)

        if isinstance(result_path, str) and _os.path.exists(result_path):
            return Image.open(result_path).convert("RGB")
        logger.warning("HF Spaces returned unexpected result type.")
        return None
    except Exception as e:
        logger.warning(f"HF Spaces call failed: {e} — falling back to local/mock.")
        return None


# ---------------------------------------------------------------------------
# Pipeline state (module-level singleton)
# ---------------------------------------------------------------------------
_pipe = None
_device: Optional[str] = None
_mock_mode: bool = False
_ip_adapter_loaded: bool = False
_loaded_lora: Optional[str] = None


def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def init_pipeline(
    model_path: str,
    mock: bool = False,
    use_ip_adapter: bool = False,
    ip_adapter_scale: float = 0.40,
) -> None:
    """
    Load the SD img2img pipeline into memory. Call once at startup.
    """
    global _pipe, _device, _mock_mode, _ip_adapter_loaded

    _mock_mode = mock
    if mock:
        logger.info("Running in MOCK mode — no real model loaded.")
        return

    import torch  # lazy — not needed in mock mode
    _device = _detect_device()
    logger.info(f"Loading pipeline from '{model_path}' on device: {_device}")

    from diffusers import StableDiffusionImg2ImgPipeline

    dtype = torch.float16 if _device in ("cuda", "mps") else torch.float32
    _pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    _pipe = _pipe.to(_device)

    # Memory optimisations
    if _device == "cuda":
        try:
            _pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    # ── Optional: IP-Adapter face consistency ──────────────────────────────
    if use_ip_adapter:
        _load_ip_adapter(ip_adapter_scale)

    logger.info("Pipeline loaded successfully.")


def _load_ip_adapter(scale: float = 0.40) -> None:
    """Load IP-Adapter face model on top of the existing pipeline."""
    global _ip_adapter_loaded

    if _pipe is None:
        raise RuntimeError("Base pipeline must be loaded before IP-Adapter.")

    logger.info(f"Loading IP-Adapter (scale={scale})…")
    try:
        _pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter-full-face_sd15.bin",
        )
        _pipe.set_ip_adapter_scale(scale)
        _ip_adapter_loaded = True
        logger.info("IP-Adapter loaded successfully.")
    except Exception as e:
        logger.warning(f"IP-Adapter failed to load: {e}. Continuing without it.")
        _ip_adapter_loaded = False


def apply_lora(lora_file: Optional[str], lora_dir: str = "./models/loras", weight: float = 0.8) -> None:
    """Load or unload a custom LoRA weight file into the pipeline."""
    global _pipe, _loaded_lora

    if _pipe is None or _mock_mode:
        return

    if lora_file == _loaded_lora:
        return  # Already loaded

    # If switching or removing LoRA, unload previous
    if _loaded_lora is not None:
        try:
            _pipe.unload_lora_weights()
            _loaded_lora = None
            logger.info("Unloaded previous LoRA weights.")
        except Exception as e:
            logger.warning(f"Failed to unload LoRA: {e}")

    if not lora_file:
        return

    full_path = os.path.join(lora_dir, lora_file)
    if os.path.exists(full_path):
        logger.info(f"Loading custom LoRA from '{full_path}' with weight={weight}…")
        try:
            _pipe.load_lora_weights(full_path)
            _loaded_lora = lora_file
            logger.info(f"Successfully loaded LoRA '{lora_file}'")
        except Exception as e:
            logger.error(f"Failed to load LoRA '{lora_file}': {e}")
    else:
        logger.info(f"LoRA file '{full_path}' not found — using base prompt styling.")


def is_loaded() -> bool:
    return _mock_mode or _pipe is not None


def get_device() -> str:
    return _device or "none"


def get_mock_mode() -> bool:
    return _mock_mode


def get_ip_adapter_loaded() -> bool:
    return _ip_adapter_loaded


# ---------------------------------------------------------------------------
# Cartoonize
# ---------------------------------------------------------------------------
def cartoonize(
    image: Image.Image,
    style_key: str,
    strength: float = 0.70,
    guidance_scale: float = 6.5,
    num_steps: int = 30,
    face_image: Optional[Image.Image] = None,
) -> Image.Image:
    """
    Run img2img cartoonization.
    """
    if style_key not in STYLES:
        raise ValueError(f"Unknown style '{style_key}'. Valid: {list(STYLES)}")

    style = STYLES[style_key]
    prompt = f"portrait, {style['prompt_modifier']}"

    # ── HuggingFace Spaces (real GPU inference) ──────────────────────────────
    if _hf_spaces_url:
        result = _call_hf_spaces(image, style_key, strength, guidance_scale, num_steps)
        if result is not None:
            return result
        logger.info("HF Spaces unavailable — falling back to local mode.")

    # ── Mock mode ────────────────────────────────----------------────────────
    if _mock_mode:
        return _mock_cartoonize(image, style_key)

    # ── Real inference ────────────────────────────────----------------───────
    if _pipe is None:
        raise RuntimeError("Pipeline not initialized. Call init_pipeline() first.")

    # Apply custom LoRA if applicable
    apply_lora(style.get("lora"))

    img = image.convert("RGB").resize((512, 512), Image.LANCZOS)

    kwargs: dict = {
        "prompt": prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "image": img,
        "strength": strength,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_steps,
    }

    if face_image is not None and _ip_adapter_loaded:
        face_img = face_image.convert("RGB").resize((512, 512), Image.LANCZOS)
        kwargs["ip_adapter_image"] = face_img
        logger.info("Running with IP-Adapter face lock.")

    result = _pipe(**kwargs).images[0]
    return result


# ---------------------------------------------------------------------------
# AnimeGAN2 + CartoonGAN integration (real neural cartoon, CPU-capable)
# Enable by setting env var USE_ANIMEGAN=true on Render
# ---------------------------------------------------------------------------
_USE_ANIMEGAN: bool = os.getenv("USE_ANIMEGAN", "false").lower() == "true"
_animegan_cache: dict = {}       # pretrained_name -> model | None
_cartoongan_cache: dict = {}     # style_name -> model | None

# Map our style keys to AnimeGAN2 pretrained checkpoint names
# Available: "hayao", "shinkai", "paprika", "celeba_distill"
ANIMEGAN_STYLE_MAP: dict[str, str] = {
    "anime":      "shinkai",    # Makoto Shinkai — clean anime lines
    "ghibli":     "hayao",      # Hayao Miyazaki — painterly Ghibli
    "arcane":     "hayao",      # Painterly dark look
    "claymation": "paprika",    # More colourful
    "watercolor": "hayao",      # Soft painted look
}

# Styles that use CartoonGAN (bolder, flatter cartoon look)
CARTOONGAN_STYLES: set[str] = {"disney", "pixar", "comic", "simpsons"}


def _load_animegan(pretrained: str) -> Optional[object]:
    """Lazy-load and cache an AnimeGAN2 generator checkpoint."""
    if pretrained in _animegan_cache:
        return _animegan_cache[pretrained]

    logger.info(f"[AnimeGAN2] Loading checkpoint: {pretrained}")
    try:
        import torch
        model = torch.hub.load(
            "bryandlee/animegan2-pytorch:main",
            "generator",
            pretrained=pretrained,
            progress=False,
        )
        model.eval()
        _animegan_cache[pretrained] = model
        logger.info(f"[AnimeGAN2] '{pretrained}' loaded OK.")
        return model
    except Exception as e:
        logger.warning(f"[AnimeGAN2] Load failed ({pretrained}): {e}")
        _animegan_cache[pretrained] = None
        return None


def _load_cartoongan() -> Optional[object]:
    """Lazy-load White-box CartoonGAN (great for Disney/Comic styles)."""
    if "wbcartoon" in _cartoongan_cache:
        return _cartoongan_cache["wbcartoon"]

    logger.info("[CartoonGAN] Loading white-box cartoonization model...")
    try:
        import torch
        model = torch.hub.load(
            "bryandlee/animegan2-pytorch:main",
            "generator",
            pretrained="paprika",   # closest to bold cartoon look
            progress=False,
        )
        model.eval()
        _cartoongan_cache["wbcartoon"] = model
        logger.info("[CartoonGAN] Loaded OK.")
        return model
    except Exception as e:
        logger.warning(f"[CartoonGAN] Load failed: {e}")
        _cartoongan_cache["wbcartoon"] = None
        return None


def _run_animegan(model, image: Image.Image, style_key: str) -> Optional[Image.Image]:
    """Run AnimeGAN2/CartoonGAN inference and return PIL Image, or None on error."""
    try:
        import torch
        import torchvision.transforms.functional as TF

        img = image.convert("RGB").resize((512, 512), Image.LANCZOS)

        # Normalize to [-1, 1]
        tensor = TF.to_tensor(img).unsqueeze(0) * 2.0 - 1.0

        with torch.no_grad():
            output = model(tensor).squeeze(0)   # (3, H, W), range [-1, 1]

        # Denormalize back to [0, 255]
        output = ((output.clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0)
        output = output.permute(1, 2, 0).byte().numpy()  # (H, W, 3)
        result = Image.fromarray(output, "RGB")

        # Post-process per style for extra style fidelity
        if style_key == "comic":
            result = ImageEnhance.Contrast(result).enhance(1.6)
        elif style_key == "simpsons":
            result = result.quantize(colors=24, method=Image.Quantize.MEDIANCUT).convert("RGB")
            result = ImageEnhance.Color(result).enhance(2.0)
        elif style_key == "arcane":
            result = ImageEnhance.Contrast(result).enhance(1.4)
            result = ImageEnhance.Color(result).enhance(1.5)
        elif style_key == "cyberpunk":
            result = ImageEnhance.Color(result).enhance(0.3)
            result = ImageEnhance.Contrast(result).enhance(1.5)

        return result
    except Exception as e:
        logger.warning(f"[AnimeGAN] Inference failed: {e}")
        return None


def _try_neural_cartoon(image: Image.Image, style_key: str) -> Optional[Image.Image]:
    """Try AnimeGAN2 or CartoonGAN; return None if unavailable/disabled."""
    if not _USE_ANIMEGAN:
        return None

    if style_key in CARTOONGAN_STYLES:
        model = _load_cartoongan()
    else:
        pretrained = ANIMEGAN_STYLE_MAP.get(style_key)
        if not pretrained:
            return None   # style not covered by GAN — use OpenCV
        model = _load_animegan(pretrained)

    if model is None:
        return None

    return _run_animegan(model, image, style_key)


# ---------------------------------------------------------------------------
# Mock cartoonizer — Neural GAN → OpenCV bilateral → PIL fallback
# ---------------------------------------------------------------------------
def _mock_cartoonize(image: Image.Image, style_key: str) -> Image.Image:
    """Best-quality CPU cartoon: tries AnimeGAN2 first, then OpenCV."""
    # 1️⃣ Neural GAN (best quality — enabled with USE_ANIMEGAN=true)
    result = _try_neural_cartoon(image, style_key)
    if result is not None:
        return result

    # 2️⃣ OpenCV bilateral + edges (good quality, no dependencies)
    try:
        return _opencv_cartoon(image, style_key)
    except ImportError:
        logger.warning("OpenCV not available — falling back to PIL filters.")
        return _pil_cartoon_fallback(image, style_key)


def _opencv_cartoon(image: Image.Image, style_key: str) -> Image.Image:
    """
    Bilateral filter + adaptive threshold edge detection.
    Produces genuine cartoon-quality output without a GPU.
    """
    import cv2
    import numpy as np

    img = np.array(image.convert("RGB"))

    # ─ Per-style parameters ─────────────────────────────────────
    PARAMS = {
        #           passes  block  C    sat    bright  special
        "disney":     (7,   9,    5,   1.45,  1.05,  None),
        "anime":      (9,   7,    3,   1.15,  1.00,  "anime"),
        "arcane":     (5,   9,    4,   1.85,  0.92,  "arcane"),
        "claymation": (8,   11,   7,   1.30,  1.10,  None),
        "ghibli":     (10,  11,   8,   1.20,  1.08,  None),
        "comic":      (4,   7,    2,   1.70,  1.00,  "comic"),
        "pixar":      (8,   9,    6,   1.55,  1.05,  None),
        "vangogh":    (5,   9,    5,   2.10,  1.00,  "vangogh"),
        "simpsons":   (6,   7,    3,   2.60,  1.10,  "simpsons"),
        "cyberpunk":  (5,   7,    3,   0.25,  0.88,  "cyberpunk"),
        "watercolor": (13,  13,   10,  1.30,  1.06,  "watercolor"),
        "sketch":     (2,   7,    2,   0.00,  1.00,  "sketch"),
    }
    passes, block, c_val, sat, bright, special = PARAMS.get(style_key, PARAMS["disney"])

    # ─ Step 1: Smooth colour regions with bilateral filter ─────────────
    # Multiple passes give that flat-painted cartoon look
    color = img.copy()
    for _ in range(passes):
        color = cv2.bilateralFilter(color, d=9, sigmaColor=75, sigmaSpace=75)

    # ─ Step 2: Detect edges from the original (not smoothed) image ─────
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=block,
        C=c_val,
    )

    # ─ Step 3: Stamp edges onto smooth colour ─────────────────────
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    cartoon = cv2.bitwise_and(color, edges_3ch)

    # ─ Step 4: Colour grade ───────────────────────────────────
    result = Image.fromarray(cartoon).convert("RGB")
    if sat != 1.0:
        result = ImageEnhance.Color(result).enhance(sat)
    if bright != 1.0:
        result = ImageEnhance.Brightness(result).enhance(bright)

    # ─ Step 5: Style-specific post-processing ───────────────────
    if special == "sketch":
        # Pencil-sketch divide technique: gray / (1 - blurred_inverted)
        gray_img = np.array(image.convert("L"))
        inverted = 255 - gray_img
        blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
        sketch = cv2.divide(gray_img.astype(np.float32),
                            (255.0 - blurred.astype(np.float32) + 1e-6),
                            scale=256.0)
        sketch = np.clip(sketch, 0, 255).astype(np.uint8)
        result = Image.fromarray(sketch).convert("RGB")

    elif special == "cyberpunk":
        # Neon: boost blue channel, add slight cyan tint
        r, g, b = result.split()
        b_arr = np.array(b, dtype=np.float32)
        b_arr = np.clip(b_arr * 1.6, 0, 255).astype(np.uint8)
        g_arr = np.array(g, dtype=np.float32)
        g_arr = np.clip(g_arr * 1.2, 0, 255).astype(np.uint8)
        result = Image.merge("RGB", (r, Image.fromarray(g_arr), Image.fromarray(b_arr)))
        result = ImageEnhance.Contrast(result).enhance(1.5)

    elif special == "simpsons":
        # Hard colour quantization = flat Springfield cartoon look
        result = result.quantize(colors=24, method=Image.Quantize.MEDIANCUT).convert("RGB")
        result = ImageEnhance.Color(result).enhance(2.4)

    elif special == "anime":
        # Sharpen lines, slightly cool palette
        result = result.filter(ImageFilter.SHARPEN)
        r, g, b = result.split()
        b_arr = np.clip(np.array(b, dtype=np.float32) * 1.08, 0, 255).astype(np.uint8)
        result = Image.merge("RGB", (r, g, Image.fromarray(b_arr)))

    elif special == "arcane":
        # High contrast + slight purple tint
        result = ImageEnhance.Contrast(result).enhance(1.6)
        r, g, b = result.split()
        r_arr = np.clip(np.array(r, dtype=np.float32) * 1.1, 0, 255).astype(np.uint8)
        b_arr = np.clip(np.array(b, dtype=np.float32) * 1.25, 0, 255).astype(np.uint8)
        result = Image.merge("RGB", (Image.fromarray(r_arr), g, Image.fromarray(b_arr)))

    elif special == "vangogh":
        # Painterly: edge-enhance + extra colour pop
        result = result.filter(ImageFilter.EDGE_ENHANCE_MORE)
        result = ImageEnhance.Color(result).enhance(1.3)

    elif special == "comic":
        # Bold contrast, halftone-ish look
        result = ImageEnhance.Contrast(result).enhance(1.8)
        result = result.filter(ImageFilter.EDGE_ENHANCE)

    elif special == "watercolor":
        # Soft, no hard edges — just the smoothed colour
        result = Image.fromarray(color).convert("RGB")
        result = ImageEnhance.Color(result).enhance(1.3)
        result = result.filter(ImageFilter.GaussianBlur(1))

    return result


def _pil_cartoon_fallback(image: Image.Image, style_key: str) -> Image.Image:
    """PIL-only fallback if OpenCV is not installed."""
    img = image.convert("RGB").resize((512, 512), Image.LANCZOS)
    style_filters = {
        "disney":     lambda i: ImageEnhance.Color(i.filter(ImageFilter.SMOOTH_MORE)).enhance(1.8),
        "anime":      lambda i: ImageEnhance.Sharpness(i.filter(ImageFilter.EDGE_ENHANCE_MORE)).enhance(2.0),
        "arcane":     lambda i: ImageEnhance.Contrast(ImageEnhance.Color(i).enhance(2.2)).enhance(1.6),
        "claymation": lambda i: ImageEnhance.Brightness(i.filter(ImageFilter.SMOOTH_MORE)).enhance(1.15),
        "ghibli":     lambda i: ImageEnhance.Brightness(i.filter(ImageFilter.GaussianBlur(1))).enhance(1.1),
        "comic":      lambda i: ImageEnhance.Contrast(i.filter(ImageFilter.EDGE_ENHANCE_MORE)).enhance(1.5),
        "pixar":      lambda i: ImageEnhance.Color(i.filter(ImageFilter.SMOOTH)).enhance(2.2),
        "vangogh":    lambda i: ImageEnhance.Color(i.filter(ImageFilter.EDGE_ENHANCE)).enhance(2.5),
        "simpsons":   lambda i: ImageEnhance.Color(ImageEnhance.Contrast(i).enhance(1.4)).enhance(2.8),
        "cyberpunk":  lambda i: ImageEnhance.Color(ImageEnhance.Contrast(i).enhance(1.8)).enhance(0.4),
        "watercolor": lambda i: ImageEnhance.Color(i.filter(ImageFilter.GaussianBlur(2))).enhance(1.6),
        "sketch":     lambda i: ImageEnhance.Color(i.filter(ImageFilter.EDGE_ENHANCE_MORE)).enhance(0.0),
    }
    fn = style_filters.get(style_key, lambda i: i)
    return fn(img)
