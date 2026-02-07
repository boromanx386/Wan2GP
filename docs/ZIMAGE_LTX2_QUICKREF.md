# Z-Image i LTX-2 - Brza Referenca

## 🚀 Brzo Startovanje

### Z-Image (Osnovni)
```python
from models.z_image.z_image_handler import family_handler

pipe_processor, pipe = family_handler.load_model(
    model_filename=["ZImageTurbo_bf16.safetensors"],
    base_model_type="z_image",
    dtype=torch.bfloat16,
)

images = pipe_processor.generate(
    seed=42,
    input_prompt="Your prompt",
    sampling_steps=8,
    width=1024,
    height=1024,
)
```

### Z-Image Control (ControlNet v2.1)
```python
pipe_processor, pipe = family_handler.load_model(
    model_filename=["ZImageTurbo_bf16.safetensors"],
    base_model_type="z_image_control2_1",
    model_def={
        "modules": [["Z-Image-Turbo-Fun-Controlnet-Union2.1_bf16.safetensors"]]
    },
    dtype=torch.bfloat16,
)

images = pipe_processor.generate(
    seed=42,
    input_prompt="Your prompt",
    input_frames=control_image,
    context_scale=[0.65],
    sampling_steps=9,
    width=1920,
    height=1088,
)
```

### LTX-2
```python
from models.ltx2.ltx2_handler import family_handler

ltx2_model, pipe = family_handler.load_model(
    model_filename=["ltx-2-19b-dev.safetensors"],
    base_model_type="ltx2_19B",
    model_def={"ltx2_pipeline": "two_stage"},
    dtype=torch.bfloat16,
)

result = ltx2_model.generate(
    input_prompt="Your prompt",
    sampling_steps=40,
    frame_num=241,
    height=1024,
    width=1536,
    fps=24.0,
    seed=42,
)
```

## 📦 Ključne Zavisnosti

```txt
mmgp==3.6.11          # KRITIČNO!
diffusers==0.34.0
transformers==4.53.1
torch>=2.6.0
```

## 📁 Handler Putanje

```python
family_handlers = [
    "models.z_image.z_image_handler",
    "models.ltx2.ltx2_handler",
]
```

## 🎯 Model Tipovi (TAČNO KOJI SE KORISTE)

### Z-Image
- `z_image` - Osnovni (ZImageTurbo_bf16.safetensors)
- `z_image_control` - ControlNet v1 (Z-Image-Turbo-Fun-Controlnet-Union_bf16.safetensors)
- `z_image_control2` - ControlNet v2 (Z-Image-Turbo-Fun-Controlnet-Union2_bf16.safetensors)
- `z_image_control2_1` - ControlNet v2.1 (Z-Image-Turbo-Fun-Controlnet-Union2.1_bf16.safetensors)
- `z_image` (TwinFlow) - TwinFlow-Z-Image-Turbo_bf16.safetensors

### LTX-2
- `ltx2_19B` (two_stage) - ltx-2-19b-dev.safetensors
- `ltx2_19B` (distilled) - ltx-2-19b-distilled.safetensors

## 🔑 Ključni Parametri

### Z-Image
- `guide_scale`: 0.0 (uvek!)
- `sampling_steps`: 8-9 (Turbo)
- `sample_solver`: "default" | "unified" | "twinflow"
- `NAG_scale/tau/alpha`: Za z_image varijantu

### LTX-2
- `sampling_steps`: 40 (two_stage) | 8 (distilled)
- `guidance_phases`: 2 (two_stage) | 1 (distilled)
- `sliding_window_size`: 481
- `sliding_window_overlap`: 9
- `audio_scale`: 1.0 (ako koristiš audio)

## 📦 HuggingFace Repos i Fajlovi

### Z-Image (`DeepBeepMeep/Z-Image`)
- Base: `ZImageTurbo_bf16.safetensors`
- Control v1: `Z-Image-Turbo-Fun-Controlnet-Union_bf16.safetensors`
- Control v2: `Z-Image-Turbo-Fun-Controlnet-Union2_bf16.safetensors`
- Control v2.1: `Z-Image-Turbo-Fun-Controlnet-Union2.1_bf16.safetensors`
- TwinFlow: `TwinFlow-Z-Image-Turbo_bf16.safetensors`
- VAE: `ZImageTurbo_VAE_bf16.safetensors`
- Text Encoder: `Qwen3/qwen3_bf16.safetensors`

### LTX-2 (`DeepBeepMeep/LTX-2`)
- Two-stage: `ltx-2-19b-dev.safetensors`
- Distilled: `ltx-2-19b-distilled.safetensors`
- Spatial Upscaler: `ltx-2-spatial-upscaler-x2-1.0.safetensors`
- LoRA: `ltx-2-19b-distilled-lora-384.safetensors` (opciono)
- Text Encoder: `gemma-3-12b-it-qat-q4_0-unquantized/`

## 🗂️ LoRA Direktorijumi

- Z-Image: `loras/z_image/`
- LTX-2: `loras/ltx2/`

## ⚙️ Default Settings

### Z-Image
```python
{
    "guidance_scale": 0.0,
    "num_inference_steps": 9,
    "NAG_scale": 1.0,
    "NAG_tau": 3.5,
    "NAG_alpha": 0.5
}
```

### LTX-2
```python
{
    "sliding_window_size": 481,
    "sliding_window_overlap": 9,
    "audio_scale": 1.0,
    "guidance_phases": 2  # ili 1 za distilled
}
```

## 🔧 Text Encoders

- **Z-Image**: Qwen3 (`Qwen3/qwen3_bf16.safetensors`)
- **LTX-2**: Gemma-3-12B-IT (`gemma-3-12b-it-qat-q4_0-unquantized/`)

## 📝 Pipeline Tipovi (LTX-2)

- `"two_stage"` - Dvostepeni, bolji kvalitet
- `"distilled"` - Brži, 8 koraka umesto 40

## ⚠️ Važno

1. **Z-Image**: `guide_scale` uvek 0.0!
2. **LTX-2**: Audio se generiše automatski
3. **MMGP**: Obavezno `mmgp==3.6.11`
4. **VAE**: Z-Image koristi custom AutoencoderKL
5. **RGB Faktori**: Z-Image koristi "flux", LTX-2 koristi "ltx2"
