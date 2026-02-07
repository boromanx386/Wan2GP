# Z-Image i LTX-2 Modeli - Migracija Dokumentacija

Ovaj dokument sadrži sve potrebne informacije za korišćenje Z-Image i LTX-2 modela u drugom projektu sa drugim venv okruženjem.

## 📋 Pregled Modela - TAČNO KOJI SE KORISTE

### Z-Image Modeli (Korišćeni u projektu)

#### 1. Z-Image Turbo 6B (Osnovni)
- **Architecture**: `z_image`
- **Model File**: `ZImageTurbo_bf16.safetensors` ili `ZImageTurbo_quanto_bf16_int8.safetensors`
- **Config**: `defaults/z_image.json`
- **Opis**: Osnovni Z-Image Turbo text-to-image model
- **Steps**: 8-9 koraka
- **Resolution**: 1024x1024

#### 2. Z-Image Turbo Fun ControlNet v1 6B
- **Architecture**: `z_image_control`
- **Model File**: `Z-Image-Turbo-Fun-Controlnet-Union_bf16.safetensors` (kao modul)
- **Base Model**: Koristi `z_image` base model + ControlNet modul
- **Config**: `defaults/z_image_control.json`
- **Opis**: ControlNet Union sa Pose, Canny, Depth, Scribble kontrolama
- **Control Weight**: 0.75

#### 3. Z-Image Turbo Fun ControlNet v2 6B
- **Architecture**: `z_image_control2`
- **Model File**: `Z-Image-Turbo-Fun-Controlnet-Union2_bf16.safetensors` (kao modul)
- **Base Model**: Koristi `z_image` base model + ControlNet v2 modul
- **Config**: `defaults/z_image_control2.json`
- **Opis**: ControlNet v2 sa više control layera i inpainting podrškom
- **Control Weight**: 0.75

#### 4. Z-Image Turbo Fun ControlNet v2.1 6B
- **Architecture**: `z_image_control2_1`
- **Model File**: `Z-Image-Turbo-Fun-Controlnet-Union2.1_bf16.safetensors` (kao modul)
- **Base Model**: Koristi `z_image` base model + ControlNet v2.1 modul
- **Config**: `defaults/z_image_control2_1.json`
- **Opis**: ControlNet v2.1 sa još više control layera
- **Resolution**: 1920x1088
- **Control Weight**: 0.65

#### 5. TwinFlow Z-Image Turbo
- **Architecture**: `z_image`
- **Model File**: `TwinFlow-Z-Image-Turbo_bf16.safetensors`
- **Config**: `defaults/z_image_twinflow_turbo.json`
- **Opis**: Distilled finetune optimizovan za 1-4 koraka
- **Steps**: 2 koraka
- **Unified Solver**: true

### LTX-2 Modeli (Korišćeni u projektu)

#### 1. LTX-2 Dev 19B (Two-Stage)
- **Architecture**: `ltx2_19B`
- **Model File**: `ltx-2-19b-dev.safetensors` ili `ltx-2-19b-dev-fp8.safetensors`
- **Pipeline**: `two_stage`
- **Config**: `defaults/ltx2_19B.json`
- **Opis**: Generiše video do 20s sa audio soundtrack-om
- **Steps**: 40 koraka
- **Guidance Phases**: 2
- **LoRA**: `ltx-2-19b-distilled-lora-384.safetensors` (opciono)

#### 2. LTX-2 Distilled 19B
- **Architecture**: `ltx2_19B`
- **Model File**: `ltx-2-19b-distilled.safetensors` ili `ltx-2-19b-distilled-fp8.safetensors`
- **Pipeline**: `distilled`
- **Config**: `defaults/ltx2_distilled.json`
- **Opis**: Brža verzija sa manje koraka
- **Steps**: 8 koraka
- **Guidance Phases**: 1

---

## 🔧 Zavisnosti (Dependencies)

### Core Dependencies
```txt
# Core AI stack
diffusers==0.34.0
transformers==4.53.1
torch>=2.6.0  # CUDA 12.4+ potreban
mmgp==3.6.11  # KRITIČNO - za offloading i quantization

# Text Encoders
# Z-Image koristi Qwen3
# LTX-2 koristi Gemma-3-12B-IT

# Utilities
omegaconf
hydra-core
accelerate>=1.1.1
```

### Z-Image Specifične Zavisnosti
- `Qwen3ForCausalLM` iz transformers
- `FlowMatchEulerDiscreteScheduler` iz diffusers
- `AutoencoderKL` (custom implementacija u `models/z_image/autoencoder_kl.py`)

### LTX-2 Specifične Zavisnosti
- `gemma-3-12b-it-qat-q4_0-unquantized` text encoder
- Audio processing: `torchaudio`
- Video/Audio VAE komponente

---

## 📁 Struktura Fajlova

### Z-Image Struktura
```
models/z_image/
├── __init__.py
├── z_image_handler.py          # Handler za registraciju modela
├── z_image_main.py             # Model factory i glavna logika
├── pipeline_z_image.py         # Diffusion pipeline
├── z_image_transformer2d.py    # Transformer arhitektura
├── autoencoder_kl.py            # VAE implementacija
├── unified_sampler.py          # Sampler za unified solver
├── pipeline_output.py          # Output klasa
└── configs/
    ├── z_image.json            # Osnovna konfiguracija
    ├── z_image_control.json
    ├── z_image_control2.json
    └── z_image_control2_1.json
```

### LTX-2 Struktura
```
models/ltx2/
├── __init__.py
├── ltx2_handler.py             # Handler za registraciju modela
├── ltx2.py                     # Glavna LTX2 klasa
├── ltx_core/                   # Core komponente
│   ├── components/             # Osnovne komponente
│   ├── conditioning/           # Conditioning logika
│   ├── guidance/               # Guidance mehanizmi
│   ├── loader/                 # Model loading
│   ├── model/                  # Model arhitekture
│   │   ├── audio_vae.py
│   │   ├── video_vae.py
│   │   └── transformer/        # Transformer implementacija
│   ├── text_encoders/          # Text encoder (Gemma)
│   └── tools.py
└── ltx_pipelines/              # Pipeline implementacije
    ├── distilled.py            # Distilled pipeline
    ├── ti2vid_two_stages.py    # Two-stage pipeline
    ├── ti2vid_one_stage.py
    ├── keyframe_interpolation.py
    ├── ic_lora.py
    └── utils/
```

---

## 🔌 Registracija Modela

### U glavnom fajlu (wgp.py ekvivalent)
```python
family_handlers = [
    "models.z_image.z_image_handler",
    "models.ltx2.ltx2_handler",
    # ... ostali handleri
]
```

### Handler Interface
Oba handlera implementiraju `family_handler` klasu sa sledećim metodama:

#### Obavezni Metodi:
- `query_supported_types()` - Vraća listu podržanih tipova modela
- `query_model_family()` - Vraća naziv familije ("z_image" ili "ltx2")
- `query_family_infos()` - Vraća dict sa info o modelima
- `query_model_def()` - Vraća dodatne definicije modela
- `load_model()` - Učitava model i vraća pipeline + pipe dict
- `query_model_files()` - Vraća listu fajlova za download sa HuggingFace

#### Opcioni Metodi:
- `register_lora_cli_args()` - Registruje CLI argumente za LoRA direktorijume
- `get_lora_dir()` - Vraća putanju do LoRA direktorijuma
- `get_rgb_factors()` - Vraća RGB faktore za latent space
- `update_default_settings()` - Ažurira default UI postavke
- `validate_generative_settings()` - Validira input parametre

---

## 🚀 Učitavanje Modela

### Z-Image Loading

#### Osnovni Model (z_image)
```python
from models.z_image.z_image_handler import family_handler

pipe_processor, pipe = family_handler.load_model(
    model_filename=["ZImageTurbo_bf16.safetensors"],
    model_type=None,
    base_model_type="z_image",
    model_def={},
    quantizeTransformer=False,
    text_encoder_quantization=None,  # ili "int8" za quantized text encoder
    dtype=torch.bfloat16,
    VAE_dtype=torch.float32,
    mixed_precision_transformer=False,
    save_quantized=False,
)
```

#### Control Varijante (z_image_control, z_image_control2, z_image_control2_1)
```python
# Control varijante koriste base model + ControlNet modul
# model_def mora da sadrži "modules" sa putanjom do ControlNet modula
pipe_processor, pipe = family_handler.load_model(
    model_filename=["ZImageTurbo_bf16.safetensors"],  # Base model
    model_type=None,
    base_model_type="z_image_control2_1",  # ili "z_image_control", "z_image_control2"
    model_def={
        "modules": [
            ["Z-Image-Turbo-Fun-Controlnet-Union2.1_bf16.safetensors"]  # ControlNet modul
        ]
    },
    quantizeTransformer=False,
    text_encoder_quantization=None,
    dtype=torch.bfloat16,
    VAE_dtype=torch.float32,
    mixed_precision_transformer=False,
    save_quantized=False,
)
```

**Važno**: Control varijante automatski učitavaju base model iz `z_image` konfiguracije, a zatim dodaju ControlNet modul iz `modules` liste.

**Kako funkcioniše učitavanje modula**:
- `model_def.get("modules")` sadrži listu listi sa putanjama do ControlNet modula
- Prvi element `model_filename` je base model
- `module_source` se ekstraktuje iz `modules[0][0]` (prvi modul)
- Modul se merge-uje sa base modelom tokom učitavanja
- Control varijante imaju `is_control=True` što aktivira control layer-e u transformer-u

# pipe_processor ima:
# - transformer (ZImageTransformer2DModel)
# - text_encoder (Qwen3ForCausalLM)
# - vae (AutoencoderKL)
# - scheduler (FlowMatchEulerDiscreteScheduler)
# - pipeline (ZImagePipeline)

# pipe dict sadrži:
# {
#     "transformer": pipe_processor.transformer,
#     "text_encoder": pipe_processor.text_encoder,
#     "vae": pipe_processor.vae,
# }
```

### LTX-2 Loading
```python
from models.ltx2.ltx2_handler import family_handler

# Handler poziva LTX2 klasu iz ltx2.py
ltx2_model, pipe = family_handler.load_model(
    model_filename=["ltx-2-19b-dev.safetensors"],
    model_type=None,
    base_model_type="ltx2_19B",
    model_def={
        "ltx2_pipeline": "two_stage"  # ili "distilled" za distilled verziju
    },
    quantizeTransformer=False,
    text_encoder_quantization=None,  # ili "int8"
    dtype=torch.bfloat16,
    VAE_dtype=torch.float32,
    mixed_precision_transformer=False,
    save_quantized=False,
)

# ltx2_model ima:
# - model (transformer)
# - model2 (drugi stage transformer, ako je two_stage)
# - text_encoder (Gemma)
# - text_embedding_projection
# - text_embeddings_connector
# - video_encoder
# - video_decoder (vae)
# - audio_encoder
# - audio_decoder
# - vocoder
# - spatial_upsampler

# pipe dict za two_stage:
# {
#     "pipe": {
#         "transformer": ltx2_model.model,
#         "transformer2": ltx2_model.model2,  # ako postoji
#         "text_encoder": ltx2_model.text_encoder,
#         "text_embedding_projection": ltx2_model.text_embedding_projection,
#         "text_embeddings_connector": ltx2_model.text_embeddings_connector,
#         "vae": ltx2_model.video_decoder,
#         "video_encoder": ltx2_model.video_encoder,
#         "audio_encoder": ltx2_model.audio_encoder,
#         "audio_decoder": ltx2_model.audio_decoder,
#         "vocoder": ltx2_model.vocoder,
#         "spatial_upsampler": ltx2_model.spatial_upsampler,
#     },
#     "loras": ["text_embedding_projection", "text_embeddings_connector"]
# }
```

---

## 🎯 Generisanje (Generation)

### Z-Image Generation
```python
# Kroz pipe_processor.generate()
images = pipe_processor.generate(
    seed=42,
    input_prompt="A beautiful landscape",
    n_prompt="blurry, low quality",
    sampling_steps=8,  # Default za Turbo
    sample_solver="default",  # ili "unified", "unified_2s", "unified_4s", "twinflow"
    width=1024,
    height=1024,
    guide_scale=0.0,  # Z-Image koristi 0.0 (no guidance)
    batch_size=1,
    callback=None,
    max_sequence_length=512,
    VAE_tile_size=None,  # ili int za tiling
    cfg_normalization=False,
    cfg_truncation=1.0,
    input_frames=None,  # Za control varijante
    input_masks=None,   # Za inpainting
    context_scale=[0.75],  # Za control net weight
    input_ref_images=None,
    NAG_scale=1.0,      # Negative Adaptive Guidance
    NAG_tau=3.5,
    NAG_alpha=0.5,
    loras_slists=None,
)

# Vraća torch.Tensor shape: [batch_size, channels, height, width]
```

### LTX-2 Generation
```python
# Kroz ltx2_model.generate()
result = ltx2_model.generate(
    input_prompt="A warm sunny backyard...",
    n_prompt=None,
    image_start=None,  # PIL Image ili tensor
    image_end=None,    # PIL Image ili tensor
    sampling_steps=40,  # 8 za distilled
    guide_scale=4.0,
    frame_num=241,      # Broj frameova
    height=1024,
    width=1536,
    fps=24.0,
    seed=42,
    callback=None,
    VAE_tile_size=None,
    input_video=None,      # Prefix frames
    prefix_frames_count=0,
    audio_guide=None,      # Audio conditioning
    audio_scale=1.0,
    sliding_window_size=481,
    sliding_window_overlap=9,
    guidance_phases=2,     # 1 za distilled
    loras_slists=None,
)

# Vraća dict sa:
# {
#     "video": torch.Tensor,  # [batch, frames, channels, height, width]
#     "audio": torch.Tensor,  # [channels, samples] ili None
# }
```

---

## ⚙️ Konfiguracija Modela

### Z-Image Default Settings
```json
{
    "guidance_scale": 0.0,
    "num_inference_steps": 9,
    "NAG_scale": 1.0,
    "NAG_tau": 3.5,
    "NAG_alpha": 0.5
}
```

### Z-Image Control Varijante
```json
{
    "control_net_weight": 0.75,
    "mask_preprocessing": {
        "selection": ["", "A", "NA"],  // za control2
        "visible": false
    },
    "guide_preprocessing": {
        "selection": ["", "PV", "DV", "EV", "V"],
        "labels": {"V": "Use Z-Image Raw Format"}
    }
}
```

### LTX-2 Default Settings
```json
{
    "sliding_window_size": 481,
    "sliding_window_overlap": 9,
    "audio_scale": 1.0,
    "guidance_phases": 2,  // 1 za distilled
    "fps": 24,
    "frames_minimum": 17,
    "frames_steps": 8
}
```

### LTX-2 Model Def Properties
```python
{
    "dtype": "bf16",
    "fps": 24,
    "frames_minimum": 17,
    "frames_steps": 8,
    "sliding_window": True,
    "image_prompt_types_allowed": "TSEV",
    "returns_audio": True,
    "any_audio_prompt": True,
    "audio_prompt_choices": True,
    "one_speaker_only": True,
    "ltx2_pipeline": "two_stage"  // ili "distilled"
}
```

---

## 📦 Model Fajlovi (HuggingFace)

### Z-Image Fajlovi
**Repository**: `DeepBeepMeep/Z-Image`

#### Base Model Fajlovi (za sve Z-Image varijante):
```
Qwen3/
├── tokenizer.json
├── tokenizer_config.json
├── vocab.json
├── config.json
├── merges.txt
└── qwen3_bf16.safetensors  # ili qwen3_quanto_bf16_int8.safetensors za int8

ZImageTurbo_VAE_bf16_config.json
ZImageTurbo_VAE_bf16.safetensors
ZImageTurbo_scheduler_config.json
```

#### Model Checkpoint Fajlovi (zavisno od varijante):

**Za `z_image` (osnovni)**:
- `ZImageTurbo_bf16.safetensors` ili `ZImageTurbo_quanto_bf16_int8.safetensors`

**Za `z_image_control` (ControlNet v1)**:
- Base: `ZImageTurbo_bf16.safetensors` (koristi se iz `z_image`)
- Modul: `Z-Image-Turbo-Fun-Controlnet-Union_bf16.safetensors` ili quantized verzija

**Za `z_image_control2` (ControlNet v2)**:
- Base: `ZImageTurbo_bf16.safetensors` (koristi se iz `z_image`)
- Modul: `Z-Image-Turbo-Fun-Controlnet-Union2_bf16.safetensors` ili quantized verzija

**Za `z_image_control2_1` (ControlNet v2.1)**:
- Base: `ZImageTurbo_bf16.safetensors` (koristi se iz `z_image`)
- Modul: `Z-Image-Turbo-Fun-Controlnet-Union2.1_bf16.safetensors` ili quantized verzija

**Za TwinFlow**:
- `TwinFlow-Z-Image-Turbo_bf16.safetensors` ili quantized verzija

**Napomena**: Control varijante koriste `"URLs": "z_image"` što znači da koriste base model iz `z_image.json`, a dodaju samo ControlNet modul.

### LTX-2 Fajlovi
**Repository**: `DeepBeepMeep/LTX-2`

**Potrebni fajlovi**:
```
ltx-2-19b-dev.safetensors  # ili ltx-2-19b-dev-fp8.safetensors
ltx-2-19b-distilled.safetensors  # za distilled verziju
ltx-2-spatial-upscaler-x2-1.0.safetensors

gemma-3-12b-it-qat-q4_0-unquantized/
├── added_tokens.json
├── chat_template.json
├── config.json
├── generation_config.json
├── preprocessor_config.json
├── processor_config.json
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer.model
├── tokenizer_config.json
└── gemma-3-12b-it-qat-q4_0-unquantized.safetensors  # ili _quanto_bf16_int8 verzija
```

---

## 🔑 Ključni Parametri

### Z-Image Parametri

#### Sampling Solver
- `"default"` - Standardni solver
- `"unified"` - Unified solver
- `"unified_2s"` - Unified sa 2 stepa
- `"unified_4s"` - Unified sa 4 stepa
- `"twinflow"` - TwinFlow solver

#### NAG (Negative Adaptive Guidance)
- `NAG_scale`: 1.0 - Skaliranje NAG efekta
- `NAG_tau`: 3.5 - Temperature parametar
- `NAG_alpha`: 0.5 - Alpha blending parametar

#### Control Varijante
- `control_net_weight`: 0.75 - Težina control net-a
- `context_scale`: [0.75] - Context scaling za control

### LTX-2 Parametri

#### Pipeline Tipovi
- `"two_stage"` - Dvostepeni pipeline (default)
- `"distilled"` - Distilled pipeline (brži, 8 koraka)

#### Sliding Window
- `sliding_window_size`: 481 - Veličina prozora
- `sliding_window_overlap`: 9 - Preklapanje između prozora
- `window_min`: 5, `window_max`: 501
- `overlap_min`: 1, `overlap_max`: 97

#### Audio Conditioning
- `audio_guide`: Audio tensor ili fajl
- `audio_scale`: 1.0 - Jačina audio conditioning-a
- `audio_prompt_type`: "" ili "A" - Tip audio prompta

#### Guidance Phases
- `guidance_phases`: 2 (two_stage) ili 1 (distilled)
- `virtual_higher_phases`: True (two_stage)
- `lock_guidance_phases`: True (two_stage)

---

## 🛠️ MMGP Offloading

Oba modela koriste `mmgp.offload` za memory management:

```python
from mmgp import offload

# Model loading sa offloading
offload.load_model_data(
    model,
    model_filename,
    dtype=torch.bfloat16,
    # ... ostali parametri
)

# LoRA step tracking
offload.set_step_no_for_lora(transformer, step_index)

# CPU offload sequence za Z-Image
# "text_encoder->transformer->vae"
```

---

## 🎨 RGB Faktori

Oba modela koriste RGB faktore za latent-to-RGB konverziju:

```python
from shared.RGB_factors import get_rgb_factors

# Z-Image koristi flux faktore
latent_rgb_factors, latent_rgb_factors_bias = get_rgb_factors("flux")

# LTX-2 koristi svoje faktore
latent_rgb_factors, latent_rgb_factors_bias = get_rgb_factors("ltx2")
```

---

## 🔄 LoRA Podrška

### Z-Image LoRA
- LoRA direktorijum: `loras/z_image/`
- CLI argument: `--lora-dir-z-image`

### LTX-2 LoRA
- LoRA direktorijum: `loras/ltx2/`
- CLI argument: `--lora-dir-ltx2`
- LoRA targeti: `["text_embedding_projection", "text_embeddings_connector"]`

---

## 📝 Primeri Korišćenja

### Z-Image - Osnovni Primer
```python
from models.z_image.z_image_handler import family_handler
import torch

# Učitaj osnovni model
pipe_processor, pipe = family_handler.load_model(
    model_filename=["ZImageTurbo_bf16.safetensors"],
    base_model_type="z_image",
    dtype=torch.bfloat16,
)

# Generiši sliku
images = pipe_processor.generate(
    seed=42,
    input_prompt="A beautiful sunset over mountains",
    sampling_steps=8,
    width=1024,
    height=1024,
)

# Konvertuj u PIL Image
from shared.utils.utils import convert_tensor_to_image
pil_image = convert_tensor_to_image(images[0])
pil_image.save("output.png")
```

### Z-Image Control - ControlNet Primer
```python
# Učitaj ControlNet v2.1 model
pipe_processor, pipe = family_handler.load_model(
    model_filename=["ZImageTurbo_bf16.safetensors"],  # Base model
    base_model_type="z_image_control2_1",
    model_def={
        "modules": [
            ["Z-Image-Turbo-Fun-Controlnet-Union2.1_bf16.safetensors"]
        ]
    },
    dtype=torch.bfloat16,
)

# Generiši sa control image-om
images = pipe_processor.generate(
    seed=42,
    input_prompt="A red car",
    input_frames=control_image,  # Control image (pose, canny, depth, scribble)
    context_scale=[0.65],  # Control weight
    sampling_steps=9,
    width=1920,
    height=1088,
)
```

### Z-Image TwinFlow - Brzi Primer
```python
# Učitaj TwinFlow model (optimizovan za 1-4 koraka)
pipe_processor, pipe = family_handler.load_model(
    model_filename=["TwinFlow-Z-Image-Turbo_bf16.safetensors"],
    base_model_type="z_image",
    model_def={"unified_solver": True},
    dtype=torch.bfloat16,
)

# Generiši sa samo 2 koraka
images = pipe_processor.generate(
    seed=42,
    input_prompt="A beautiful landscape",
    sampling_steps=2,  # Veoma brzo!
    sample_solver="unified",
    width=1024,
    height=1024,
)
```

### LTX-2 - Osnovni Primer
```python
from models.ltx2.ltx2_handler import family_handler
import torch

# Učitaj model
ltx2_model, pipe = family_handler.load_model(
    model_filename=["ltx-2-19b-dev.safetensors"],
    base_model_type="ltx2_19B",
    model_def={"ltx2_pipeline": "two_stage"},
    dtype=torch.bfloat16,
)

# Generiši video
result = ltx2_model.generate(
    input_prompt="A cat playing in a garden",
    sampling_steps=40,
    frame_num=121,
    height=1024,
    width=1536,
    fps=24.0,
    seed=42,
)

# Sa audio
result = ltx2_model.generate(
    input_prompt="A cat playing in a garden",
    audio_guide=audio_tensor,
    audio_scale=1.0,
    audio_prompt_type="A",
    sampling_steps=40,
    frame_num=241,
    height=1024,
    width=1536,
    fps=24.0,
    seed=42,
)
```

### LTX-2 - Sa Start/End Frameovima
```python
from PIL import Image

image_start = Image.open("start_frame.png")
image_end = Image.open("end_frame.png")

result = ltx2_model.generate(
    input_prompt="A cat playing in a garden",
    image_start=image_start,
    image_end=image_end,
    sampling_steps=40,
    frame_num=241,
    height=1024,
    width=1536,
    fps=24.0,
    seed=42,
)
```

### LTX-2 - Distilled (Brži)
```python
ltx2_model, pipe = family_handler.load_model(
    model_filename=["ltx-2-19b-distilled.safetensors"],
    base_model_type="ltx2_19B",
    model_def={"ltx2_pipeline": "distilled"},
    dtype=torch.bfloat16,
)

result = ltx2_model.generate(
    input_prompt="A cat playing in a garden",
    sampling_steps=8,  # Mnogo manje koraka!
    frame_num=241,
    height=1024,
    width=1536,
    fps=24.0,
    seed=42,
)
```

---

## ⚠️ Važne Napomene

### Z-Image
1. **Guidance Scale**: Uvek 0.0 - Z-Image ne koristi guidance
2. **NAG**: Samo za `z_image` varijantu (ne za control)
3. **Control Weight**: Koristi `context_scale` parametar, ne `control_net_weight` direktno
4. **Inpainting**: Samo `z_image_control2` i `z_image_control2_1` podržavaju inpainting
5. **Text Encoder**: Qwen3, može biti quantized (int8)

### LTX-2
1. **Audio**: Model generiše i video i audio soundtrack
2. **Sliding Window**: Automatski se koristi za duže video-e
3. **Two Stage**: Zahteva 2 guidance phase-a (osim distilled)
4. **Frame Count**: Minimum 17 frameova, koraci po 8
5. **Text Encoder**: Gemma-3-12B-IT, može biti quantized (int8)
6. **VAE Block Size**: 64 (koristi se za tiling)

---

## 🔍 Debugging Tips

### Z-Image
- Proveri da li je `unified_solver` uključen u model_def
- Proveri NAG parametre ako koristiš `z_image` varijantu
- Za control varijante, proveri da li `input_frames` ima ispravan format

### LTX-2
- Proveri `ltx2_pipeline` tip u model_def
- Proveri sliding window parametre za duže video-e
- Proveri audio sample rate (AUDIO_SAMPLE_RATE konstanta)
- Proveri da li su start/end frameovi u ispravnom formatu

---

## 📚 Dodatni Resursi

- Z-Image HuggingFace: `DeepBeepMeep/Z-Image`
- LTX-2 HuggingFace: `DeepBeepMeep/LTX-2`
- MMGP dokumentacija: Potrebna za offloading i quantization
- Default konfiguracije: `defaults/z_image*.json` i `defaults/ltx2*.json`

---

## ✅ Checklist za Migraciju

- [ ] Instalirati sve dependencies (posebno `mmgp==3.6.11`)
- [ ] Kopirati `models/z_image/` folder
- [ ] Kopirati `models/ltx2/` folder
- [ ] Kopirati `shared/RGB_factors.py` (ako postoji)
- [ ] Kopirati `shared/utils/files_locator.py` (za file locating)
- [ ] Registrovati handler-e u glavnom fajlu
- [ ] Preuzeti model fajlove sa HuggingFace
- [ ] Preuzeti text encoder fajlove (Qwen3 za Z-Image, Gemma za LTX-2)
- [ ] Testirati osnovno učitavanje modela
- [ ] Testirati generisanje sa osnovnim parametrima
- [ ] Konfigurisati LoRA direktorijume (ako koristiš LoRA-e)

---

**Napomena**: Ova dokumentacija je bazirana na trenutnoj implementaciji u Wan2GP projektu. Proveri verzije paketa i eventualne izmene u API-ju pri migraciji.

---

## 📊 Sažetak - Tačno Korišćeni Modeli

### Z-Image Modeli u Projektu

| Varijanta | Architecture | Base Model | ControlNet Modul | Config File |
|-----------|-------------|------------|-------------------|-------------|
| Osnovni | `z_image` | `ZImageTurbo_bf16.safetensors` | - | `defaults/z_image.json` |
| Control v1 | `z_image_control` | `ZImageTurbo_bf16.safetensors` | `Z-Image-Turbo-Fun-Controlnet-Union_bf16.safetensors` | `defaults/z_image_control.json` |
| Control v2 | `z_image_control2` | `ZImageTurbo_bf16.safetensors` | `Z-Image-Turbo-Fun-Controlnet-Union2_bf16.safetensors` | `defaults/z_image_control2.json` |
| Control v2.1 | `z_image_control2_1` | `ZImageTurbo_bf16.safetensors` | `Z-Image-Turbo-Fun-Controlnet-Union2.1_bf16.safetensors` | `defaults/z_image_control2_1.json` |
| TwinFlow | `z_image` | `TwinFlow-Z-Image-Turbo_bf16.safetensors` | - | `defaults/z_image_twinflow_turbo.json` |

**Zajednički fajlovi za sve Z-Image varijante**:
- VAE: `ZImageTurbo_VAE_bf16.safetensors`
- Text Encoder: `Qwen3/qwen3_bf16.safetensors`
- Scheduler: `ZImageTurbo_scheduler_config.json`

### LTX-2 Modeli u Projektu

| Varijanta | Architecture | Model File | Pipeline | Config File |
|-----------|-------------|------------|----------|-------------|
| Dev 19B | `ltx2_19B` | `ltx-2-19b-dev.safetensors` | `two_stage` | `defaults/ltx2_19B.json` |
| Distilled 19B | `ltx2_19B` | `ltx-2-19b-distilled.safetensors` | `distilled` | `defaults/ltx2_distilled.json` |

**Zajednički fajlovi za LTX-2**:
- Text Encoder: `gemma-3-12b-it-qat-q4_0-unquantized/`
- Spatial Upscaler: `ltx-2-spatial-upscaler-x2-1.0.safetensors`
- LoRA (opciono): `ltx-2-19b-distilled-lora-384.safetensors`

### Ključne Razlike

**Z-Image**:
- Svi modeli koriste isti base model (`ZImageTurbo_bf16.safetensors`)
- Control varijante dodaju samo ControlNet modul
- TwinFlow je zaseban checkpoint optimizovan za brzu generaciju

**LTX-2**:
- Dva različita checkpoint-a (dev i distilled)
- Različiti pipeline tipovi (`two_stage` vs `distilled`)
- Distilled koristi 8 koraka umesto 40
