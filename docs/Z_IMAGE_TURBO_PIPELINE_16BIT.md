# Z Image Turbo – ceo pipeline sa 16-bit (bf16) preciznošću

Dokument opisuje **ceo pipeline** Z Image Turbo u aplikaciji: od učitavanja modela, preko scheduler-a i VAE-a, do generisanja slike sa **bf16** preciznošću. Možeš ga koristiti kao referencu za drugi projekat.

---

## 1. Pregled – šta se koristi

| Komponenta | Fajl / vrednost | Napomena |
|------------|------------------|----------|
| **Model** | `ZImageTurbo_bf16.safetensors` | Transformer 6B, već u bf16 |
| **Arhitektura** | `z_image` | `ZImageTransformer2DModel` |
| **Text encoder** | Qwen3 (npr. `qwen3_bf16.safetensors`) | Bf16 |
| **VAE** | `ZImageTurbo_VAE_bf16.safetensors` | U kodu može biti float32 ili fp16 (vae_precision) |
| **Scheduler** | `FlowMatchEulerDiscreteScheduler` | Config: `ZImageTurbo_scheduler_config.json` |
| **Sampler** | Default (Flow Match Euler), ne unified | 8 koraka, guidance_scale=0 |
| **dtype** | `torch.bfloat16` | Za transformer i text encoder |

---

## 2. Tok podataka (visok nivo)

```
[Prompt] → Tokenizer (Qwen3) → Text Encoder (Qwen3, bf16) → prompt_embeds
                                                                  ↓
[Generator] → latents (float32, 16 kanala) ← prepare_latents
                                                                  ↓
                    Transformer (bf16) + Scheduler → denoising (8 steps)
                                                                  ↓
                    latents (float32) → VAE decode → image (tensor / PIL)
```

- **16-bit** u pipeline-u: transformer i text encoder rade u **bf16**; latents tokom denoisinga su **float32**; na kraju VAE dekoduje u sliku (tip zavisi od `output_type`).

---

## 3. Gde se šta dešava (fajlovi)

### 3.1 Registracija i konfiguracija

- **Handler (familija modela)**  
  `models/z_image/z_image_handler.py`  
  - `query_supported_types()` → `["z_image", "z_image_base", "z_image_control", ...]`  
  - `query_model_def()`, `query_model_files()`, `load_model()`, `update_default_settings()`  
  - Za običan Turbo bez kontroli koristi se tip **`z_image`**.

- **Default UI / inference parametri**  
  `defaults/z_image.json`:
  - `architecture`: `"z_image"`
  - `num_inference_steps`: **8**
  - `guidance_scale`: **0**
  - `resolution`: `"1024x1024"`
  - URLs za model: `ZImageTurbo_bf16.safetensors` (i opciono quanto verzija).

- **Konfiguracija transformer arhitekture**  
  `models/z_image/configs/z_image.json`  
  - Dimenzije, broj slojeva, patch size, itd. za `ZImageTransformer2DModel`.

---

### 3.2 Učitavanje modela (16-bit)

Poziv iz glavne app (npr. `wgp.py`) ide preko:

- `load_models(model_type, ...)`  
  → `model_type_handler.load_model(...)`  
  → `models.z_image.z_image_handler.family_handler.load_model(...)`  
  → `models/z_image/z_image_main.py` → **`model_factory`**.

U **`z_image_main.py`** (model_factory):

1. **dtype za transformer**  
   - Prosleđuje se iz app-a: `dtype=torch.bfloat16` (npr. iz `get_transformer_dtype()` ili `--bf16`).  
   - U factory-ju: `dtype=torch.bfloat16` kao default, pa:
     - `offload.load_model_data(transformer, ...)` učitava težine,
     - zatim **`transformer.to(dtype)`** → transformer ostaje u **bf16**.

2. **Text encoder**  
   - `offload.fast_load_transformers_model(text_encoder_filename, ..., modelClass=Qwen3ForCausalLM)`  
   - U praksi se koristi bf16 verzija (`qwen3_bf16.safetensors`), dakle **16-bit**.

3. **VAE**  
   - `offload.fast_load_transformers_model(vae_filename, ..., default_dtype=VAE_dtype)`.  
   - `VAE_dtype` iz app-a: `torch.float16` ako je `vae_precision == "16"`, inače `torch.float32`.  
   - Za “16-bit pipeline” u smislu u kojem ti koristiš, VAE može biti fp16 ili float32; transformer i text encoder su ono što je fiksno bf16.

4. **Scheduler**  
   - `FlowMatchEulerDiscreteScheduler(**scheduler_config)` iz `ZImageTurbo_scheduler_config.json`.

5. **Pipeline**  
   - Jedan objekat tipa **`ZImagePipeline`** (iz `pipeline_z_image.py`) sa:  
     `scheduler, vae, text_encoder, tokenizer, transformer` – sve komponente su već učitane sa željenim dtype-ovima.

Znači: **ceo “16-bit” deo za inference jeste: transformer i text encoder u bf16; opciono VAE u fp16.**

---

### 3.3 Pipeline – jedan poziv generisanja

Korisnik u app-u bira model (npr. Z Image Turbo) i pokrene generisanje. App poziva:

- `wan_model.generate(...)`  
  gde je `wan_model` zapravo **`model_factory`** instanca iz `z_image_main.py`.

U **`z_image_main.py`**, **`model_factory.generate()`**:

- Postavlja generator (seed),
- opciono VAE tiling (`VAE_tile_size`),
- za običan Turbo: `sample_solver = "default"`, `guide_scale = 0`,
- poziva **`self.pipeline(...)`** iz `models/z_image/pipeline_z_image.py` sa parametrima kao što su:
  - `prompt`, `negative_prompt`, `num_inference_steps=8`, `sample_solver="default"`, `guidance_scale=0`, `height`, `width`, `max_sequence_length=512`, itd.

Znači **ceo pipeline jedne slike** je u `ZImagePipeline.__call__()` u **`pipeline_z_image.py`**.

---

### 3.4 ZImagePipeline – korak po korak (`pipeline_z_image.py`)

Redosled u **`__call__`**:

1. **Dtype za embedding**  
   - `dtype` za tok = dtype text encodera (bf16 kada koristiš bf16 encoder).

2. **Encode prompt**  
   - `encode_prompt()` → `_encode_prompt()`:  
     tokenizer (Qwen3 chat template) → text encoder → **prompt_embeds** (bf16).  
   - Za `guidance_scale=0` nema CFG, ali pipeline i dalje može da enkoduje negativni prompt; za sam Turbo sa scale 0 korak denoisinga koristi samo pozitivne embedinge.

3. **Prepare latents**  
   - `prepare_latents(...)`:  
     - shape iz height/width i `vae_scale_factor * 2`,  
     - `num_channels_latents` iz transformer-a (16).  
   - Latents se generišu u **float32** (`randn_tensor(..., dtype=...)` u kontekstu gde se za latents koristi float32).

4. **Scheduler timesteps (default solver)**  
   - Za `sample_solver="default"` (nije unified):  
     - `calculate_shift(image_seq_len, ...)` za Flow Match,  
     - `retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=..., mu=...)`  
   - Dobijaš `timesteps` za 8 koraka.

5. **Denoising loop (default)**  
   - Za svaki korak:  
     - `timestep = (1000 - t) / 1000`,  
     - `latent_model_input = latents.to(dtype)` (dtype = bf16 od text encodera) → ulaz u transformer je **bf16**,  
     - `self.transformer(latent_model_input, timestep, prompt_embeds, ...)` → izlaz se vodi dalje u float32 za scheduler,  
     - `noise_pred = -noise_pred`,  
     - `latents = self.scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]`  
   - Scheduler radi u **float32**; unutra transformer radi u **bf16** (mixed precision u smislu: ulaz/izlaz prema scheduler-u u float32, računanje u bf16).

6. **VAE decode**  
   - `latents = latents.to(dtype)` (dtype opet bf16),  
   - za sliku:  
     - `latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor`,  
     - `image = self.vae.decode(latents, return_dict=False)[0]`,  
     - zatim `image_processor.postprocess(image, output_type=...)` (npr. `"pt"` ili `"pil"`).

7. **Povratak**  
   - Za `output_type="pt"` app dobija tensor slike; za `"pil"` – PIL slike.

---

## 4. Gde je eksplicitno 16-bit (bf16)

- **Transformer**: učitavanje i `transformer.to(dtype)` u `z_image_main.model_factory` sa `dtype=torch.bfloat16`; u pipeline-u `latent_model_input` i prompt_embeds su u tom dtype-u, pa je forward u bf16.
- **Text encoder**: učitavan preko `fast_load_transformers_model` sa bf16 težinama (npr. `qwen3_bf16.safetensors`), pa encode prompta je u bf16.
- **VAE**: opciono 16-bit ako app postavi `VAE_dtype=torch.float16` (npr. kada je `vae_precision == "16"`). To ne menja činjenicu da je “glavni” 16-bit deo transformer + text encoder u bf16.

---

## 5. CPU RAM offload i MMGP

Pipeline koristi **mmgp** (`mmgp.offload`) za upravljanje memorijom: modeli mogu da žive na CPU (RAM), a na GPU se prebacuju samo kada treba. To omogućava rad sa manje VRAM-a (npr. 12 GB) uz dovoljno RAM-a.

### 5.1 Redosled CPU offload-a (pipeline)

U **`pipeline_z_image.py`** je definisano:

```python
class ZImagePipeline(DiffusionPipeline, FromSingleFileMixin):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
```

To znači: prvo se na GPU šalje **text encoder** (encode prompta), pa **transformer** (denoising), pa **VAE** (decode). Posle svakog koraka komponenta se može vratiti na CPU, zavisno od profila i budžeta.

### 5.2 Učitavanje preko mmgp (težine na CPU)

U **`z_image_main.py`** (model_factory):

- **Transformer**: `offload.load_model_data(transformer, model_filename, ...)` – učitava težine u memoriju (po defaultu na CPU, mmgp kasnije prebacuje na GPU po potrebi).
- **Text encoder**: `offload.fast_load_transformers_model(text_encoder_filename, ..., modelClass=Qwen3ForCausalLM)` – isto, model se učitava tako da mmgp može da ga offload-uje.
- **VAE**: `offload.fast_load_transformers_model(vae_filename, ..., defaultConfigPath=..., default_dtype=VAE_dtype)` – isto.

Svi delovi **pipe**-a (`transformer`, `text_encoder`, `vae`) se prosleđuju u **`offload.profile(...)`**, koji na osnovu profila i budžeta odlučuje šta držati na GPU, a šta na CPU.

### 5.3 Aktivacija offload-a posle učitavanja (wgp.py)

U glavnoj app-u, posle `model_type_handler.load_model(...)`:

1. **`init_pipe(pipe, kwargs, profile)`**  
   Postavlja parametre za mmgp u zavisnosti od **profila** (1–5):
   - **Profile 2** (HighRAM_LowVRAM): `budgets = { "transformer": ..., "text_encoder": ..., "*": ... }` – procenti/limiti koliko koja komponenta može da koristi VRAM; `*` je opšti budžet za RAM.
   - **Profile 3** (LowRAM_HighVRAM): slično, drugačiji budžeti.
   - **Profile 3.5**: kao 3, ali `mmgp_profile = 3`, `pinnedMemory = False` (manje rezervisane RAM).
   - **Profile 4** (LowRAM_LowVRAM): preporučeno za malo VRAM-a; **Profile 4.5**: varijanta sa manje VRAM-a.
   - **Profile 5** (VeryLowRAM_LowVRAM): najstrožiji – malo RAM-a i VRAM-a, sve više offload-uje na CPU.

2. **`offload.profile(pipe, profile_no=mmgp_profile, ...)`**  
   Glavni poziv koji **uključuje CPU offload** za ceo `pipe`:

   ```python
   offloadobj = offload.profile(
       pipe,
       profile_no=mmgp_profile,
       compile=compile_modules,
       quantizeTransformer=False,
       loras=loras_transformer,
       perc_reserved_mem_max=perc_reserved_mem_max,
       vram_safety_coefficient=vram_safety_coefficient,
       convertWeightsFloatTo=transformer_dtype,
       **kwargs   # budgets, pinnedMemory, asyncTransfers, itd.
   )
   ```

   - **`pipe`**: dict sa `transformer`, `text_encoder`, `vae` (za Z-Image).
   - **`profile_no`**: 1–5 (ili 3.5, 4.5) – odgovara MMGP profilu (koliko agresivno koristiti CPU vs VRAM).
   - **`loras`**: lista imena modula u koju se učitavaju LoRA-e (npr. `["transformer"]`).
   - **`perc_reserved_mem_max`**, **`vram_safety_coefficient`**: ograničenja VRAM-a.
   - **`convertWeightsFloatTo`**: dtype težina (npr. `torch.bfloat16`).
   - **`kwargs`**: iz `init_pipe` – npr. `budgets`, `pinnedMemory`, `asyncTransfers`.

Tokom inference-a mmgp prebacuje komponente na GPU pre poziva (text_encoder → transformer → vae) i po želji ih vraća na CPU, u skladu sa `model_cpu_offload_seq` i budžetima.

### 5.4 LoRA i step broj u petlji

U **`pipeline_z_image.py`** u denoising petlji (i za unified solver):

```python
offload.set_step_no_for_lora(self.transformer, i)
```

To omogućava mmgp-u da po stepu menja aktivne LoRA-e (ako koristiš step-based LoRA), tj. da zna koji je trenutni korak pri offload-u.

### 5.5 Profili – kratak pregled

| Profil | Naziv                  | RAM / VRAM          | Namena |
|--------|------------------------|---------------------|--------|
| 1      | HighRAM_HighVRAM       | 64 GB RAM, 24 GB VRAM | Sve na GPU, maksimalna brzina. |
| 2      | HighRAM_LowVRAM        | 64 GB RAM, 12 GB VRAM | Dosta RAM-a, manje VRAM-a; offload po budžetima. |
| 3      | LowRAM_HighVRAM        | 32 GB RAM, 24 GB VRAM | Manje RAM-a, dovoljno VRAM-a. |
| 3.5    | VeryLowRAM_HighVRAM    | 32 GB RAM, 24 GB VRAM | Kao 3, bez rezervisane RAM (manje RAM). |
| 4      | LowRAM_LowVRAM         | 32 GB RAM, 12 GB VRAM | Preporučeno za malo VRAM-a; više CPU offload-a. |
| 4.5    | LowRAM_LowVRAM+        | 32 GB RAM, 12 GB VRAM | Kao 4, još manje VRAM-a. |
| 5      | VeryLowRAM_LowVRAM     | 24 GB RAM, 10 GB VRAM | Fail-safe: minimum VRAM/RAM, sve preko CPU. |

Za Z Image Turbo sa 16-bit obično se koristi **profile 4** (ili 2 ako imaš puno RAM-a). **image_profile** u config-u (npr. `server_config["image_profile"]`) određuje koji profil će biti korišćen za slike.

### 5.6 Rezime offload toka

1. **Učitavanje**: `load_model` → transformer, text_encoder, VAE učitani preko **mmgp** (težine u memoriji, po defaultu na CPU).
2. **Inicijalizacija**: `init_pipe(pipe, kwargs, profile)` → postavlja **budgets** i **mmgp_profile** za izabrani profil.
3. **Aktivacija**: `offload.profile(pipe, profile_no=..., **kwargs)` → uključuje CPU/VRAM offload za ceo pipeline; tokom `generate()` mmgp prebacuje **text_encoder → transformer → vae** na GPU po redosledu iz `model_cpu_offload_seq`.
4. **Inference**: u petlji `offload.set_step_no_for_lora(transformer, i)` za LoRA step tracking.

Ako u drugom projektu želiš isto ponašanje (malo VRAM-a, više RAM-a), moraš koristiti **mmgp** i pozvati `offload.profile()` na isti `pipe` sa odgovarajućim profilom i budžetima; bez mmgp-a sve komponente ostaju na GPU i treba dovoljno VRAM-a.

---

## 6. Kako da koristiš ovaj pipeline u drugom projektu

### 6.1 Minimalan set fajlova (Z Image Turbo, default, 16-bit)

- **Modeli (safetensors + config)**  
  - Transformer: `ZImageTurbo_bf16.safetensors`  
  - Config transformera: `models/z_image/configs/z_image.json`  
  - VAE: `ZImageTurbo_VAE_bf16.safetensors` + `ZImageTurbo_VAE_bf16_config.json`  
  - Scheduler: `ZImageTurbo_scheduler_config.json`  
  - Text encoder: npr. Qwen3 iz istog repoa (npr. `qwen3_bf16.safetensors`) + tokenizer (config/tokenizer iz tog foldera)

- **Kod**  
  - `models/z_image/z_image_main.py` (model_factory)  
  - `models/z_image/pipeline_z_image.py` (ZImagePipeline)  
  - `models/z_image/z_image_transformer2d.py` (ZImageTransformer2DModel)  
  - `models/z_image/autoencoder_kl.py` (VAE)  
  - `models/z_image/z_image_handler.py` – možeš ga iskopirati ili u drugom projektu samo pozivati `model_factory` i `ZImagePipeline` direktno.

- **Zavisnosti**  
  - `torch`, `diffusers` (FlowMatchEulerDiscreteScheduler, osnove pipeline-a), `transformers` (Qwen3, tokenizer), `mmgp` (offload / load_model_data ako želiš isto ponašanje učitavanja).

### 6.2 Jednostavan “standalone” primer (isti redosled kao u app-u)

```python
import torch
from models.z_image.z_image_main import model_factory
from shared.utils import files_locator as fl  # ili svoj locator za putanje

# 1. Učitavanje (16-bit: transformer + text encoder u bf16)
factory = model_factory(
    checkpoint_dir="ckpts",
    model_filename=fl.locate_file("ZImageTurbo_bf16.safetensors"),
    model_type=None,
    base_model_type="z_image",
    model_def={},
    text_encoder_filename=fl.locate_file("Qwen3/qwen3_bf16.safetensors"),
    dtype=torch.bfloat16,
    VAE_dtype=torch.float32,  # ili torch.float16 za VAE 16-bit
)

# 2. Generisanje (default: 8 koraka, guidance_scale=0)
images = factory.generate(
    seed=42,
    input_prompt="Your prompt",
    sampling_steps=8,
    sample_solver="default",
    width=1024,
    height=1024,
    guide_scale=0.0,
    batch_size=1,
    max_sequence_length=512,
    NAG_scale=1.0,
    NAG_tau=3.5,
    NAG_alpha=0.5,
)

# images: torch.Tensor [batch, C, H, W]
```

U drugom projektu možeš zameniti `fl.locate_file` svojim putanjama do istih fajlova.

### 6.3 Šta tačno preneti

- **Ceo folder** `models/z_image/` (svi .py + `configs/z_image.json`).  
- **Shared**: bar ono što pipeline koristi, npr. `shared.utils.files_locator`, `shared.utils.text_encoder_cache` (ako želiš cache), i sve što pipeline importuje iz `shared`.  
- **Default config** za Turbo: `num_inference_steps=8`, `guidance_scale=0`, `sample_solver="default"` – to je “Z Image Turbo default” sa 16-bit preciznošću kako ga koristiš u app-u.

---

## 7. Rezime – ceo pipeline u jednoj rečenici

**Prompt → Qwen3 tokenizer + text encoder (bf16) → embedingi; generator → latents (float32); 8× (transformer u bf16 + FlowMatchEulerDiscreteScheduler step u float32); VAE decode → slika.**  
Sve što ti treba za reprodukciju u drugom projektu je u ovom dokumentu i u navedenim fajlovima (`z_image_main`, `pipeline_z_image`, handler, configs i model fajlovi).
