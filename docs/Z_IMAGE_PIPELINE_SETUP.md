# Z Image Turbo – uputstvo: 16-bit pipeline sa Qwen3 i Memory Profile 3

Ovaj dokument objašnjava kako je u Wan2GP podešen **Z Image Turbo** pipeline u **16-bit (bf16)** sa **Qwen3** enkoderom (takođe 16-bit) i **Memory Profile 3**. Sa ovim profilom modeli se učitavaju u **RAM**, a na **VRAM** se prebacuje samo ono što je potrebno za trenutni korak – tako pipeline radi i kada su modeli veći od raspoloživog VRAM-a.

---

## 1. Šta dobijaš sa ovim setup-om

| Komponenta | Preciznost | Gde živi po učitavanju |
|------------|------------|-------------------------|
| **Transformer** (ZImageTurbo) | bf16 | RAM → na GPU samo tokom denoisinga |
| **Text encoder** (Qwen3) | bf16 | RAM → na GPU samo tokom encode_prompt |
| **VAE** | float32 ili fp16 | RAM → na GPU samo tokom decode |
| **Scheduler** | config (Flow Match Euler) | Nije model, u memoriji |

- **Memory Profile 3** (LowRAM_HighVRAM): namenjen za **32 GB+ RAM** i **24 GB VRAM**. Budžet za VRAM je **70%** (`"*": "70%"`), što znači da mmgp drži težine u RAM-u i prebacuje na GPU samo aktivnu komponentu po redosledu: text_encoder → transformer → vae.
- Rezultat: možeš pokrenuti Z Image (6B transformer + Qwen3 + VAE) i kada ti VRAM nije dovoljan da sve stane odjednom – sve je u RAM-u, na GPU ide samo ono što pipeline trenutno koristi.

---

## 2. Zahtevi (hardver i softver)

### Hardver (za Profile 3)

- **RAM**: najmanje 32 GB (preporučeno 64 GB ako radiš i druge stvari).
- **VRAM**: 24 GB (npr. RTX 3090 / 4090). Sa 12 GB VRAM koristi Profile 4.

### Softver (pip)

```
torch
diffusers>=0.34.0   # 0.36.0 u Wan2GP
transformers        # sa Qwen3 (npr. 4.54.0)
accelerate>=0.31.0
mmgp                # 3.7.x – ključan za učitavanje u RAM i offload
safetensors
Pillow
numpy
```

Opciono: `tokenizers`, `huggingface-hub` (često već dolaze uz `transformers` i `diffusers`).

---

## 3. Model fajlovi koje treba da imaš

Sve možeš preuzeti iz repozitorijuma **DeepBeepMeep/Z-Image** (Hugging Face) ili iz lokalnog Wan2GP checkpoint foldera.

| Šta | Fajl(ovi) |
|-----|-----------|
| Transformer | `ZImageTurbo_bf16.safetensors` |
| Config transformera | `models/z_image/configs/z_image.json` |
| VAE | `ZImageTurbo_VAE_bf16.safetensors` + `ZImageTurbo_VAE_bf16_config.json` |
| Scheduler | `ZImageTurbo_scheduler_config.json` |
| Text encoder (Qwen3, 16-bit) | `qwen3_bf16.safetensors` (u folderu npr. `Qwen3/`) |
| Tokenizer | `tokenizer_config.json`, `tokenizer.json`, `vocab.json`, `config.json`, `merges.txt` (u istom folderu kao Qwen3) |

Struktura foldera može izgledati ovako:

```
ckpts/
  ZImageTurbo_bf16.safetensors
  ZImageTurbo_VAE_bf16.safetensors
  ZImageTurbo_VAE_bf16_config.json
  ZImageTurbo_scheduler_config.json
  Qwen3/
    qwen3_bf16.safetensors
    tokenizer_config.json
    tokenizer.json
    ...
```

Config za transformer (`z_image.json`) mora biti u kodu (npr. u `models/z_image/configs/`).

---

## 4. Kod: šta preneti iz Wan2GP

### 4.1 Fajlovi iz `models/z_image/`

- `z_image_main.py` – `model_factory`: učitavanje transformer, Qwen3, VAE, scheduler, gradnja pipeline-a.
- `pipeline_z_image.py` – `ZImagePipeline` (redosled: text_encoder → transformer → vae).
- `z_image_transformer2d.py` – `ZImageTransformer2DModel`.
- `autoencoder_kl.py` – lokalni VAE (baziran na diffusers).
- `unified_sampler.py` – sampler.
- `pipeline_output.py` – `ZImagePipelineOutput`.
- `configs/z_image.json` – konfiguracija transformera.

Handler (`z_image_handler.py`) ti treba samo ako gradiš sličan sistem kao Wan2GP (biranje modela, download, itd.). Za “samo pipeline” dovoljno je pozivati `model_factory` i zatim `offload.profile()`.

### 4.2 Zavisnosti od `shared/`

U `pipeline_z_image.py` i `z_image_main.py` koriste se:

- `shared.utils.utils`: `get_outpainting_frame_location`, `resize_lanczos`, `calculate_new_dimensions`, `convert_image_to_tensor`, `fit_image_into_canvas`
- `shared.utils.loras_mutipliers`: `update_loras_slists`
- `shared.utils.text_encoder_cache`: `TextEncoderCache`
- `shared.utils.files_locator` (npr. `fl.locate_file(...)`)
- `shared.attention`: `pay_attention` (u transformeru)

Za drugi projekat: ili prebaciš te module, ili zameniš jednostavnijim verzijama / stubovima (npr. bez outpaintinga ili LoRA ako ti ne trebaju).

---

## 5. Učitavanje u RAM i Memory Profile 3 (kako Wan2GP radi)

### 5.1 Zašto sve može da bude veće od VRAM-a

1. **Učitavanje**  
   Svi modeli se učitavaju preko **mmgp**:
   - **Transformer**: `offload.load_model_data(transformer, model_filename, ...)` – težine idu u memoriju (podrazumevano na **CPU/RAM**).
   - **Text encoder**: `offload.fast_load_transformers_model(..., modelClass=Qwen3ForCausalLM)` – isto, u RAM.
   - **VAE**: `offload.fast_load_transformers_model(vae_filename, ..., modelClass=AutoencoderKL, ...)` – isto, u RAM.

2. **Aktivacija profila**  
   Posle učitavanja, umesto da sve ostane na GPU, poziva se **`offload.profile(pipe, profile_no=3, ...)`**. Objekat `pipe` je rečnik: `{"transformer": ..., "text_encoder": ..., "vae": ...}`.

3. **Profile 3 u Wan2GP**  
   U `wgp.py`, `init_pipe(pipe, kwargs, profile)` za **profile == 3** postavlja:
   - `kwargs["budgets"] = { "*": "70%" }`  
   Znači: mmgp koristi do 70% VRAM-a; ostalo ostaje u RAM-u i prebacuje se na GPU samo po potrebi.

4. **Redosled na GPU**  
   U `pipeline_z_image.py`:
   ```python
   model_cpu_offload_seq = "text_encoder->transformer->vae"
   ```
   Tokom jednog `generate()`: prvo na GPU ide **text_encoder** (encode prompta), pa **transformer** (denoising), pa **VAE** (decode). Po želji mmgp može da vraća komponente na CPU između koraka, u skladu sa budžetom.

Zbog toga možeš imati modele veće od VRAM-a: sve leži u RAM-u, na GPU se uvek nalazi samo aktivna komponenta (ili nekoliko, ako 70% VRAM-a dozvoli).

### 5.2 Tačan redosled u kodu (Wan2GP)

1. **Učitavanje modela**  
   `model_type_handler.load_model(...)` → za Z Image to je `z_image_handler.load_model()` → kreira `model_factory` u `z_image_main.py`. U factory-ju:
   - transformer: `init_empty_weights` + `offload.load_model_data` + `transformer.to(dtype)` (bf16),
   - text encoder: `offload.fast_load_transformers_model(..., Qwen3ForCausalLM)` (bf16 težine),
   - VAE: `offload.fast_load_transformers_model(..., AutoencoderKL)`,
   - scheduler i pipeline sastavljaju se od tih komponenti.

2. **Rečnik `pipe`**  
   Handler vraća `(pipe_processor, pipe)` gde je:
   ```python
   pipe = {
       "transformer": pipe_processor.transformer,
       "text_encoder": pipe_processor.text_encoder,
       "vae": pipe_processor.vae,
   }
   ```

3. **Inicijalizacija profila**  
   `mmgp_profile = init_pipe(pipe, kwargs, profile)`. Za **profile 3**: `kwargs["budgets"] = { "*": "70%" }`, `mmgp_profile = 3`.

4. **Aktivacija offload-a**  
   ```python
   offload.profile(
       pipe,
       profile_no=mmgp_profile,
       compile=...,
       quantizeTransformer=False,
       loras=["transformer"],
       perc_reserved_mem_max=...,
       vram_safety_coefficient=...,
       convertWeightsFloatTo=torch.bfloat16,
       **kwargs
   )
   ```
   Od ovog trenutka mmgp upravlja premeštanjem: težine ostaju u RAM-u, na GPU idu po `model_cpu_offload_seq` i budžetu.

5. **Generisanje**  
   `wan_model.generate(...)` → `factory.generate(...)` → `self.pipeline(...)`. Unutar pipeline-a mmgp prebacuje text_encoder → transformer → vae na GPU pre svakog bloka koji ih koristi.

---

## 6. Kako napraviti taj pipeline u drugom projektu (korak po korak)

### Korak 1: Zavisnosti

Instaliraj: `torch`, `diffusers`, `transformers`, `accelerate`, `mmgp`, `safetensors`, `Pillow`, `numpy`. Verzije kao u Wan2GP `requirements.txt` ako želiš maksimalnu kompatibilnost (npr. diffusers 0.36.0, transformers 4.54.0).

### Korak 2: Prebaci kod i config

- Prebaci ceo folder `models/z_image/` (svi navedeni .py + `configs/z_image.json`).
- Zameni ili prilagodi importe iz `shared.*` (files_locator, utils, loras_mutipliers, text_encoder_cache, attention) – bilo kopijom iz Wan2GP bilo svojim stubovima.

### Korak 3: Putanje do modela

Implementiraj način da kod nađe fajlove (npr. jedan folder `ckpts/` sa Z Image + Qwen3 fajlovima). U Wan2GP se koristi `shared.utils.files_locator` (npr. `fl.locate_file("ZImageTurbo_bf16.safetensors")`). U drugom projektu možeš prosleđivati apsolutne putanje ili ime root foldera.

### Korak 4: Učitavanje (16-bit, sve u RAM)

```python
import torch
from mmgp import offload
# tvoj import model_factory (npr. iz models.z_image.z_image_main)

factory = model_factory(
    checkpoint_dir="ckpts",
    model_filename="ckpts/ZImageTurbo_bf16.safetensors",  # ili tvoj locator
    model_type="z_image",
    base_model_type="z_image",
    model_def={},
    text_encoder_filename="ckpts/Qwen3/qwen3_bf16.safetensors",
    dtype=torch.bfloat16,
    VAE_dtype=torch.float32,
)
```

Ovo učitava transformer, Qwen3 i VAE preko mmgp – težine su u RAM-u.

### Korak 5: Pipe i Memory Profile 3

```python
pipe = {
    "transformer": factory.transformer,
    "text_encoder": factory.text_encoder,
    "vae": factory.vae,
}

kwargs = {"budgets": { "*": "70%" }}
mmgp_profile = 3

offload.profile(
    pipe,
    profile_no=mmgp_profile,
    quantizeTransformer=False,
    loras=["transformer"],
    convertWeightsFloatTo=torch.bfloat16,
    **kwargs
)
```

Opciono dodaj `perc_reserved_mem_max`, `vram_safety_coefficient`, `compile`, itd. ako ih koristiš.

### Korak 6: Generisanje

```python
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
)
```

Pipeline će interno koristiti mmgp: text_encoder → transformer → vae prebacuju se na GPU po redosledu, tako da ceo 16-bit Z Image + Qwen3 pipeline radi i kada su modeli veći od tvog VRAM-a.

---

## 7. Primer izlaza mmgp (Profile 3 + LoRA)

Kad pokreneš Z Image sa Memory Profile 3 i LoRA-om, mmgp ispisuje nešto ovako:

```
************ Memory Management for the GPU Poor (mmgp 3.7.3) by DeepBeepMeep ************
Pinning data of 'transformer' to reserved RAM
The whole model was pinned to reserved RAM: 52 large blocks spread across 11796.03 MB
Hooked to model 'text_encoder' (Qwen3ForCausalLM)
Hooked to model 'vae' (AutoencoderKL)
Lora 'loras\z_image\ZIT_Midjourney_Luneva_Cinematic_v1_r128.safetensors' was loaded in model '...'
'loras\z_image\ZIT_Midjourney_Luneva_Cinematic_v1_r128.safetensors' was pinned entirely to reserved RAM: 3 large blocks spread across 648.75 MB
```

Šta to znači:

| Poruka | Značenje |
|--------|----------|
| **Pinning data of 'transformer' to reserved RAM** | Transformer (Z Image 6B bf16) se drži u rezervisanoj RAM – ne na GPU. |
| **52 large blocks ... 11796.03 MB** | ~11,8 GB RAM za transformer (ceo model u bf16). |
| **Hooked to model 'text_encoder'** / **'vae'** | Qwen3 i VAE su “hookovani” – mmgp ih prebacuje na GPU samo kad treba (po `model_cpu_offload_seq`). |
| **Lora ... was loaded** / **pinned entirely to reserved RAM ... 648.75 MB** | LoRA je učitana u transformer i takođe pinovana u RAM (~649 MB), ne zauzima VRAM dok ne radi denoising. |

Generisanje onda ide 10 (ili 8) koraka; progress bar ponekad ostane na 0%, ali slike se normalno snimaju (npr. u `outputs\`). Vreme po slici reda veličine ~35–40 s zavisi od hardvera.

---

## 8. Rezime

- **Z Image 16-bit** = transformer + Qwen3 u **bf16**, VAE po želji fp16/float32.
- **Memory Profile 3** = `budgets = { "*": "70%" }`, mmgp drži težine u **RAM-u** i na **VRAM** šalje samo ono što pipeline trenutno koristi (text_encoder → transformer → vae).
- Da bi to radilo, **sve učitavanje** mora preko **mmgp** (`load_model_data`, `fast_load_transformers_model`), a posle učitavanja obavezno **`offload.profile(pipe, profile_no=3, budgets={ "*": "70%" }, ...)`**.

Za detaljan opis 16-bit toka i svih fajlova vidi **Z_IMAGE_TURBO_PIPELINE_16BIT.md**.
