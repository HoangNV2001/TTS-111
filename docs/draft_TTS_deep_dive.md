Mình đã làm một vòng research khá sâu theo đúng 4 nhánh bạn đưa ra. Có một kết luận quan trọng ngay từ đầu: **không nên gộp “SOTA TTS” thành một bảng xếp hạng duy nhất**. Model tốt nhất về naturalness, model tốt nhất cho realtime serving, model phù hợp tiếng Việt, và model tốt nhất cho on-device hiện là những nhóm khá khác nhau.

Tính đến **29/08/2026**, mình sẽ định hướng R&D của team theo ba track song song:

> **Server-grade / quality:** OmniVoice + Fish Audio S2 Pro
> **Inference/serving engineering:** OmniVoice là bài tập rất hay vì NAR diffusion khác hẳn LLM serving
> **Edge/on-device:** Supertonic 3 + Piper; Kokoro để tham khảo kiến trúc lightweight
> **Vietnamese training:** ưu tiên OmniVoice LoRA/full fine-tune, không train TTS from scratch ngay.

---

# 1. Tối ưu inference và serving OmniVoice

Sau khi đọc code và documentation hiện tại, mình nghĩ đây nên là **project ưu tiên số 1**.

## 1.1. Bottleneck của OmniVoice đã khá rõ

OmniVoice dùng Qwen3-0.6B làm backbone và 8 acoustic codebooks. Default inference là **32 iterative unmasking steps**. Documentation chính thức cũng nói có thể dùng `num_step=16` để inference nhanh hơn. ([GitHub][1])

Khác LLM AR:

```text
LLM
token 1 → token 2 → token 3
          ↑
       KV cache
```

OmniVoice gần với:

```text
[M M M M M M]
      │
 Transformer full forward
      ↓
[M 7 M M 4 M]
      │
 Transformer full forward
      ↓
[2 7 M 8 4 M]
      │
      ...
     ~32x
```

Do Transformer là **bidirectional và toàn bộ sequence thay đổi ở mỗi iteration**, normal KV cache không phải optimization chính.

Một request về mặt Transformer compute có thể hình dung gần như:

```text
number of diffusion steps
×
full sequence forward
×
conditional / unconditional CFG
```

Vì vậy optimization space khác hẳn vLLM cho causal LLM.

---

## 1.2. Upstream đã chứng minh optimization quan trọng nhất: FlashInfer

Official OmniVoice hiện có optimized FlashInfer path với các trick quan trọng:

```text
CFG sequence packing
        +
ragged attention
        +
fused RMSNorm / RoPE / GEMM
        +
audio-head optimization
        +
optional CUDA Graph
```

Đặc biệt, conditional và unconditional sequence của classifier-free guidance được **pack** thay vì pad về cùng length. Với attention có cost tăng mạnh theo sequence length, việc loại padding này rất có giá trị.

Official benchmark trên **1 × H100, FP16, 32 diffusion steps**:

|          Batch | PyTorch baseline RTF | FlashInfer RTF |  Speedup |
| -------------: | -------------------: | -------------: | -------: |
|              1 |               0.0899 |         0.0430 |     2.1× |
| 1 + CUDA Graph |                    — |     **0.0367** | **2.4×** |
|              2 |               0.0480 |         0.0245 |     2.0× |
|              4 |               0.0331 |         0.0152 |     2.2× |
|              8 |               0.0298 |     **0.0115** | **2.6×** |

Upstream khuyến nghị CUDA Graph đặc biệt cho batch=1; batch ≥4 thì FlashInfer không graph đã rất tốt. ([GitHub][2])

RTF 0.0367 nghĩa là về throughput trung bình:

```text
10 giây audio
≈ 0.367 giây inference
```

Nhưng lưu ý đây **không đồng nghĩa TTFA = 367 ms**, và càng không đồng nghĩa streaming native.

---

# 2. Optimization plan cho OmniVoice mà mình sẽ thực hiện

Mình sẽ không bắt đầu bằng TensorRT hay FP8.

Thứ tự ROI nên là:

```text
PyTorch FP16 baseline
        ↓
FlashInfer
        ↓
FlashInfer + CUDA Graph
        ↓
32 → 16 diffusion steps
        ↓
dynamic / length-aware batching
        ↓
voice-prompt cache
        ↓
chunked generation
        ↓
torch.compile experiments
        ↓
FP8 / TensorRT / custom kernels
```

Đặc biệt `num_step` có khả năng là knob mạnh nhất sau FlashInfer:

```text
32 steps = quality/reference mode

16 steps = candidate production mode

8–12 steps = aggressive low-latency experiment
```

Không được đánh giá bằng latency đơn thuần. Phải kiểm tra WER/CER + speaker similarity + listening test.

---

# 3. Serving architecture cho OmniVoice

Official CLI hiện có batch inference và multi-GPU distribution, nhưng chưa phải một online serving scheduler kiểu vLLM. ([GitHub][3])

vLLM-Omni đã support OmniVoice qua OpenAI-compatible `/v1/audio/speech`, nhưng ở thời điểm hiện tại **online OmniVoice mới expose auto-voice; voice cloning và voice design vẫn là offline**, và chưa có native streaming trong bảng support. Trong khi Fish S2 Pro đã support cả voice cloning + streaming. ([vLLM][4])

Vậy với production OmniVoice cloning, mình nghiêng về kiến trúc:

```text
             API frontend
                  │
                  ▼
           Request Queue
                  │
                  ▼
        Duration Estimator
                  │
                  ▼
        Length-aware Scheduler
             │         │
             │ batches │
             ▼         ▼
        ┌─────────────────┐
        │ 1 GPU worker    │
        │                 │
        │ OmniVoice FP16  │
        │ + FlashInfer    │
        │ + CUDA Graph    │
        └────────┬────────┘
                 │
                 ▼
           Codec Decoder
                 │
                 ▼
        PCM/audio response
```

**Một GPU chỉ nên có một model-serving worker chính.** Không nên để 8 Gunicorn processes mỗi process load một bản OmniVoice và cùng tranh H100.

---

# 4. Dynamic batching phải làm khác LLM

Giả sử queue:

```text
A = 1.2 sec audio
B = 1.7 sec
C = 12 sec
D = 1.4 sec
```

Nếu batch:

```text
[A B C D]
```

thì A/B/D phải chịu padding gần bằng C.

Với full-sequence diffusion attention, việc này cực phí.

Nên bucket:

```text
0–2 s       → A B D
2–4 s
4–8 s
8–16 s      → C
```

Chính OmniVoice training implementation cũng dùng length-grouped padding cho SDPA để giảm padding waste. ([GitHub][5])

Đây là một trong những chỗ mình nghĩ team mình có thể tạo ra giá trị ngoài upstream.

---

# 5. Voice cloning cần cache trước

Production API không nên làm lại toàn bộ:

```text
reference.wav
     ↓
resample
     ↓
audio tokenizer
     ↓
reference acoustic tokens
     ↓
TTS
```

mỗi request.

Nên có abstraction:

```text
POST /voices
ref.wav + transcript
        ↓
precompute
        ↓
voice_id
```

sau đó:

```text
POST /tts
text + voice_id
      ↓
cached prompt
      ↓
OmniVoice
```

Điều này vừa giảm latency, vừa giảm GPU work, vừa làm request path ổn định hơn.

---

# 6. Streaming là vấn đề riêng của OmniVoice

Đây là thứ mình muốn nghiên cứu kỹ hơn nữa.

AR model:

```text
token → audio
token → audio
token → audio
```

có thể streaming khá tự nhiên.

OmniVoice cần refine một target sequence nhiều lần:

```text
chunk
 ↓
diffusion step 1
 ↓
...
 ↓
diffusion step 16/32
 ↓
codec
 ↓
audio
```

Nên “streaming” thực tế sẽ gần **chunked/pseudo streaming**:

```text
LLM:
"Xin chào anh."
       ↓
TTS chunk 1
       ↓
client hears audio

"Em đang kiểm tra..."
       ↓
TTS chunk 2
       ↓
client hears audio
```

Official long-form config mặc định dùng chunk khá lớn, nên cho conversational agent mình sẽ benchmark lại khoảng:

```text
~1.5 s
~2 s
~3 s
~4 s
```

và đánh đổi:

```text
chunk nhỏ
→ TTFA ↓
→ prosody continuity ↓
→ overhead ↑

chunk lớn
→ TTFA ↑
→ prosody tốt hơn
→ GPU efficiency ↑
```

Đây có thể trở thành một sub-project riêng.

---

# 7. Benchmark suite cho OmniVoice

Mình sẽ dựng một benchmark matrix cố định như sau:

| Dimension       | Cases                                   |
| --------------- | --------------------------------------- |
| Runtime         | PyTorch / FlashInfer / FlashInfer+Graph |
| diffusion steps | 32 / 16 / 12 / 8                        |
| batch           | 1 / 2 / 4 / 8                           |
| target duration | 1 / 3 / 5 / 10 / 15 s                   |
| concurrency     | 1 / 2 / 4 / 8 / 16                      |
| voice           | auto / clone                            |
| reference       | 3 / 5 / 10 s                            |
| GPU condition   | idle / shared-load                      |
| language        | Vietnamese / English                    |

Metrics không chỉ RTF:

| Loại          | Metrics                           |
| ------------- | --------------------------------- |
| User latency  | TTFA, E2E p50/p95/p99             |
| Throughput    | requests/s, audio-sec/s, RTF      |
| GPU           | utilization, VRAM, SM utilization |
| Quality       | WER/CER                           |
| Voice cloning | speaker similarity                |
| Naturalness   | UTMOS + listening                 |
| Stability     | repeat/skip/hallucination rate    |

Nếu cần phục vụ chatbot/realtime voice, **TTFA p95 quan trọng hơn average RTF**.

---

# 8. Landscape SOTA TTS hiện tại

Đây là snapshot **29/08/2026** và ranking có thể thay đổi nhanh.

Theo Artificial Analysis Speech Arena, provider-voice ranking hiện đứng đầu là:

| Rank | Model                           |  Elo |
| ---: | ------------------------------- | ---: |
|    1 | Cartesia Sonic 3.6              | 1288 |
|    2 | Qwen-Audio-3.0-TTS-Plus         | 1243 |
|    3 | Speechify Simba 3.2             | 1239 |
|    4 | Inworld Realtime TTS-2 Flash    | 1226 |
|    5 | VUI Luna TTS                    | 1225 |
|    6 | **Breeze TTS 2 — open weights** | 1221 |

Artificial Analysis dùng blind human preference / Elo, nên đây chủ yếu là **naturalness preference**, không phải benchmark tiếng Việt hay voice cloning hoàn chỉnh. ([Artificial Analysis][6])

Trong **open weights**, ranking hiện tại đứng đầu là:

```text
Breeze TTS 2
Fish Audio S2 Pro
Step Audio EditX
Voxtral TTS
NVIDIA Magpie-Multilingual 357M
Kokoro 82M
...
```

([Artificial Analysis][6])

Nhưng với bài toán của team, mình không chọn model chỉ dựa vào Elo.

---

# 9. Các model open hiện đáng nghiên cứu nhất

| Model            | Điểm mạnh                                                | Vietnamese    | Serving           | Mình xếp              |
| ---------------- | -------------------------------------------------------- | ------------- | ----------------- | --------------------- |
| **OmniVoice**    | 646 languages, zero-shot clone, NAR diffusion, rất nhanh | ✅ rất mạnh    | FlashInfer        | **P0**                |
| **Fish S2 Pro**  | quality + expressiveness + streaming                     | ✅             | SGLang/vLLM-Omni  | **P0/P1**             |
| **Breeze TTS 2** | #1 open-weight naturalness, <40 ms TTFA                  | ❌ chỉ EN/ZH   | rất tối ưu        | **P1 để học serving** |
| **Qwen3-TTS**    | 0.6B/1.7B, native streaming, voice design/clone          | ❌             | vLLM work ongoing | **P1**                |
| CosyVoice3       | 0.5B, streaming/clone tốt                                | ❌ official VI | vLLM-Omni         | P2                    |
| IndexTTS2        | expressive + duration/emotion control                    | hạn chế       | research          | P2                    |
| Kokoro           | 82M, cực nhỏ                                             | ❌             | edge-friendly     | edge reference        |

OmniVoice support **646 languages / 581k giờ**, trong đó riêng Vietnamese khoảng **8,481.98 giờ**. Đây là lý do nó đặc biệt đáng đầu tư cho use case Việt Nam. ([github.com][7])

---

# 10. Fish Audio S2 Pro là competitor mình sẽ benchmark nghiêm túc nhất

S2 Pro hiện là model rất đáng chú ý:

```text
4B
Dual-Autoregressive
10M+ hours audio
80+ languages
Vietnamese supported
native voice cloning
fine-grained prosody/emotion tags
multi-speaker
multi-turn
streaming
```

([Fish Audio][8])

Điểm thú vị cho bạn học inference là architecture của Fish lại **gần LLM serving hơn OmniVoice**.

Do là AR, serving engine dựa trên SGLang có thể tận dụng:

```text
continuous batching
Paged KV Cache
CUDA Graph
RadixAttention / prefix caching
```

Official Fish report trên một H200 khoảng **RTF 0.195, TTFA ~100 ms**, với >3000 acoustic token/s; đây là số tự công bố và khác hardware H100 nên chỉ nên dùng tham khảo. ([Fish Audio][9])

Do đó so sánh:

```text
OmniVoice
NAR diffusion
↓
full-sequence optimization

vs.

Fish S2
AR
↓
KV-cache / continuous batching optimization
```

sẽ là một bài học inference serving cực kỳ tốt.

Fish S2 cũng support Vietnamese trực tiếp. ([Fish Audio][10])

Caveat: weights/code dùng **Fish Audio Research License**, commercial deployment cần xem licensing kỹ. ([GitHub][11])

---

# 11. Breeze TTS 2 rất đáng đọc source dù không có tiếng Việt

Breeze TTS 2 vừa open-weight ngày **25/08/2026** và hiện đứng #1 open-weight trên Artificial Analysis. ([GitHub][12])

Điều mình thấy đáng học nhất không phải model quality mà là **serving implementation**.

Official H100 numbers:

```text
TTFA < 40 ms
RTF ≈ 0.32
~3.1× realtime
```

Fast mode khoảng 14.4 GiB VRAM. ([GitHub][12])

Họ tối ưu từng stage riêng:

```text
Text encoder
→ static CUDA Graph by text-length bucket

Backbone prefill
→ CUDA Graph by prompt bucket

Backbone decoding
→ StaticCache CUDA Graph

...
```

Đây là một codebase mình nghĩ rất nên đọc để học tư duy **latency-oriented TTS inference**.

Nhược điểm lớn cho mình: hiện chỉ **English + Chinese** và model weights/non-commercial output có license hạn chế. ([GitHub][12])

---

# 12. Qwen3-TTS cũng rất đáng học

Qwen3-TTS có các variant 0.6B và 1.7B, dùng tokenizer acoustic chỉ khoảng 12 Hz và hỗ trợ true streaming. Official report end-to-end latency thấp nhất khoảng **97 ms**. ([GitHub][13])

Điểm mạnh:

```text
Voice Clone
Voice Design
Custom Voice
Streaming
Fine-tuning officially supported
0.6B option khá nhỏ
```

Nhưng official language list hiện chỉ:

```text
Chinese
English
Japanese
Korean
German
French
Russian
Portuguese
Spanish
Italian
```

**không có Vietnamese**. ([GitHub][13])

Do đó mình xem Qwen3-TTS là **architecture/serving reference**, không phải candidate tiếng Việt đầu tiên.

---

# 13. Model on-device: shortlist khác hoàn toàn

Nếu mục tiêu là:

```text
Android / iOS
browser
embedded CPU
laptop CPU
edge device
```

thì không nên cố deploy OmniVoice/Fish.

Mình đang shortlist như sau:

| Model            | Size | Vietnamese | Runtime      | Nhận xét                            |
| ---------------- | ---: | ---------- | ------------ | ----------------------------------- |
| **Supertonic 3** | ~99M | ✅          | ONNX Runtime | tốt nhất để benchmark edge VI ngay  |
| **Piper**        |  nhỏ | ✅ voices   | ONNX/VITS    | dễ train custom voice nhất          |
| **Kokoro**       |  82M | ❌          | PyTorch/ONNX | quality/size rất đáng học           |
| Chatterbox Nano  | 110M | ❌ English  | CPU          | ~3× realtime / 8 cores              |
| Qwen3-TTS 0.6B   | 600M | ❌ VI       | GPU          | “small server”, chưa thật sự mobile |

---

# 14. Supertonic 3: candidate on-device tiếng Việt tốt nhất để thử ngay

Supertonic 3 chỉ khoảng **99M parameters**, chạy ONNX Runtime trên desktop, browser/WebGPU, mobile, Raspberry Pi và thậm chí e-reader; không bắt buộc GPU. Nó support **31 languages bao gồm Vietnamese**. ([GitHub][14])

Official benchmark của chính Supertonic trên Vietnamese:

| Model        | Vietnamese WER ↓ |
| ------------ | ---------------: |
| OmniVoice    |         **0.79** |
| VoxCPM2      |             1.48 |
| Supertonic 3 |             4.49 |

([GitHub][14])

Tức là nó đổi:

```text
quality ↓

để lấy

99M params
CPU
ONNX
on-device
```

Đây chính xác là trade-off mình muốn nghiên cứu.

Nhưng có một vấn đề rất mới: ngày **23/07/2026**, Supertone thông báo repository sẽ được archive và không còn official development/support; Voice Builder cũng đóng sau **31/08/2026**. ([GitHub][15])

Nên:

> **Benchmark Supertonic — có. Xây product roadmap dài hạn phụ thuộc vào upstream Supertonic — cần cân nhắc.**

---

# 15. Piper lại rất phù hợp nếu mục tiêu là “tự train tiếng Việt rồi chạy edge”

Current Piper successor vẫn active, release v1.4.2 vào tháng 4/2026, hỗ trợ:

```text
local inference
ONNX
C/C++
Python
training new voices
```

và dùng espeak-ng phonemization. ([GitHub][16])

Quality không ở cùng tier với OmniVoice/Fish.

Nhưng nếu project là:

> “Lấy 10 giờ một speaker tiếng Việt → train → export ONNX → chạy Android/CPU”

thì Piper là baseline cực hợp lý.

Nó giúp tách hai câu hỏi:

```text
Server TTS frontier?
→ OmniVoice

Small fixed voice TTS for edge?
→ Piper
```

Không nhất thiết một model phải giải quyết cả hai.

---

# 16. Kokoro: đáng nghiên cứu nhưng chưa phải Vietnamese candidate

Kokoro chỉ **82M params**, Apache 2.0, weights khoảng 327 MB và có ONNX variants. ([Hugging Face][17])

Điều thú vị là dù rất nhỏ, nó vẫn đang nằm khá cao trong open-weight human-preference ranking.

Nhưng current voice/language support gồm English, Japanese, Mandarin, Spanish, French, Hindi, Italian, Portuguese — **không có Vietnamese**. ([Hugging Face][18])

Mình sẽ dùng nó để trả lời câu hỏi:

> “Một modern TTS chất lượng khá có thể nhỏ đến mức nào?”

chứ chưa chọn làm Vietnamese P0.

---

# 17. Training TTS tiếng Việt: OmniVoice là điểm bắt đầu rất tốt

Điều mình thấy quan trọng nhất:

**OmniVoice đã pretrain trên ~8.5k giờ Vietnamese.**

([github.com][7])

Do đó bài toán không phải:

```text
dạy model học tiếng Việt
```

mà gần hơn:

```text
adapt model tốt hơn cho distribution của mình

+ pronunciation
+ numbers / abbreviations
+ names
+ code switching
+ regional accents
+ speaker cloning
+ target speaking style
```

Đây là bài toán dễ hơn rất nhiều so với language adaptation từ zero.

---

# 18. Official OmniVoice đã có pipeline fine-tuning và LoRA

Không cần tự viết trainer.

Official support:

```text
full fine-tuning
LoRA fine-tuning
init_from_checkpoint
WebDataset
BF16
FlexAttention / SDPA
```

([GitHub][5])

Data đầu vào cực đơn giản:

```json
{
  "id": "sample_001",
  "audio_path": "/data/001.wav",
  "text": "Xin chào Việt Nam",
  "language_id": "vi"
}
```

Pipeline:

```text
wav + transcript
      │
      ▼
Higgs audio tokenizer
      │
      ▼
8 × acoustic token sequence
      │
      ▼
WebDataset shards
      │
      ▼
OmniVoice LoRA / full FT
```

Official script sử dụng `eustlb/higgs-audio-v2-tokenizer` và đóng acoustic tokens thành WebDataset shards. ([GitHub][19])

Official fine-tune config còn khá nhỏ:

```text
init_from_checkpoint = k2-fsa/OmniVoice
steps = 5000
mixed_precision = bf16
batch_tokens = 8192
```

([GitHub][20])

Vì vậy **1 H100 hoàn toàn là môi trường hợp lý để bắt đầu fine-tune**, ít nhất ở LoRA/small experiment.

---

# 19. Dữ liệu tiếng Việt hiện khá thú vị

Có một số lựa chọn đáng khảo sát.

| Dataset           |          Size | Speakers | License / caveat              | Dùng làm gì               |
| ----------------- | ------------: | -------: | ----------------------------- | ------------------------- |
| VLSP 2020         |        ~5–6 h |        1 | challenge data                | controlled single-speaker |
| VieNeu-TTS public |   **140.7 h** |  **193** | Apache 2.0 theo dataset card  | rất đáng thử              |
| VieNeu-TTS 1000h  |       ~1000 h |    multi | CC-BY-NC, gated research      | research                  |
| viVoice           | **1016.97 h** |    nhiều | CC-BY-NC-SA, YouTube-derived  | research                  |
| Dolly-Audio       |       ~1000 h |  **152** | CC-BY-NC-SA, synthetic-tagged | research                  |

VLSP 2020 chỉ khoảng 5–6 giờ một speaker. ([VLSP][21])

**VieNeu-TTS-140h** công bố 140.7 giờ / 193 speaker và Apache 2.0 trên dataset card, nên đây là dataset mình sẽ kiểm tra kỹ đầu tiên. ([Hugging Face][22])

viVoice có **1,016.97 giờ**, 24 kHz, nhưng lấy từ YouTube và chính dataset card ước tính transcription-error khoảng 1.8%; license CC-BY-NC-SA nên không phù hợp để mặc định coi là commercial-clean. ([GitHub][23])

VieNeu cũng có bản 1000 giờ nhưng gated và CC-BY-NC, chỉ cho research/education. ([Hugging Face][24])

Dolly-Audio công bố gần 1000 giờ / 152 speaker và cũng là non-commercial; metadata còn tag `synthetic`, vì vậy mình sẽ đặc biệt audit provenance/quality trước khi dùng. ([Hugging Face][25])

Một chi tiết thú vị: đã có community checkpoint **OmniVoice Vietnamese fine-tuned trên Dolly 1000h**, report 8k steps / H200. Nó có thể dùng làm reference experiment, nhưng không nên coi số liệu/model card community là benchmark chuẩn. ([Hugging Face][26])

---

# 20. Mình không muốn bắt đầu bằng 1000 giờ

Experiment đầu tiên mình đề xuất là:

```text
OmniVoice base
      │
      ├── zero-shot Vietnamese benchmark
      │
      ▼
Curate 10–50 h very clean Vietnamese
      │
      ▼
LoRA
      │
      ▼
base vs LoRA A/B
      │
      ▼
Nếu gain rõ
      │
      ├── 100 h
      └── full FT / larger dataset
```

TTS cực kỳ nhạy với:

```text
transcription mismatch
bad segmentation
background noise
music
multiple speakers
wrong speaker IDs
bad punctuation
bad number normalization
clipped audio
reverb
```

Một tập **20 giờ sạch** có thể đáng giá hơn một tập 1000 giờ noisy cho experiment đầu tiên.

---

# 21. Vietnamese evaluation set nên tự xây

Đây mới có thể trở thành asset quan trọng của team.

Không chỉ lấy 200 câu Wikipedia.

Mình sẽ xây khoảng **500–1000 câu**, cố tình stress các trường hợp:

| Category    | Ví dụ                        |
| ----------- | ---------------------------- |
| dấu thanh   | hỏi/ngã/nặng, từ gần âm      |
| số          | `12.345`, `3,14`, `0987...`  |
| ngày tháng  | `29/08/2026`                 |
| tiền        | `1.250.000 đồng`             |
| đơn vị      | `10 GB`, `2.4 GHz`, `5 km/h` |
| acronym     | `VTV`, `AI`, `GPU`, `5G`     |
| English mix | “deploy model lên GPU H100”  |
| tên riêng   | tên người/địa danh           |
| địa chỉ     | phố/phường/quận              |
| câu hỏi     | prosody interrogative        |
| cảm xúc     | vui/buồn/ngạc nhiên          |
| long form   | 30–60 giây                   |
| miền        | Bắc/Trung/Nam                |
| voice clone | nhiều gender/accent          |

Rồi đo:

```text
CER
WER
speaker similarity
UTMOS
duration error
repeat / skip rate
```

và cuối cùng vẫn phải có **native Vietnamese A/B listening test**.

WER thấp không đồng nghĩa giọng hay.

---

# 22. Experiment suite mình đề xuất cho Vietnamese

Có 3 tầng rất rõ.

### Track A — Frontier Vietnamese

```text
OmniVoice base
vs
OmniVoice VI LoRA
vs
Fish S2 Pro
```

Objective:

```text
naturalness
pronunciation
voice clone
accent
code switching
```

### Track B — On-device Vietnamese

```text
Supertonic 3
vs
Piper Vietnamese
vs
custom Piper model
```

Objective:

```text
CPU RTF
RAM
model size
cold start
WER
MOS
```

### Track C — Vietnamese adaptation

```text
10h clean
→ 30h
→ 100h

LoRA
vs
full FT
```

Objective là tìm **data scaling curve**, chứ không chỉ train checkpoint cuối cùng.

---

# 23. Một matrix rất đáng làm

Ví dụ:

| Model                | Vietnamese CER/WER | MOS | SIM | TTFA | RTF | VRAM/RAM |
| -------------------- | -----------------: | --: | --: | ---: | --: | -------: |
| OmniVoice 32         |                    |     |     |      |     |          |
| OmniVoice 16         |                    |     |     |      |     |          |
| OmniVoice VI LoRA 16 |                    |     |     |      |     |          |
| Fish S2              |                    |     |     |      |     |          |
| Supertonic 3         |                    |     | N/A |      |     |          |
| Piper                |                    |     | N/A |      |     |          |

Kết quả này về lâu dài có giá trị hơn rất nhiều so với việc chỉ đọc paper.

---

# 24. Nếu mình phải chọn thứ tự ưu tiên R&D

Mình sẽ làm theo thứ tự này:

1. **Reproduce OmniVoice trên một H100**, build benchmark harness chuẩn và reproduce official FlashInfer speedup.
2. **Profile và serving OmniVoice**, tập trung FlashInfer, CUDA Graph, diffusion steps, length-aware batching, prompt caching và chunking.
3. **Tạo Vietnamese TTS evaluation suite** trước khi fine-tune bất kỳ model nào.
4. Benchmark **OmniVoice vs Fish S2 Pro** trên chính tiếng Việt.
5. Fine-tune **OmniVoice LoRA trên 10–50 giờ sạch**, rồi mới quyết định full FT/larger dataset.
6. Song song benchmark **Supertonic 3 + Piper** cho on-device.
7. Đọc serving code của **Breeze TTS 2 và Fish/SGLang**, vì hai model này cho hai tư duy serving rất khác OmniVoice.
8. Chỉ sau khi profile chứng minh cần thiết mới đụng tới **torch.compile sâu, FP8, TensorRT hoặc custom Triton kernels**.

---

# 25. Architecture map mà mình nghĩ bạn nên hình thành trong đầu

Sau vòng research này, mình sẽ chia modern TTS thành ba “family” để học:

```text
                 MODERN TTS
                    │
       ┌────────────┼────────────┐
       │            │            │
       ▼            ▼            ▼
     AR LM      NAR/Diffusion   Classical/
                               lightweight
       │            │            │
    Fish S2      OmniVoice       Piper
    Qwen3-TTS                   Kokoro
       │            │          Supertonic
       ▼            ▼            │
 KV cache       full-seq          ▼
 continuous     iterative       ONNX/CPU
 batching       decoding       quantization
 prefix cache   packing        small model
 streaming      CUDA Graph
```

Với background DL/NLP, học cả ba nhánh này sẽ giúp bạn chuyển sang speech nhanh hơn rất nhiều: phần speech-specific chủ yếu nằm ở **audio representation/codec, duration/prosody, vocoder và evaluation**; còn phần Transformer/inference thì bạn đã có nền khá tốt.

---

## Kết luận hiện tại

Nếu phải chốt ngay hướng cho team, mình sẽ chọn:

**OmniVoice = main server-side Vietnamese R&D model.** Nó đã có lượng Vietnamese pretraining rất lớn, official fine-tuning/LoRA và một optimized H100 FlashInfer implementation rất tốt. ([GitHub][7])

**Fish S2 Pro = strongest open competitor cần benchmark**, đặc biệt cho realtime streaming, emotion và voice cloning tiếng Việt. ([Fish Audio][10])

**Breeze TTS 2 = serving codebase đáng học**, dù không dùng trực tiếp cho tiếng Việt. ([GitHub][12])

**Supertonic 3 + Piper = on-device track**, trong đó Supertonic cho performance baseline còn Piper cho custom-training/deployment baseline. ([GitHub][14])

**Vietnamese fine-tuning = OmniVoice LoRA trước**, bắt đầu từ curated clean data thay vì lao thẳng vào 1000h.

Bước tiếp theo mình đề xuất là mình đi sâu riêng **phần 1: OmniVoice inference optimization**, đọc trực tiếp các hàm inference + FlashInfer implementation, dựng cho bạn **tensor-shape flow, profiling plan, benchmark script structure và serving architecture tới mức code**. Sau đó ta chuyển sang một tài liệu riêng kiểu **“TTS fundamentals for an NLP engineer”** để bạn nắm codec/vocoder/prosody/duration/AR-vs-NAR mà không phải học speech từ đầu.

Landscape TTS đang thay đổi rất nhanh; mình có thể theo dõi release/benchmark upstream và chỉ báo cho bạn khi có model hoặc optimization thực sự đáng chú ý.

[1]: https://github.com/k2-fsa/OmniVoice/blob/master/docs/generation-parameters.md?utm_source=chatgpt.com "OmniVoice/docs/generation-parameters.md at master · k2-fsa/OmniVoice · GitHub"
[2]: https://github.com/k2-fsa/OmniVoice?ref=huayutui.com&utm_source=chatgpt.com "GitHub - k2-fsa/OmniVoice at huayutui.com · GitHub"
[3]: https://github.com/k2-fsa/OmniVoice/blob/master/omnivoice/cli/infer_batch.py?utm_source=chatgpt.com "OmniVoice/omnivoice/cli/infer_batch.py at master · k2-fsa/OmniVoice · GitHub"
[4]: https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/online_serving/text_to_speech/?utm_source=chatgpt.com "Text-To-Speech (Online Serving) - vLLM-Omni"
[5]: https://github.com/k2-fsa/OmniVoice/blob/master/docs/training.md?utm_source=chatgpt.com "OmniVoice/docs/training.md at master · k2-fsa/OmniVoice · GitHub"
[6]: https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice?utm_source=chatgpt.com "Text to Speech Leaderboard - Top AI Speech Models | Artificial Analysis"
[7]: https://github.com/k2-fsa/OmniVoice/blob/master/docs/languages.md?utm_source=chatgpt.com "OmniVoice/docs/languages.md at master · k2-fsa/OmniVoice · GitHub"
[8]: https://docs.fish.audio/developer-guide/models-pricing/models-overview?utm_source=chatgpt.com "Models Overview - Fish Audio"
[9]: https://fish.audio/zh-CN/s2/?utm_source=chatgpt.com "Fish Audio S2 - The Most Expressive Open-Source TTS Model"
[10]: https://fish.audio/ko/blog/fish-audio-open-sources-s2/?articleLocale=en&utm_source=chatgpt.com "Fish Audio Open-Sources S2: Fine-Grained Control Meets Production Streaming - Fish Audio Blog"
[11]: https://github.com/fishaudio/fish-speech?utm_source=chatgpt.com "GitHub - fishaudio/fish-speech: SOTA Open Source TTS · GitHub"
[12]: https://github.com/breezeblue-ai/breeze-tts?utm_source=chatgpt.com "breezeblue-ai/breeze-tts: Official PyTorch inference for ..."
[13]: https://github.com/QwenLM/Qwen3-TTS?utm_source=chatgpt.com "Qwen3-TTS is an open-source series ..."
[14]: https://github.com/supertone-inc/supertonic?utm_source=chatgpt.com "GitHub - supertone-inc/supertonic: Lightning-Fast, On-Device, Multilingual TTS — running natively via ONNX. · GitHub"
[15]: https://github.com/supertone-inc/supertonic/?ref=blog.tjtl.io&utm_source=chatgpt.com "GitHub - supertone-inc/supertonic at blog.tjtl.io · GitHub"
[16]: https://github.com/OHF-voice/piper1-gpl?utm_source=chatgpt.com "GitHub - OHF-Voice/piper1-gpl: Fast and local neural text-to-speech engine · GitHub"
[17]: https://huggingface.co/hexgrad/Kokoro-82M/tree/main?utm_source=chatgpt.com "hexgrad/Kokoro-82M at main"
[18]: https://huggingface.co/hexgrad/Kokoro-82M/blame/main/VOICES.md?utm_source=chatgpt.com "VOICES.md · hexgrad/Kokoro-82M at main"
[19]: https://github.com/k2-fsa/OmniVoice/blob/master/examples/run_finetune_lora.sh?utm_source=chatgpt.com "OmniVoice/examples/run_finetune_lora.sh at master · k2-fsa/OmniVoice · GitHub"
[20]: https://github.com/k2-fsa/OmniVoice/blob/master/examples/config/train_config_finetune.json?utm_source=chatgpt.com "OmniVoice/examples/config/train_config_finetune.json at master · k2-fsa/OmniVoice · GitHub"
[21]: https://vlsp.org.vn/node/134?utm_source=chatgpt.com "Vietnamese Text-To-Speech on Common Datasets | Association for Vietnamese Language and Speech Processing"
[22]: https://huggingface.co/datasets/pnnbao-ump/VieNeu-TTS-140h?utm_source=chatgpt.com "pnnbao-ump/VieNeu-TTS-140h · Datasets at Hugging Face"
[23]: https://github.com/dieptran43/viVoice?utm_source=chatgpt.com "GitHub - dieptran43/viVoice: A 1000 Hours Cleaned Vietnamese Speech Dataset ✨ · GitHub"
[24]: https://huggingface.co/datasets/pnnbao-ump/VieNeu-TTS-1000h/tree/main?utm_source=chatgpt.com "pnnbao-ump/VieNeu-TTS-1000h at main"
[25]: https://huggingface.co/datasets/dolly-vn/dolly-audio-1000h-vietnamese?utm_source=chatgpt.com "dolly-vn/dolly-audio-1000h-vietnamese · Datasets at Hugging Face"
[26]: https://huggingface.co/splendor1811/omnivoice-vietnamese?utm_source=chatgpt.com "splendor1811/omnivoice-vietnamese · Hugging Face"
