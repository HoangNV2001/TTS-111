Đúng trọng tâm của OmniVoice là: **voice clone không phải “rút speaker embedding rồi đưa embedding đó vào TTS” theo kiểu truyền thống**. Nó dùng chính **audio tokens của câu tham chiếu làm context**, ghép cùng transcript tham chiếu và text cần đọc, rồi để backbone kiểu LLM dự đoán **audio tokens của câu mới** bằng iterative masked decoding.

Có thể hình dung pipeline thực tế như sau:

```text
ref_audio.wav
   │
   ├─ resample / mono / silence processing
   │
   ├─ Audio Tokenizer
   │       ↓
   │   ref_audio_tokens: [8, T_ref]
   │
   └─ Whisper ASR (chỉ khi không có ref_text)
           ↓
       ref_text

target text
   ↓

          build conditioning sequence
─────────────────────────────────────────────
style tokens
+ [ref_text + target_text]
+ ref_audio_tokens
+ MASK MASK MASK ...     ← audio cần sinh
─────────────────────────────────────────────
                 ↓
               Qwen3
                 ↓
        logits cho 8 codebooks
                 ↓
 iterative masked token generation
                 ↓
 target_audio_tokens [8, T_target]
                 ↓
 Higgs Audio Tokenizer decoder
                 ↓
 waveform
```

Và điểm rất đáng chú ý: **thứ tự thực tế là TEXT trước, AUDIO reference sau**, chứ không phải `ref_text → ref_audio → target_text → output`. Code hiện tại dựng sequence thành:

```text
STYLE
→ TEXT(ref_text + target_text)
→ REF_AUDIO_TOKENS
→ TARGET_AUDIO_MASKS
```

Đây là điều quan trọng nếu bạn đang nghiên cứu tối ưu inference. 

---

## 1. Đầu vào voice cloning

Ví dụ của bạn:

```bash
omnivoice-infer \
  --model k2-fsa/OmniVoice \
  --text "Hôm nay tôi bắt đầu học về tối ưu suy luận..." \
  --ref_audio vi_ref_01.wav \
  --ref_text "Xin chào, tôi đang thử giọng nói này..." \
  --output clone_01.wav
```

Về logic, OmniVoice có 3 dữ liệu chính:

```text
ref_audio
ref_text
target_text
```

Ví dụ:

```text
ref_text:
"Xin chào, đây là giọng nói tham chiếu."

ref_audio:
[waveform của đúng câu trên]

target_text:
"Hôm nay tôi bắt đầu học về tối ưu suy luận."
```

`ref_text` và `ref_audio` cần **khớp nhau về nội dung**.

---

# 2. Xử lý reference audio

`create_voice_clone_prompt()` là phần quan trọng đầu tiên.

OmniVoice cuối cùng tạo object:

```python
VoiceClonePrompt(
    ref_audio_tokens,
    ref_text,
    ref_rms,
)
```

Trong implementation hiện tại, object này chứa đúng ba thứ:

```text
ref_audio_tokens : Tensor [C, T]
ref_text         : string
ref_rms          : float
```

Không có trường kiểu:

```text
speaker_embedding
voice_embedding
style_embedding
```

riêng biệt. 

---

## 2.1 Load và chuẩn hóa audio

Nếu audio là file:

```python
ref_wav = load_audio(ref_audio, self.sampling_rate)
```

Nếu input là `(waveform, sample_rate)` thì OmniVoice:

```text
stereo → mono
resample → sampling rate của audio tokenizer
```

Code thực tế:

```python
if waveform.shape[0] > 1:
    waveform = np.mean(waveform, axis=0, keepdims=True)

if sr != self.sampling_rate:
    waveform = torchaudio.functional.resample(...)
```

([GitHub][1])

---

# 3. RMS normalization

OmniVoice tính:

```python
ref_rms = sqrt(mean(ref_wav ** 2))
```

Nếu:

```text
0 < RMS < 0.1
```

thì nó nâng volume reference lên:

```python
ref_wav *= 0.1 / ref_rms
```

Điểm thú vị là **RMS gốc vẫn được giữ lại** trong `VoiceClonePrompt`.

Sau khi sinh xong audio, OmniVoice lại scale output về mức tương ứng:

```python
generated_audio *= ref_rms / 0.1
```

nếu reference ban đầu nhỏ hơn 0.1 RMS. ([GitHub][1])

Vậy `ref_rms` không tham gia trực tiếp vào conditioning giọng; nó chủ yếu dùng cho **volume matching hậu xử lý**.

---

# 4. Silence removal / trim

Nếu:

```python
preprocess_prompt=True
```

mặc định, OmniVoice xử lý silence.

Đáng chú ý:

```text
nếu ref_text KHÔNG được cung cấp
    có thể trim reference >20 s

nếu ref_text đã được cung cấp
    không trim kiểu đó
```

Lý do rất hợp lý: nếu cắt audio nhưng giữ nguyên transcript thì `ref_text` và `ref_audio` mất alignment.

Sau đó nó chạy silence removal với khoảng giữ silence tương đối ngắn ở đầu/cuối. Repo cũng cảnh báo reference >20 giây có thể làm tăng memory, chậm inference và giảm cloning quality; họ khuyên khoảng **3–10 giây**. ([GitHub][1])

---

# 5. Nếu không có `ref_text`: Whisper

Đoạn này đúng với câu hỏi trước của bạn.

```python
if ref_text is None:
    if self._asr_pipe is None:
        self.load_asr_model()

    ref_text = self.transcribe(...)
```

ASR mặc định hiện tại là:

```text
openai/whisper-large-v3-turbo
```



Vậy:

```text
ref_audio
   │
   ├──────────────┐
   ↓              ↓
Whisper        Audio tokenizer
   ↓              ↓
ref_text      ref_audio_tokens
```

Hai nhánh này sau đó hội tụ lại trong prompt.

---

# 6. Audio reference được biến thành cái gì?

Đây là phần quan trọng nhất.

OmniVoice hiện dùng **Higgs Audio V2 tokenizer**.

Reference waveform được encode:

```python
ref_audio_tokens = self.audio_tokenizer.encode(
    ref_wav_tensor.unsqueeze(0)
).audio_codes.squeeze(0)
```

Output:

```text
(C, T)
```

Với OmniVoice:

```text
C = 8 audio codebooks
```

Repo training cũng kiểm tra rõ:

```python
assert audio_tokens.size(0) == 8
```



Nó giống ma trận:

```text
time →       t0   t1   t2   t3   ... tN

codebook 0   25  311   98  502   ...
codebook 1  812   72  644  122   ...
codebook 2  123  532   91  227   ...
codebook 3  ...
...
codebook 7  ...
```

Nên mỗi **audio frame** không phải một token duy nhất.

Nó là:

```text
8 discrete token IDs
```

một token cho mỗi codebook.

---

# 7. Audio vocab

Model config hiện khai báo:

```text
num_audio_codebook = 8
audio_vocab_size   = 1025
audio_mask_id      = 1024
```

Nghĩa là mỗi codebook về logic có:

```text
0 ... 1023    = audio codes
1024          = MASK
```

([GitHub][1])

Đây là chìa khóa để hiểu generation.

OmniVoice **không generate autoregressive từng audio token từ trái sang phải** như:

```text
t0 → t1 → t2 → t3 → ...
```

Nó bắt đầu bằng:

```text
MASK MASK MASK MASK MASK ...
MASK MASK MASK MASK MASK ...
...
8 hàng
```

rồi dần dần điền các mask.

---

# 8. Reference audio được ghép với text thế nào?

Đây chính là phần bạn hỏi về **thứ tự prompt**.

Trong `_prepare_inference_inputs()`, đầu tiên nó tạo `style_text`.

Nếu voice clone có reference audio và `denoise=True`:

```text
<|denoise|>
```

Sau đó:

```text
<|lang_start|>LANG<|lang_end|>
<|instruct_start|>INSTRUCT<|instruct_end|>
```

Code:

```python
style_text = ""

if denoise and ref_audio_tokens is not None:
    style_text += "<|denoise|>"

style_text += f"<|lang_start|>{lang_str}<|lang_end|>"
style_text += f"<|instruct_start|>{instruct_str}<|instruct_end|>"
```



Với trường hợp voice clone bình thường và `language=vi`, đại khái:

```text
<|denoise|>
<|lang_start|>vi<|lang_end|>
<|instruct_start|>None<|instruct_end|>
```

---

# 9. `ref_text` và target text được nối vào nhau trước

Đây là một chi tiết rất đáng chú ý.

Code gọi:

```python
full_text = _combine_text(
    ref_text=ref_text,
    text=text
)
```

Implementation:

```python
if ref_text:
    full_text = ref_text.strip() + " " + text.strip()
else:
    full_text = text.strip()
```

Sau đó:

```python
wrapped_text =
    "<|text_start|>"
    + full_text
    + "<|text_end|>"
```



Ví dụ:

```text
ref_text =
"Xin chào, đây là giọng tham chiếu."

target_text =
"Hôm nay tôi bắt đầu học về tối ưu suy luận."
```

thành:

```text
<|text_start|>
Xin chào, đây là giọng tham chiếu. Hôm nay tôi bắt đầu học về tối ưu suy luận.
<|text_end|>
```

**Không có separator rõ kiểu `<REF_TEXT>` / `<TARGET_TEXT>` ở giữa.**

Nó đơn giản nối:

```text
ref_text + " " + target_text
```

---

# 10. Đây mới là thứ tự prompt đầy đủ

Sau khi text tokenize, code:

```python
parts = [
    style_tokens,
    text_tokens
]

if ref_audio_tokens is not None:
    parts.append(ref_audio_tokens)

parts.append(target_audio_tokens)
```

sau đó:

```python
cond_input_ids = torch.cat(parts, dim=2)
```



Vì vậy sequence chính xác về logic là:

```text
┌──────────────────────────────────────┐
│ STYLE                                │
│                                      │
│ <|denoise|>                          │
│ <|lang_start|>vi<|lang_end|>         │
│ <|instruct_start|>None<|...|>        │
├──────────────────────────────────────┤
│ TEXT                                 │
│                                      │
│ <|text_start|>                       │
│ REF_TEXT                             │
│ TARGET_TEXT                          │
│ <|text_end|>                         │
├──────────────────────────────────────┤
│ REFERENCE AUDIO TOKENS               │
│                                      │
│ [8 × T_ref]                          │
├──────────────────────────────────────┤
│ TARGET AUDIO TOKENS                  │
│                                      │
│ [MASK MASK MASK ...]                 │
│ [MASK MASK MASK ...]                 │
│ ... 8 codebooks                      │
└──────────────────────────────────────┘
```

Hay viết một dòng:

```text
STYLE
→ TEXT_START
→ REF_TEXT
→ TARGET_TEXT
→ TEXT_END
→ REF_AUDIO
→ MASKED_TARGET_AUDIO
```

**Đó là ordering thật trong inference code hiện tại.**

---

# 11. Nhưng text token và audio token làm sao cùng đi vào Qwen?

Đây là một phần thiết kế rất hay.

Input có shape:

```text
[B, C, S]
```

với:

```text
C = 8
```

Nhưng Qwen bình thường cần:

```text
[B, S, hidden]
```

OmniVoice có hai đường embedding.

### Với vị trí text

Nó lấy channel đầu:

```python
text_embeds =
    self.get_input_embeddings()(input_ids[:, 0, :])
```

Tức là dùng **Qwen text embedding bình thường**.

### Với vị trí audio

Nó có riêng:

```python
self.audio_embeddings = nn.Embedding(
    num_audio_codebook * audio_vocab_size,
    hidden_size
)
```

Mỗi codebook có offset riêng:

```python
offset = codebook_id * audio_vocab_size
```

Sau đó:

```text
audio_token codebook 0
audio_token codebook 1
...
audio_token codebook 7
```

được embed riêng và **SUM lại**:

```python
audio_embeds =
    self.audio_embeddings(shifted_ids).sum(dim=1)
```

Rồi chọn:

```python
torch.where(
    audio_mask,
    audio_embeds,
    text_embeds
)
```

([GitHub][1])

Tức là tại một timestep audio:

```text
codebook0 token ─→ embedding ┐
codebook1 token ─→ embedding │
codebook2 token ─→ embedding │
...                          ├─ SUM → một vector hidden
codebook7 token ─→ embedding ┘
```

Kết quả:

```text
8 audio codes
       ↓
1 hidden vector
       ↓
Qwen position
```

Đây là lý do sequence length theo **audio time frames**, không phải `8 × T` positions.

---

# 12. `audio_mask` phân biệt text/audio

OmniVoice tạo một boolean mask:

```text
False = vị trí text/style
True  = vị trí audio
```

Với sequence:

```text
STYLE | TEXT | REF AUDIO | TARGET AUDIO
```

thì:

```text
0 0 0 0 0 0 0 | 0 0 0 ... | 1 1 1 ... | 1 1 1 ...
```

Code tính điểm bắt đầu audio sao cho **cả reference audio và target audio** đều được đánh `True`. 

Nghĩa là Qwen nhận một chuỗi embedding duy nhất:

```text
text embeddings
text embeddings
...
audio embeddings
audio embeddings
...
```

---

# 13. Target audio length được xác định trước

Đây cũng là điểm khác AR-TTS.

Trước khi generation, OmniVoice phải biết:

```text
T_target ≈ bao nhiêu audio frames
```

Nó dùng `RuleDurationEstimator`.

Khi có reference:

```python
estimate_duration(
    target_text,
    ref_text,
    num_ref_audio_tokens
)
```



Nói dễ hiểu:

```text
Reference:
"Xin chào..." = 4 s = T_ref frames

Target:
"Hôm nay tôi..." dài hơn ~1.8×

→ estimate target ≈ 1.8 × T_ref
```

Sau đó target audio ban đầu được tạo:

```python
target_audio_tokens = torch.full(
    [1, 8, T_target],
    audio_mask_id
)
```

hay:

```text
8 × T_target toàn bộ = MASK
```

---

# 14. Generation không phải autoregressive

Đây có lẽ là phần quan trọng nhất khi bạn muốn **tối ưu inference**.

OmniVoice gọi:

```python
_generate_iterative(...)
```

Docstring ghi:

```text
N-step iterative unmasked decoding
```

Mặc định:

```text
num_step = 32
```



Ban đầu:

```text
Codebook 0:  M M M M M M M M M
Codebook 1:  M M M M M M M M M
...
Codebook 7:  M M M M M M M M M
```

Step 1:

```text
Codebook 0:  M 3 M M 91 M M M
Codebook 1:  M M M 8 M M M M
...
```

Step 2:

```text
Codebook 0:  7 3 M 2 91 M M M
codebook 1:  M 5 M 8 M 6 M M
...
```

...

Step 32:

```text
Codebook 0: 7 3 12 2 91 38 ...
Codebook 1: 2 5 14 8 21 6 ...
...
```

Không còn MASK.

---

# 15. Mỗi inference step Qwen output cái gì?

Qwen tạo:

```text
hidden_states:
[B, S, hidden]
```

Sau đó OmniVoice có:

```python
self.audio_heads = nn.Linear(
    hidden_size,
    8 * 1025
)
```

Output được reshape:

```text
[B, S, 8, 1025]

→

[B, 8, S, 1025]
```

([GitHub][1])

Nghĩa là tại **mọi audio time position**, model predict:

```text
codebook 0 → distribution 1025 classes
codebook 1 → distribution 1025 classes
...
codebook 7 → distribution 1025 classes
```

Tức là về mặt output head:

```text
hidden(t)
   │
   ├→ logits CB0 [1025]
   ├→ logits CB1 [1025]
   ├→ logits CB2 [1025]
   ...
   └→ logits CB7 [1025]
```

---

# 16. Classifier-Free Guidance

Một chi tiết nữa rất quan trọng với performance:

OmniVoice thực ra tạo **2 batch copies**:

```text
conditional
unconditional
```

Nó allocate:

```python
batch_input_ids:
[2 * B, 8, max_len]
```



Conditional:

```text
STYLE + TEXT + REF_AUDIO + TARGET_MASK
```

Unconditional chỉ giữ:

```text
TARGET_MASK
```

Sau forward:

```text
conditional logits = c_logits
unconditional logits = u_logits
```

Rồi:

```python
c_log_probs
+ guidance_scale * (c_log_probs - u_log_probs)
```



Với default:

```text
guidance_scale = 2.0
```

Đây là **CFG**, gần giống tư tưởng trong diffusion.

Và đây cũng là chỗ đáng tối ưu vì effective batch inference bị nhân khoảng:

```text
B → 2B
```

---

# 17. Token nào được unmask trước?

OmniVoice không đi từ trái sang phải.

Nó tính confidence cho từng:

```text
(codebook, time_position)
```

rồi chọn `top-k` vị trí để mở mask.

Có thêm penalty:

```python
scores =
    scores
    - layer_id * layer_penalty_factor
```

Default:

```text
layer_penalty_factor = 5.0
```



Do đó codebook sớm được ưu tiên unmask trước.

Sơ đồ:

```text
coarse-ish layers
CB0 ███████████
CB1 █████████
CB2 ███████
CB3 █████
...
CB7 ██
```

Không nên hiểu cứng rằng mỗi codebook đúng nghĩa “pitch”, “speaker”, “noise” riêng biệt, nhưng hierarchical codec thường khiến codebook đầu mang phần thông tin nền tảng hơn.

---

# 18. Voice cloning xảy ra ở đâu?

Đây là câu hỏi khái niệm quan trọng.

Không có bước:

```text
reference audio
   ↓
speaker encoder
   ↓
speaker vector = [0.2, ...]
```

Thay vào đó:

```text
ref_text
+
raw discrete audio codes của reference
         ↓
      context
         ↓
Transformer học quan hệ:
"text này ↔ audio này ↔ đặc tính speaker/prosody"
         ↓
predict audio cho target text
```

Có thể hiểu kiểu **in-context voice cloning**:

```text
"Đây là text người đó đã nói"
+
"Đây là cách âm thanh của họ khi nói text đó"
+
"Đây là text tôi muốn họ nói"
         ↓
"hãy hoàn thiện audio tiếp theo"
```

Đây giống tinh thần của LLM few-shot hơn là hệ speaker-embedding cổ điển.

---

# 19. Một ví dụ đầy đủ với câu của bạn

Giả sử:

```text
REF TEXT:
"Tôi xin chào tất cả mọi người."

REF AUDIO:
3.5 giây

TARGET:
"Hôm nay tôi bắt đầu học về tối ưu suy luận."
```

Audio tokenizer:

```text
ref_audio
    ↓
8 × ~N_ref codes
```

Giả sử minh họa:

```text
CB0: 71 92 15 44 ...
CB1:  2 81 39 72 ...
CB2: 43 12 91 18 ...
...
CB7: 55 72 19  4 ...
```

Prompt:

```text
<|denoise|>
<|lang_start|>vi<|lang_end|>
<|instruct_start|>None<|instruct_end|>

<|text_start|>
Tôi xin chào tất cả mọi người.
Hôm nay tôi bắt đầu học về tối ưu suy luận.
<|text_end|>

[REF_AUDIO_FRAME_0]
[REF_AUDIO_FRAME_1]
...
[REF_AUDIO_FRAME_N]

[MASK_FRAME_0]
[MASK_FRAME_1]
...
[MASK_FRAME_M]
```

Mỗi audio frame thực chất chứa:

```text
[token_cb0,
 token_cb1,
 token_cb2,
 ...
 token_cb7]
```

---

# 20. Điều rất dễ hiểu nhầm: audio reference không “nằm cạnh” ref_text

Ở mức semantic ta thường vẽ:

```text
ref_text → ref_audio
target_text → target_audio
```

Nhưng actual sequence lại là:

```text
ref_text + target_text
       ↓
   text tokens
       ↓
ref audio tokens
       ↓
target audio tokens
```

Tức:

```text
[TEXT của cả reference + target]
[KNOWN AUDIO reference]
[UNKNOWN AUDIO target]
```

Mô hình học bài toán gần với:

```text
conditional audio infilling/completion
```

hơn một decoder AR truyền thống.

---

# 21. Sau 32 steps: decode audio

Kết quả:

```text
generated tokens:
[8, T_target]
```

được đưa trở lại Higgs tokenizer decoder:

```python
audio_tokenizer.decode(
    tokens.unsqueeze(0)
)
```

→ waveform. ([GitHub][1])

Sau đó OmniVoice:

```text
remove long silence
↓
volume matching theo ref_rms
↓
fade-in / fade-out
↓
pad silence
↓
output waveform
```

---

# 22. Toàn bộ pipeline compact

Nếu vẽ sát implementation nhất:

```text
                        ┌───────────────┐
ref_audio.wav ─────────▶│ Load/Resample │
                        └──────┬────────┘
                               │
                        mono waveform
                               │
                 ┌─────────────┴──────────────┐
                 │                            │
                 ▼                            ▼
           Whisper ASR                Silence/RMS preprocess
       (nếu thiếu ref_text)                  │
                 │                            ▼
                 │                    Higgs Audio Tokenizer
                 │                            │
                 ▼                            ▼
             ref_text                ref_audio_tokens
                                           [8,Tref]
                 │                            │
                 └────────────┬───────────────┘
                              │

target_text ──────────────────┤

                              ▼
                     Duration estimator
                              │
                              ▼
                       T_target frames
                              │
                              ▼
                     [8,Ttarget] MASK
                              │
                              ▼

┌──────────────────────────────────────────────────────────┐
│                  CONDITIONING SEQUENCE                   │
│                                                          │
│ <denoise>                                                │
│ <lang_start>vi<lang_end>                                 │
│ <instruct_start>None<instruct_end>                       │
│                                                          │
│ <text_start>                                             │
│ REF_TEXT + TARGET_TEXT                                   │
│ <text_end>                                               │
│                                                          │
│ REF AUDIO TOKENS [8,Tref]                                │
│                                                          │
│ TARGET MASK TOKENS [8,Ttarget]                           │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
                   Audio/Text Embedding
                         │
            text → Qwen text embedding
            audio → 8 codebook embeddings → SUM
                         │
                         ▼
                       Qwen3
                         │
                  audio output head
                         │
                [8,Ttarget,1025]
                         │
                         ▼
                   CFG cond/uncond
                         │
                         ▼
               confidence + top-k
                         │
                         ▼
              fill selected MASKs
                         │
                         ├──── repeat ~32 steps ────┐
                         │                          │
                         ◀──────────────────────────┘
                         │
                         ▼
              generated tokens [8,Ttarget]
                         │
                         ▼
                Higgs audio decoder
                         │
                         ▼
                     waveform
                         │
                 post-processing
                         │
                         ▼
                   clone_01.wav
```

---

## 23. Với mục tiêu tối ưu inference của bạn, 5 hotspot lớn nhất

Nếu bạn đang chuẩn bị tối ưu OmniVoice inference, architecture này cho thấy ngay các vùng đáng đo:

1. **Reference preprocessing** — Whisper + audio tokenizer có thể cache hoàn toàn. `VoiceClonePrompt` thậm chí đã hỗ trợ save/load `ref_audio_tokens`, `ref_text`, `ref_rms`. ([GitHub][1])

2. **Qwen forward lặp `num_step` lần** — default 32. Đây rất có thể là phần dominant compute.

3. **CFG làm batch thành `2B`** — conditional + unconditional chạy đồng thời.

4. **Sequence length = style/text + T_ref + T_target** — reference càng dài, Qwen attention mỗi step càng nặng. Đây là một lý do reference 3–10 s hợp lý về cả cloning quality lẫn compute.

5. **Không phải AR decoding**, nên các kỹ thuật tối ưu kiểu KV-cache autoregressive truyền thống **không áp dụng nguyên xi**. Mỗi iteration đang cập nhật nhiều vị trí masked trong toàn target sequence và chạy lại transformer trên sequence. Đây là khác biệt quan trọng nhất khi thiết kế TensorRT/vLLM/`torch.compile`/CUDA Graph optimization cho OmniVoice. 

Nếu mục tiêu tiếp theo là **tối ưu inference thực sự**, mình sẽ bắt đầu từ `_prepare_inference_inputs()` và `_generate_iterative()`, vì gần như toàn bộ bài toán latency nằm ở hai đoạn đó.

[1]: https://github.com/k2-fsa/OmniVoice/blob/master/omnivoice/models/omnivoice.py "OmniVoice/omnivoice/models/omnivoice.py at master · k2-fsa/OmniVoice · GitHub"
