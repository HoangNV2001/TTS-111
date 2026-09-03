# RESULTS

> Số liệu đo được. Mọi bảng đều **trống cho tới khi bạn chạy thật** — không điền số ước lượng vào đây.
> Xem [PLAN.md](PLAN.md) cho lệnh sinh ra từng bảng, [PROGRESS.md](PROGRESS.md) cho nhật ký.

---

## Quy tắc ghi số

1. Mỗi run phải ghi đủ **context**: GPU, driver, torch, commit, test set, `warmup`, sampling temperature.
2. Benchmark chất lượng chạy **greedy** (`position_temperature=0.0 class_temperature=0.0`), nếu không thì delta có thể là sampling noise.
3. Benchmark tốc độ luôn có `--warmup >= 3`.
4. Số không reproduce được thì ghi rõ `(1 lần đo)`. Số quan trọng chạy lại 3 lần, ghi median.
5. So với upstream thì so **tỉ lệ speedup**, không so số tuyệt đối — khác test set, khác ngôn ngữ, khác độ dài câu.

---

## Bối cảnh đo

| Mục | Giá trị |
|---|---|
| GPU | *(chưa đo)* |
| Driver | *(chưa đo)* |
| torch | *(chưa đo)* |
| omnivoice version / commit | *(chưa đo)* |
| flashinfer | *(chưa đo)* |
| Test set | *(chưa đo)* |
| Ngày đo | *(chưa đo)* |

---

## B0 — Smoke test (Phase 0)

| Kiểm tra | Kết quả | Ghi chú |
|---|---|---|
| `torch.cuda.is_available()` | | |
| Thời gian download weights | | |
| Kích thước `$HF_HOME` sau download | | |
| Sinh 1 câu VI auto-voice | | |
| Nghe thử — phát âm tiếng Việt | | *nhận xét chủ quan* |

---

## B1 — Baseline PyTorch fp16 (Phase 1)

Cấu hình: `num_step=32`, `enable_flashinfer=false`, `warmup=3`, test set `vi_smoke.jsonl` (12 câu).

| batch_size | Average RTF | p50 latency (s) | p95 latency (s) | peak VRAM (GB) |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 4 | | | | |
| 8 | | | | |

Voice cloning:

| Kiểm tra | Kết quả |
|---|---|
| Ref audio dùng | |
| Giống ref? (chủ quan 1–5) | |
| Có giữ được accent VI? | |

---

## B2 — FlashInfer speedup (Phase 2)

**Mục tiêu:** reproduce tỉ lệ speedup của upstream, không phải số tuyệt đối.

| batch | baseline RTF (đo) | FlashInfer RTF (đo) | speedup (đo) | speedup upstream |
|---|---|---|---|---|
| 1 | | | | 2.1× |
| 1 + CUDA graph | — | | | 2.4× |
| 2 | | | | 2.0× |
| 4 | | | | 2.2× |
| 8 | | | | 2.6× |

*(Upstream đo trên seed-tts zh, 2020 sample / 3.3h audio, single H100 fp16 num_step=32)*

Ba runtime, batch=1, từ `bench/bench_graph.py`:

| runtime | Average RTF | p50 (s) | p95 (s) | peak VRAM (GB) |
|---|---|---|---|---|
| pytorch | | | | |
| flashinfer | | | | |
| flashinfer + cuda graph | | | | |

### Bốn câu hỏi phải trả lời được

| # | Câu hỏi | Câu trả lời của bạn |
|---|---|---|
| 1 | CFG sequence packing ghép cond/uncond thế nào, tiết kiệm gì so với pad? | |
| 2 | Ragged attention: vì sao chuỗi packed không cần padding mask? | |
| 3 | Vì sao KV cache không phải optimization chính ở đây? | |
| 4 | Vì sao CUDA graph chỉ ăn rõ ở batch=1? | |

---

## B3 — `num_step` sweep (Phase 3)

Cấu hình: `batch_size=4`, `enable_flashinfer=true`, `warmup=3`, greedy sampling.

| num_step | Average RTF | speedup vs 32 | Nhận xét khi nghe | WER (điền ở Phase 4) | SIM | UTMOS |
|---|---|---|---|---|---|---|
| 32 | | 1.00× | | | | |
| 24 | | | | | | |
| 16 | | | | | | |
| 12 | | | | | | |
| 8 | | | | | | |
| 4 | | | | | | |

**Kết luận tạm — production candidate:** *(chưa có)*

### Knob khác

| Knob | Giá trị thử | RTF | Nhận xét |
|---|---|---|---|
| `guidance_scale` | 1.5 / 2.0 / 3.0 | | |
| `t_shift` | 0.1 / 0.3 / 0.5 | | |
| `position_temperature` | 5.0 / 0.0 | | |

### Batching strategy

Test set `vi_duration.jsonl` (độ dài 1/3/5/10/15 s trộn lẫn).

| Strategy | Average RTF | Tổng thời gian | Padding waste ước tính |
|---|---|---|---|
| fixed `batch_size=8` | | | |
| duration-based `batch_duration=120` | | | |

---

## B4 — Vietnamese quality (Phase 4)

Eval set: `vi_eval_v1.jsonl` — *(chưa xây)*

### WER nền của ASR (sanity check)

| ASR model | Audio người thật (VieNeu) | WER | Kết luận |
|---|---|---|---|
| | | | |

> Nếu WER nền cao, mọi số WER của TTS bên dưới đều vô nghĩa.

### Kết quả chính

| Cấu hình | CER | WER | SIM-o | UTMOS | RTF | Ghi chú |
|---|---|---|---|---|---|---|
| OmniVoice ns=32 | | | | | | |
| OmniVoice ns=16 | | | | | | |
| OmniVoice ns=8 | | | | | | |

### WER theo category

| Category | CER | WER | Lỗi điển hình |
|---|---|---|---|
| dấu thanh | | | |
| số | | | |
| ngày tháng | | | |
| tiền tệ | | | |
| đơn vị | | | |
| acronym | | | |
| chèn tiếng Anh | | | |
| tên riêng | | | |
| địa chỉ | | | |
| câu hỏi | | | |
| cảm xúc | | | |
| long-form | | | |
| vùng miền | | | |

> Bảng này mới là thứ có giá trị nhất. Số WER tổng thể che mất chỗ model thật sự yếu.

### A/B listening test

| Cặp so sánh | n | Thắng A | Thắng B | Hoà | Kết luận |
|---|---|---|---|---|---|
| ns32 vs ns16 | | | | | |
| ns16 vs ns8 | | | | | |

---

## B5 — Serving (Phase 5)

### Voice prompt cache

| Cấu hình | TTFA p50 | TTFA p95 | E2E p50 | E2E p95 |
|---|---|---|---|---|
| không cache (encode ref mỗi request) | | | | |
| có cache (`VoiceClonePrompt.load`) | | | | |
| **delta** | | | | |

### Chunk size sweep

| chunk (s) | TTFA p50 | TTFA p95 | E2E p95 | Prosody continuity (chủ quan) |
|---|---|---|---|---|
| 1.5 | | | | |
| 2.0 | | | | |
| 3.0 | | | | |
| 4.0 | | | | |
| 15.0 (default) | | | | |

### Load test

| concurrency | req/s | audio-sec/s | E2E p50 | p95 | p99 | GPU util % | VRAM |
|---|---|---|---|---|---|---|---|
| 1 | | | | | | | |
| 2 | | | | | | | |
| 4 | | | | | | | |
| 8 | | | | | | | |
| 16 | | | | | | | |

---

## B6 — Cross-model comparison (sau)

| Model | VI CER | VI WER | UTMOS | SIM | TTFA | RTF | VRAM/RAM |
|---|---|---|---|---|---|---|---|
| OmniVoice ns=32 | | | | | | | |
| OmniVoice ns=16 | | | | | | | |
| OmniVoice VI LoRA ns=16 | | | | | | | |
| Fish S2 Pro | | | | | | | |
| Supertonic 3 (CPU) | | | N/A | | | | |
| Piper VI (CPU) | | | N/A | | | | |

---

## Bài học rút ra

Ghi lại mỗi khi có kết quả bất ngờ — phần này về lâu dài giá trị hơn các bảng số.

| # | Bài học | Từ phase nào |
|---|---|---|
| 1 | | |
