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
| GPU | NVIDIA GeForce RTX 5080 · 16 GB · compute_cap 12.0 (Blackwell sm_120) |
| Driver / CUDA | 595.71.05 · toolkit 12.8 + nvcc 12.9 (`CUDA_HOME=/usr/local/cuda-12.9`) |
| torch | 2.8.0+cu129 |
| omnivoice | 0.2.1 (editable, `projects/OmniVoice`) |
| flashinfer | 0.6.18.post1 + jit-cache cu129 |
| Máy | Vast.ai container, 64 core, 251 GB RAM, Slurm 23.02.8 single-node |
| Test set | `bench/vi_smoke.jsonl` — 12 câu VI, voice clone, tổng ~48.7 s audio |
| Reference | `cand2_female` · spk `jellyfish1010_1082` · 5.13 s · VieNeu-TTS-140h |
| Sampling | **greedy** (`position_temperature=0.0`, `class_temperature=0.0`) |
| warmup | 3 |
| Ngày đo | 2026-09-06 |

> ⚠️ Lần đo baseline đầu tiên (RTF 0.2185, tổng audio 101.32 s) **đã bỏ** — `vi_smoke.jsonl` lúc đó còn placeholder `<dán từ candidates.json>` ở `ref_text`. Mọi số dưới đây đo lại với transcript đúng.

---

## B0 — Smoke test (Phase 0)

| Kiểm tra | Kết quả |
|---|---|
| `torch.cuda.is_available()` | ✅ True · `capability (12,0)` · `arch_list` có `sm_120` |
| Sinh 1 câu VI auto-voice | ✅ `outputs/phase0/vi_auto.wav`, 200 KB ≈ 4.2 s, sinh trong ~4.8 s |
| FlashInfer trên sm_120 | ✅ sau khi cài nvcc 12.9 — xem PLAN §0.7 |
| Slurm single-node | ✅ 23.02.8 build từ source; queueing + `--time` enforce đã verify |
| Nghe thử — phát âm tiếng Việt | *(chưa điền — nhận xét chủ quan của bạn)* |

---

## B1 — Baseline PyTorch fp16 (Phase 1)

Cấu hình: `num_step=32`, `enable_flashinfer=false`, `warmup=3`, test set `vi_smoke.jsonl` (12 câu).

| batch_size | Average RTF | Tổng synth (s) | peak VRAM (MiB) |
|---|---|---|---|
| 1 | **0.4648** | 22.64 | 2818 |
| 2 | 0.2398 | 11.68 | 2884 |
| 4 | **0.1411** ← tốt nhất | 6.88 | 3214 |
| 8 | 0.1554 ⚠️ *kém hơn bs=4* | 7.57 | 3954 |

Tổng audio 48.7 s cho cả 4 cấu hình (cùng test set, greedy nên deterministic).

⚠️ **bs=8 tệ hơn bs=4** — không phải nhiễu đo. Test set chỉ có 12 câu, nên bs=8 chia thành 2 batch **8 + 4**: batch thứ hai chỉ lấp nửa GPU, và padding trong batch 8 câu độ dài lệch nhau cũng phí. Đây là minh hoạ sớm cho vấn đề length-aware batching ở §3.3 — nhưng muốn kết luận chắc thì cần test set lớn hơn (≥ 64 câu).

Voice cloning:

| Kiểm tra | Kết quả |
|---|---|
| Ref audio dùng | `cand2_female.wav` → `vi_ref_01.wav` · spk `jellyfish1010_1082` · nữ · 5.13 s · SNR~73 dB |
| REF_TEXT | `26% trong số những người được Trung tâm Kennedy vinh danh.` |
| Nguồn | VieNeu-TTS-140h shard 0 (Apache-2.0) |
| Giống ref? (chủ quan 1–5) | *(chưa điền)* |
| Có giữ được accent VI? | *(chưa điền)* |

---

## B2 — FlashInfer speedup (Phase 2)

**Mục tiêu:** reproduce tỉ lệ speedup của upstream, không phải số tuyệt đối.

Đo **hai lần độc lập** (run 1 = job 34, run 2 = người dùng chạy lại cùng script).

| batch | baseline (r1 / r2) | FlashInfer (r1 / r2) | **speedup (r1 / r2)** | upstream H100 |
|---|---|---|---|---|
| 1 | 0.4648 / 0.4423 | 0.2404 / 0.2400 | **1.93× / 1.84×** | 2.1× |
| 1 + CUDA graph | 0.4477¹ / 0.4402¹ | 0.1839¹ / 0.1808¹ | **2.43× / 2.43×** | 2.4× |
| 2 | 0.2398 / 0.2294 | 0.1345 / 0.1369 | **1.78× / 1.68×** | 2.0× |
| 4 | 0.1411 / 0.1362 | 0.0836 / 0.0758 | **1.69× / 1.80×** | 2.2× |
| 8 | 0.1554 / 0.1529 | 0.0658 / 0.0638 | **2.36× / 2.40×** | 2.6× |

**Độ lặp lại:** chênh lệch run-to-run 0.2–9.3%, phần lớn dưới 5%. Cấu hình lệch nhiều nhất là FlashInfer bs=4 (0.0836 vs 0.0758, 9.3%). → **Đừng kết luận từ chênh lệch dưới ~10% giữa hai cấu hình** nếu chỉ đo một lần.

¹ đo bằng `bench/bench_graph.py` (Python API), các dòng khác đo bằng `omnivoice-infer-batch`.

**Kết luận: reproduce được xu hướng của upstream.** Tỉ lệ speedup 1.69–2.43× so với 2.0–2.6× trên H100 — cùng bậc, lệch xuống một chút, hợp lý vì 5080 có băng thông/FLOPS thấp hơn nên phần overhead mà FlashInfer cắt bỏ chiếm tỉ trọng khác.

Đáng chú ý: **cấu hình batch=1 + CUDA graph khớp gần như chính xác** (2.43× đo vs 2.4× upstream). Đây là cấu hình dành cho low-latency serving.

VRAM FlashInfer gần như phẳng ở 3520 MiB cho bs 1→4 (workspace buffer cấp phát trước), chỉ nhích lên 3552 ở bs=8. Baseline thì tăng dần 2818 → 3954. **Không cấu hình nào chạm trần 16 GB** — còn nhiều chỗ để tăng batch.

*(Upstream đo trên seed-tts zh, 2020 sample / 3.3h audio, single H100 fp16 num_step=32)*

Ba runtime, batch=1, từ `bench/bench_graph.py`:

| runtime | RTF (r1 / r2) | p50 (r1 / r2) | p95 (r1 / r2) | VRAM (GB) | speedup |
|---|---|---|---|---|---|
| pytorch | 0.4477 / 0.4402 | 1.821 / 1.781 | 1.868 / 1.828 | 2.24 | 1.00× |
| flashinfer | 0.2427 / 0.2369 | 0.978 / 0.963 | 1.008 / 0.994 | 2.97 | 1.84× |
| flashinfer + cuda graph | **0.1839 / 0.1808** | **0.858 / 0.880** | 1.055 / 1.000 | 4.03 | **2.43×** |

📌 **Graph giúp câu NGẮN nhiều nhất.** Ở run 2, `flashinfer_graph` có mean `8.814/12 = 0.735 s` nhưng p50 `0.880 s` — **mean thấp hơn median**, phân phối lệch trái. Với `flashinfer` thường thì mean `0.962` ≈ p50 `0.963`, phân phối cân.

Giải thích: graph cắt kernel-launch overhead, một chi phí gần như cố định mỗi kernel. Câu càng ngắn → compute mỗi kernel càng nhỏ → tỉ lệ overhead càng cao → graph càng ăn. Đây cũng là câu trả lời cho "vì sao graph chỉ ăn ở batch=1" (xem 4 câu hỏi bên dưới).

⚠️ **Đính chính (2026-09-06).** Sau run 1 tôi ghi rằng CUDA graph có "nghịch lý p95" — p50 tốt nhất nhưng p95 xấu hơn flashinfer thường. Run 2 **không ủng hộ kết luận đó**:

```text
              p95 run1   p95 run2
flashinfer      1.008      0.994
+ cuda graph    1.055      1.000     ← khoảng cách thu từ 4.6% xuống 0.55%
```

0.55% nằm trong nhiễu đo. **Chưa đủ bằng chứng để nói graph làm xấu p95.** Với n=12 và một lần đo, đuôi phân phối không đáng tin. Muốn kết luận cần test set ≥ 100 câu và ≥ 3 lần lặp — để Phase 5.

Cross-check: `bench_graph.py` (flashinfer, bs=1) cho RTF 0.2427, CLI cho 0.2404 — lệch 1%. Hai đường đo độc lập khớp nhau, tin được.

### Bốn câu hỏi phải trả lời được

**1. CFG sequence packing.** Baseline pad `uncond` lên bằng `cond` rồi chạy batch=2 với mask `(2,1,S,S)`. Điểm mấu chốt: hai document dài rất khác nhau — `cond` = ref audio + text + target (`c_lens`), `uncond` = **chỉ target** (`u_lens`, xem dòng 297-302: `inp["input_ids"][0, :, -u_len:]`). Với ref 5 s và câu ngắn, `c_len` có thể gấp 2-3 lần `u_len`. Packing xếp `[cond_0, uncond_0, cond_1, uncond_1, ...]` vào **một row duy nhất** dài đúng tổng thật. Tiết kiệm: bỏ token pad, và vì attention là **O(S²)** nên cắt độ dài một nửa là cắt compute bốn lần; cộng thêm bỏ được ma trận mask.

**2. Ragged attention không cần mask** vì ranh giới document nằm trong **`indptr`** chứ không nằm trong mask. `PackedAttnRunner.plan()` dựng `indptr = cumsum(doc_lens)` rồi truyền vào `wrapper.plan(indptr, indptr, ..., causal=False)`. Kernel đọc offset để biết mỗi doc bắt đầu/kết thúc ở đâu và **chỉ chạy attention trong phạm vi đó**. Khác biệt: mask thì *tính hết S×S rồi cộng -inf vào ô không hợp lệ* (tốn compute rồi vứt); indptr thì *không sinh ra compute thừa*. Nên ragged thắng cả bộ nhớ lẫn FLOPS.

**3. KV cache không giúp** vì `causal=False`. Ở LLM tự hồi quy, token i chỉ thấy token ≤ i nên K/V quá khứ không bao giờ đổi → cache được. Ở đây attention **hai chiều**: mỗi bước unmasking thay một số `[MASK]` bằng token thật, và thay đổi đó làm K/V của **mọi** vị trí khác đổi theo. Cache lại thứ sẽ bị vứt ngay bước sau là vô nghĩa. Chi phí một request ≈ `num_step (32) × full-sequence forward × (cond + uncond)` — đó là lý do `num_step` là knob mạnh nhất, nó nhân trực tiếp vào tổng compute.

**4. CUDA graph chỉ ăn ở batch=1** vì hai lý do. *(a) Bản chất:* graph loại bỏ kernel-launch overhead, chi phí gần như **cố định mỗi kernel**. Số lần launch không đổi theo batch nhưng compute mỗi kernel tăng tuyến tính → tỉ lệ overhead/compute lớn ở bs=1, nhỏ ở bs=8. Số đo khớp: 2.43× ở bs=1 vs 1.80× (FlashInfer thường) ở bs=4. Bằng chứng phụ: ở run 2, graph có mean (0.735 s) **thấp hơn** p50 (0.880 s) — nó giúp câu ngắn nhiều nhất, đúng logic overhead. *(b) Từ code:* `_get_or_capture_graph()` cache graph theo khoá `doc_lens_key` — **một graph một shape**. Ở bs=1 shape là `(c_len, u_len)`, ít tổ hợp. Ở bs=8 shape là tuple 16 phần tử, gần như mỗi batch một shape mới → cache miss → capture lại, đắt hơn phần tiết kiệm. Upstream có `_get_or_capture_bucket_graph()` nhét item vào slot cố định để tái dùng graph, nhưng đó là **đánh đổi ngược**: chấp nhận padding để dùng lại graph — đúng thứ packing ở câu 1 vừa loại bỏ. Câu dài quá bucket thì `use_graph = False`, fallback eager.

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
| 1 | **Thông báo lỗi có thể trỏ sai hoàn toàn.** FlashInfer báo `requires GPUs with sm75 or higher` trên GPU sm120 — thực chất là `TARGET_CUDA_ARCHS` rỗng vì nvcc < 12.9. Đọc source nhanh hơn đoán. | 0.7 |
| 2 | **`torch.version.cuda` ≠ CUDA toolkit.** Thư viện nào cần *biên dịch* kernel thì đọc `nvcc`, không đọc torch. | 0.7 |
| 3 | **Placeholder trong file cấu hình là bug thầm lặng.** `ref_text` còn `<dán từ ...>` vẫn chạy trót lọt, cho ra RTF trông hợp lý nhưng vô nghĩa. Luôn assert giá trị thật trước khi đo. | 1.3 |
| 4 | **Batch lớn hơn không luôn nhanh hơn.** bs=8 tệ hơn bs=4 ở baseline vì test set 12 câu chia thành 8+4, batch cuối lấp nửa GPU. Kích thước test set ảnh hưởng kết luận benchmark. | 1.4 |
| 5 | **Một lần đo không đủ để kết luận về đuôi phân phối.** Từ run 1 tôi kết luận CUDA graph làm xấu p95; run 2 cho thấy khoảng cách chỉ 0.55%, nằm trong nhiễu. Với n=12, p95 thực chất là mẫu thứ 11 — cực kỳ nhạy nhiễu. | 2.3 |
| 7 | **Đo lặp lại trước khi tin delta.** Chênh lệch run-to-run tới 9.3% (FlashInfer bs=4). Mọi so sánh dưới ~10% cần ít nhất 3 lần lặp. | 2.2 |
| 6 | **Container không có cgroup ghi được thì Slurm 23.11 không launch nổi job.** Phải lùi về 23.02. Tooling HPC ngầm giả định systemd + cgroup. | 0.3 |
