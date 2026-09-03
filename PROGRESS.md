# PROGRESS

> Nhật ký tiến độ. Cập nhật mỗi buổi làm việc. Xem [PLAN.md](PLAN.md) cho chi tiết từng bước, [RESULTS.md](RESULTS.md) cho số liệu.
>
> Quy ước trạng thái: `[ ]` chưa làm · `[~]` đang làm · `[x]` xong · `[!]` bị chặn

---

## Trạng thái tổng quan

| Phase | Nội dung | Trạng thái | Ngày xong |
|---|---|---|---|
| 0 | Environment setup | `[ ]` | |
| 1 | Baseline PyTorch fp16 | `[ ]` | |
| 2 | FlashInfer + CUDA Graph | `[ ]` | |
| 3 | Knob sweep | `[ ]` | |
| 4 | Vietnamese eval suite | `[ ]` | |
| 5 | Serving prototype | `[ ]` | |
| 6 | LoRA VI (optional) | `[ ]` | |
| 7 | On-device (optional) | `[ ]` | |

---

## Môi trường đã chốt

Điền dần khi biết. Những ô `?` là thứ chỉ xác định được khi vào worker node.

**Hạ tầng hiện tại: Vast.ai** (đã bỏ HPC nội bộ — xem nhật ký 2026-09-03 chiều)

| Mục | Giá trị |
|---|---|
| Host | `182.224.239.168:52162` · root · container `dd301f17c69f` |
| OS | Ubuntu 24.04.4 |
| CPU / RAM | 16 core · 62 GB |
| GPU | 1× RTX 5070 Ti · 16 GB VRAM · **compute_cap 12.0 (sm_120 Blackwell)** |
| Driver | 580.126.09 |
| CUDA toolkit | 12.8 (nvcc có sẵn) → wheel `cu128` |
| Disk | overlay **32 GB** ← ràng buộc chặt nhất |
| Persistent? | **KHÔNG** (`workspace_is_volume=false`) — recycle/destroy là mất sạch |
| Python | `/venv/main` 3.12.14 (dùng thẳng, không tạo venv riêng vì tiết kiệm đĩa) |
| TTS_ROOT | `/workspace/tts` |
| HF_HOME | `/workspace/.hf_home` |
| Slurm | tự dựng single-node · `MaxTime=12:00:00` `DefaultTime=01:00:00` |
| torch | `?` (dự kiến 2.8.0+cu128, cần `sm_120` trong `arch_list`) |
| omnivoice | `?` (pyproject upstream ghi 0.2.1) |
| OmniVoice commit | `?` |
| flashinfer | `?` — **sm_120 support chưa rõ, probe ở Phase 0.7** |

---

## Nhật ký

### 2026-09-03 — Khảo sát & lập plan

**Đã làm**
- [x] Đọc `docs/draft_TTS_deep_dive.md` và `docs/practices_with_slurm_on_HPC.md`
- [x] Verify SSH tới `10.254.152.71` — key auth hoạt động, vào được `login-0`
- [x] Khảo sát cluster: 10 worker × 8 H100 80GB, partition `main`, TIMELIMIT infinite
- [x] Khảo sát filesystem: home 448G free (78% đầy), `/mnt/data` 12T free (89% đầy), user thuộc group `speech` nên ghi được `/mnt/data`
- [x] Khảo sát tooling: có pip/git/git-lfs/tmux/enroot/ffmpeg/sox; **không** có conda/uv/docker/module
- [x] Kiểm tra network egress: pypi ✅ github ✅ huggingface ✅ · astral.sh ❌ (timeout)
- [x] Verify các repo trong draft đều tồn tại thật (OmniVoice 9.6k★ Apache-2.0 push 2026-08-31, fish-speech, supertonic, piper1-gpl, Qwen3-TTS, breeze-tts)
- [x] Đọc README + `infer_batch.py` argparse + `docs/generation-parameters.md` + `docs/evaluation.md` của OmniVoice để viết lệnh chính xác
- [x] Soạn `PLAN.md`, `PROGRESS.md`, `RESULTS.md`

**Phát hiện đáng chú ý**
1. **80/80 GPU đang bị allocate** tại thời điểm khảo sát → `srun --gres=gpu:1` sẽ phải xếp hàng. Đừng ghim `--nodelist`.
2. **Có đồng nghiệp đang làm đúng lĩnh vực này**: `anhntc2` chạy job `omnivoice-triton` (worker-4, 6 ngày) và `train_TTS` (worker-7, 4 GPU, 7 ngày); `baovd5` chạy `cpu-omni-data`. → nên hỏi trước, có thể đã có weights cache trong `/mnt/data`.
3. **CLI `omnivoice-infer-batch` không có flag CUDA Graph** — chỉ có `--enable_flashinfer`. Muốn đo cấu hình 2.4× (batch=1 + graph) phải viết script Python gọi `apply_flashinfer(model, enable_cuda_graph=True)`. Đã đưa script `bench/bench_graph.py` vào PLAN mục 2.3.
4. **`Average RTF` của upstream là throughput, không phải TTFA** — bảng benchmark 0.0367 không có nghĩa TTFA 367 ms. Ghi rõ trong PLAN §5.
5. Home volume đã 78% đầy → mọi thứ nặng bắt buộc nằm ở `/mnt/data/hoangnv242/`.

**Chưa làm / đang chờ**
- [ ] Chưa chạm vào worker node (chờ bạn chạy `srun` trực tiếp)
- [ ] Chưa biết NVIDIA driver version → chưa chốt được CUDA wheel

**Bước tiếp theo**
- Bạn chạy PLAN Phase 0 mục 0.1 → 0.4
- Gửi lại: `hostname`, `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv`

---

### 2026-09-03 (chiều) — Job GPU đầu tiên bị kẹt vô thời hạn

**Đã làm**
- [x] `srun --job-name=tts-dev-hoangnv242 --gres=gpu:1 --cpus-per-task=16 --mem=128G --pty bash` → job **15728**

**Kết quả**
```text
JobState=PENDING   Reason=Resources   Priority=1
StartTime=2026-10-14T17:52:06
```
- Là job pending **duy nhất** trên cluster — không ai xếp hàng trước.
- `StartTime` xa hơn một tháng **không phải dự báo thật**: partition `main` có `TIMELIMIT=infinite`, mọi job đang chạy không khai báo thời hạn nên Slurm không tính được khi nào GPU được trả.

**Vấn đề gặp phải**
- ⚠️ Đánh giá lại rủi ro R1: không phải "xếp hàng một lúc" mà là **chờ vô thời hạn cho tới khi có người tự tay tắt job**. Trong `squeue` có job đã chạy 14 ngày.
- Nhầm lẫn về cơ chế `srun --pty bash`: đây là lệnh **chặn**, chính terminal đó biến thành shell trên worker khi được cấp; không có bước ssh sang worker, không biết trước tên worker (vì cố ý bỏ `--nodelist`). Xác nhận bằng `hostname` sau khi prompt quay lại.

**Bước tiếp theo**
- [~] Giữ job 15728 queued trong tmux window riêng
- [ ] Song song: xin worker **CPU-only** (`--gres=gpu:0`, vào được ngay) để làm Phase 0.6–0.7 — venv, `pip install`, download weights về `$HF_HOME`. Việc này không cần GPU.
- [ ] Liên hệ `anhntc2` (giữ `omnivoice-triton` worker-4 1 GPU + `train_TTS` worker-7 4 GPU) — hỏi nhường 1 GPU và hỏi vị trí weights cache trong `/mnt/data`

---

### 2026-09-03 (tối) — Chuyển sang Vast.ai, dựng Slurm single-node

**Đã làm**
- [x] Bỏ HPC nội bộ. Xác nhận đã release sạch: 0 job, `sacct` ghi 15728 CANCELLED (chạy 2:38 trên worker-7), `/mnt/data/hoangnv242` 0 byte
- [x] Chuyển sang Vast.ai instance. SSH ban đầu fail (key trên máy local là `...KmoA` còn Vast giữ `...e7l9`) → người dùng thêm key máy local vào Vast console (cách A) → OK
- [x] Khảo sát instance đầy đủ
- [x] Đọc `/etc/vast-agents-guide.md` — container unprivileged, supervisor quản service, storage không persistent
- [x] Viết lại PLAN §1 (hạ tầng), §2 (nguyên tắc), **toàn bộ Phase 0**, patch Phase 2/3/7 + bảng rủi ro

**Phát hiện đáng chú ý**
1. **GPU là RTX 5070 Ti, compute_cap 12.0 (Blackwell sm_120)** — không phải H100. FlashInfer lịch sử tối ưu sm_80/sm_90; **có kernel sm_120 hay không là rủi ro số 1** vì Phase 2 dựa hoàn toàn vào nó. → Đưa probe lên Phase 0.7, làm trước khi dựng harness.
2. **Đĩa chỉ 32 GB.** torch+deps ~7GB, flashinfer-jit-cache ~2GB, weights ~3-5GB, eval models Phase 4 ~5-8GB → sát trần. Đây sẽ là thứ cắn trước cả VRAM.
3. **Không có gì persistent** (`workspace_is_volume=false`). stop/start giữ được, recycle/destroy mất sạch. → Eval set tiếng Việt bắt buộc phải `git push`.
4. **Container unprivileged**: không systemd (PID 1 = bash), không ghi được cgroup, không perf/eBPF. → Slurm phải dùng `ProctrackType=proctrack/linuxproc` + `TaskPlugin=task/none`; `nsys` profiling có thể bị hạn chế.
5. **Slurm single-node practice được nhiều hơn dự kiến**: queueing thật (1 GPU + 3 job → 2 job PD), `--time` enforce thật (job bị TIMEOUT), gres bookkeeping. Không practice được multi-node, fair-share, `sacct` (cần slurmdbd), `--mem` enforce.
6. Cố ý đặt `MaxTime=12:00:00` + `DefaultTime=01:00:00` trong `slurm.conf` — **sửa đúng lỗi thiết kế** đã làm cluster cũ tắc.

**Bước tiếp theo**
- [ ] Người dùng chạy Phase 0.1 → 0.3 (dựng Slurm), gửi `sinfo`
- [ ] Phase 0.4: 4 bài practice Slurm
- [ ] Phase 0.7: **probe FlashInfer trên sm_120** — kết quả quyết định Phase 2 đi hướng nào

---

### `<ngày>` — `<tiêu đề buổi làm việc>`

**Đã làm**
-

**Kết quả**
-

**Vấn đề gặp phải**
-

**Bước tiếp theo**
-

---

## Câu hỏi mở

| # | Câu hỏi | Trạng thái |
|---|---|---|
| Q1 | ~~`anhntc2` cache OmniVoice ở đâu~~ | đóng — đã rời HPC |
| Q2 | Driver 580.126.09 + CUDA 12.8 → wheel `cu128` OK; nhưng torch có `sm_120` trong `arch_list` không? | mở, Phase 0.5 |
| Q3 | **FlashInfer có kernel cho sm_120 (Blackwell) không?** | **mở — rủi ro số 1, probe ở Phase 0.7** |
| Q4 | ASR nào chấm WER tiếng Việt đáng tin? WER nền của nó trên audio người thật là bao nhiêu? | mở |
| Q5 | Repo này có push lên remote không, hay chỉ local + rsync lên server? | mở |
| Q6 | ~~Bao giờ có GPU trống trên HPC~~ | đóng — đã rời HPC |
| Q7 | 32 GB đĩa có đủ tới hết Phase 4 (eval models) không? | mở |

---

## Ghi chú vận hành

```bash
# vào instance
ssh -p 52162 root@182.224.239.168 -L 8080:localhost:8080
tmux a -t tts || tmux new -s tts
source /workspace/tts/env.sh

# Slurm
sinfo
squeue -o "%.8i %.14j %.2t %.10M %.12l %.18b %.20R"
scontrol show job <jobid>
scancel <jobid>
cat /var/log/slurm/jobcomp.log        # thay cho sacct (không có slurmdbd)
supervisorctl status | grep -E "munged|slurm"

# chạy job
srun --time=00:10:00 --gres=gpu:1 --pty bash      # interactive
sbatch /workspace/tts/slurm/<script>.sbatch       # batch

# kiểm tra đĩa — LÀM THƯỜNG XUYÊN, chỉ có 32 GB
df -h /
du -sh /workspace/.hf_home /workspace/tts/outputs
```

🔒 **Mọi `srun`/`sbatch` phải có `--time`.** Trên Vast lý do gấp đôi: kỷ luật scheduler + instance tính tiền theo giờ. Xem PLAN §2.1.

⚠️ **Instance KHÔNG persistent.** `recycle`/`destroy` là mất sạch, kể cả `/workspace`. Thứ không thể mất → `git push`.
