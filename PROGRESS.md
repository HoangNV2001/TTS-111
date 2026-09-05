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
| Host | `14.227.95.149:33308` · root · container `fd52e2b5d9ec` |
| OS | Ubuntu 24.04.4 |
| CPU / RAM | 64 core (Threadripper PRO 5975WX) · 251 GB |
| GPU | 1× RTX 5080 · 16 GB VRAM · **compute_cap 12.0 (sm_120 Blackwell)** |
| Driver | 595.71.05 |
| CUDA toolkit | 12.8 mặc định + **cuda-nvcc-12-9 cài thêm** · `CUDA_HOME=/usr/local/cuda-12.9` |
| Disk | overlay **32 GB** ← ràng buộc chặt nhất |
| Persistent? | **KHÔNG** (`workspace_is_volume=false`) — recycle/destroy là mất sạch |
| Python | `/venv/main` 3.12.14 (dùng thẳng, không tạo venv riêng vì tiết kiệm đĩa) |
| TTS_ROOT | `/workspace/tts` |
| HF_HOME | `/workspace/.hf_home` |
| Slurm | **23.02.8 build từ source** tại `/opt/slurm` · `cgroup/v1` + cây giả `/var/lib/slurmcg` · `MaxTime=12:00:00` `DefaultTime=01:00:00` · đã verify job GPU + queueing + TIMEOUT |
| torch | **2.8.0+cu129** (đổi từ cu128 — xem Q3) · `capability (12,0)` · `arch_list` có `sm_120` |
| omnivoice | `?` (pyproject upstream ghi 0.2.1) |
| OmniVoice commit | `?` |
| flashinfer | **0.6.18.post1 + jit-cache cu129** ✅ chạy được trên sm_120 (cần nvcc 12.9) |

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

### 2026-09-05 — Instance mới; Slurm 23.11 KHÔNG chạy job được trong container

**Hạ tầng mới:** `14.227.95.149:33308` (container `fd52e2b5d9ec`) — RTX 5080 16GB **vẫn sm_120**, 64 core, 251 GB RAM, disk vẫn 32 GB, vẫn `workspace_is_volume=false`.

**Đã làm** — chẩn đoán `slurmctld`/`slurmd` FATAL trong supervisor, test trên config sandbox ở `/tmp` (không đụng config thật), đã dọn sạch sau khi test.

**Bốn lỗi tìm được, ba cái sửa được:**

| # | Lỗi | Sửa |
|---|---|---|
| 1 | `CLUSTER NAME MISMATCH` — state file ghi `localcluster` (từ config mặc định của gói) nhưng conf ghi `vastbox` | `rm /var/spool/slurmctld/clustername` |
| 2 | `Node configuration differs from hardware: CPUs=16:64(hw)` — conf copy từ instance cũ | `CPUs=64 Sockets=1 CoresPerSocket=32 ThreadsPerCore=2 RealMemory=240000` |
| 3 | `Ignoring file-less GPU` → node `INVAL`, `gres/gpu count reported lower than configured (0 < 1)` | **`File=/dev/nvidia0` là bắt buộc** trong `gres.conf` (quyết định bỏ `File=` trước đó của tôi sai) |
| 4 | **cgroup — không sửa được** | xem dưới |

**Lỗi 4 — chặn cứng.** `/sys/fs/cgroup` mount **read-only**, không có systemd/dbus. Slurm 23.11 khởi tạo cgroup **vô điều kiện** dù đã đặt `ProctrackType=proctrack/linuxproc` + `TaskPlugin=task/none`. Đã thử theo thứ tự:

1. `IgnoreSystemd=yes` → hết lỗi dbus, nhưng chết ở `Could not create scope directory /sys/fs/cgroup/...` (read-only)
2. `CgroupMountPoint=/run/slurmcg` + dựng giả cây cgroup v2 (`cgroup.controllers`, `cgroup.procs`, `system.slice/...`) → **`slurmd version 23.11.4 started`** ✅
3. Nhưng khi launch job: `slurmstepd` fatal — `Could not move slurmstepd pid to a Slurm's delegated cgroup. Should be in /run/slurmcg/, we are in /run/slurmcg/system.slice/...scope/system`

Bước 3 kiểm tra cgroup **thật** của process qua `/proc/self/cgroup` (`0::/`) và so với cây giả → không giả được. `CgroupPlugin=disabled` **không tồn tại** trong 23.11 (man 5 cgroup.conf chỉ có `cgroup/v1|cgroup/v2|autodetect`).

→ **Kết luận: Slurm 23.11 (bản duy nhất trong apt của Ubuntu 24.04) chạy được daemon nhưng KHÔNG launch được job trong container unprivileged này.**

**Giải pháp — ĐÃ VERIFY end-to-end trên máy:** build **Slurm 23.02.8 từ source** vào `/opt/slurm` (~4 phút, 123 MB).

23.02 vẫn khởi tạo cgroup nhưng **không có bước kiểm tra "delegated cgroup"** của slurmstepd (đó là tính năng mới của 23.11). Chỉ cần trỏ `CgroupPlugin=cgroup/v1` + `CgroupMountpoint=/var/lib/slurmcg` vào một cây thư mục giả là qua.

Kết quả test thật:

| Test | Kết quả |
|---|---|
| `slurmd`/`slurmctld` 23.02.8 khởi động | ✅ node `idle` |
| `srun --gres=gpu:1 nvidia-smi -L` | ✅ `CUDA_VISIBLE_DEVICES=0`, thấy RTX 5080 |
| Queueing 3 job / 1 GPU | ✅ `q1 R`, `q2 PD (Resources)`, `q3 PD (Priority)` |
| `--time=00:01:00` với `sleep 300` | ✅ bị giết, `JobState=TIMEOUT`, ctld log `Time limit exhausted for JobId=5` |

Lưu ý phát sinh: **phải gỡ gói apt `slurm-wlm`/`slurm-client`** — client 23.11 nói chuyện với daemon 23.02 sẽ lỗi protocol version. Giữ lại `munge`.

Công thức đầy đủ đã ghi vào PLAN §0.3 (đã verify từng bước). Sandbox test đã dọn sạch, config thật của người dùng chưa bị đụng.

**Bước tiếp theo** — người dùng chạy PLAN §0.3.2 → §0.3.9 (build đã xong sẵn ở /opt/slurm), rồi §0.4 (4 bài practice), rồi §0.7 (probe FlashInfer sm_120).

---

### 2026-09-06 — Gỡ được rào cản FlashInfer trên Blackwell sm_120

**Kết quả: FlashInfer chạy được.** Rủi ro số 1 của plan đã đóng.

Mất 4 vòng chẩn đoán vì thông báo lỗi trỏ sai chỗ:

| Vòng | Triệu chứng | Phát hiện |
|---|---|---|
| 1 | `RuntimeError: FlashInfer requires GPUs with sm75 or higher` (trên GPU sm120!) | `jit/core.py:96-108` duyệt `TARGET_CUDA_ARCHS`, tập **rỗng** thì ném lỗi này |
| 2 | `TARGET_CUDA_ARCHS = set()` + `SM 12.x requires CUDA >= 12.9` | torch đang là cu128 → đổi sang **torch 2.8.0+cu129** |
| 3 | Vẫn lỗi dù `torch.version.cuda = 12.9` | `jit/cpp_ext.py:68-80` đọc **`$CUDA_HOME/bin/nvcc --version`**, không đọc torch. nvcc hệ thống = 12.8 |
| 4 | Ép `FLASHINFER_CUDA_ARCH_LIST=12.0f` → `nvcc fatal: Unsupported gpu architecture 'compute_120f'` | suffix `f` là tính năng CUDA 12.9 → **bắt buộc nvcc ≥ 12.9** |

**Fix:** `cuda-nvcc-12-9` + `cuda-cudart-dev-12-9` từ apt repo NVIDIA (~300 MB, không cần cả toolkit) + `CUDA_HOME=/usr/local/cuda-12.9`. Không đổi symlink `/usr/local/cuda`.

**Kết quả probe (RTX 5080):**
```text
first call (plan+run): 0.17s
flashinfer ragged : 0.201 ms/call
SDPA (x2 seq)     : 0.239 ms/call   → 1.19x
max|diff| vs SDPA : 0.000244140625
rmsnorm / silu_and_mul: OK
```

**Ba bài học ghi lại:**
1. Thông báo lỗi có thể trỏ sai hoàn toàn — "requires sm75" thực chất là tập arch rỗng. Đọc source nhanh hơn đoán.
2. `torch.version.cuda` ≠ CUDA toolkit. Thư viện cần *biên dịch* kernel thì đọc `nvcc`.
3. `nvidia-cuda-nvcc-cu12` trên PyPI **chỉ có `ptxas`**, không có `nvcc`.
4. Probe sai API cũng cho kết luận sai: `single_prefill_with_kv_cache` không nằm trong jit-cache (0/906 file) nên luôn rơi xuống JIT. OmniVoice dùng `BatchPrefillWithRaggedKVCacheWrapper` + `norm.rmsnorm` + `rope.apply_rope_pos_ids_inplace` + `activation.silu_and_mul`.

⚠️ **1.19× là một op đơn lẻ, KHÔNG phải 2.1-2.6× của upstream.** Con số upstream đến từ cả pipeline (CFG packing bỏ padding + fused kernels + CUDA graph). Phase 2 phải đo end-to-end.

**Bước tiếp theo:** §0.8 smoke test OmniVoice qua sbatch → Phase 1.

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
| Q2 | ~~torch có `sm_120` không?~~ | **đóng — CÓ** (`['sm_70','sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']`, verify 2026-09-06) |
| Q3 | FlashInfer có kernel cho sm_120 không? | **ĐÓNG — CÓ, đã verify 2026-09-06.** Cần torch cu129 **và** nvcc ≥ 12.9 (`cuda-nvcc-12-9`, ~300MB). Ragged attention 0.201 ms/call vs SDPA 0.239 (1.19×), numerics khớp. Công thức ở PLAN §0.7.1 |
| Q4 | ASR nào chấm WER tiếng Việt đáng tin? WER nền của nó trên audio người thật là bao nhiêu? | mở |
| Q5 | Repo này có push lên remote không, hay chỉ local + rsync lên server? | mở |
| Q6 | ~~Bao giờ có GPU trống trên HPC~~ | đóng — đã rời HPC |
| Q7 | 32 GB đĩa có đủ tới hết Phase 4 (eval models) không? | mở |
| Q8 | ~~Slurm 23.02 có tránh được ràng buộc cgroup không?~~ | **đóng — CÓ, đã verify end-to-end 2026-09-05** |

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
