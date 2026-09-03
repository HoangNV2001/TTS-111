# PLAN — TTS & Inference Optimization (hands-on)

> Chủ repo: `hoangnv242` · Bắt đầu: **2026-09-03**
> Nguồn scope: [docs/draft_TTS_deep_dive.md](docs/draft_TTS_deep_dive.md) · Quy tắc server: [docs/practices_with_slurm_on_HPC.md](docs/practices_with_slurm_on_HPC.md)
>
> **Cách dùng file này:** mỗi Phase có block lệnh copy-paste được. Bạn chạy, dán output lại, tôi phân tích và cùng bạn ghi vào [RESULTS.md](RESULTS.md) / [PROGRESS.md](PROGRESS.md).

---

## 0. Mục tiêu

Bạn đến từ nền DL/NLP, mục tiêu **không phải** làm ra một sản phẩm TTS, mà là:

| # | Mục tiêu | Đo bằng gì |
|---|---|---|
| G1 | Hiểu inference của một model TTS non-autoregressive ở mức tensor/kernel | tự giải thích được vì sao FlashInfer nhanh 2.x lần |
| G2 | Tự dựng được benchmark harness đáng tin (TTFA, RTF, p95) | script chạy lại được, số ổn định giữa 2 lần đo |
| G3 | Biết knob nào đổi latency bao nhiêu, đổi chất lượng bao nhiêu | bảng trade-off `num_step` × WER trong RESULTS.md |
| G4 | Có eval suite tiếng Việt của riêng mình | 300–500 câu + WER/CER/SIM/UTMOS pipeline |
| G5 | Chạm được vào serving thật (batching, cache, chunking) | một prototype server đo được p95 |

Model chủ lực: **OmniVoice** (`k2-fsa/OmniVoice`, Apache-2.0). Lý do chọn đã có trong draft; đã verify repo tồn tại, 9.6k sao, push gần nhất 2026-08-31, tiếng Việt 8481.98h trong pretraining.

**Không làm ngay:** TensorRT, FP8, custom Triton kernel, train from scratch. Chỉ đụng tới sau khi profiling chứng minh cần.

---

## 1. Hạ tầng

### 1.1. Hiện tại: Vast.ai instance (khảo sát 2026-09-03)

```text
host          182.224.239.168:52162 · root · container dd301f17c69f
OS            Ubuntu 24.04.4 · Python 3.12.3 (system) / 3.12.14 (/venv/main)
CPU/RAM       16 core · 62 GB
GPU           1× RTX 5070 Ti · 16 GB VRAM · driver 580.126.09 · compute_cap 12.0 (Blackwell sm_120)
CUDA          toolkit 12.8 có sẵn, nvcc OK → khớp wheel cu128
disk          overlay 32 GB (dùng 28 MB) ← RÀNG BUỘC CHẶT NHẤT
venv          /venv/main (image default) · uv có sẵn tại /usr/local/bin/uv
container     unprivileged Docker · KHÔNG systemd (PID 1 = bash) · KHÔNG cgroup ghi được
                KHÔNG kernel module / perf / eBPF / docker-in-docker
services      supervisor quản lý long-running process; caddy reverse-proxy
network       pypi ✅ github ✅ huggingface ✅ flashinfer.ai ✅
port forward  ssh -L 8080:localhost:8080 → dùng cho omnivoice-demo sau này
```

### ⚠️ Ba ràng buộc phải thuộc lòng

**(1) Disk chỉ 32 GB.** Đây là thứ sẽ cắn bạn trước tiên, không phải VRAM.

| Hạng mục | Ước tính |
|---|---|
| torch 2.8 cu128 + nvidia-* deps | ~7 GB |
| flashinfer-jit-cache cu128 | ~1–2 GB |
| OmniVoice weights | ~3–5 GB |
| Whisper (nếu auto-transcribe ref) | ~1.5 GB |
| Eval models (WavLM/UTMOS/ASR) — Phase 4 | ~5–8 GB |

Cộng lại đã sát trần. Quy tắc: `df -h /` sau mỗi lần cài lớn, `uv cache clean` khi xong, và **không giữ audio output của mọi sweep**.

**(2) Không có gì persistent.** `vast-capabilities | jq '.instance.workspace_is_volume'` trả về **`false`**.

```text
stop/start          → giữ nguyên toàn bộ filesystem  ✅
recycle / destroy   → XOÁ SẠCH, kể cả /workspace     ❌
```

→ Eval set tiếng Việt (Phase 4) là asset không thể mất. Phải `git push` hoặc đẩy lên HF Hub, **không** để nằm một chỗ trên instance.

**(3) GPU là RTX 5070 Ti, không phải H100.** Hai hệ quả:

- **16 GB VRAM** (H100: 80 GB) → sweep `batch_size=8` có thể OOM. Tăng dần và theo dõi.
- **compute capability 12.0 (Blackwell)** → torch cu128 hỗ trợ, nhưng **FlashInfer có kernel cho sm_120 hay không là câu hỏi mở**. Đây là rủi ro lớn nhất của cả plan vì Phase 2 phụ thuộc hoàn toàn vào nó. Vì vậy ta **probe nó ngay ở Phase 0.8**, trước khi đầu tư công vào benchmark harness.
- Số RTF tuyệt đối sẽ **chậm hơn H100 nhiều lần** (băng thông bộ nhớ và FLOPS thấp hơn hẳn). Không sao — cái ta so là **tỉ lệ speedup**, không phải số tuyệt đối. Bảng upstream chỉ là mốc tham chiếu.

### 1.2. Đã bỏ: HPC cluster nội bộ (10.254.152.71)

Giữ lại làm ghi chú, có thể quay lại sau.

```text
login-0 + worker-0..9 · mỗi node 8× H100 80GB · partition main · TIMELIMIT=infinite
```

Lý do bỏ: **80/80 GPU bị allocate liên tục**, partition không có time limit nên GPU chỉ trống khi người khác tự tắt job. Job đầu tiên (15728) chờ vô thời hạn, chạy được 2 phút 38 giây trên worker-7 rồi bị huỷ. Không phải môi trường để học hands-on theo nhịp riêng.

Bài học mang sang: **`TIMELIMIT=infinite` là một lỗi thiết kế cluster.** Trên Vast ta sẽ tự cấu hình `MaxTime=12:00:00` để không lặp lại.

Nếu quay lại: mọi thứ từ Phase 1 trở đi dùng lại được nguyên vẹn, chỉ Phase 0 phải đổi.

---

## 2. Nguyên tắc làm việc

```text
Một máy duy nhất  → vừa là login node vừa là worker (khác HPC, không còn golden rule tách đôi)
Slurm             → vẫn dùng, để practice: sbatch/squeue/--time/--gres
Mọi thứ nặng      → /workspace/tts/
Thứ không được mất → git push hoặc HF Hub (instance KHÔNG persistent)
```

### 2.1. 🔒 Quy tắc cứng: MỌI job phải có `--time`

**Không có ngoại lệ.** Mọi `srun` và mọi `sbatch` trong repo này đều mang `--time`.

Lý do gốc: trên HPC cũ, partition `main` khai báo `TIMELIMIT=infinite` nên Slurm **không bao giờ** tự thu hồi tài nguyên của job không đặt `--time`. Với `srun --pty bash`, job chỉ chết khi cái shell chết — không phải khi lệnh benchmark chạy xong. Một shell bị quên trong tmux giữ GPU vô thời hạn. Đó là lời giải thích cho tình trạng 80/80 GPU bận và job chạy 14 ngày trên cluster đó.

Trên Vast, lý do còn mạnh hơn: **instance tính tiền theo giờ, chạy idle vẫn mất tiền.** `--time` giờ vừa là kỷ luật Slurm vừa là kỷ luật ví tiền.

Slurm ta tự dựng ở Phase 0 sẽ đặt `MaxTime=12:00:00` và `DefaultTime=01:00:00` — cố ý **không** để `INFINITE`, để không lặp lại lỗi của cluster cũ.

`--time` là cái phanh tay: quên `exit` thì Slurm tự thu hồi, mất tối đa vài tiếng GPU thay vì vài tuần.

Giá trị mặc định dùng trong repo này:

| Loại job | `--time` | Ghi chú |
|---|---|---|
| Dev / debug interactive (`--pty`) | `08:00:00` | một ngày làm việc |
| Setup CPU-only (`--pty`) | `04:00:00` | cài đặt, download weights |
| Benchmark sweep (`sbatch`) | `04:00:00` | đặt ~2× thời gian ước tính |
| LoRA fine-tune (`sbatch`) | `24:00:00` | chốt lại sau khi biết throughput thật |

> ⚠️ Job bị Slurm giết vì hết `--time` **không được cứu**. Với training, luôn có checkpoint định kỳ trước khi đặt `--time` sát.

### 2.2. Khi nào `srun`, khi nào `sbatch`

| | `srun --pty` | `sbatch` |
|---|---|---|
| Dùng cho | mò mẫm, debug, đọc lỗi ngay | sweep/train đã biết chắc chạy gì |
| Kết thúc khi | shell `exit`/chết/hết `--time` | **script chạy xong** → tự trả tài nguyên |
| Giữ tài nguyên khi rảnh | ❌ có, rất phí | ✅ không |
| Cần tmux sống | có | không |

→ Phase 0–1 chủ yếu `srun`. Phase 2 trở đi, các sweep dài nên chuyển sang `sbatch`.

### 2.3. Reflex khi rời máy

```bash
exit                                   # trong worker shell — trả tài nguyên ngay
squeue -u hoangnv242 -o "%.8i %.20j %.2t %.10M %.12l %.18b %.20R"   # kiểm tra còn giữ gì
scancel <jobid>                        # dọn job không dùng nữa
```

Môi trường trong `/workspace/tts/` không mất đi đâu — xin worker lại là dùng tiếp được ngay.

Layout thư mục sẽ dựng:

```text
/workspace/tts/
├── envs/omnivoice/          # venv
├── huggingface/             # HF_HOME (model cache, ~10-20GB)
├── projects/OmniVoice/      # clone upstream để đọc code
├── projects/TTS-111/        # repo này, clone lên server
├── datasets/                # ref audio, eval set VI
├── outputs/                 # audio sinh ra
└── bench/                   # log benchmark, jsonl kết quả
```

---

## 3. Roadmap tổng thể

```text
Phase 0  Environment          ─ 1 buổi   ─ venv + OmniVoice chạy được 1 câu
Phase 1  Baseline             ─ 1 buổi   ─ RTF PyTorch fp16, hiểu output CLI
Phase 2  FlashInfer + Graph   ─ 1-2 buổi ─ reproduce bảng 2.1x / 2.4x / 2.6x
Phase 3  Knob sweep           ─ 1-2 buổi ─ num_step × batch × duration matrix
Phase 4  VI eval suite        ─ 2-3 buổi ─ 300-500 câu + WER/SIM/UTMOS
Phase 5  Serving prototype    ─ 3-5 buổi ─ TTFA, length-aware batching, voice cache
Phase 6  LoRA VI (optional)   ─ sau      ─ chỉ khi Phase 4 cho baseline rõ ràng
Phase 7  On-device (optional) ─ sau      ─ Supertonic 3 / Piper trên CPU
```

Ta đi tuần tự. **Không nhảy cóc sang Phase 5 khi Phase 2 chưa có số ổn định** — không có baseline thì mọi optimization đều là cảm tính.

---

# Phase 0 — Vast instance + Slurm single-node

**Done khi:** `sbatch` một job xin `--gres=gpu:1 --time=...` và job đó sinh ra được file wav.

Ta cố ý **không** chạy trực tiếp `python infer.py`. Mọi thứ đi qua Slurm, kể cả khi chỉ có một máy — vì mục tiêu là practice workflow.

---

## 0.1. Kết nối

```bash
ssh -p 52162 root@182.224.239.168 -L 8080:localhost:8080
tmux new -s tts
```

> `-L 8080:localhost:8080` để dành cho `omnivoice-demo` sau này: chạy trên instance ở port 8080, mở `http://localhost:8080` trên máy bạn.

Định vị máy:

```bash
hostname
nvidia-smi
df -h /
vast-capabilities | jq '.instance.workspace_is_volume'   # kỳ vọng: false
```

## 0.2. Layout + biến môi trường

```bash
mkdir -p /workspace/tts/{projects,datasets,outputs,bench/logs,slurm}

cat > /workspace/tts/env.sh <<'EOF'
export TTS_ROOT=/workspace/tts
export HF_HOME=/workspace/.hf_home
export UV_CACHE_DIR=/workspace/.uvcache
source /venv/main/bin/activate
EOF

echo 'source /workspace/tts/env.sh' >> ~/.bashrc
source /workspace/tts/env.sh
python -V
```

> Dùng thẳng `/venv/main` của image chứ **không** tạo venv riêng. Lý do: đĩa chỉ 32 GB, một bản torch nữa là ~7 GB đi tong.

---

## 0.3. Dựng Slurm single-node

Ý tưởng: `slurmctld` (scheduler) + `slurmd` (worker daemon) + `munge` (auth) chạy trên cùng một container. Bạn submit job vào chính máy mình đang ngồi.

### 0.3.1. Cài gói

```bash
apt-get update
apt-get install -y munge slurm-wlm
sinfo --version   # kỳ vọng: slurm-wlm 23.11.x
```

### 0.3.2. Munge — lớp authentication

Slurm không nói chuyện được nếu munge chưa chạy. Munge xác thực bằng một khoá bí mật chia sẻ:

```bash
dd if=/dev/urandom bs=1 count=1024 > /etc/munge/munge.key 2>/dev/null
chown munge:munge /etc/munge/munge.key
chmod 400 /etc/munge/munge.key
mkdir -p /run/munge /var/log/munge /var/lib/munge
chown -R munge:munge /etc/munge /run/munge /var/log/munge /var/lib/munge

sudo -u munge /usr/sbin/munged
munge -n | unmunge | head -6      # STATUS: Success (0) là đạt
```

> Không có systemd (PID 1 của container là `bash`), nên `systemctl start munge` sẽ thất bại. Ta chạy daemon trực tiếp.

### 0.3.3. `slurm.conf`

```bash
cat > /etc/slurm/slurm.conf <<'EOF'
ClusterName=vastbox
SlurmctldHost=localhost(127.0.0.1)
SlurmUser=root
SlurmdUser=root

AuthType=auth/munge
CredType=cred/munge

StateSaveLocation=/var/spool/slurmctld
SlurmdSpoolDir=/var/spool/slurmd
SlurmctldPidFile=/run/slurmctld.pid
SlurmdPidFile=/run/slurmd.pid
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log

# Container unprivileged: KHÔNG ghi được cgroup → không dùng plugin cgroup
ProctrackType=proctrack/linuxproc
TaskPlugin=task/none

SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_CPU_Memory
GresTypes=gpu

# sacct cần slurmdbd + MySQL (quá nặng cho box này).
# jobcomp/filetxt cho ta lịch sử job ở dạng file, đủ để practice.
JobCompType=jobcomp/filetxt
JobCompLoc=/var/log/slurm/jobcomp.log
AccountingStorageType=accounting_storage/none
JobAcctGatherType=jobacct_gather/none

ReturnToService=2
SlurmctldTimeout=120
SlurmdTimeout=300
MinJobAge=300
KillWait=30
MaxJobCount=1000

NodeName=localhost NodeAddr=127.0.0.1 CPUs=16 RealMemory=56000 Gres=gpu:rtx5070ti:1 State=UNKNOWN
PartitionName=main Nodes=localhost Default=YES State=UP DefaultTime=01:00:00 MaxTime=12:00:00
EOF
```

Hai dòng cuối là chỗ đáng đọc kỹ:

| Thiết lập | Ý nghĩa |
|---|---|
| `Gres=gpu:rtx5070ti:1` | khai báo có đúng 1 GPU → submit 2 job cùng xin GPU là job thứ hai phải xếp hàng |
| `RealMemory=56000` | 56 GB (chừa headroom từ 62 GB) |
| `DefaultTime=01:00:00` | quên `--time` thì mặc định 1 tiếng, **không phải vô hạn** |
| `MaxTime=12:00:00` | trần cứng — cố ý sửa lỗi thiết kế của cluster cũ |

### 0.3.4. `gres.conf`

```bash
cat > /etc/slurm/gres.conf <<'EOF'
NodeName=localhost Name=gpu Type=rtx5070ti Count=1
EOF
```

> Không khai `File=/dev/nvidiaN` vì container chỉ thấy 1 GPU và ánh xạ device node không chắc chắn. Với 1 GPU, `Count=1` là đủ và an toàn hơn.

### 0.3.5. Thư mục + khởi động

```bash
mkdir -p /var/spool/slurmctld /var/spool/slurmd /var/log/slurm

/usr/sbin/slurmctld
sleep 2
/usr/sbin/slurmd -N localhost
sleep 2

sinfo
```

Kỳ vọng:
```text
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
main*        up   12:00:00      1   idle localhost
```

Nếu `STATE` là `down` hoặc `drain`:
```bash
tail -30 /var/log/slurm/slurmctld.log
tail -30 /var/log/slurm/slurmd.log
scontrol update NodeName=localhost State=RESUME    # sau khi đã sửa nguyên nhân
```

📥 Dán log cho tôi nếu vướng — bước này hay lỗi vặt về quyền/thư mục.

### 0.3.6. Cho daemon sống qua disconnect (supervisor)

Image này dùng **supervisor** để quản long-running process. Đăng ký Slurm vào đó thì daemon tự chạy lại sau `stop/start` instance:

```bash
cat > /etc/supervisor/conf.d/slurm.conf <<'EOF'
[program:munged]
command=/usr/sbin/munged -F
user=munge
autostart=true
autorestart=true
priority=10

[program:slurmctld]
command=/usr/sbin/slurmctld -D
autostart=true
autorestart=true
priority=20

[program:slurmd]
command=/usr/sbin/slurmd -D -N localhost
autostart=true
autorestart=true
priority=30
EOF

pkill slurmctld; pkill slurmd; pkill munged; sleep 2
supervisorctl reread && supervisorctl update
supervisorctl status | grep -E "munged|slurm"
sinfo
```

> Đừng `supervisorctl stop caddy / instance_portal / tunnel_manager` — đó là lớp quản lý và auth của Vast.

---

## 0.4. Practice Slurm — bốn bài, làm hết

Đây là phần chính của Phase 0. Bốn bài này tái hiện đúng những gì bạn vấp trên HPC thật.

### Bài 1 — `srun` chạy một lệnh rồi trả tài nguyên ngay

```bash
srun --time=00:02:00 --gres=gpu:1 nvidia-smi -L
squeue          # rỗng ngay sau khi lệnh xong
```

→ Đây là dạng `srun [tài nguyên] [lệnh]`: chạy xong là trả. Khác hẳn `--pty bash`.

### Bài 2 — `srun --pty bash` giữ tài nguyên tới khi shell chết

```bash
srun --time=00:10:00 --gres=gpu:1 --pty bash
# prompt đổi → bạn đang trong "job"
echo $CUDA_VISIBLE_DEVICES
squeue                       # thấy job của chính mình, ST=R
exit                         # ← đây mới là thứ trả GPU
squeue                       # rỗng
```

→ Xác nhận bằng tay điều đã bàn: **lệnh chạy xong ≠ job xong**.

### Bài 3 — Queueing thật với 1 GPU

```bash
for i in 1 2 3; do
  sbatch --job-name=q$i --gres=gpu:1 --time=00:03:00 \
    --output=$TTS_ROOT/bench/logs/q$i.out \
    --wrap="hostname; date; sleep 60; date"
done

squeue -o "%.8i %.10j %.2t %.10M %.12l %.18b %.20R"
```

Kỳ vọng: 1 job `R`, 2 job `PD` với `REASON=(Resources)`. Đợi rồi xem chúng chạy lần lượt.

→ Đây chính là cơ chế đã bắt bạn chờ trên HPC, nhưng lần này bạn thấy toàn cảnh trong 3 phút.

### Bài 4 — `--time` bị enforce thật

```bash
sbatch --job-name=timeout-test --time=00:01:00 \
  --output=$TTS_ROOT/bench/logs/timeout.out \
  --wrap="echo start; sleep 300; echo 'DÒNG NÀY SẼ KHÔNG BAO GIỜ IN'"

watch -n 10 'squeue -o "%.8i %.14j %.2t %.10M %.12l"'
```

Sau ~1 phút job bị giết. Kiểm chứng:

```bash
cat $TTS_ROOT/bench/logs/timeout.out
grep timeout-test /var/log/slurm/jobcomp.log     # JobState=TIMEOUT
```

→ Thấy tận mắt: **job hết `--time` bị giết giữa chừng, không cứu được.** Đó là lý do Phase 6 (fine-tune) bắt buộc phải checkpoint định kỳ.

### Lệnh tra cứu hằng ngày

```bash
sinfo                                              # node/partition
squeue -o "%.8i %.14j %.2t %.10M %.12l %.18b %.20R"  # job đang chạy/chờ
scontrol show job <jobid>                          # chi tiết một job
scontrol show node localhost                       # tài nguyên còn lại
scancel <jobid>                                    # huỷ
cat /var/log/slurm/jobcomp.log                     # lịch sử (thay cho sacct)
```

⚠️ **`sacct` sẽ KHÔNG chạy** trên setup này (`accounting_storage/none` — muốn có phải dựng slurmdbd + MySQL, không đáng cho box 32 GB). Dùng `jobcomp.log` thay thế.

### Slurm ở đây khác cluster thật chỗ nào

| Practice được | Không practice được |
|---|---|
| `sbatch` / `srun` / `squeue` / `scancel` / `scontrol` | Multi-node (`--nodes=2`, NCCL cross-node) |
| `--time` enforce thật, job bị `TIMEOUT` thật | Fair-share, QOS, nhiều user cạnh tranh |
| `--gres=gpu:1` và queueing khi hết GPU | Backfill ở quy mô lớn |
| Viết sbatch script đúng chuẩn, `%x_%j` log | Tách login node / worker node |
| `jobcomp.log` để tra lịch sử | `sacct` đầy đủ (cần slurmdbd) |
| | `--mem` enforce (không cgroup → chỉ là bookkeeping) |

---

## 0.5. Python env + PyTorch

```bash
source /workspace/tts/env.sh
uv pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128

df -h /        # kiểm tra đĩa NGAY sau bước này
```

Verify qua Slurm (chứ không chạy thẳng — practice):

```bash
srun --time=00:05:00 --gres=gpu:1 python -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('available', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0))
print('capability', torch.cuda.get_device_capability(0))
print('arch_list', torch.cuda.get_arch_list())
"
```

📌 **Nhìn kỹ hai dòng cuối.** `capability` phải là `(12, 0)`, và `arch_list` phải có `sm_120`. Nếu **không** có `sm_120`, torch sẽ không chạy được trên GPU này — báo tôi ngay, ta đổi sang nightly cu128/cu129.

## 0.6. Cài OmniVoice

```bash
cd $TTS_ROOT/projects
git clone https://github.com/k2-fsa/OmniVoice.git
cd OmniVoice
git log -1 --oneline          # ghi hash vào PROGRESS.md
uv pip install -e .
df -h /
```

> Cài `-e .` để Phase 2–5 còn đọc và sửa `omnivoice/models/omnivoice_flashinfer.py`.

## 0.7. 🚨 Probe sm_120 + FlashInfer — LÀM NGAY, ĐỪNG HOÃN

Toàn bộ Phase 2 (trái tim của plan) dựa trên FlashInfer. FlashInfer lịch sử tối ưu cho sm_80/sm_90 (A100/H100); **có kernel cho sm_120 Blackwell hay không là câu hỏi chưa có lời đáp**. Biết sớm thì còn xoay, biết muộn thì phí công dựng harness.

```bash
uv pip install flashinfer-python==0.6.15.post1 "flashinfer-jit-cache==0.6.15.post1+cu128" \
  --extra-index-url https://flashinfer.ai/whl/cu128/

srun --time=00:10:00 --gres=gpu:1 python -c "
import torch, flashinfer
print('flashinfer', flashinfer.__version__)
print('cap', torch.cuda.get_device_capability(0))
import flashinfer.jit as J
print('jit ok')
"
df -h /
```

Ba kết cục có thể xảy ra:

| Kết cục | Nghĩa là | Ta làm gì |
|---|---|---|
| import OK, kernel chạy | 🎉 tốt nhất | Phase 2 giữ nguyên |
| import OK nhưng JIT compile lâu/lỗi lúc chạy thật | thiếu prebuilt kernel cho sm_120 | thử build JIT tại chỗ, chấp nhận warmup lâu |
| không hỗ trợ sm_120 | Phase 2 phải đổi trục | chuyển trọng tâm sang `torch.compile` + CUDA Graph thuần PyTorch + `num_step` — vẫn học được inference optimization, chỉ khác công cụ |

Kể cả kết cục xấu nhất, plan **không chết** — mục tiêu là học tối ưu inference, FlashInfer chỉ là một đường đi.

📥 **Gửi tôi output của bước này trước khi đi tiếp.**

## 0.8. Smoke test — qua `sbatch`

```bash
cat > $TTS_ROOT/slurm/p0_smoke.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=p0-smoke
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/workspace/tts/bench/logs/%x_%j.out

set -euo pipefail
source /workspace/tts/env.sh
mkdir -p $TTS_ROOT/outputs/phase0

nvidia-smi --query-gpu=name,memory.used --format=csv
date

omnivoice-infer \
  --model k2-fsa/OmniVoice \
  --text "Xin chào, đây là bài kiểm tra đầu tiên của hệ thống tổng hợp tiếng nói." \
  --output $TTS_ROOT/outputs/phase0/vi_auto.wav

date
ls -lh $TTS_ROOT/outputs/phase0/
df -h /
EOF

sbatch $TTS_ROOT/slurm/p0_smoke.sbatch
squeue
```

Lần đầu sẽ tải weights về `$HF_HOME` (vài GB). Theo dõi:

```bash
tail -f $TTS_ROOT/bench/logs/p0-smoke_*.out
```

> `--time=00:30:00` cho lần đầu vì phải download. Các lần sau hạ xuống `00:10:00`.

Nghe thử — chạy trên **máy local**:

```bash
scp -P 52162 root@182.224.239.168:/workspace/tts/outputs/phase0/vi_auto.wav .
```

## ✅ Checklist Phase 0

- [ ] SSH + tmux OK
- [ ] `sinfo` báo `localhost  idle`, partition `main` TIMELIMIT `12:00:00`
- [ ] 4 bài practice Slurm đã làm hết, đặc biệt bài 3 (queueing) và bài 4 (timeout)
- [ ] supervisor quản munged/slurmctld/slurmd
- [ ] `torch.cuda.get_arch_list()` có `sm_120`
- [ ] **kết quả probe FlashInfer đã ghi vào PROGRESS.md** ← quan trọng nhất
- [ ] OmniVoice commit hash đã ghi
- [ ] `sbatch` smoke test ra file wav, nghe được
- [ ] `df -h /` còn ít nhất 10 GB trống

📥 **Gửi tôi:** output `sinfo`, kết quả bài 3 + bài 4, output probe ở 0.7, và `df -h /`.

---

# Phase 1 — Baseline PyTorch fp16

**Done khi:** có số RTF baseline của chính bạn, và bạn giải thích được nó đo cái gì.

### 1.1. Chuẩn bị reference audio để voice cloning

Cần 1 file wav tiếng Việt sạch, 3–10 giây, kèm transcript chính xác. Ba cách:

```bash
mkdir -p $TTS_ROOT/datasets/ref
```

**Cách A — tự thu** (tốt nhất, giọng bạn, không vướng license): thu trên máy local, rồi
```bash
# chạy trên máy local
scp myvoice.wav hoangnv242@10.254.152.71:/workspace/tts/datasets/ref/
```

**Cách B — lấy 1 sample từ VieNeu-TTS-140h** (Apache-2.0 theo dataset card):
```bash
uv pip install datasets soundfile
python - <<'EOF'
from datasets import load_dataset
import soundfile as sf, os
ds = load_dataset("pnnbao-ump/VieNeu-TTS-140h", split="train", streaming=True)
os.makedirs("/workspace/tts/datasets/ref", exist_ok=True)
for i, x in enumerate(ds):
    if i >= 3: break
    print(i, {k: (type(v).__name__ if k=="audio" else v) for k,v in x.items()})
EOF
```
> In ra schema trước rồi mới ghi file — tên cột (`audio`/`text`/`transcription`) tuỳ dataset, đừng đoán.

**Cách C — dùng chính output auto-voice ở Phase 0** làm ref tạm để chạy pipeline thông trước, rồi thay bằng audio thật sau.

Chuẩn hoá format (OmniVoice sinh ở 24 kHz):
```bash
ffmpeg -i input.wav -ar 24000 -ac 1 -c:a pcm_s16le $TTS_ROOT/datasets/ref/vi_ref_01.wav
soxi $TTS_ROOT/datasets/ref/vi_ref_01.wav
```

### 1.2. Voice cloning một câu

```bash
omnivoice-infer \
  --model k2-fsa/OmniVoice \
  --text "Hôm nay tôi bắt đầu học về tối ưu suy luận cho mô hình tổng hợp tiếng nói." \
  --ref_audio $TTS_ROOT/datasets/ref/vi_ref_01.wav \
  --ref_text "<transcript chính xác của file ref>" \
  --output $TTS_ROOT/outputs/phase1/clone_01.wav
```

> Bỏ `--ref_text` thì nó tự gọi Whisper transcribe — tiện nhưng **thêm latency và thêm một nguồn sai số**. Khi benchmark luôn đưa `ref_text` tường minh.

### 1.3. Dựng test list cho batch inference

Đây là input chuẩn của `omnivoice-infer-batch`. Tạo 20 câu tiếng Việt trải đều độ dài:

```bash
mkdir -p $TTS_ROOT/bench
python - <<'EOF'
import json, os
REF = "/workspace/tts/datasets/ref/vi_ref_01.wav"
REF_TEXT = "<transcript chính xác của file ref>"
short = [
    "Chào bạn.", "Cảm ơn nhiều.", "Tôi không biết.", "Được thôi.", "Hẹn gặp lại.",
]
mid = [
    "Hôm nay trời khá đẹp, tôi định đi dạo một vòng quanh hồ.",
    "Mô hình này hỗ trợ hơn sáu trăm ngôn ngữ khác nhau.",
    "Bạn có thể gửi lại cho tôi báo cáo trước năm giờ chiều không?",
    "Chi phí dự kiến khoảng một triệu hai trăm năm mươi nghìn đồng.",
    "Chúng tôi sẽ triển khai mô hình lên máy chủ trong tuần này.",
]
long = [
    "Trong vài năm gần đây, các mô hình tổng hợp tiếng nói đã tiến bộ rất nhanh, "
    "đặc biệt là ở khả năng sao chép giọng nói chỉ từ một đoạn âm thanh ngắn.",
    "Việc tối ưu suy luận cho mô hình khuếch tán khác khá nhiều so với mô hình ngôn ngữ tự hồi quy, "
    "bởi vì toàn bộ chuỗi được cập nhật lại ở mỗi bước thay vì sinh lần lượt từng token.",
]
rows = []
for i, t in enumerate(short + mid + long):
    rows.append({"id": f"vi_{i:03d}", "text": t, "ref_audio": REF,
                 "ref_text": REF_TEXT, "language_id": "vi"})
out = "/workspace/tts/bench/vi_smoke.jsonl"
with open(out, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", out, len(rows), "rows")
EOF
```

### 1.4. Baseline run

```bash
cd $TTS_ROOT
mkdir -p outputs/phase1_base bench/logs

omnivoice-infer-batch \
  --model k2-fsa/OmniVoice \
  --test_list bench/vi_smoke.jsonl \
  --res_dir outputs/phase1_base \
  --num_step 32 \
  --batch_size 1 \
  --warmup 3 \
  --enable_flashinfer false \
  2>&1 | tee bench/logs/p1_baseline_bs1.log

grep -E "RTF|Average" bench/logs/p1_baseline_bs1.log | tail -20
```

> `--warmup 3` rất quan trọng. Không warmup thì lần chạy đầu dính JIT/cuDNN autotune và số RTF sẽ sai lệch mạnh.

### 1.5. Hiểu con số

`infer_batch.py` tính:
```text
rtf         = synthesis_time / audio_duration        (mỗi sample)
Average RTF = total_synthesis_time / total_audio_duration
```

Nghĩa là **RTF ở đây là throughput-oriented**, không phải latency người dùng cảm nhận.
- RTF 0.09 → sinh 10 s audio mất 0.9 s
- Nhưng **TTFA** (time to first audio) là chuyện khác hoàn toàn, Phase 5 mới đo được đúng.

Đây là điểm bẫy lớn nhất khi đọc benchmark TTS. Ghi nhớ nó.

### ✅ Checklist Phase 1

- [ ] có ref audio + transcript đúng
- [ ] voice cloning nghe ra giống ref
- [ ] `bench/vi_smoke.jsonl` chạy hết 12 sample không lỗi
- [ ] `Average RTF` baseline đã ghi vào [RESULTS.md](RESULTS.md) bảng B1

📥 **Gửi tôi:** dòng `Average RTF`, vài dòng per-sample RTF, và `nvidia-smi` lúc đang chạy (VRAM dùng bao nhiêu).

---

# Phase 2 — FlashInfer + CUDA Graph

**Done khi:** bạn reproduce được (hoặc giải thích được vì sao lệch) bảng speedup upstream.

Bảng chính thức của upstream (H100, fp16, `num_step=32`, seed-tts zh 2020 sample):

| batch | baseline | FlashInfer | speedup |
|---|---|---|---|
| 1 | 0.0899 | 0.0430 | 2.1× |
| 1 + CUDA graph | — | 0.0367 | 2.4× |
| 2 | 0.0480 | 0.0245 | 2.0× |
| 4 | 0.0331 | 0.0152 | 2.2× |
| 8 | 0.0298 | **0.0115** | **2.6×** |

### 2.1. FlashInfer đã cài ở Phase 0.7

Nếu chưa, quay lại **Phase 0.7** — bước probe sm_120 phải xong trước khi đầu tư vào phase này.

```bash
source /workspace/tts/env.sh
python -c "import flashinfer; print(flashinfer.__version__)"
```

⚠️ **Nếu probe 0.7 cho kết quả FlashInfer không hỗ trợ sm_120**, đừng chạy tiếp §2.2. Thay vào đó Phase 2 đổi trục sang:

```text
PyTorch fp16 baseline
        ↓
torch.compile (mode="reduce-overhead" → dùng CUDA Graph bên dưới)
        ↓
CUDA Graph thủ công qua torch.cuda.CUDAGraph
        ↓
num_step sweep
        ↓
đọc code CFG packing của upstream để hiểu ý tưởng, dù không chạy được kernel
```

Vẫn học đúng tư duy latency-oriented inference, chỉ đổi công cụ. Báo tôi để tôi soạn lại §2.2–2.3.

### 2.2. Sweep baseline vs FlashInfer

```bash
cd $TTS_ROOT
for FI in false true; do
  for BS in 1 2 4 8; do
    tag="p2_fi${FI}_bs${BS}"
    echo "=== $tag ==="
    omnivoice-infer-batch \
      --model k2-fsa/OmniVoice \
      --test_list bench/vi_smoke.jsonl \
      --res_dir outputs/$tag \
      --num_step 32 \
      --batch_size $BS \
      --warmup 3 \
      --enable_flashinfer $FI \
      2>&1 | tee bench/logs/$tag.log
  done
done

grep -H "Average RTF" bench/logs/p2_*.log
```

⏱ 8 lần chạy, mỗi lần vài phút → nên gói vào một `sbatch` thay vì ngồi canh (xem mẫu ở PLAN §2.6).

⚠️ **16 GB VRAM, không phải 80 GB.** `batch_size=8` có thể OOM. Chạy tăng dần 1 → 2 → 4 → 8, theo dõi bằng `nvidia-smi` ở window khác. Gặp `CUDA out of memory` thì dừng ở batch lớn nhất chạy được và ghi rõ trần đó vào RESULTS.md — bản thân cái trần cũng là một kết quả.

### 2.3. CUDA Graph — phải viết script riêng

⚠️ **CLI `omnivoice-infer-batch` không có flag `--enable_cuda_graph`.** Nó chỉ gọi `apply_flashinfer(model)` mặc định. Muốn đo cấu hình 2.4× (batch=1 + graph) phải dùng Python API:

```bash
cat > $TTS_ROOT/bench/bench_graph.py <<'EOF'
"""Đo RTF batch=1 cho 3 runtime: pytorch / flashinfer / flashinfer+cudagraph."""
import argparse, json, time, statistics
import torch, numpy as np
from omnivoice import OmniVoice

p = argparse.ArgumentParser()
p.add_argument("--runtime", choices=["pytorch", "flashinfer", "flashinfer_graph"], required=True)
p.add_argument("--test_list", required=True)
p.add_argument("--num_step", type=int, default=32)
p.add_argument("--warmup", type=int, default=3)
p.add_argument("--out_json", default=None)
a = p.parse_args()

rows = [json.loads(l) for l in open(a.test_list, encoding="utf-8")]

model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
if a.runtime != "pytorch":
    from omnivoice.models.omnivoice_flashinfer import apply_flashinfer
    apply_flashinfer(model, enable_cuda_graph=(a.runtime == "flashinfer_graph"))

def gen(r):
    kw = dict(text=r["text"], num_step=a.num_step)
    if r.get("ref_audio"):
        kw.update(ref_audio=r["ref_audio"], ref_text=r.get("ref_text"))
    return model.generate(**kw)

# warmup — bắt buộc, nhất là với cuda graph (lần đầu phải capture)
for i in range(a.warmup):
    gen(rows[i % len(rows)])
torch.cuda.synchronize()

per, tot_syn, tot_aud = [], 0.0, 0.0
for r in rows:
    torch.cuda.synchronize(); t0 = time.perf_counter()
    audio = gen(r)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    dur = len(audio[0]) / 24000
    per.append({"id": r["id"], "synth_s": dt, "audio_s": dur, "rtf": dt / dur})
    tot_syn += dt; tot_aud += dur

res = {
    "runtime": a.runtime, "num_step": a.num_step, "n": len(per),
    "average_rtf": tot_syn / tot_aud,
    "latency_p50": statistics.median([x["synth_s"] for x in per]),
    "latency_p95": sorted([x["synth_s"] for x in per])[int(0.95 * (len(per) - 1))],
    "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
    "per_sample": per,
}
print(json.dumps({k: v for k, v in res.items() if k != "per_sample"}, indent=2))
if a.out_json:
    json.dump(res, open(a.out_json, "w"), indent=2, ensure_ascii=False)
EOF
```

Chạy:

```bash
cd $TTS_ROOT
for RT in pytorch flashinfer flashinfer_graph; do
  python bench/bench_graph.py --runtime $RT \
    --test_list bench/vi_smoke.jsonl \
    --num_step 32 --warmup 3 \
    --out_json bench/logs/p2_graph_$RT.json \
    2>&1 | tee bench/logs/p2_graph_$RT.log
done

python - <<'EOF'
import json, glob
for f in sorted(glob.glob("/workspace/tts/bench/logs/p2_graph_*.json")):
    d = json.load(open(f))
    print(f"{d['runtime']:<20} RTF={d['average_rtf']:.4f}  p50={d['latency_p50']:.3f}s  "
          f"p95={d['latency_p95']:.3f}s  VRAM={d['peak_vram_gb']:.1f}GB")
EOF
```

> Script này chưa chắc chạy đúng ngay lần đầu — signature `apply_flashinfer` hoặc `model.generate` có thể khác. Đó là **chủ đích**: đọc traceback rồi mở `omnivoice/models/omnivoice_flashinfer.py` ra sửa chính là phần học được nhiều nhất của Phase 2. Cứ dán lỗi cho tôi.

### 2.6. Gói sweep vào `sbatch`

Từ Phase 2 trở đi, sweep dài nên chạy bằng `sbatch` — job tự trả tài nguyên khi script xong, không phụ thuộc tmux:

```bash
cat > $TTS_ROOT/slurm/p2_sweep.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=p2-sweep
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/workspace/tts/bench/logs/%x_%j.out

set -euo pipefail
source /workspace/tts/env.sh
cd $TTS_ROOT

for FI in false true; do
  for BS in 1 2 4 8; do
    tag="p2_fi${FI}_bs${BS}"
    echo "=== $tag ==="
    omnivoice-infer-batch \
      --model k2-fsa/OmniVoice \
      --test_list bench/vi_smoke.jsonl \
      --res_dir outputs/$tag \
      --num_step 32 --batch_size $BS --warmup 3 \
      --enable_flashinfer $FI || echo "FAILED: $tag"
  done
done
df -h /
EOF

sbatch $TTS_ROOT/slurm/p2_sweep.sbatch
squeue
```

> `|| echo "FAILED: $tag"` để một cấu hình OOM không giết cả sweep.

### 2.4. Đọc code — phần quan trọng nhất

Sau khi có số, đọc để hiểu **tại sao**:

```bash
cd $TTS_ROOT/projects/OmniVoice
wc -l omnivoice/models/omnivoice.py omnivoice/models/omnivoice_flashinfer.py
grep -n "def " omnivoice/models/omnivoice_flashinfer.py
grep -n -iE "cuda_graph|packing|ragged|rmsnorm|rope|append_paged|plan\(" omnivoice/models/omnivoice_flashinfer.py | head -40
```

Bốn câu hỏi cần trả lời được sau Phase 2:

1. **CFG sequence packing** — cond và uncond được ghép thế nào, tiết kiệm bao nhiêu so với pad?
2. **Ragged attention** — vì sao attention trên chuỗi packed không cần padding mask?
3. **Vì sao KV cache không giúp?** — bidirectional + toàn bộ sequence đổi mỗi step.
4. **CUDA graph chỉ ăn ở batch=1?** — kernel launch overhead vs compute-bound.

### 2.5. Profiling (nếu số lệch nhiều so với upstream)

```bash
nsys profile -o $TTS_ROOT/bench/logs/p2_nsys_fi \
  --trace=cuda,nvtx,osrt --force-overwrite true \
  python bench/bench_graph.py --runtime flashinfer \
    --test_list bench/vi_smoke.jsonl --num_step 32 --warmup 2
```
(nếu `nsys` không có sẵn, dùng `torch.profiler` — tôi sẽ đưa script khi tới bước đó)

### ✅ Checklist Phase 2

- [ ] flashinfer import được
- [ ] bảng 8 ô baseline×FlashInfer × bs{1,2,4,8} đã điền RESULTS.md B2
- [ ] cấu hình `flashinfer_graph` chạy được
- [ ] trả lời được 4 câu hỏi ở 2.4

📥 **Gửi tôi:** output `grep -H "Average RTF" bench/logs/p2_*.log` và bảng so sánh 3 runtime.

---

# Phase 3 — Knob sweep

**Done khi:** có bảng trade-off latency ↔ chất lượng, không chỉ latency.

### 3.1. `num_step` — knob mạnh nhất sau FlashInfer

```bash
cd $TTS_ROOT
for NS in 32 24 16 12 8 4; do
  omnivoice-infer-batch \
    --model k2-fsa/OmniVoice \
    --test_list bench/vi_smoke.jsonl \
    --res_dir outputs/p3_ns${NS} \
    --num_step $NS --batch_size 4 --warmup 3 \
    --enable_flashinfer true \
    2>&1 | tee bench/logs/p3_ns${NS}.log
done
grep -H "Average RTF" bench/logs/p3_ns*.log
```

⚠️ **Bắt buộc nghe.** RTF giảm tuyến tính theo `num_step` là chuyện hiển nhiên và vô nghĩa nếu chất lượng sập. Nghe ít nhất `ns32`, `ns16`, `ns8` cùng một câu:

```bash
# trên máy local
scp 'hoangnv242@10.254.152.71:/workspace/tts/outputs/p3_ns{32,16,8}/vi_010*.wav' .
```

Ghi cảm nhận chủ quan vào RESULTS.md trước — Phase 4 mới có WER để đối chiếu.

### 3.2. Các knob khác

```bash
# guidance_scale: 1.5 / 2.0 (default) / 3.0  → giống/khác ref, độ ổn định
# t_shift: 0.1 (default) / 0.3 / 0.5         → phân bổ effort giữa các step
# position_temperature 5.0 → 0.0             → deterministic, tốt cho reproducibility khi benchmark
```

Ví dụ chạy deterministic để đo lặp lại được:
```bash
omnivoice-infer-batch --model k2-fsa/OmniVoice \
  --test_list bench/vi_smoke.jsonl --res_dir outputs/p3_greedy \
  --num_step 32 --batch_size 4 --warmup 3 --enable_flashinfer true \
  --position_temperature 0.0 --class_temperature 0.0 \
  2>&1 | tee bench/logs/p3_greedy.log
```

> Đây là một điểm thực hành quan trọng: **benchmark chất lượng nên chạy greedy**, không thì mỗi lần chạy ra một kết quả và bạn không biết delta đến từ optimization hay từ sampling noise.

### 3.3. Ảnh hưởng độ dài & batching

Sinh test list phân tầng theo độ dài mục tiêu (1/3/5/10/15 s) bằng `duration`:

```bash
python - <<'EOF'
import json
REF = "/workspace/tts/datasets/ref/vi_ref_01.wav"
REF_TEXT = "<transcript>"
base = "Hệ thống tổng hợp tiếng nói đang được kiểm tra với nhiều độ dài khác nhau. "
rows = []
for d in [1, 3, 5, 10, 15]:
    for k in range(4):
        n = max(1, round(d / 3))
        rows.append({"id": f"dur{d:02d}_{k}", "text": (base * n).strip(),
                     "ref_audio": REF, "ref_text": REF_TEXT,
                     "language_id": "vi", "duration": float(d)})
with open("/workspace/tts/bench/vi_duration.jsonl", "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(rows))
EOF
```

Rồi so **fixed-size batching** vs **duration-based batching** — đây chính là chỗ draft nói team có thể tạo giá trị ngoài upstream:

```bash
# fixed batch size: padding waste khi trộn 1s với 15s
omnivoice-infer-batch --model k2-fsa/OmniVoice --test_list bench/vi_duration.jsonl \
  --res_dir outputs/p3_fixed8 --batch_size 8 --num_step 32 --warmup 3 \
  --enable_flashinfer true 2>&1 | tee bench/logs/p3_fixed8.log

# duration-based batching (batch_size=0 → dùng batch_duration)
omnivoice-infer-batch --model k2-fsa/OmniVoice --test_list bench/vi_duration.jsonl \
  --res_dir outputs/p3_durbatch --batch_size 0 --batch_duration 120 --num_step 32 --warmup 3 \
  --enable_flashinfer true 2>&1 | tee bench/logs/p3_durbatch.log
```

### ✅ Checklist Phase 3

- [ ] bảng `num_step` × RTF điền RESULTS.md B3
- [ ] đã nghe và ghi nhận xét chủ quan cho ns 32/16/8
- [ ] so sánh fixed vs duration batching
- [ ] xác định được cấu hình "production candidate" tạm thời

---

# Phase 4 — Vietnamese eval suite

**Done khi:** có 300–500 câu + pipeline tính WER/CER, SIM, UTMOS chạy một lệnh.

Đây là **asset dài hạn nhất** của toàn bộ project. Số RTF sẽ lỗi thời khi đổi GPU; eval set tiếng Việt thì không.

### 4.1. Cài eval extras

```bash
uv pip install "omnivoice[eval]"
```

### 4.2. Đọc pipeline eval upstream trước khi tự viết

```bash
cd $TTS_ROOT/projects/OmniVoice
cat docs/evaluation.md
sed -n '1,80p' examples/run_eval.sh
ls omnivoice/eval/wer/
```

Upstream đã có: WER (HuBERT/Whisper/Paraformer/Omnilingual-ASR), SIM (ECAPA-TDNN+WavLM), UTMOS. **Không viết lại** — chỉ cần cắm test set tiếng Việt vào.

### 4.3. Xây eval set

Cấu trúc mục tiêu, 300–500 câu, cố tình stress:

| Category | Số câu | Ví dụ |
|---|---|---|
| dấu thanh khó | 40 | hỏi/ngã, cặp gần âm |
| số | 40 | `12.345`, `3,14`, `0987654321` |
| ngày tháng | 25 | `29/08/2026`, `quý 3` |
| tiền tệ | 25 | `1.250.000 đồng` |
| đơn vị | 25 | `10 GB`, `2.4 GHz`, `5 km/h` |
| acronym | 30 | `VTV`, `AI`, `GPU`, `5G` |
| chèn tiếng Anh | 40 | "deploy model lên GPU H100" |
| tên riêng / địa danh | 40 | tên người, tỉnh thành |
| địa chỉ | 25 | phố/phường/quận |
| câu hỏi | 30 | prosody nghi vấn |
| cảm xúc | 30 | vui/buồn/ngạc nhiên |
| long-form | 20 | 30–60 giây |
| giọng vùng miền | 30 | Bắc/Trung/Nam |

File: `datasets/vi_eval/vi_eval_v1.jsonl`, mỗi dòng:
```json
{"id":"num_001","category":"number","text":"Tổng chi phí là 1.250.000 đồng.","expected_read":"một triệu hai trăm năm mươi nghìn đồng","language_id":"vi"}
```

> Trường `expected_read` không phải input của model — nó là **ground truth để chấm WER sau khi ASR**, vì ASR sẽ trả về chữ chứ không trả về `1.250.000`. Đây là chi tiết mà nhiều eval pipeline làm sai.

Bước này ta sẽ làm cùng nhau — tôi sinh draft, bạn (native speaker) review và sửa. **Không tự động hoá 100%**: eval set sai thì mọi kết luận sau đó sai theo.

### 4.4. Chạy eval

Lệnh cụ thể sẽ chốt sau khi đọc `run_eval.sh` ở 4.2. Khung dự kiến:

```bash
# 1. sinh audio
omnivoice-infer-batch --model k2-fsa/OmniVoice \
  --test_list datasets/vi_eval/vi_eval_v1.jsonl \
  --res_dir outputs/p4_eval_ns32 --num_step 32 --batch_size 8 \
  --enable_flashinfer true --position_temperature 0.0 --class_temperature 0.0

# 2. chấm WER/CER + SIM + UTMOS  (module chính xác chốt ở 4.2)
# 3. lặp lại cho ns16, ns8 → bảng chất lượng vs tốc độ
```

⚠️ ASR nào chấm tiếng Việt? Upstream có FLEURS/Omnilingual-ASR (102 ngôn ngữ, có VI). Nhưng cần kiểm tra WER của **chính ASR đó trên tiếng Việt** trước — nếu ASR yếu, số WER đo được phản ánh ASR chứ không phản ánh TTS. Cách kiểm tra: cho ASR chạy trên audio người thật (từ VieNeu-TTS) và xem WER nền là bao nhiêu.

### ✅ Checklist Phase 4

- [ ] eval set v1 ≥ 300 câu, đã review thủ công
- [ ] xác định được ASR chấm điểm + WER nền của nó
- [ ] bảng chất lượng `num_step` ∈ {32,16,8} × {WER, SIM, UTMOS}
- [ ] A/B listening test ít nhất 20 cặp

---

# Phase 5 — Serving prototype

**Done khi:** có server đo được TTFA p50/p95 dưới tải đồng thời.

Kiến trúc mục tiêu (từ draft):

```text
API frontend → Request Queue → Duration Estimator → Length-aware Scheduler
                                                          │
                                              1 GPU worker (1 model duy nhất)
                                            OmniVoice fp16 + FlashInfer + Graph
                                                          │
                                                   Codec decoder → PCM
```

Ba việc theo thứ tự ROI:

1. **Voice prompt cache** — `create_voice_clone_prompt()` + `prompt.save()`. Upstream đã có API sẵn:
   ```python
   prompt = model.create_voice_clone_prompt(ref_audio=..., ref_text=...)
   prompt.save("my_voice.pt")
   # request sau: model.generate(text=..., voice_clone_prompt=VoiceClonePrompt.load("my_voice.pt"))
   ```
   → đo delta latency giữa có cache và không cache. Đây là win rẻ nhất.

2. **TTFA thật** — với chunked generation (`audio_chunk_duration`), TTFA = thời gian tới chunk đầu tiên, không phải tổng thời gian. Sweep chunk 1.5 / 2 / 3 / 4 s và đo trade-off TTFA ↔ prosody continuity.

3. **Length-aware bucketing** — bucket 0–2 / 2–4 / 4–8 / 8–16 s, so với naive batching ở Phase 3.3.

Load test:
```bash
# concurrency 1/2/4/8/16, đo p50/p95/p99 e2e và TTFA
```
Script cụ thể ta viết khi tới đây — phụ thuộc kết luận Phase 3.

> **Quy tắc:** một GPU = một model worker. Không bao giờ 8 process Gunicorn mỗi process load một bản OmniVoice.

---

# Phase 6 — LoRA tiếng Việt (optional, sau Phase 4)

Chỉ bắt đầu khi Phase 4 cho baseline zero-shot rõ ràng. Nguyên tắc từ draft: **20 giờ sạch > 1000 giờ noisy** cho experiment đầu tiên.

```text
zero-shot VI benchmark  →  curate 10–50h sạch  →  LoRA  →  A/B vs base
                                                            │
                                             nếu gain rõ → 100h / full FT
```

Tài liệu upstream: `docs/lora_finetuning.md`, `docs/data_preparation.md`, `examples/run_finetune_lora.sh`, `examples/config/train_config_finetune_lora.json`.

Dataset ứng viên (kiểm license kỹ trước khi dùng):

| Dataset | Size | License | Ghi chú |
|---|---|---|---|
| VieNeu-TTS-140h | 140.7h / 193 spk | Apache-2.0 (theo card) | **kiểm tra đầu tiên** |
| viVoice | 1016.97h | CC-BY-NC-SA | YouTube-derived, ~1.8% transcript error |
| VieNeu-TTS-1000h | ~1000h | CC-BY-NC, gated | research |
| Dolly-Audio | ~1000h / 152 spk | CC-BY-NC-SA | tag `synthetic`, cần audit provenance |
| VLSP 2020 | ~5–6h | challenge data | single-speaker sạch |

⚠️ Trước khi dùng bất kỳ dataset nào cho mục đích ngoài học tập cá nhân: đọc license. CC-BY-NC ≠ dùng cho công việc công ty.

---

# Phase 7 — On-device (optional)

Track hoàn toàn khác, chạy CPU, không tranh GPU với ai:

```bash
srun --job-name=tts-cpu --gres=gpu:0 --cpus-per-task=8 --time=04:00:00 --pty bash
```

> Trên box này `--gres=gpu:0` không giải phóng máy nào cả (chỉ có 1 máy), nhưng nó **giữ GPU rảnh cho job khác trong hàng đợi** — vẫn là thói quen đúng.

- **Supertonic 3** (~99M, ONNX, có tiếng Việt) — nhưng repo đã thông báo archive từ 2026-07-23, chỉ dùng để benchmark, không xây roadmap phụ thuộc.
- **Piper** (`OHF-Voice/piper1-gpl`, GPL-3.0, active) — baseline tự train giọng tiếng Việt rồi export ONNX.

Đo: CPU RTF, RAM, model size, cold start, WER.

---

## 4. Rủi ro & cách xử lý

| Rủi ro | Xác suất | Xử lý |
|---|---|---|
| **FlashInfer không hỗ trợ sm_120 (Blackwell)** | **cao — chưa biết** | Probe ngay ở Phase 0.7. Nếu hỏng, Phase 2 đổi trục sang `torch.compile` + CUDA Graph thuần PyTorch. Plan không chết, chỉ đổi công cụ |
| **Hết đĩa (32 GB)** | **cao** | `df -h /` sau mỗi lần cài lớn; `uv cache clean`; xoá `outputs/` của sweep cũ; Phase 4 eval models có thể phải tải chọn lọc |
| **Mất sạch dữ liệu khi recycle/destroy** | trung bình | `workspace_is_volume=false`. Eval set + script phải `git push`; weights chấp nhận tải lại |
| OOM VRAM ở batch lớn (16 GB) | trung bình | tăng batch dần, ghi trần thực tế vào RESULTS.md như một kết quả |
| Số RTF lệch xa upstream | **chắc chắn** | 5070 Ti ≠ H100, test set khác, ngôn ngữ khác. So **tỉ lệ speedup**, không so số tuyệt đối |
| Chi phí Vast tính theo giờ | trung bình | `--time` trên mọi job; `stop` instance khi nghỉ dài; gộp sweep vào `sbatch` thay vì mò interactive |
| Slurm daemon chết sau restart | thấp | đã đăng ký supervisor ở 0.3.6; kiểm tra `supervisorctl status` |
| `sacct` không dùng được | chắc chắn | dùng `/var/log/slurm/jobcomp.log`; dựng slurmdbd không đáng cho box này |
| `--mem` không được enforce | chắc chắn | container không ghi được cgroup → `--mem` chỉ là bookkeeping. Đừng dựa vào nó để chống OOM RAM |
| ASR chấm WER tiếng Việt yếu | trung bình | đo WER nền trên audio người thật trước |
| Script `bench_graph.py` không chạy ngay | **cao** | có chủ đích — đọc code là mục tiêu học |

---

## 5. Từ điển metric

| Metric | Định nghĩa | Bẫy thường gặp |
|---|---|---|
| **RTF** | `synthesis_time / audio_duration` | throughput-oriented, KHÔNG phải latency người dùng |
| **TTFA** | thời gian tới mẫu audio đầu tiên | với NAR diffusion, chỉ có nghĩa khi chunked |
| **E2E p95** | latency toàn request, percentile 95 | quan trọng hơn average cho realtime |
| **WER/CER** | ASR transcript vs ground truth | phụ thuộc chất lượng ASR; VI thường dùng cả CER |
| **SIM-o** | cosine sim speaker embedding (ECAPA+WavLM) | chỉ có nghĩa ở chế độ voice cloning |
| **UTMOS** | MOS dự đoán bằng NN | proxy, không thay được listening test |

> **WER thấp không đồng nghĩa giọng hay.** Luôn có listening test bởi người bản ngữ.

---

## 6. Bước tiếp theo ngay bây giờ

1. Phase 0.1 → 0.3: kết nối, dựng Slurm, chạy `sinfo`.
2. Phase 0.4: làm hết 4 bài practice — nhất là bài 3 (queueing) và bài 4 (`--time` bị enforce).
3. Phase 0.5 → 0.7: cài torch, OmniVoice, rồi **probe FlashInfer trên sm_120**.
4. Gửi tôi output probe trước khi đi tiếp — nó quyết định Phase 2 đi hướng nào.
