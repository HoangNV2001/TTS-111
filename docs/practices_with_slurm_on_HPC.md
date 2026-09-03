# 🚀 Practices with Slurm on H100 / A100

> **Making training feel like a local machine — using Slurm on the server**

Chào mừng bạn! 👋
Đây là **hướng dẫn sinh tồn trong 2 phút** để bạn có thể làm việc hiệu quả trên server mà không vô tình làm treo login node 😄

---

## 📚 Table of Contents

1. [SSH into the server](#1-ssh-into-the-server)
2. [Check GPU availability](#2-check-gpu-availability)
3. [Open a tmux session](#3-open-a-tmux-session)
4. [Request a GPU worker](#4-request-a-gpu-worker)
5. [Nested tmux = productivity mode](#5-nested-tmux--productivity-mode)
6. [Put big stuff in `/mnt/data`](#6-put-big-stuff-in-mntdata)
7. [No GPU? Still request Slurm](#7-no-gpu-still-request-slurm)
8. [Golden Rule](#-golden-rule)

---

# 1. SSH into the server

Đầu tiên, SSH vào server như bình thường.

Sau khi đăng nhập, bạn sẽ dừng chân tại **login node**.

> ⚠️ **Quan trọng**
>
> Login node **KHÔNG dành cho các tác vụ nặng**, ví dụ:
>
> * Training model
> * Data preprocessing lớn
> * Chạy script dài / tốn CPU
> * Load model lớn
> * Benchmark
> * Các tác vụ sử dụng nhiều RAM / CPU / GPU
>
> Login node chủ yếu dùng để:
>
> * SSH vào hệ thống
> * Kiểm tra trạng thái cluster
> * Submit / request Slurm job
> * Chỉnh sửa file
> * Chạy các command nhẹ

---

# 2. Check GPU availability

Để kiểm tra GPU, worker hoặc job nào đang được sử dụng:

```bash
squeue -o "%.8i %.9P %.20j %.8u %.2t %.10M %.6D %R %b"
```

Nếu muốn **sắp xếp theo worker node** và hiển thị thêm:

* CPU
* Memory
* GPU
* Node

thì dùng:

```bash
squeue \
  -o '%.8i %.9P %.20j %.8u %.2t %.10M %.4D %.6C %.10m %.18b %R' \
  -S +N
```

Bạn nên kiểm tra trạng thái cluster **trước khi request resource**, đặc biệt khi cần H100/A100.

---

# 3. Open a tmux session

Trước khi làm bất cứ việc gì quan trọng, hãy mở một `tmux` session:

```bash
tmux new -s train_llm
```

## Tại sao nên dùng tmux?

Nếu bạn đang chạy job mà:

* SSH bị mất kết nối → ✅ Job vẫn sống
* Laptop hết pin → ✅ Job vẫn sống
* Wi-Fi disconnect → ✅ Job vẫn sống
* Internet quyết định nghỉ làm → ✅ Job vẫn sống

Để quay lại session sau đó:

```bash
tmux a -t train_llm
```

Để xem danh sách các tmux session:

```bash
tmux ls
```

> 💡 **Khuyến nghị**
>
> Luôn mở `tmux` **trước khi chạy `srun`**.

---

# 4. Request a GPU worker

Ví dụ: yêu cầu **1× GPU H100/A100**, `32 CPU`, `128 GB RAM` và mở shell trực tiếp trên `worker-0`:

```bash
srun \
  --job-name=function_calling \
  --nodes=1 \
  --gres=gpu:1 \
  --nodelist=worker-0 \
  --partition=main \
  --cpus-per-task=32 \
  --mem=128G \
  --pty bash
```

Sau khi command thành công, bạn đã ở bên trong **worker node** 🎉

Bạn có thể kiểm tra bằng:

```bash
hostname
```

và kiểm tra GPU:

```bash
nvidia-smi
```

## Ý nghĩa một số tham số

| Parameter             | Ý nghĩa                      |
| --------------------- | ---------------------------- |
| `--job-name`          | Tên job hiển thị trong Slurm |
| `--nodes=1`           | Số worker node cần sử dụng   |
| `--gres=gpu:1`        | Request 1 GPU                |
| `--nodelist=worker-0` | Chỉ định worker cụ thể       |
| `--partition=main`    | Slurm partition              |
| `--cpus-per-task=32`  | Số CPU cấp cho job           |
| `--mem=128G`          | RAM cấp cho job              |
| `--pty bash`          | Mở interactive Bash shell    |

---

# 5. Nested tmux = productivity mode

Sau khi đã vào worker node, bạn có thể tạo thêm một `tmux` session bên trong worker:

```bash
tmux new -s runs
```

Session này có thể dùng để chia nhiều cửa sổ cho:

* 🏋️ Training
* 📊 `nvidia-smi`
* 📝 Logs
* 📈 Monitoring
* 🧪 Experiments
* 🐛 Debugging

Để attach lại:

```bash
tmux a -t runs
```

---

## ⌨️ Useful tmux shortcuts

Trong trường hợp đang sử dụng **nested tmux**:

```text
Login node tmux
└── Worker node
    └── Worker tmux
```

Muốn gửi command tới **tmux lớp thứ hai**, nhấn:

```text
Ctrl + B
Ctrl + B
```

sau đó nhấn phím command tương ứng.

| Action          | Shortcut |
| --------------- | -------: |
| New window      |      `c` |
| Next window     |      `n` |
| Previous window |      `p` |
| Detach tmux     |      `d` |

Ví dụ, để tạo một window mới trong nested tmux:

```text
Ctrl+B → Ctrl+B → c
```

> 💡 Bạn có thể dùng mỗi window cho một tác vụ:
>
> ```text
> 0: training
> 1: nvidia-smi
> 2: logs
> 3: shell
> ```

---

# 6. Put big stuff in `/mnt/data`

> 🙏 **Làm ơn đừng làm đầy ổ đĩa home.**

Các dữ liệu hoặc dependency lớn nên được đặt trong:

```text
/mnt/data/
```

Ví dụ:

* 📦 Datasets
* 🐍 Conda environments
* 📚 Pip packages
* 🤗 Hugging Face cache
* 💾 Model checkpoints
* 🧠 Downloaded models
* 🗃️ Temporary caches
* 📊 Training outputs

Ví dụ cấu trúc thư mục:

```text
/mnt/data/
├── datasets/
├── conda/
├── huggingface/
├── models/
├── checkpoints/
└── projects/
```

## Hugging Face cache

Nên đặt Hugging Face cache tại:

```bash
export HF_HOME=/mnt/data/huggingface
```

Có thể thêm dòng trên vào:

```bash
~/.bashrc
```

nếu muốn sử dụng mặc định.

---

# 7. No GPU? Still request Slurm 🚨

Ngay cả khi workload của bạn **chỉ sử dụng CPU**, các tác vụ nặng vẫn nên chạy thông qua Slurm.

> ❌ **Không chạy CPU-heavy job trên login node.**
>
> Việc này có thể:
>
> * Làm login node bị lag
> * Làm SSH chậm
> * Chiếm hết RAM / CPU
> * Ảnh hưởng đến tất cả người dùng khác
> * Trong trường hợp xấu, khiến mọi người không thể SSH vào server

Ví dụ request một worker chỉ dùng CPU:

```bash
srun \
  --job-name=function_calling \
  --nodes=1 \
  --gres=gpu:0 \
  --nodelist=worker-0 \
  --partition=main \
  --cpus-per-task=32 \
  --mem=128G \
  --pty bash
```

Sau đó bạn có thể chạy các tác vụ CPU-heavy như:

```bash
python preprocess.py
```

hoặc:

```bash
python build_index.py
```

trên worker node thay vì login node.

---

# ⭐ Golden Rule

Hãy nhớ một nguyên tắc đơn giản:

```text
Login node
    ↓
Các tác vụ nhẹ / quản lý / submit job

Worker node
    ↓
Training / preprocessing / inference / workload nặng
```

Hay ngắn gọn hơn:

> ### 🧑‍💻 Login node = quản lý công việc
>
> ### 🏗️ Worker node + Slurm = thực hiện công việc

---

## ✅ Recommended workflow

Workflow nên sử dụng hằng ngày:

```text
1. SSH vào server
        ↓
2. Mở tmux trên login node
        ↓
3. Kiểm tra tài nguyên bằng squeue
        ↓
4. Request worker bằng srun
        ↓
5. Mở tmux bên trong worker
        ↓
6. Chạy training / experiment / monitoring
        ↓
7. Lưu dataset, model và cache vào /mnt/data
```

Ví dụ:

```bash
# 1. SSH
ssh user@server

# 2. Login-node tmux
tmux new -s train_llm

# 3. Check cluster
squeue -o "%.8i %.9P %.20j %.8u %.2t %.10M %.6D %R %b"

# 4. Request worker
srun \
  --job-name=train_llm \
  --nodes=1 \
  --gres=gpu:1 \
  --nodelist=worker-0 \
  --partition=main \
  --cpus-per-task=32 \
  --mem=128G \
  --pty bash

# 5. Worker tmux
tmux new -s runs

# 6. Check GPU
nvidia-smi

# 7. Start training
python train.py
```

---

# 🎯 TL;DR

> **Đừng chạy workload nặng trên login node.**
>
> Luôn đi theo flow:
>
> **SSH → tmux → `squeue` → `srun` → worker → tmux → work**
>
> Và nhớ:
>
> **Big files → `/mnt/data`**

Hãy tử tế với chính bạn trong tương lai — và với tất cả mọi người đang dùng chung server 😄
