# Parallel Recurrence: LRU vs. GRU on AMD ROCm

A from-scratch implementation of parallel-scan Linear Recurrent Units in HIP/C++, benchmarked against GRU on an AMD RX 7700 XT.

> LRUs were introduced by Orvieto et al. (2023) as a parallelisable alternative to gated RNNs. This project implements them at the kernel level on the AMD ROCm platform, comparing CPU, GPU-with-Python-loop, and custom-kernel execution paths.

> Note: PyTorch ships `nn.GRU` and `nn.LSTM` but no native LRU module, and no native associative scan primitive. Every LRU implementation in the wild is custom code. This one is a HIP kernel rather than a Python loop, which turns out to matter quite a lot.

---

## What this is

I wanted to know how much performance was being left on the table when running an RNN through a Python `for` loop on the GPU. The short version: a lot. Standard GRUs can't be parallelised across the time dimension because each hidden state depends non-linearly on the one before it. Linear Recurrent Units (LRUs) break that dependency, so the time loop can be replaced by a parallel scan. If you implement that scan in a HIP kernel rather than dispatching one timestep at a time from Python, most of the wall-clock time disappears.

This repo benchmarks five configurations of the same problem and digs into why the numbers come out the way they do.

---

## Results

Batch size 128, hidden size 256, sequence length 300.

| Configuration | 10 epochs | 20 epochs | 30 epochs |
|---|---|---|---|
| GRU — CPU baseline | — | — | 468.7s |
| GRU — GPU (PyTorch) | 25.7s | 51.5s | 78.1s |
| LRU — CPU | 17.1s | 35.5s | 52.2s |
| LRU — GPU (Python loop) | 7.7s | 15.2s | 22.5s |
| **LRU — Custom ROCm kernel** | **4.7s** | **8.8s** | **12.8s** |

The custom kernel is roughly 6x faster than the GRU on GPU and nearly 2x faster than the Python-dispatched LRU. The timings on their own are not the most interesting part. What's more useful is looking at what the GPU is actually doing during each run, because the picture isn't what you'd expect.

### What the GPU is actually doing

Live `rocm-smi` readings during each run:

| Run | Power | GFX% | MEM% | VRAM | Notes |
|---|---|---|---|---|---|
| GRU — GPU (PyTorch) | ~99W | 98% | 10–13% | 2.1 GB | Compute busy on dense gate matmuls. Memory pipe lightly used because gates are sequential, scheduled one per timestep. |
| LRU — GPU (Python loop) | ~70W | 94–99% | 1–2% | 1.5 GB | Looks like the GPU is busy, but it's running tiny one-step kernels back-to-back. Memory is barely touched. The Python `for` loop is the bottleneck; every timestep crosses the CPU↔GPU boundary. |
| LRU — Custom ROCm kernel | ~139W | 96% | 30–34% | 2.1 GB | The whole sequence is fused into one launch. Compute and memory bandwidth both working. This is what saturation actually looks like. |

GFX% on its own is misleading. Both the GRU-GPU run and the Python-loop LRU sit near 100% GPU usage, which would normally make you say "the GPU is working hard, this is as fast as it gets." It isn't. The GPU is reporting "busy" because it's *constantly executing kernels*, but each kernel is tiny and the gaps between launches are full of Python overhead. Memory bandwidth is a much better signal here. Only 1–13% utilised in those runs.

The custom kernel pushes memory bandwidth past 30% and power draw nearly doubles, from 70W to 139W. Compute % and memory % are independent signals, and you really want both high before claiming a GPU is saturated. I missed this for a while when I first started looking at the data.

### Why this proves the case for the custom kernel

The Python-loop LRU is the smoking gun. It's already on a GPU. It's already using the parallelisable LRU algorithm. The numerical recurrence is mathematically the same as in the kernel version. And yet it runs at 70W and is nearly 2x slower than the kernel running at 139W. The only thing that changed between the two is where the time-axis loop lives: in the Python interpreter, or compiled into HIP code on the GPU.

That single change accounts for most of the speedup. The LRU formulation, the parallel scan, the RDNA 3 tuning are all necessary, but moving the loop off the Python side is what actually pays off.

---

## The CPU LRU beats the GPU GRU

Something the table above quietly shows: LRU on CPU (52.2s) beats GRU on GPU (78.1s). I didn't expect this when I first ran it.

A GRU on GPU pays for non-trivial CPU↔GPU transfer, kernel launch overhead per timestep, and a sequential dependency chain that the GPU can't unroll anyway. The LRU on CPU has none of that. It runs in cache-friendly memory, doesn't dispatch anything, and benefits from modern x86 vectorisation (AVX2/AVX-512). For small batches, short sequences, or low-latency inference, the CPU LRU is genuinely competitive with a GPU GRU and is much easier to deploy. The GPU only pulls decisively ahead once batch size and sequence length grow enough to hide its dispatch overhead.

The takeaway for me was that algorithmic choice and hardware choice are coupled decisions. Picking the right pair matters more than picking the fastest hardware on its own.

---

## Why LRUs can be parallelised

A GRU's hidden state update looks like this:

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

The gates $z_t$ and $\tilde{h}_t$ are both functions of $h_{t-1}$ passed through sigmoid/tanh. The non-linearity breaks associativity, so step 2 has to wait for step 1, step 3 has to wait for step 2, and so on. The hardware ends up sitting idle.

The LRU replaces the gated update with a pair of linear equations, a state equation and an output equation:

$$h_t = \Lambda h_{t-1} + B x_t$$

$$y_t = \mathrm{Re}(C h_t) + D x_t$$

Where:

- $\Lambda$ is a diagonal matrix of complex eigenvalues. Each diagonal entry $\lambda_i$ governs one hidden unit independently. Complex values allow oscillation; real values would only allow exponential decay.
- $B$ is the input projection. It maps $x_t$ into the (complex) hidden space.
- $C$ is the output projection. It maps the complex hidden state back to a real-valued output, with $\mathrm{Re}(\cdot)$ taking the real part.
- $D$ is a feed-through term, a direct skip from input to output, similar to a residual.

**Why this is parallelisable.** The state equation $h_t = \Lambda h_{t-1} + B x_t$ has the form of an affine recurrence, which is associative. Define the scan operator $\bullet$ on pairs $(\lambda, u)$:

$$(\lambda_a, u_a) \bullet (\lambda_b, u_b) = (\lambda_a \lambda_b, \\; \lambda_b u_a + u_b)$$

Apply this operator across the time axis with a parallel prefix scan and you recover exactly the same hidden states as the sequential recurrence, but in $O(\log N)$ depth instead of $O(N)$. This is the operator implemented in `CustomCombine` inside the HIP kernel.

---

## The mathematics behind the design

**Why $\Lambda$ is a diagonal matrix of eigenvalues, not just a scalar**

A general RNN hidden state update is $h_t = W h_{t-1} + B x_t$, where $W$ is a dense matrix. Every hidden unit is coupled to every other, and that entanglement is what makes parallelisation hard.

Eigen-decomposition lets you escape it. Any diagonalisable matrix $W$ can be written as:

$$W = V \Lambda V^{-1}$$

where $V$ is the matrix of eigenvectors and $\Lambda$ is the diagonal matrix of eigenvalues. Working in the eigenvector basis (i.e. changing coordinates via $V$), the update equation decouples completely. Each hidden unit evolves independently under its own scalar $\lambda_i$, and independent scalar multiplications are trivially parallel.

So the $\Lambda$ in $h_t = \Lambda h_{t-1} + B x_t$ isn't just a simplification. It's the eigenvalue diagonal after a change of basis. The eigenvectors get absorbed into the input and output projections $B$ and $C$.

**Why $\Lambda$ is complex-valued**

Real eigenvalues can only produce exponential growth or decay along the time axis. Complex eigenvalues introduce rotation in the complex plane, which lets the unit model oscillations and periodic patterns. That sort of behaviour shows up everywhere in sensor data, audio, and time-series with rhythm. A real-valued $\Lambda$ can't represent it without a much larger hidden dimension.

**Why the backward pass uses $\lambda^*$ (Wirtinger calculus)**

Standard calculus assumes real-valued functions. When your parameters are complex, the usual partial derivative $\partial L / \partial \lambda$ is not well-defined in the classical sense, because a complex function $f(\lambda)$ that depends on both $\lambda$ and $\bar{\lambda}$ (its conjugate) is not holomorphic.

Wirtinger calculus resolves this by defining two operators:

$$\frac{\partial}{\partial \lambda} \quad (\text{treat } \bar{\lambda} \text{ as constant}) \qquad \frac{\partial}{\partial \bar{\lambda}} \quad (\text{treat } \lambda \text{ as constant})$$

For gradient descent on a real loss $L$, the update direction is $\partial L / \partial \bar{\lambda}$, which in the LRU backward pass works out to involve the complex conjugate $\lambda^*$. That's why the backward kernel uses $\lambda^*$ rather than $\lambda$ itself. It's not a hack, it's what Wirtinger calculus actually requires for a complex-valued recurrence.

The Cauchy-Riemann equations are the underlying reason the split into $\partial / \partial \lambda$ and $\partial / \partial \bar{\lambda}$ is clean. A holomorphic function satisfies them automatically, making $\partial / \partial \bar{\lambda} = 0$. The LRU recurrence is *not* holomorphic (it mixes $\lambda$ and $\bar{\lambda}$), so both Wirtinger derivatives are non-zero and both matter.

---

## The tensor layout trick

Before the kernel can run, the data has to be in the right shape.

PyTorch stores sequence data as `[Batch, Seq, Hidden]` by default, with the sequence dimension in the middle. `hipcub::BlockScan` performs an inclusive prefix scan along a contiguous block of memory, so the dimension being scanned has to be last. That means the kernel expects `[Batch, Hidden, Seq]`.

The forward pass transposes the input before the scan and transposes back afterward. Sounds trivial. It's the difference between the scan reading contiguous memory (correct and fast) and striding across non-contiguous memory (slow, and quietly wrong if you don't notice).

---

## What the kernel actually does

The forward pass is `lru_forward_kernel` and the backward pass is `lru_backward_kernel`, both built for AMD's RDNA 3 architecture (RX 7700 XT). The forward uses `hipcub::BlockScan` for the parallel scan. The backward is a sequential reverse-time loop per (batch, hidden) pair, parallelised across that 2D grid.

**Hardware target**

- 54 Compute Units, organised into Workgroup Processors (WGPs)
- Each CU: 4 × SIMD32 units running 32-thread wavefronts
- LDS (Local Data Share): fast SRAM shared within a WGP

**The hard part**

I lost most of an afternoon to a pointer arithmetic bug. The kernel was reading 16-byte tuples from memory allocated for 8-byte complex floats, walking off the end of the buffer. The fix was to stop "zipping" tensors in Python (which creates copies and overhead anyway) and just pass separate `λ` and `u` pointers directly to the kernel, then do the zip in thread-local registers right before the scan. Once I stopped trying to be clever in Python, the C++ side became simpler too.

The backward pass uses Wirtinger calculus for the complex-valued gradients (see the maths section for why), with the complex conjugate `λ*` propagating gradients through the recurrence.

---

## Dataset

Training and evaluation use the
[Anomaly Detection in Oil and Gas / Chemical Plants dataset](https://www.kaggle.com/datasets/programmer3/anomaly-detection-in-oil-and-gas-chemical-plants)
from Kaggle.

The raw CSV is not included in this repository.
Place `plant_sensor_data.csv` under the `data/` directory before running training.

---

## Repo structure

```
lru-rocm-experiment/
├── data/                              # put plant_sensor_data.csv here (not committed)
├── native/
│   ├── build/
│   └── src/
│       ├── extension.cpp              # PyBind11 bridge (forward)
│       ├── extensionbackward.cpp      # PyBind11 bridge (backward)
│       ├── kernel.hip                 # lru_forward_kernel — parallel scan (blackward)
│       └── kernelbackward.hip         # lru_backward_kernel — reverse-time gradient pass
├── src/
│   └── lru_experiment/
│       ├── common/
│       ├── models/
│       │   ├── lru_native_model.py    # Model using compiled ROCm extension
│       │   └── lru_python_model.py    # Pure-Python LRU for comparison
│       ├── native/
│       │   └── lru_scan.py            # Python wrapper around the compiled extension
│       └── training/
│           ├── train_gru_python.py
│           ├── train_lru_native.py
│           └── train_lru_python.py
├── tests/
├── setup.py                           # ROCm/HIP compilation script
└── pyproject.toml
```

---

## Setup

Requires ROCm toolkit on your PATH.

```bash
cd native
python setup.py install
```

This compiles the HIP source via `torch.utils.cpp_extension` and drops the compiled `.so` files into `native/src/`.

**Known issue: manual `.so` copy required**

After compilation, the `.so` files need to be copied into `src/lru_experiment/training/` before any training script will find them. I haven't worked out how to make `setup.py` install them in the right place automatically, so for now it's a manual step.

```bash
cp native/src/lru_scan_forward*.so src/lru_experiment/training/
cp native/src/lru_scan_backward*.so src/lru_experiment/training/
```

---

## Reference

The LRU architecture is based on:

> Orvieto et al. (2023). *Resurrecting Recurrent Neural Networks for Long Sequences.*
> https://arxiv.org/pdf/2303.06349
