# Deep Learning-Based Channel Estimation for 6G-Oriented Massive MIMO Wireless Communication Systems

An end-to-end, reproducible PyTorch and NumPy/SciPy simulation framework upgrading a single-coefficient SISO channel estimator into a complete **Massive MIMO** and **6G XL-MIMO (Extremely Large-Scale MIMO)** channel estimation research pipeline.

---

## 1. Research Motivation & Existing vs. Proposed System

Modern 5G-Advanced and emerging 6G wireless links deploy large antenna arrays ($N_t \in \{8, 16, 32, 64\}$) to achieve high spectral efficiency via spatial multiplexing and beamforming. Accurate Channel State Information (CSI) at the transmitter/receiver is critical for precoding and multi-stream detection, yet pilot overhead and matrix-inversion complexity grow rapidly with array size.

| Aspect | Existing SISO System | Upgraded Massive MIMO & 6G XL-MIMO System |
| :--- | :--- | :--- |
| **Channel Model** | Single complex Rayleigh coefficient $h \in \mathbb{C}$ | Full complex MIMO matrix $H \in \mathbb{C}^{N_r \times N_t}$ (i.i.d. Rayleigh + 6G XL-MIMO near-field spherical-wave & spatial non-stationarity) |
| **Pilot Design** | Fixed all-ones BPSK vector | Orthogonal Hadamard BPSK pilots ($X X^H = T_p I_{N_t}$), DFT pilots, and full-column-rank underdetermined pilots ($T_p < N_t$) |
| **Baselines** | Scalar LS only | Batched MIMO **LS** + exact Bayesian **MMSE** (plus minimum-norm pseudoinverse LS when $T_p < N_t$) |
| **AI Architecture** | Fixed `32 -> 128 -> 64 -> 2` MLP | Configurable residual-skip **AI-DNN** (`2*Nr*Tp -> 512 -> 256 -> 256 -> 2*Nr*Nt` + linear skip) and optional **2D CNN** (`CNNChannelEstimator`) |
| **Training Pipeline** | Basic 30-epoch loop | Train-only normalization stats saved in checkpoint, `ReduceLROnPlateau`, early stopping, best-model checkpointing, loss CSV & plots |
| **Evaluation** | Scalar MSE & SISO BPSK BER | Linear/dB **NMSE**, **MSE**, downlink **RZF-precoded multi-stream MIMO BER** (vs. Perfect CSI), batch inference time, parameter counts |
| **Research Scope** | Single SNR curve (`0..20 dB`) | 6 modular experiments: SNR (`0..30 dB`), antenna scaling (`8..64`), SNR interpolation/extrapolation, cross-antenna transfer, pilot overhead, and 6G XL-MIMO near-field vs. far-field |

> **Scope clarification**: Standard Massive MIMO with i.i.d. Rayleigh fading is treated strictly as the **conventional 5G/MIMO baseline**; the **6G-oriented research contribution** is the modular **XL-MIMO near-field spherical-wave and spatially non-stationary visibility-region extension** at 28 GHz.

---

## 2. Mathematical System Model & Estimators

### 2.1 Massive MIMO Pilot Transmission Model

$$Y = H X + N$$

- $H \in \mathbb{C}^{N_r \times N_t}$: Complex MIMO channel matrix (baseline: $N_t = 32$, $N_r = 4$, configurable for $N_t \in \{8, 16, 32, 64\}$ and $N_r \in \{2, 4, 8\}$).
  - Under i.i.d. Rayleigh fading: $H = \frac{1}{\sqrt{2}}(H_r + j H_i)$, with $H_r, H_i \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, 1)$, so $\mathbb{E}[|H_{i,j}|^2] = 1$.
- $X \in \mathbb{C}^{N_t \times T_p}$: Transmitted pilot matrix ($T_p = 32$ by default).
  - Using unnormalized Hadamard BPSK construction ($X_{m,k} \in \{+1, -1\}$) for $T_p \ge N_t$ and power-of-2 $T_p$:
    $$X X^H = T_p I_{N_t}$$
- $N \in \mathbb{C}^{N_r \times T_p}$: Complex circularly symmetric AWGN, $N = \sqrt{\frac{\sigma^2}{2}}(N_r + j N_i)$ with $\mathbb{E}[|N_{i,k}|^2] = \sigma^2$.
- **Consistent Power & Receive SNR Convention**: Because $\mathbb{E}[|(HX)_{i,k}|^2] = \frac{1}{T_p}\mathrm{tr}(X X^H) = N_t$ for unnormalized pilots, receive SNR is:
  $$\mathrm{SNR}_{\text{linear}} = \frac{N_t}{\sigma^2} \implies \sigma^2 = \frac{N_t}{10^{\mathrm{SNR}_{\text{dB}} / 10}}$$
  This keeps receive SNR fair and constant across $N_t \in \{8, 16, 32, 64\}$.

### 2.2 Least Squares (LS) Estimation

$$H_{\text{LS}} = Y X^H (X X^H)^{-1}$$

When orthogonal pilots are used ($X X^H = T_p I_{N_t}$):

$$H_{\text{LS}} = \frac{1}{T_p} Y X^H$$

Under $H_{i,j} \sim \mathcal{CN}(0,1)$ and $\sigma^2 = N_t / \mathrm{SNR}$, for $T_p = N_t$ the theoretical LS NMSE is:

$$\mathrm{NMSE}_{\text{LS}} = \frac{\sigma^2}{T_p} = \frac{1}{\mathrm{SNR}_{\text{linear}}} \quad \left(-10 \log_{10}(\mathrm{SNR}_{\text{linear}}) = -\mathrm{SNR}_{\text{dB}}\right)$$

When $T_p < N_t$, row orthogonality is mathematically impossible ($\mathrm{rank}(X) \le T_p < N_t$), $X X^H$ is singular, and `traditional/ls_estimator.py` falls back to the minimum-norm Moore-Penrose pseudoinverse $H_{\text{LS}} = Y X^+$.

### 2.3 Minimum Mean Square Error (MMSE) Estimation

Under the i.i.d. Rayleigh prior ($\mathbb{E}[h_r^H h_r] = I_{N_t}$) and AWGN variance $\sigma^2$:

$$H_{\text{MMSE}} = Y X^H (X X^H + \sigma^2 I_{N_t})^{-1}$$

For orthogonal pilots $X X^H = T_p I_{N_t}$, this reduces to optimal linear shrinkage $H_{\text{MMSE}} = \frac{T_p}{T_p + \sigma^2} H_{\text{LS}}$ with theoretical $\mathrm{NMSE}_{\text{MMSE}} = \frac{\sigma^2}{T_p + \sigma^2}$.

### 2.4 AI-Based Channel Estimator (`ChannelEstimator` & `CNNChannelEstimator`)

- **Real-Valued Representation (Row-Major Flattening)**:
  - Input: $\mathbf{x} = [\mathrm{vec}(\Re(Y))^T,\; \mathrm{vec}(\Im(Y))^T]^T \in \mathbb{R}^{2 N_r T_p}$
  - Target: $\mathbf{y} = [\mathrm{vec}(\Re(H))^T,\; \mathrm{vec}(\Im(H))^T]^T \in \mathbb{R}^{2 N_r N_t}$
- **DNN Architecture (`models/neural_network.py`)**:
  - Nonlinear backbone: `2*Nr*Tp -> 512 (ReLU) -> 256 (ReLU) -> 256 (ReLU) -> 2*Nr*Nt`
  - Linear residual skip connection: `Linear(2*Nr*Tp, 2*Nr*Nt)` added directly to the MLP output, eliminating the ReLU rank bottleneck when representing linear pilot projections while allowing the ReLU layers to learn SNR-adaptive shrinkage and structured spatial priors.
  - Default `Nt=32, Nr=4, Tp=32` parameter count: **460,288** trainable parameters.
- **Optional 2D CNN (`models/cnn_estimator.py`)**: Reshapes inputs to `(B, 2, Nr, Tp)` and applies 2D convolutions across receive antennas and pilot symbols.
- **Normalization & Leakage Prevention**: Per-feature mean and standard deviation are computed strictly on the training split (`80%` of `50,000` samples) and stored inside the `.pth` checkpoint; independent test datasets (`seed=10042` / per-experiment seeds) are never used during training or normalization.

### 2.5 Performance Metrics & Downlink BER Simulation

Implemented in `evaluation/metrics.py`:
- **MSE**: $\frac{1}{B N_r N_t}\sum |H - \hat{H}|^2$
- **NMSE**: $\frac{\mathbb{E}[\|H - \hat{H}\|_F^2]}{\mathbb{E}[\|H\|_F^2]}$, reported in linear scale and $\mathrm{NMSE}_{\text{dB}} = 10 \log_{10}(\mathrm{NMSE})$.
- **MIMO BER Simulation**: Downlink $N_s = N_r$ spatial-multiplexing BPSK transmission using Regularized Zero-Forcing (RZF) transmit precoding $W = \alpha \hat{H}^H (\hat{H}\hat{H}^H + \frac{N_r}{\mathrm{SNR}} I_{N_r})^{-1}$ constructed from $\hat{H}$ (unit-normalized total precoded power, true channel propagation $Y_{\text{data}} = H W S + N_{\text{data}}$ with $\sigma_{\text{data}}^2 = N_t / \mathrm{SNR}$, and MMSE equalization at the receiver using $\hat{H} W$).

---

## 3. 6G XL-MIMO Near-Field Extension (`utils/channel.py`, `experiments/xl_mimo_experiment.py`)

At 6G mmWave/sub-THz frequencies and large array apertures, users lie inside the **Fraunhofer / Rayleigh distance**:

$$d_R = \frac{2 D^2}{\lambda}$$

Our modular XL-MIMO physical configuration (`config/config.py -> XLMIMOConfig`):
- Carrier frequency: $f_c = 28\text{ GHz}$ ($\lambda = 10.71\text{ mm}$)
- Transmit ULA: $N_t = 64$ elements, spacing $d = \lambda/2 = 5.35\text{ mm}$, aperture $D = (N_t - 1)d = 0.337\text{ m}$
- **Rayleigh distance**: $d_R = 21.25\text{ m}$
- **Near-field regime**: $r \in [3, 20]\text{ m} < d_R$, modeling exact 3D Euclidean element-pair distances $d_{m,n} = \|\mathbf{p}_{\text{rx}}[m] - \mathbf{p}_{\text{tx}}[n]\|_2$ for spherical wavefront phase $\exp(-j \frac{2\pi}{\lambda}(d_{m,n} - r))$ and amplitude $r / d_{m,n}$, combined with **spatial non-stationarity** via contiguous Visibility Regions (VR fraction $\in [0.4, 1.0]$) and Rician $K = 10\text{ dB}$.
- **Far-field regime**: $r = 200\text{ m} \gg d_R$ using planar-wave steering vectors.

---

## 4. Project Structure

```text
aiChannelEstimation/
├── config/
│   ├── __init__.py
│   └── config.py                      # Centralized MIMO, Model, Experiment, XL-MIMO, Path configs
├── data/
│   ├── __init__.py
│   ├── dataset.py                     # Massive MIMO & XL-MIMO dataset generator + NPZ save/load
│   ├── channel_dataset.npz            # 50,000 train/val samples (Nt=32, Nr=4, Tp=32)
│   └── channel_test_dataset.npz       # 5,000 independent test samples
├── models/
│   ├── __init__.py
│   ├── neural_network.py              # Configurable ChannelEstimator (DNN + residual skip)
│   └── cnn_estimator.py               # Optional 2D CNNChannelEstimator
├── traditional/
│   ├── __init__.py
│   ├── ls_estimator.py                # Batched MIMO LS (+ SISO backward compat + pinv fallback)
│   └── mmse_estimator.py              # Batched MIMO Bayesian LMMSE
├── utils/
│   ├── __init__.py
│   ├── channel.py                     # SISO Rayleigh, MIMO Rayleigh, 6G XL-MIMO near/far-field
│   ├── modulation.py                  # BPSK & QPSK bit/symbol utilities
│   ├── noise.py                       # AWGN & SNR-to-noise-variance utilities
│   └── pilots.py                      # Hadamard BPSK & DFT pilot generators + orthogonality check
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                     # MSE, NMSE, parameter count, timing, MIMO RZF BER simulator
│   └── visualization.py               # Publication-quality Matplotlib plots + CSV writer
├── experiments/
│   ├── snr_experiment.py              # Exp 1: NMSE, MSE, BER, inference time vs SNR (0..30 dB)
│   ├── antenna_experiment.py          # Exp 2: Nt in {8, 16, 32, 64} scaling
│   ├── generalization_experiment.py   # Exp 3 & 5: SNR interp/extrap & cross-antenna transfer
│   ├── pilot_experiment.py            # Exp 4: Pilot overhead Tp in {8, 16, 32, 64}
│   └── xl_mimo_experiment.py          # Exp 6: 6G XL-MIMO near-field vs far-field & Rayleigh
├── results/
│   ├── checkpoints/                   # Saved .pth checkpoints (with .bak backup protection)
│   ├── plots/                         # All generated PNG figures (300 DPI)
│   └── tables/                        # Numerical CSV results for every experiment
├── tests/                             # Pytest unit test suite (31 tests)
├── check_dataset.py                   # NPZ dataset verifier
├── train.py                           # Reproducible training pipeline
├── evaluate.py                        # Master experiment runner
├── main.py                            # Quick physical & AI demo / full pipeline CLI
├── test_ls.py                         # Legacy SISO LS verification script
└── requirements.txt
```

---

## 5. Installation & Reproducible Execution

### Installation
```bash
pip install -r requirements.txt
```

### Generate Datasets & Verify
```bash
python -m data.dataset
python check_dataset.py
python check_dataset.py --path data/channel_test_dataset.npz
```

### Train Baseline Massive MIMO DNN (`Nt=32, Nr=4, Tp=32`)
```bash
python train.py --epochs 50
```

### Run All Unit Tests
```bash
python -m pytest tests/ -v
python test_ls.py
```

### Run All Experiments (Generates CSVs & Plots in `results/`)
```bash
python evaluate.py --experiment all
```
Individual experiments can also be run via `--experiment snr|antenna|pilot|generalization|xl_mimo`.

---

## 6. Verified Experimental Results

All numbers below are directly measured from simulations in `results/tables/`.

### Experiment 1: NMSE, BER & Inference Time vs. SNR (`Nt=32, Nr=4, Tp=32`, 5,000 test samples/SNR)

| SNR (dB) | LS NMSE (dB) | MMSE NMSE (dB) | AI-DNN NMSE (dB) | Perfect CSI BER | LS BER | MMSE BER | AI-DNN BER |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 0.00 | -3.01 | **-1.89** | 0.25052 | 0.31935 | 0.31652 | 0.32170 |
| 5 | -5.00 | -6.20 | **-5.64** | 0.11859 | 0.15457 | 0.15395 | 0.15675 |
| 10 | -10.01 | -10.42 | -9.79 | 0.01838 | 0.02485 | 0.02483 | 0.02549 |
| 15 | -15.00 | -15.13 | -13.33 | 0.00010 | 0.00016 | 0.00016 | 0.00019 |
| 20 | -19.99 | -20.03 | -15.64 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| 25 | -24.99 | -25.00 | -16.72 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| 30 | -30.00 | -30.01 | -17.10 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |

- **Observation on i.i.d. Rayleigh**: At low SNR (0–5 dB), AI-DNN beats LS by **1.89 dB** (at 0 dB) and **0.64 dB** (at 5 dB) by learning MMSE shrinkage without knowing $\sigma^2$. Because $(H, Y)$ are jointly Gaussian under i.i.d. Rayleigh fading, oracle-SNR LMMSE is the Bayes-optimal lower bound. Above 15 dB, unweighted MSE training across `[0, 30]` dB is dominated by low-SNR errors, producing an error floor around `-17.10 dB`.
- **Batch Inference Time (5,000 samples, CPU)**: LS `~24.7 ms`, AI-DNN `~43.6 ms` (`460,288` params), general per-sample MMSE solve `~201.3 ms`.

### Experiment 2: Transmit Antenna Scaling (`Nr=4, Tp=Nt, SNR=10 dB`, 4,000 test samples)

| $N_t = T_p$ | AI-DNN Params | LS NMSE (dB) | MMSE NMSE (dB) | AI-DNN NMSE (dB) | LS Time (ms) | MMSE Time (ms) | AI-DNN Time (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 8 | 251,008 | -9.99 | -10.40 | -9.88 | 3.70 | 17.57 | 16.49 |
| 16 | 312,576 | -10.00 | -10.42 | -9.97 | 8.36 | 47.13 | 23.48 |
| 32 | 460,288 | -9.99 | -10.40 | -9.94 | 19.42 | 163.56 | 35.12 |
| 64 | 854,016 | -10.00 | -10.41 | -9.95 | 64.55 | 654.51 | **68.22** |

As $N_t$ scales to 64, general per-sample MMSE matrix inversion scales cubically (`654.5 ms`), whereas AI-DNN forward inference (`68.2 ms`) is **9.6x faster** than MMSE while matching LS NMSE.

### Experiment 3 & 5: SNR Generalization & Cross-Antenna Generalization

- **SNR Interpolation** (trained on `[0, 10] U [20, 30]` dB): On unseen intermediate SNRs `12.5, 15.0, 17.5` dB, AI-DNN achieves `-11.22, -12.36, -13.18` dB respectively (smooth interpolation).
- **SNR Extrapolation** (trained on `[5, 20]` dB): At unseen `0` dB, NMSE degrades to `-0.68` dB (vs. `-1.88` dB for `[0, 30]` dB training); at unseen `25, 30` dB, it reaches `-18.36, -18.94` dB (outperforming the `[0, 30]` dB model at high SNR because training is not dominated by `0` dB noise).
- **Cross-Antenna Generalization** (Target `Nt=16`, `SNR=15 dB`): A matched `Nt=16` AI-DNN achieves `-13.23 dB`, whereas zero-padding a `Nt=32`-trained fixed-dimension DNN degrades sharply to `-5.35 dB` (`LS = -14.99 dB`, `MMSE = -15.13 dB`), confirming that fixed-dimension MLPs require matched input/output dimensions or dimension-agnostic architectures.

### Experiment 4: Pilot Overhead (`Nt=32, Nr=4, SNR=15 dB`)

| Pilot Length $T_p$ | Orthogonal $X X^H = T_p I$ Possible? | LS NMSE (dB) | MMSE NMSE (dB) | AI-DNN NMSE (dB) |
| :---: | :---: | :---: | :---: | :---: |
| 8 | No ($\mathrm{rank}(X) \le 8 < 32$; uses `pinv`) | -1.21 | -1.21 | -1.12 |
| 16 | No ($\mathrm{rank}(X) \le 16 < 32$; uses `pinv`) | -2.87 | -2.87 | -2.71 |
| 32 | Yes | -14.99 | -15.12 | -13.21 |
| 64 | Yes | -18.01 | -18.08 | -16.22 |

When $T_p < N_t$, orthogonal pilots cannot exist; minimum-norm LS hits the theoretical null-space error floor $10\log_{10}(1 - T_p/N_t)$ (`-1.25 dB` at $T_p=8$, `-3.01 dB` at $T_p=16$).

### Experiment 6 (6G Extension): XL-MIMO Near-Field vs. Far-Field (`Nt=64, Nr=4, f_c=28 GHz, d_R=21.25 m`)

| SNR (dB) | Near-Field LS (dB) | Near-Field i.i.d. MMSE (dB) | **Near-Field AI-DNN (dB)** | Rayleigh-Trained AI on NF (dB) | Far-Field AI-DNN (dB) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 0.00 | -3.01 | **-4.57** | -1.00 | -6.23 |
| 5 | -5.00 | -6.19 | **-8.71** | -5.71 | -10.13 |
| 10 | -10.01 | -10.42 | **-11.76** | -9.94 | -12.68 |
| 15 | -15.00 | -15.13 | -13.42 | -13.13 | -13.91 |
| 20 | -20.00 | -20.05 | -14.14 | -14.96 | -14.38 |
| 25 | -25.01 | -25.03 | -14.38 | -15.75 | -14.56 |
| 30 | -30.00 | -30.01 | -14.43 | -16.04 | -14.60 |

**Key 6G Research Finding**: Unlike i.i.d. Rayleigh channels, near-field XL-MIMO channels exhibit spherical wavefront curvature and non-stationary visibility sparsity ($R_H \neq I_{N_t}$). Consequently, i.i.d. MMSE is prior-mismatched, and **Near-Field AI-DNN outperforms i.i.d. MMSE by +1.56 dB at 0 dB, +2.52 dB at 5 dB, and +1.34 dB at 10 dB** (and outperforms LS by up to **+4.57 dB**). Conversely, evaluating a Rayleigh-trained DNN on near-field channels (`-1.00 dB` at 0 dB) causes severe domain-shift degradation.

---

## 7. Generated Visualizations (`results/plots/`)

- `training_loss_dnn_nt32_nr4_tp32.png` — Training vs. validation MSE convergence
- `nmse_vs_snr.png` & `mse_vs_snr.png` — LS vs. MMSE vs. AI-DNN across 0–30 dB
- `ber_vs_snr.png` — Downlink RZF-precoded multi-stream BER vs. SNR (including Perfect CSI)
- `nmse_vs_antennas.png` & `inference_time_vs_antennas.png` — Scaling across $N_t \in \{8, 16, 32, 64\}$
- `inference_time_comparison.png` — Batch inference latency comparison
- `snr_generalization.png` — SNR interpolation vs. extrapolation curves
- `nmse_vs_pilots.png` — Pilot overhead $T_p \in \{8, 16, 32, 64\}$
- `xl_mimo_near_vs_far_field.png` — 6G XL-MIMO near-field vs. far-field & domain shift

---

## 8. Limitations & Future Work

1. **High-SNR Floor in Joint-SNR Unweighted Training**: Training a single DNN with unweighted MSE over `[0, 30]` dB biases gradients toward low-SNR samples; future work can incorporate explicit SNR conditioning (`sigma^2` concatenated to inputs) or NMSE/SNR-weighted loss functions.
2. **Fixed-Dimension MLP Input/Output**: A fully connected DNN is tied to `(Nr, Nt, Tp)` used during training; future extensions can explore dimension-agnostic graph neural networks or per-antenna shared convolutional/transformer architectures.
3. **XL-MIMO Physics Scope**: Our 6G extension models near-field 3D spherical-wave LoS, Rician NLOS, and contiguous visibility regions at 28 GHz; it does not claim full 3GPP TR 38.901 stochastic multi-cluster delay/Doppler or sub-THz molecular absorption modeling.
