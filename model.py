"""
model.py
========
Single NPU calcium oscillation model and NPU-array reservoir.

Mathematical basis:
- Modified FitzHugh–Nagumo (FHN) with a slow calcium (Ca²⁺) recovery variable.
- The FHN model is a 2D reduction of the Hodgkin–Huxley equations and is widely
  used to capture excitable / oscillatory membrane dynamics
  (FitzHugh, 1961; Nagumo et al., 1962).
- To mimic cytosolic Ca²⁺ oscillations observed in astrocytes and neuronal
  ensembles, we append a slow adaptive variable `c` that models Ca²⁺ influx /
  clearance with time constant τ_ca (Dupont & Goldbeter, 1993).
- The resulting 3D system exhibits limit-cycle oscillations whose frequency
  and amplitude can be modulated by external photothermal input I(t).

References
----------
[1] FitzHugh, R. (1961). Impulses and physiological states in theoretical
    models of nerve membrane. *Biophysical Journal*, 1(6), 445–466.
[2] Nagumo, J., Arimoto, S., & Yoshizawa, S. (1962). An active pulse
    transmission line simulating nerve axon. *Proc. IRE*, 50(10), 2061–2070.
[3] Dupont, G., & Goldbeter, A. (1993). One-pool model for Ca²⁺ oscillations
    involving Ca²⁺ and inositol 1,4,5-trisphosphate. *Chaos*, 3(2), 219–230.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert, butter, filtfilt
from scipy.integrate import solve_ivp
from typing import Callable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Global random state helpers (reproducibility)
# ---------------------------------------------------------------------------

_rng = np.random.default_rng(seed=42)


def set_seed(seed: int = 42) -> None:
    global _rng
    _rng = np.random.default_rng(seed=seed)


# ---------------------------------------------------------------------------
# 1. Single NPU model  –  modified FHN + Ca²⁺ slow variable
# ---------------------------------------------------------------------------

class SingleNPU:
    """
    Single Neuro-Photonic Unit (NPU).

    State vector:  x = [v, w, c]
        v  – fast membrane-like variable (analogous to voltage)
        w  – slow recovery variable
        c  – slow Ca²⁺ variable (controls threshold / excitability)

    Dynamics (dimensionless):
        dv/dt = v - v**3/3 - w + c + g*I(t) + ξ_v(t)
        dw/dt = (v + a - b*w) / τ_w
        dc/dt = (-c + κ * max(0, v)) / τ_ca  +  ξ_c(t)

    Parameters
    ----------
    f0 : float
        Intrinsic oscillation frequency (Hz), used to set τ_w.
    tau_ca : float
        Calcium recovery time constant (s).
    gain : float
        Input gain g (dimensionless).
    sigma : float
        Additive Gaussian noise standard deviation.
    a, b, kappa : float
        FHN / Ca²⁺ shape parameters (default values give ~1 Hz limit cycle).
    dt : float
        Simulation time step (s).
    """

    def __init__(
        self,
        f0: float = 1.0,
        tau_ca: float = 1.0,
        gain: float = 1.0,
        sigma: float = 0.02,
        a: float = 0.7,
        b: float = 0.8,
        kappa: float = 0.5,
        dt: float = 0.01,
    ):
        self.f0 = f0
        self.tau_ca = tau_ca
        self.gain = gain
        self.sigma = sigma
        self.a = a
        self.b = b
        self.kappa = kappa
        self.dt = dt

        # τ_w is chosen so that the intrinsic frequency ≈ f0 in the uncoupled,
        # noise-free limit cycle (empirically calibrated).
        self.tau_w = max(0.1, 2.5 / (2.0 * np.pi * f0))

        # State variables
        self.v = 0.0
        self.w = 0.0
        self.c = 0.0
        self.t = 0.0

        # History buffers (for feature extraction)
        self.history: dict = {"t": [], "v": [], "w": [], "c": [], "I": []}

    # ------------------------------------------------------------------
    # Core ODE RHS
    # ------------------------------------------------------------------
    def _rhs(self, state: np.ndarray, I: float) -> np.ndarray:
        v, w, c = state
        dv = v - (v ** 3) / 3.0 - w + c + self.gain * I
        dw = (v + self.a - self.b * w) / self.tau_w
        dc = (-c + self.kappa * max(0.0, v)) / self.tau_ca
        return np.array([dv, dw, dc])

    # ------------------------------------------------------------------
    # Single Euler–Maruyama step
    # ------------------------------------------------------------------
    def step(self, I: float) -> float:
        """Advance one time step with input I. Returns v (the Ca²⁺-proxy output)."""
        state = np.array([self.v, self.w, self.c])
        drift = self._rhs(state, I)
        # Additive noise on v and c (w is usually deterministic in FHN)
        noise = np.array([
            _rng.normal(0.0, self.sigma),
            0.0,
            _rng.normal(0.0, self.sigma * 0.3),
        ])
        state_new = state + self.dt * drift + np.sqrt(self.dt) * noise
        self.v, self.w, self.c = state_new
        self.t += self.dt

        # Record history
        self.history["t"].append(self.t)
        self.history["v"].append(self.v)
        self.history["w"].append(self.w)
        self.history["c"].append(self.c)
        self.history["I"].append(I)

        # The "calcium oscillation signal" is taken as the slow variable c,
        # which tracks cytosolic Ca²⁺ and shows smooth oscillatory transients.
        return float(self.c)

    # ------------------------------------------------------------------
    # Run for a duration
    # ------------------------------------------------------------------
    def run(self, duration: float, I_func: Callable[[float], float]) -> np.ndarray:
        """Simulate for `duration` seconds with input function I_func(t)."""
        n_steps = int(round(duration / self.dt))
        out = np.empty(n_steps)
        for k in range(n_steps):
            t = self.t
            out[k] = self.step(I_func(t))
        return out

    def reset(self, v0: float = 0.0, w0: float = 0.0, c0: float = 0.0) -> None:
        """Reset state and clear history."""
        self.v = v0
        self.w = w0
        self.c = c0
        self.t = 0.0
        self.history = {"t": [], "v": [], "w": [], "c": [], "I": []}

    def get_signals(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return arrays (t, v, w, c, I) from history."""
        t = np.array(self.history["t"])
        v = np.array(self.history["v"])
        w = np.array(self.history["w"])
        c = np.array(self.history["c"])
        I = np.array(self.history["I"])
        return t, v, w, c, I


# ---------------------------------------------------------------------------
# 2. NPU Array (Oscillatory Reservoir)
# ---------------------------------------------------------------------------

class NPUArray:
    """
    Reservoir of N SingleNPUs with optional weak global / diffusive coupling.

    Coupling scheme (default zero):
        For each unit i, an extra term is added to dv_i/dt:
            + g_couple * (mean(v_j) - v_i)
        This mimics residual cross-talk in a microfluidic chamber without
        explicit synaptic weights.
    """

    def __init__(
        self,
        N: int = 8,
        f0_range: Tuple[float, float] = (0.5, 2.5),
        tau_ca_range: Tuple[float, float] = (0.5, 2.0),
        gain_range: Tuple[float, float] = (0.8, 1.2),
        sigma_range: Tuple[float, float] = (0.01, 0.05),
        g_couple: float = 0.0,
        dt: float = 0.01,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            set_seed(seed)

        self.N = N
        self.dt = dt
        self.g_couple = g_couple

        # Sample heterogeneous parameters
        self.f0s = _rng.uniform(*f0_range, size=N)
        self.tau_cas = _rng.uniform(*tau_ca_range, size=N)
        self.gains = _rng.uniform(*gain_range, size=N)
        self.sigmas = _rng.uniform(*sigma_range, size=N)

        self.units: List[SingleNPU] = []
        for i in range(N):
            npu = SingleNPU(
                f0=self.f0s[i],
                tau_ca=self.tau_cas[i],
                gain=self.gains[i],
                sigma=self.sigmas[i],
                dt=dt,
            )
            self.units.append(npu)

        self.state = np.zeros((N, 3))  # [v, w, c] for each unit
        self.t = 0.0

    # ------------------------------------------------------------------
    # Vectorised step with coupling
    # ------------------------------------------------------------------
    def step(self, I_vec: np.ndarray) -> np.ndarray:
        """
        Advance all NPUs by one dt.

        Parameters
        ----------
        I_vec : np.ndarray, shape (N,)
            Photothermal input intensity for each NPU at current time.

        Returns
        -------
        c_vec : np.ndarray, shape (N,)
            The calcium-proxy output of each NPU.
        """
        # Gather current v
        v_vec = np.array([u.v for u in self.units])
        v_mean = v_vec.mean()

        c_out = np.empty(self.N)
        for i, npu in enumerate(self.units):
            # Add coupling term to input
            I_eff = I_vec[i] + self.g_couple * (v_mean - v_vec[i])
            c_out[i] = npu.step(I_eff)

        self.t += self.dt
        return c_out

    def run(self, duration: float, I_func: Callable[[float], np.ndarray]) -> np.ndarray:
        """
        Simulate reservoir for `duration` seconds.

        Returns
        -------
        C : np.ndarray, shape (n_steps, N)
            Calcium signal of every NPU.
        """
        n_steps = int(round(duration / self.dt))
        C = np.empty((n_steps, self.N))
        for k in range(n_steps):
            t = self.t
            C[k, :] = self.step(I_func(t))
        return C

    def reset(self) -> None:
        """Reset every NPU and clear history."""
        self.t = 0.0
        for u in self.units:
            u.reset()

    def get_all_signals(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Return list of (t, v, w, c, I) for each unit."""
        return [u.get_signals() for u in self.units]


# ---------------------------------------------------------------------------
# 3. Input encoding interfaces
# ---------------------------------------------------------------------------

class InputEncoder:
    """
    Converts a scalar or vector input stream u(t) ∈ [0,1] into photothermal
    stimulus patterns.

    Supported modes
    ---------------
    - "frequency" : pulse frequency ∈ [f_min, f_max] Hz.
    - "phase"     : sinusoidal carrier with phase offset per channel.
    - "pulsewidth": fixed-frequency pulse train with duty-cycle modulation.
    """

    def __init__(
        self,
        mode: str = "frequency",
        f_min: float = 0.5,
        f_max: float = 4.0,
        base_freq: float = 1.0,
        amplitude: float = 0.5,
        n_channels: int = 1,
    ):
        assert mode in ("frequency", "phase", "pulsewidth")
        self.mode = mode
        self.f_min = f_min
        self.f_max = f_max
        self.base_freq = base_freq
        self.amplitude = amplitude
        self.n_channels = n_channels

        # Phase offsets for multi-channel phase encoding
        self._phases = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)

    def encode_scalar(self, t: float, u: float, channel: int = 0) -> float:
        """Return stimulus intensity for a single channel."""
        u = np.clip(u, 0.0, 1.0)
        if self.mode == "frequency":
            f = self.f_min + u * (self.f_max - self.f_min)
            # Rectified sine burst at frequency f
            return self.amplitude * max(0.0, np.sin(2.0 * np.pi * f * t))
        elif self.mode == "phase":
            phi = self._phases[channel % self.n_channels]
            return self.amplitude * max(0.0, np.sin(2.0 * np.pi * self.base_freq * t + phi))
        elif self.mode == "pulsewidth":
            duty = 0.1 + 0.8 * u  # duty ratio ∈ [0.1, 0.9]
            period = 1.0 / self.base_freq
            phase = (t % period) / period
            return self.amplitude if phase < duty else 0.0
        else:
            raise ValueError(f"Unknown encoding mode: {self.mode}")

    def encode_vector(self, t: float, u_vec: np.ndarray) -> np.ndarray:
        """
        Encode a vector input for all channels.
        If u_vec is scalar-like, broadcast to all channels.
        """
        if np.isscalar(u_vec) or u_vec.size == 1:
            u_vec = np.full(self.n_channels, float(u_vec))
        return np.array([self.encode_scalar(t, u_vec[i], i) for i in range(self.n_channels)])


# ---------------------------------------------------------------------------
# 4. Feature extraction (read-out layer pre-processing)
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Online / sliding-window feature extraction from NPU calcium signals.

    For each NPU i and each time window we compute:
        1. Instantaneous amplitude envelope  A_i(t)   (Hilbert transform)
        2. Instantaneous phase               Φ_i(t)   (Hilbert transform)
        3. Dominant frequency                f_i(t)   (peak of FFT in window)
        4. Band-pass energies                E_{i,b}  (0.5–1, 1–2, 2–4 Hz)
        5. Phase-locking value (PLV)         ρ_{ij}   (when N>1)

    The final feature vector per window is concatenated across all NPUs.
    """

    def __init__(
        self,
        N: int,
        dt: float = 0.01,
        window_size: float = 2.0,
        overlap: float = 1.0,
        freq_bands: Optional[List[Tuple[float, float]]] = None,
    ):
        self.N = N
        self.dt = dt
        self.window_size = window_size
        self.overlap = overlap
        self.win_samples = int(round(window_size / dt))
        self.hop_samples = int(round((window_size - overlap) / dt))
        if self.hop_samples < 1:
            self.hop_samples = 1

        if freq_bands is None:
            freq_bands = [(0.5, 1.0), (1.0, 2.0), (2.0, 4.0)]
        self.freq_bands = freq_bands
        self.n_bands = len(freq_bands)

        # Pre-compute band-pass filters
        self._filters = {}
        nyq = 0.5 / dt
        for low, high in freq_bands:
            b, a = butter(
                N=2,
                Wn=(low / nyq, high / nyq),
                btype="band",
            )
            self._filters[(low, high)] = (b, a)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _band_energy(self, sig: np.ndarray, low: float, high: float) -> float:
        b, a = self._filters[(low, high)]
        try:
            filtered = filtfilt(b, a, sig)
        except ValueError:
            # Signal too short – return NaN, caller should handle
            return np.nan
        return float(np.mean(filtered ** 2))

    def _dominant_freq(self, sig: np.ndarray) -> float:
        """Return dominant frequency (Hz) via FFT peak."""
        n = len(sig)
        if n < 8:
            return np.nan
        win = np.hanning(n)
        yf = np.fft.rfft(sig * win)
        xf = np.fft.rfftfreq(n, d=self.dt)
        # Exclude DC
        idx = np.argmax(np.abs(yf[1:])) + 1
        return float(xf[idx])

    # ------------------------------------------------------------------
    # Main extraction routine
    # ------------------------------------------------------------------
    def extract(self, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        C : np.ndarray, shape (T, N)
            Calcium signals of N NPUs over T time steps.

        Returns
        -------
        features : np.ndarray, shape (n_windows, N * K)
            K = 3 (envelope, phase, dominant_freq) + n_bands (band energies)
                + 1 (mean amplitude)  =  4 + n_bands  per NPU.
            Plus (optional) pairwise PLV appended at the end.
        times : np.ndarray, shape (n_windows,)
            Centre time of each window.
        """
        T = C.shape[0]
        n_win = max(1, (T - self.win_samples) // self.hop_samples + 1)

        # Per-NPU feature dimension
        K_per = 4 + self.n_bands  # amp_mean, amp_std, dom_freq, phase_mean + band energies
        feat_dim = self.N * K_per

        # Optionally append PLV features
        if self.N > 1:
            n_plv = self.N * (self.N - 1) // 2
            feat_dim += n_plv
        else:
            n_plv = 0

        features = np.empty((n_win, feat_dim))
        times = np.empty(n_win)

        # Hilbert transform for instantaneous amplitude / phase (whole signal)
        amp_env = np.abs(hilbert(C, axis=0))
        phase = np.angle(hilbert(C, axis=0))

        for w in range(n_win):
            start = w * self.hop_samples
            end = start + self.win_samples
            if end > T:
                end = T
                start = max(0, end - self.win_samples)
            centre_t = (start + end) / 2.0 * self.dt
            times[w] = centre_t

            idx = 0
            win_amp = amp_env[start:end, :]
            win_pha = phase[start:end, :]
            win_sig = C[start:end, :]

            for i in range(self.N):
                features[w, idx] = float(np.mean(win_amp[:, i]))
                idx += 1
                features[w, idx] = float(np.std(win_amp[:, i]))
                idx += 1
                features[w, idx] = self._dominant_freq(win_sig[:, i])
                idx += 1
                features[w, idx] = float(np.mean(win_pha[:, i]))
                idx += 1
                for low, high in self.freq_bands:
                    features[w, idx] = self._band_energy(win_sig[:, i], low, high)
                    idx += 1

            # Pairwise PLV
            if n_plv > 0:
                plv_idx = 0
                for i in range(self.N):
                    for j in range(i + 1, self.N):
                        dphi = win_pha[:, i] - win_pha[:, j]
                        plv = np.abs(np.mean(np.exp(1j * dphi)))
                        features[w, idx + plv_idx] = plv
                        plv_idx += 1

        # Drop any NaN rows (too-short signals)
        valid = ~np.isnan(features).any(axis=1)
        features = features[valid]
        times = times[valid]

        return features, times

    @property
    def feature_dim(self) -> int:
        """Return dimensionality of feature vector per window."""
        K_per = 4 + self.n_bands
        dim = self.N * K_per
        if self.N > 1:
            dim += self.N * (self.N - 1) // 2
        return dim


# ---------------------------------------------------------------------------
# 5. Utility: generate input time-series for benchmarking
# ---------------------------------------------------------------------------

def generate_narma10(
    n_steps: int,
    u_mean: float = 0.5,
    u_std: float = 0.1,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate NARMA-10 time series.

    y(t+1) = tanh(0.3*y(t) + 0.05*y(t)*sum(y(t-k)) + 1.5*u(t-9)*u(t) + 0.1)

    Bounded by tanh to prevent overflow; then normalised to zero-mean /
    unit-variance for ridge regression.
    """
    rng = np.random.default_rng(seed)
    u = u_mean + u_std * rng.standard_normal(n_steps + 100)
    u = np.clip(u, 0.0, 1.0)

    y = np.zeros(n_steps + 100)
    for t in range(10, n_steps + 100 - 1):
        y[t + 1] = np.tanh(
            0.3 * y[t]
            + 0.05 * y[t] * np.sum(y[t - 9 : t + 1])
            + 1.5 * u[t - 9] * u[t]
            + 0.1
        )
    u_out = u[100:]
    y_out = y[100:]
    y_out = (y_out - np.mean(y_out)) / (np.std(y_out) + 1e-8)
    return u_out, y_out


def generate_rhythm_classification_trials(
    n_trials: int = 200,
    trial_duration: float = 10.0,
    dt: float = 0.01,
    fA: float = 0.8,
    fB: float = 2.2,
    seed: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate trials for Task A.

    Each trial is a sinusoidal modulation of input intensity at either fA or fB.
    Returns (stimuli, labels) where stimuli shape = (n_trials, n_steps, 1)
    and labels ∈ {0,1} (0=A, 1=B).
    """
    rng = np.random.default_rng(seed)
    n_steps = int(round(trial_duration / dt))
    stimuli = np.empty((n_trials, n_steps))
    labels = rng.integers(0, 2, size=n_trials)

    t = np.arange(n_steps) * dt
    for i in range(n_trials):
        f = fA if labels[i] == 0 else fB
        # Add small phase jitter so classification is non-trivial
        phi = rng.uniform(0, 2 * np.pi)
        stimuli[i, :] = 0.5 * (1.0 + np.sin(2.0 * np.pi * f * t + phi))
    return stimuli, labels


# ---------------------------------------------------------------------------
# 6. Baseline: Echo State Network (ESN)
# ---------------------------------------------------------------------------

class ESN:
    """
    Traditional Echo State Network with tanh neurons for comparison.
    """

    def __init__(
        self,
        n_reservoir: int = 100,
        spectral_radius: float = 0.9,
        input_scaling: float = 1.0,
        leaking_rate: float = 0.3,
        dt: float = 0.01,
        seed: int = 99,
    ):
        rng = np.random.default_rng(seed)
        self.n_reservoir = n_reservoir
        self.alpha = leaking_rate
        self.dt = dt

        # Input weights
        self.W_in = input_scaling * rng.standard_normal((n_reservoir, 1))

        # Reservoir weights (sparse random, then scaled to spectral radius)
        density = 0.1
        W = rng.standard_normal((n_reservoir, n_reservoir))
        mask = rng.random((n_reservoir, n_reservoir)) < density
        W[~mask] = 0.0
        radius = max(1e-6, np.max(np.abs(np.linalg.eigvals(W))))
        self.W = W * (spectral_radius / radius)

        self.x = np.zeros(n_reservoir)

    def reset(self) -> None:
        self.x = np.zeros(self.n_reservoir)

    def step(self, u: float) -> np.ndarray:
        self.x = (1.0 - self.alpha) * self.x + self.alpha * np.tanh(
            self.W @ self.x + self.W_in.ravel() * u
        )
        return self.x.copy()

    def run(self, u_series: np.ndarray) -> np.ndarray:
        T = len(u_series)
        X = np.empty((T, self.n_reservoir))
        for t in range(T):
            X[t, :] = self.step(float(u_series[t]))
        return X
