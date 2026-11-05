import torch
import torch.nn as nn

torch.set_num_threads(4)
theta_min, theta_max = 0.0, 1.0

V_ROLL, SIG_ROLL = 6.5, 1.0
LAGS = (0, 1, 2, 3)              # candidate lags (samples) between kinematics and power; negatives allowed
ROBUST_C = 2.5                   # a window's weight drops below 0.5 once its misfit exceeds ROBUST_C x the ride median
N_IRLS = 3                       # robust re-fit iterations
N_REG = 5

p_law_index = 0.25
G = 9.81


def segment_regressors(x_phys):
    """x_phys: (..., T, 6) in physical units  ->  X: (..., T, 5), y: (..., T)."""
    P, v, a, grad, wind, rho = x_phys.unbind(-1)
    v = v.clamp(min=0.0)

    cos_t = 1.0 / torch.sqrt(1.0 + grad ** 2)       # grad = rise/run
    sin_t = grad * cos_t
    v_air = v + wind

    x1 = v * (a + G * sin_t)                                           # coefficient: m
    x2 = v * G * cos_t                                                 # coefficient: m * Crr
    x3 = 0.5 * rho * v_air * v_air.abs() * v                            # coefficient: CdA
    # F_imp_over_m (k = 1) times v -> power/mass; power-law part and bump part kept separate
    x4 = v * G * cos_t * (v + 1.0) ** p_law_index                       # coefficient: m * k
    x5 = v * G * cos_t * torch.exp(-((v - V_ROLL) / SIG_ROLL) ** 2)   # coefficient: m * k * amp

    return torch.stack([x1, x2, x3, x4, x5], dim=-1), P


def _highpass(z):
    """Residual of z about the mean of its two neighbours: removes slow trends, keeps noise. (..., T) -> (..., T-2)"""
    return z[..., 1:-1] - 0.5 * (z[..., :-2] + z[..., 2:])


class PhysicsEmbedding(nn.Module):
    # beta_plain + beta_robust + log_sd (3*K), corr (K(K-1)/2), lag profile, 5 misfit/count numbers, 8 quality numbers
    N_FEATURES = 3 * N_REG + N_REG * (N_REG - 1) // 2 + len(LAGS) + 5 + 8

    def __init__(self, x_mean=None, x_std=None, num_channels=6, output_dim=32,
                 beta_scale=(80.0, 1.0, 0.3, 0.1, 0.1), sigma_p=20.0):
        """
        x_mean, x_std : pass these ONLY if the tensor reaching forward() is z-scored;
                        the net then undoes it. Leave as None if x arrives in physical units.
        beta_scale    : typical size (prior sd) of (m, m*Crr, CdA, m*k, m*k*amp). Sets the ridge
                        strength and keeps features O(1); doesn't need to be precise.
        sigma_p       : nominal power noise [W] used to weight the regression.
        """
        super().__init__()
        self.num_channels = num_channels
        C = num_channels
        self.register_buffer("x_mean", None if x_mean is None else x_mean.reshape(1, 1, 1, C).clone())
        self.register_buffer("x_std", None if x_std is None else x_std.reshape(1, 1, 1, C).clone())
        self.register_buffer("beta_scale", torch.tensor(beta_scale, dtype=torch.float64))
        self.sigma_p = sigma_p
        self.register_buffer("iu", torch.triu_indices(N_REG, N_REG, offset=1), persistent=False)

        self.mlp = nn.Sequential(
            nn.Linear(self.N_FEATURES, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, output_dim),
        )

    # ------------------------------------------------------------------ small helpers
    @staticmethod
    def _ridge(A_r, b_r):
        """Ridge solution with a unit-normal prior on beta' = beta/scale. Batched over leading dims."""
        eye = torch.eye(N_REG, dtype=A_r.dtype, device=A_r.device)
        cov = torch.linalg.inv(A_r + eye)
        return (cov @ b_r.unsqueeze(-1)).squeeze(-1), cov

    @staticmethod
    def _window_rms(beta, A, b, c, Tn):
        """Per-window rms misfit (units of sigma_p) of the ride-level fit beta. A:(B,S,K,K) b:(B,S,K) c:(B,S)."""
        r2 = (c - 2 * torch.einsum('bsi,bi->bs', b, beta)
              + torch.einsum('bi,bsij,bj->bs', beta, A, beta))
        return (r2.clamp(min=0) / Tn).sqrt()

    # ------------------------------------------------------------------ features
    @torch.no_grad()
    def ride_features(self, x):
        """(B, S, T, C+1) -> dict of (B, n) float tensors. No learnable parameters here."""
        B, S, T, _ = x.shape
        C = self.num_channels
        m = (x[..., 0, C] > 0.5).double()                     # (B, S) valid segments
        n_seg = m.sum(1).clamp(min=1)

        x_phys = x[..., :C].double()
        if self.x_mean is not None:                           # undo z-scoring if needed
            x_phys = x_phys * self.x_std.double() + self.x_mean.double()
        P, wind, grad, accel = x_phys[..., 0], x_phys[..., 4], x_phys[..., 3], x_phys[..., 2]

        # ---- 1. per-window normal equations at every candidate lag, in units where beta'=beta/scale, noise=1
        X, y = segment_regressors(x_phys)                     # (B,S,T,K), (B,S,T)
        X = X * self.beta_scale / self.sigma_p
        y = y / self.sigma_p
        y_start = max(0, -min(LAGS))
        Tn = T - y_start - max(0, max(LAGS))                  # common length so all lags are comparable
        y_c = y[:, :, y_start:y_start + Tn]
        c = (y_c ** 2).sum(-1) * m                            # (B, S)
        A_l, b_l = [], []
        for lag in LAGS:                                      # pair P[t] with the regressors at t+lag
            X_l = X[:, :, y_start + lag: y_start + lag + Tn]
            A_l.append(torch.einsum('bsti,bstj->bsij', X_l, X_l) * m[..., None, None])
            b_l.append(torch.einsum('bsti,bst->bsi', X_l, y_c) * m[..., None])
        A_w, b_w = torch.stack(A_l, 1), torch.stack(b_l, 1)   # (B,L,S,K,K), (B,L,S,K)

        # ---- 2. lag scan: plain ridge fit at each lag, keep the lag with the smallest residual
        A_r, b_r = A_w.sum(2), b_w.sum(2)                     # (B,L,K,K), (B,L,K)
        beta_l, _ = self._ridge(A_r, b_r)                     # (B,L,K)
        rss_l = (c.sum(1)[:, None] - 2 * (b_r * beta_l).sum(-1)
                 + torch.einsum('blk,blkj,blj->bl', beta_l, A_r, beta_l))
        best = rss_l.argmin(1)
        lag_profile = torch.log((rss_l + 1) / (rss_l.min(1, keepdim=True).values + 1))   # 0 at the best lag

        ar = torch.arange(B, device=x.device)
        A, b = A_w[ar, best], b_w[ar, best]                   # (B,S,K,K), (B,S,K) at the best lag
        beta_plain, _ = self._ridge(A.sum(1), b.sum(1))

        # ---- 3. robust re-fit (Cauchy-type weights): a window whose misfit is far above the ride's median
        #         (barometer glitch, braking spike, ...) is almost rejected; ordinary windows keep weight ~0.85-1
        beta = beta_plain
        for _ in range(N_IRLS):
            r = self._window_rms(beta, A, b, c, Tn)
            s = r.masked_fill(m == 0, float('nan')).nanmedian(1).values.nan_to_num(1.0).clamp(min=0.1)
            w = m / (1.0 + (r / (ROBUST_C * s[:, None])) ** 2)                      # (B,S), ~1 = trusted, ->0 = reject
            beta, cov = self._ridge((A * w[..., None, None]).sum(1), (b * w[..., None]).sum(1))
        r = self._window_rms(beta, A, b, c, Tn)

        sd = torch.diagonal(cov, dim1=-2, dim2=-1).sqrt()
        corr = (cov / (sd[:, :, None] * sd[:, None, :]))[:, self.iu[0], self.iu[1]]

        misfit = torch.stack([(r * m).sum(1) / n_seg,                                  # mean window misfit
                              (r * m).max(1).values,                                   # worst window misfit
                              ((w < 0.5) & (m > 0)).double().sum(1) / n_seg], dim=1)   # fraction down-weighted
        counts = torch.stack([w.sum(1).clamp(min=1).log(), n_seg.log()], dim=1)       # log N_effective, log N

        # ---- 4. noise / quality numbers (ride level)
        def rms(d):                                           # pooled rms of d (B,S,T') over valid windows
            return (((d ** 2).mean(-1) * m).sum(1) / n_seg).sqrt()

        def autocorr(d, k):
            return ((d[..., :-k] * d[..., k:]).sum(-1) * m).sum(1) / ((d ** 2).sum(-1) * m).sum(1).clamp(min=1e-9)

        d_a = _highpass(accel).clamp(-1.0, 1.0)               # clamp so a few braking spikes don't dominate
        d_p = _highpass(P) / self.sigma_p
        g_rms = _highpass(grad).pow(2).mean(-1).sqrt()        # per-window grade roughness (B,S): glitch indicator
        quality = torch.stack([
            (rms(d_a) + 1e-6).log(),                          # accel noise level
            autocorr(d_a, 1), autocorr(d_a, 2),               # accel noise colour (GPS differencing signature)
            (rms(d_p) + 1e-6).log(),                          # high-frequency power level (meter noise + rider surging)
            ((wind.abs().mean(-1) * m).sum(1) / n_seg) / 5.0,           # mean |wind|
            ((wind.std(-1) * m).sum(1) / n_seg) / 5.0,                  # within-window wind variability (gusts)
            ((g_rms * m).sum(1) / n_seg + 1e-5).log(),                  # mean grade roughness
            ((g_rms * m).max(1).values + 1e-5).log(),                   # roughest window (barometer glitch?)
        ], dim=1)

        feats = {'beta_plain': beta_plain, 'beta_robust': beta, 'log_sd': sd.log(), 'corr': corr,
                 'lag_profile': lag_profile, 'misfit': misfit, 'counts': counts, 'quality': quality}
        return {k: v.float() for k, v in feats.items()}

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return self.mlp(torch.cat(list(self.ride_features(x).values()), dim=1))