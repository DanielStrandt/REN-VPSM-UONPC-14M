"""
uo_vpsm_model_ren_v5_scan.py

Final speed-oriented attention-free VPSM language model.

Why v5 exists
-------------
V4 proved that the persistent-state path can be learned and used, but its
Python token loop made real TinyStories training impractical on an RTX 3060.
V5 keeps the successful high-level ideas (trainable fixed reference space,
REN, multiple persistence scales, query-conditioned state reads, fixed-size
state, no attention/KV cache/position embeddings) while replacing the
nonlinear per-token recurrence with an associative affine recurrence that can
be evaluated by a logarithmic-depth parallel prefix scan.

For each layer and token, all expensive projections are computed over the
whole [B,T,D] tensor in parallel. The persistent state update is

    g_t = write_gain * sigmoid(Wg q_t + bias)
    c_t = tanh(Wc q_t)
    a_t = rho * (1 - g_t)
    b_t = g_t * c_t
    s_t = a_t * s_{t-1} + b_t

where rho[k] = exp(-decay_gain[k] / tau[k]). Because 0<=a<=1, 0<=g<=1,
and |c|<=1, a+g<=1 and a state initialized in [-1,1] stays bounded.

The important speed property is that affine transforms compose associatively:

    (a2,b2) o (a1,b1) = (a2*a1, b2 + a2*b1)

so a length-128 recurrence requires seven parallel scan stages instead of 128
Python recurrent steps. Resets are represented by a_t=0. Padding/held state is
represented by the identity transform a_t=1,b_t=0.

V5 uses much sparser initial write priors than the old convex-write prototype,
particularly in slow channels. This preserves the intended physical half-life
until the model learns an event-worthy write. A strong learned gate can still
replace a slow channel immediately.

Default configuration (4096 vocab):
    d_model          256
    n_layers         16
    d_ff             1024
    state            8 x 32 = 256 values/layer
    persistence tau  2,4,8,16,32,64,128,256
    reference rank   32
    REN alpha/groups 0.75 / 8

There is deliberately no:
    - self/cross attention
    - KV cache
    - RoPE/ALiBi/learned/sinusoidal positions
    - token memory bank
    - external retrieval

The only temporal mechanism is the fixed-size VPSM state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

VPSMState = Tuple[torch.Tensor, ...]


@dataclass
class UOVPSMConfig:
    vocab_size: int = 4096
    training_seq_len: int = 128
    d_model: int = 256
    n_layers: int = 16
    d_ff: int = 1024

    n_state_channels: int = 8
    state_dim: int = 32
    persistence_scales: Tuple[float, ...] = (
        2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0,
    )

    reference_rank: int = 32
    reference_gain_init: float = 0.50
    reference_gain_max: float = 2.0

    decay_gain_init: float = 1.0
    decay_gain_max: float = 4.0

    # Sparse event-write priors. Slow channels stay genuinely slow at init.
    write_gate_inits: Tuple[float, ...] = (
        0.10, 0.05, 0.02, 0.008, 0.003, 0.001, 0.0003, 0.0001,
    )
    write_gain_init: float = 0.95
    write_gain_max: float = 1.0

    read_gain_init: float = 1.0
    read_gain_max: float = 2.0
    channel_read_gate_init: float = 0.90
    feature_read_rank: int = 32
    feature_read_gate_init: float = 0.90

    ren_alpha: float = 0.75
    ren_groups: int = 8
    norm_eps: float = 1e-6
    reference_coordinate_detach_scale: bool = True
    reference_coordinate_affine: bool = False
    state_read_use_ren: bool = False
    state_read_detach_scale: bool = False
    state_read_affine: bool = False

    dropout: float = 0.0
    mlp_bias: bool = False
    init_std: float = 0.02
    tie_embeddings: bool = True
    scale_residual_projections: bool = True
    ignore_index: int = -100

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.n_layers <= 0 or self.d_ff <= 0:
            raise ValueError("model dimensions must be positive")
        if self.n_state_channels <= 0 or self.state_dim <= 0:
            raise ValueError("state dimensions must be positive")
        if self.total_state_dim != self.d_model:
            raise ValueError(
                "v5 expects n_state_channels * state_dim == d_model; "
                f"got {self.total_state_dim} != {self.d_model}"
            )
        if len(self.persistence_scales) != self.n_state_channels:
            raise ValueError("persistence_scales must have one value per channel")
        if len(self.write_gate_inits) != self.n_state_channels:
            raise ValueError("write_gate_inits must have one value per channel")
        if any(t <= 0 for t in self.persistence_scales):
            raise ValueError("persistence scales must be positive")
        if any(not (0.0 < p < 1.0) for p in self.write_gate_inits):
            raise ValueError("write gate priors must be in (0,1)")
        if not (0.0 < self.write_gain_init < self.write_gain_max <= 1.0):
            raise ValueError("require 0 < write_gain_init < write_gain_max <= 1")
        if self.d_model % self.ren_groups != 0:
            raise ValueError("d_model must be divisible by ren_groups")

    @property
    def total_state_dim(self) -> int:
        return self.n_state_channels * self.state_dim


class RelativeEnergyNorm(nn.Module):
    """Token-local non-centering grouped L1/RMS energy normalization."""

    def __init__(
        self,
        dim: int,
        *,
        groups: int = 8,
        alpha: float = 0.75,
        eps: float = 1e-6,
        detach_scale: bool = False,
        affine: bool = True,
    ) -> None:
        super().__init__()
        if dim % groups != 0:
            raise ValueError("dim must be divisible by groups")
        self.dim = int(dim)
        self.groups = int(groups)
        self.group_dim = dim // groups
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.detach_scale = bool(detach_scale)
        self.affine = bool(affine)
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected final dim {self.dim}, got {x.shape[-1]}")
        shape = x.shape
        g = x.reshape(*shape[:-1], self.groups, self.group_dim)
        # Compute statistics in fp32; return in input dtype.
        gf = g.float()
        ma = gf.abs().mean(dim=-1, keepdim=True)
        rms = gf.square().mean(dim=-1, keepdim=True).sqrt()
        scale = (1.0 - self.alpha) * ma + self.alpha * rms
        scale = scale + self.eps
        if self.detach_scale:
            scale = scale.detach()
        y = (gf / scale).to(dtype=x.dtype).reshape(shape)
        if self.affine:
            y = y * self.weight.to(dtype=x.dtype) + self.bias.to(dtype=x.dtype)
        return y

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, groups={self.groups}, alpha={self.alpha}, "
            f"eps={self.eps}, detach_scale={self.detach_scale}, affine={self.affine}"
        )


class FeedForward(nn.Module):
    def __init__(self, config: UOVPSMConfig) -> None:
        super().__init__()
        self.fc_in = nn.Linear(config.d_model, config.d_ff, bias=config.mlp_bias)
        self.fc_out = nn.Linear(config.d_ff, config.d_model, bias=config.mlp_bias)
        self.dropout_p = float(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        x = F.gelu(x)
        x = self.fc_out(x)
        if self.dropout_p > 0.0:
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x


class TrainableReferenceLayer(nn.Module):
    def __init__(self, config: UOVPSMConfig) -> None:
        super().__init__()
        self.input_norm = RelativeEnergyNorm(
            config.d_model, groups=config.ren_groups, alpha=config.ren_alpha,
            eps=config.norm_eps, detach_scale=False, affine=True,
        )
        self.fc_in = nn.Linear(config.d_model, config.d_model, bias=config.mlp_bias)
        self.fc_out = nn.Linear(config.d_model, config.d_model, bias=config.mlp_bias)
        self.output_norm = RelativeEnergyNorm(
            config.d_model, groups=config.ren_groups, alpha=config.ren_alpha,
            eps=config.norm_eps, detach_scale=False, affine=True,
        )
        self.dropout_p = float(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc_in(self.input_norm(x))
        y = F.gelu(y)
        y = self.fc_out(y)
        if self.dropout_p > 0.0:
            y = F.dropout(y, p=self.dropout_p, training=self.training)
        return self.output_norm(x + y)


class ReferenceComparator(nn.Module):
    def __init__(self, config: UOVPSMConfig) -> None:
        super().__init__()
        self.gain_max = float(config.reference_gain_max)
        self.current_norm = RelativeEnergyNorm(
            config.d_model, groups=config.ren_groups, alpha=config.ren_alpha,
            eps=config.norm_eps, detach_scale=False, affine=False,
        )
        self.anchor_norm = RelativeEnergyNorm(
            config.d_model, groups=config.ren_groups, alpha=config.ren_alpha,
            eps=config.norm_eps,
            detach_scale=config.reference_coordinate_detach_scale,
            affine=config.reference_coordinate_affine,
        )
        self.current_proj = nn.Linear(config.d_model, config.reference_rank, bias=False)
        self.anchor_proj = nn.Linear(config.d_model, config.reference_rank, bias=False)
        self.out_proj = nn.Linear(config.reference_rank, config.d_model, bias=False)
        ratio = config.reference_gain_init / config.reference_gain_max
        self.raw_gain = nn.Parameter(torch.tensor(math.log(ratio / (1.0 - ratio))))

    def gain(self) -> torch.Tensor:
        return self.gain_max * torch.sigmoid(self.raw_gain)

    def forward(self, current: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        u = self.current_proj(self.current_norm(current))
        v = self.anchor_proj(self.anchor_norm(anchor))
        relation = torch.tanh(u + v + u * v)
        return self.gain().to(dtype=current.dtype) * self.out_proj(relation)


def _affine_prefix_scan(
    a: torch.Tensor,
    b: torch.Tensor,
    state0: torch.Tensor,
) -> torch.Tensor:
    """
    Inclusive parallel scan of s_t = a_t*s_{t-1} + b_t.

    a,b: [B,T,K,S] fp32
    state0: [B,K,S] fp32
    returns state AFTER every token: [B,T,K,S]

    Hillis-Steele takes ceil(log2(T)) Python iterations; every iteration is a
    handful of large elementwise CUDA kernels, not a token-by-token loop.
    """
    if a.shape != b.shape or a.ndim != 4:
        raise ValueError("a and b must have identical [B,T,K,S] shapes")
    if state0.shape != (a.shape[0], a.shape[2], a.shape[3]):
        raise ValueError("state0 shape mismatch")

    A = a
    B = b
    offset = 1
    T = a.shape[1]
    while offset < T:
        # Compose the transform `offset` positions back with the current one.
        # All RHS slices refer to the previous scan stage.
        right_a = A[:, offset:]
        left_a = A[:, :-offset]
        right_b = B[:, offset:]
        left_b = B[:, :-offset]
        tail_a = right_a * left_a
        tail_b = right_b + right_a * left_b
        A = torch.cat((A[:, :offset], tail_a), dim=1)
        B = torch.cat((B[:, :offset], tail_b), dim=1)
        offset <<= 1

    return A * state0.unsqueeze(1) + B


class ScanVPSMCell(nn.Module):
    """Parallel-scan multi-timescale persistent state."""

    def __init__(self, config: UOVPSMConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.K = config.n_state_channels
        self.S = config.state_dim
        self.total = config.total_state_dim
        self.dropout_p = float(config.dropout)
        self.state_read_use_ren = bool(config.state_read_use_ren)
        self.decay_gain_max = float(config.decay_gain_max)
        self.write_gain_max = float(config.write_gain_max)
        self.read_gain_max = float(config.read_gain_max)

        self.register_buffer(
            "persistence_scales",
            torch.tensor(config.persistence_scales, dtype=torch.float32),
            persistent=True,
        )

        # Entire sequence projections run as big GEMMs.
        self.candidate_proj = nn.Linear(config.d_model, self.total, bias=False)
        self.write_gate_proj = nn.Linear(config.d_model, self.total, bias=False)

        write_logits = torch.tensor(
            [math.log(p / (1.0 - p)) for p in config.write_gate_inits],
            dtype=torch.float32,
        ).view(self.K, 1)
        self.write_gate_bias = nn.Parameter(write_logits.expand(self.K, self.S).clone())

        self.channel_selector = nn.Linear(config.d_model, self.K, bias=False)
        ch_logit = math.log(config.channel_read_gate_init / (1.0 - config.channel_read_gate_init))
        self.channel_selector_bias = nn.Parameter(torch.full((self.K,), ch_logit))

        self.feature_query_down = nn.Linear(config.d_model, config.feature_read_rank, bias=False)
        self.feature_query_up = nn.Linear(config.feature_read_rank, self.total, bias=False)
        feat_logit = math.log(config.feature_read_gate_init / (1.0 - config.feature_read_gate_init))
        self.feature_read_bias = nn.Parameter(torch.full((self.K, self.S), feat_logit))

        self.state_read_norm = RelativeEnergyNorm(
            self.total,
            groups=self.K,
            alpha=config.ren_alpha,
            eps=config.norm_eps,
            detach_scale=config.state_read_detach_scale,
            affine=config.state_read_affine,
        )
        self.read_proj = nn.Linear(self.total, config.d_model, bias=False)

        decay_ratio = config.decay_gain_init / config.decay_gain_max
        self.raw_decay_gain = nn.Parameter(
            torch.full((self.K,), math.log(decay_ratio / (1.0 - decay_ratio)))
        )
        write_ratio = config.write_gain_init / config.write_gain_max
        self.raw_write_gain = nn.Parameter(
            torch.full((self.K,), math.log(write_ratio / (1.0 - write_ratio)))
        )
        read_ratio = config.read_gain_init / config.read_gain_max
        self.raw_read_gain = nn.Parameter(
            torch.full((self.K,), math.log(read_ratio / (1.0 - read_ratio)))
        )

    def decay_gains(self) -> torch.Tensor:
        return self.decay_gain_max * torch.sigmoid(self.raw_decay_gain)

    def write_gains(self) -> torch.Tensor:
        return self.write_gain_max * torch.sigmoid(self.raw_write_gain)

    def read_gains(self) -> torch.Tensor:
        return self.read_gain_max * torch.sigmoid(self.raw_read_gain)

    def retention_factors(self) -> torch.Tensor:
        return torch.exp(-self.decay_gains().float() / self.persistence_scales.float())

    def nominal_half_lives(self) -> torch.Tensor:
        return math.log(2.0) * self.persistence_scales.float() / self.decay_gains().float().clamp_min(1e-8)

    def prior_effective_retention(self) -> torch.Tensor:
        prior = torch.sigmoid(self.write_gate_bias.float()).mean(dim=-1)
        strength = self.write_gains().float() * prior
        return self.retention_factors() * (1.0 - strength)

    def prior_effective_half_lives(self) -> torch.Tensor:
        r = self.prior_effective_retention().clamp(1e-8, 1.0 - 1e-8)
        return math.log(0.5) / torch.log(r)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*x.shape[:-1], self.K, self.S)

    def initial_state(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        # Persistent state stays fp32 even during bf16 language-model compute.
        return torch.zeros(batch_size, self.K, self.S, device=device, dtype=torch.float32)

    def _project_update(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate = torch.tanh(self._shape(self.candidate_proj(q)))
        logits = self._shape(self.write_gate_proj(q))
        logits = logits + self.write_gate_bias.to(device=q.device, dtype=q.dtype)
        gate = torch.sigmoid(logits)
        gain = self.write_gains().to(device=q.device, dtype=q.dtype).view(1, 1, self.K, 1)
        strength = gate * gain
        return candidate, gate, strength

    def _read_sequence(
        self,
        q: torch.Tensor,
        state_before: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        channel = torch.sigmoid(
            self.channel_selector(q)
            + self.channel_selector_bias.to(device=q.device, dtype=q.dtype)
        )
        h = F.silu(self.feature_query_down(q))
        feature = torch.sigmoid(
            self._shape(self.feature_query_up(h))
            + self.feature_read_bias.to(device=q.device, dtype=q.dtype)
        )
        gain = self.read_gains().to(device=q.device, dtype=q.dtype).view(1, 1, self.K, 1)

        state_coord = state_before.to(dtype=q.dtype)
        if self.state_read_use_ren:
            flat = state_coord.reshape(*state_coord.shape[:-2], self.total)
            state_coord = self.state_read_norm(flat).reshape_as(state_coord)

        weighted = state_coord * gain * channel.unsqueeze(-1) * feature
        memory = self.read_proj(weighted.reshape(*weighted.shape[:-2], self.total))
        if self.dropout_p > 0.0:
            memory = F.dropout(memory, p=self.dropout_p, training=self.training)
        return memory, channel, feature

    def forward_sequence(
        self,
        q: torch.Tensor,
        state0: torch.Tensor,
        *,
        reset_mask: Optional[torch.Tensor] = None,
        update_mask: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        if q.ndim != 3:
            raise ValueError("q must be [B,T,D]")
        B, T, _ = q.shape
        if state0.shape != (B, self.K, self.S):
            raise ValueError("state0 shape mismatch")
        if reset_mask is not None and reset_mask.shape != (B, T):
            raise ValueError("reset_mask must be [B,T]")
        if update_mask is not None and update_mask.shape != (B, T):
            raise ValueError("update_mask must be [B,T]")

        candidate, gate, strength = self._project_update(q)
        rho = self.retention_factors().to(device=q.device).view(1, 1, self.K, 1)

        # Scan coefficients in fp32 so the 128/256-token channels retain their
        # intended sub-percent decay precision under bf16 autocast.
        wf = strength.float()
        a = rho * (1.0 - wf)
        b = wf * candidate.float()

        if update_mask is not None:
            u = update_mask.to(device=q.device, dtype=torch.bool).view(B, T, 1, 1)
            a = torch.where(u, a, torch.ones_like(a))
            b = torch.where(u, b, torch.zeros_like(b))

        if reset_mask is not None:
            r = reset_mask.to(device=q.device, dtype=torch.bool).view(B, T, 1, 1)
            # Reset happens before this token. Its write still occurs unless the
            # update mask also disables it.
            a = torch.where(r, torch.zeros_like(a), a)

        after = _affine_prefix_scan(a, b, state0.float())

        # READ-BEFORE-WRITE: token t sees state after token t-1. A reset at t
        # forces the read state to zero before token t is processed.
        before = torch.cat((state0.float().unsqueeze(1), after[:, :-1]), dim=1)
        if reset_mask is not None:
            r = reset_mask.to(device=q.device, dtype=torch.bool).view(B, T, 1, 1)
            before = torch.where(r, torch.zeros_like(before), before)

        memory, channel, feature = self._read_sequence(q, before)
        final_state = after[:, -1]

        if not return_diagnostics:
            return memory, final_state, None

        diag = {
            "mean_write_gate": gate.float().mean(dim=-1),
            "mean_write_strength": strength.float().mean(dim=-1),
            "selector": channel.float(),
            "mean_feature_gate": feature.float().mean(dim=-1),
            "state_rms": before.square().mean(dim=-1).sqrt(),
        }
        return memory, final_state, diag

    def step(
        self,
        q: torch.Tensor,
        state: torch.Tensor,
        *,
        update_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if q.ndim != 2:
            raise ValueError("q must be [B,D]")
        B = q.shape[0]
        before = state.float()
        memory, channel, feature = self._read_sequence(q.unsqueeze(1), before.unsqueeze(1))
        candidate, gate, strength = self._project_update(q.unsqueeze(1))
        rho = self.retention_factors().to(device=q.device).view(1, 1, self.K, 1)
        wf = strength.float()
        proposed = rho * (1.0 - wf) * before.unsqueeze(1) + wf * candidate.float()
        proposed = proposed[:, 0]
        if update_mask is not None:
            m = update_mask.to(device=q.device, dtype=torch.bool).view(B, 1, 1)
            proposed = torch.where(m, proposed, before)
        return memory[:, 0], proposed, {
            "write_gate": gate[:, 0],
            "write_strength": strength[:, 0],
            "selector": channel[:, 0],
            "feature_gate": feature[:, 0],
        }


class UOVPSMBlock(nn.Module):
    def __init__(self, config: UOVPSMConfig) -> None:
        super().__init__()
        self.state_norm = RelativeEnergyNorm(
            config.d_model, groups=config.ren_groups, alpha=config.ren_alpha,
            eps=config.norm_eps, detach_scale=False, affine=True,
        )
        self.reference_comparator = ReferenceComparator(config)
        self.vpsm = ScanVPSMCell(config)
        self.mlp_norm = RelativeEnergyNorm(
            config.d_model, groups=config.ren_groups, alpha=config.ren_alpha,
            eps=config.norm_eps, detach_scale=False, affine=True,
        )
        self.mlp = FeedForward(config)

    def initial_state(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        return self.vpsm.initial_state(batch_size, device=device)

    def forward_sequence(
        self,
        x: torch.Tensor,
        anchor: torch.Tensor,
        state: torch.Tensor,
        *,
        reset_mask: Optional[torch.Tensor] = None,
        update_mask: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        z = self.state_norm(x)
        ref = self.reference_comparator(x, anchor)
        q = z + ref
        memory, new_state, diag = self.vpsm.forward_sequence(
            q,
            state,
            reset_mask=reset_mask,
            update_mask=update_mask,
            return_diagnostics=return_diagnostics,
        )
        x = x + memory
        x = x + self.mlp(self.mlp_norm(x))
        if diag is not None:
            diag["reference_context_rms"] = ref.float().square().mean(dim=-1).sqrt()
            diag["reference_gain"] = self.reference_comparator.gain()
        return x, new_state, diag

    def step(
        self,
        x: torch.Tensor,
        anchor: torch.Tensor,
        state: torch.Tensor,
        *,
        update_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        z = self.state_norm(x)
        ref = self.reference_comparator(x, anchor)
        q = z + ref
        memory, new_state, diag = self.vpsm.step(q, state, update_mask=update_mask)
        x = x + memory
        x = x + self.mlp(self.mlp_norm(x))
        diag["reference_context_rms"] = ref.float().square().mean(dim=-1).sqrt()
        return x, new_state, diag


class UOVPSMModel(nn.Module):
    def __init__(self, config: Optional[UOVPSMConfig] = None) -> None:
        super().__init__()
        self.config = config or UOVPSMConfig()
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.reference_layer = TrainableReferenceLayer(self.config)
        self.blocks = nn.ModuleList([UOVPSMBlock(self.config) for _ in range(self.config.n_layers)])
        self.final_norm = RelativeEnergyNorm(
            self.config.d_model, groups=self.config.ren_groups,
            alpha=self.config.ren_alpha, eps=self.config.norm_eps,
            detach_scale=False, affine=True,
        )
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if self.config.scale_residual_projections:
            self._scale_residual_output_projections()
        if self.config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def _scale_residual_output_projections(self) -> None:
        scale = 1.0 / math.sqrt(2.0 * self.config.n_layers)
        with torch.no_grad():
            self.reference_layer.fc_out.weight.mul_(1.0 / math.sqrt(2.0))
            for block in self.blocks:
                block.vpsm.read_proj.weight.mul_(scale)
                block.mlp.fc_out.weight.mul_(scale)

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> VPSMState:
        del dtype  # state is intentionally fp32 for slow-channel precision
        device = device or self.token_embedding.weight.device
        return tuple(block.initial_state(batch_size, device=device) for block in self.blocks)

    def _prepare_state(self, state: VPSMState, batch_size: int, device: torch.device) -> VPSMState:
        if len(state) != self.config.n_layers:
            raise ValueError("state must contain one tensor per layer")
        expected = (batch_size, self.config.n_state_channels, self.config.state_dim)
        out = []
        for i, s in enumerate(state):
            if tuple(s.shape) != expected:
                raise ValueError(f"state[{i}] expected {expected}, got {tuple(s.shape)}")
            out.append(s.to(device=device, dtype=torch.float32))
        return tuple(out)

    @staticmethod
    def detach_state(state: Optional[VPSMState]) -> Optional[VPSMState]:
        if state is None:
            return None
        return tuple(s.detach() for s in state)

    @staticmethod
    def clone_state(state: VPSMState) -> VPSMState:
        return tuple(s.clone() for s in state)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        *,
        state: Optional[VPSMState] = None,
        reset_mask: Optional[torch.Tensor] = None,
        update_mask: Optional[torch.Tensor] = None,
        return_state: bool = False,
        return_hidden: bool = False,
        return_diagnostics: bool = False,
        return_reference: bool = False,
    ):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B,T]")
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        B, T = input_ids.shape
        if T <= 0:
            raise ValueError("sequence length must be positive")
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets shape mismatch")
        if reset_mask is not None and reset_mask.shape != input_ids.shape:
            raise ValueError("reset_mask shape mismatch")
        if update_mask is not None and update_mask.shape != input_ids.shape:
            raise ValueError("update_mask shape mismatch")

        embedded = self.token_embedding(input_ids)
        reference = self.reference_layer(embedded)
        x = reference
        current = self.initial_state(B, device=x.device) if state is None else self._prepare_state(state, B, x.device)

        final_states = []
        diagnostics = [] if return_diagnostics else None
        for i, block in enumerate(self.blocks):
            x, s, diag = block.forward_sequence(
                x, reference, current[i],
                reset_mask=reset_mask,
                update_mask=update_mask,
                return_diagnostics=return_diagnostics,
            )
            final_states.append(s)
            if return_diagnostics:
                diagnostics.append(diag)

        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=self.config.ignore_index,
            )

        if not (return_state or return_hidden or return_diagnostics or return_reference):
            return logits if targets is None else (logits, loss)

        out: Dict[str, Any] = {"logits": logits}
        if loss is not None:
            out["loss"] = loss
        if return_state:
            out["state"] = tuple(final_states)
        if return_hidden:
            out["hidden_states"] = hidden
        if return_diagnostics:
            out["diagnostics"] = diagnostics
        if return_reference:
            out["reference_states"] = reference
        return out

    def step(
        self,
        input_ids: torch.Tensor,
        *,
        state: Optional[VPSMState] = None,
        update_mask: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Dict[str, Any]:
        if input_ids.ndim == 2:
            if input_ids.shape[1] != 1:
                raise ValueError("step expects [B] or [B,1]")
            input_ids = input_ids[:, 0]
        if input_ids.ndim != 1:
            raise ValueError("step expects [B] or [B,1]")
        B = input_ids.shape[0]
        embedded = self.token_embedding(input_ids.long())
        reference = self.reference_layer(embedded)
        x = reference
        current = self.initial_state(B, device=x.device) if state is None else self._prepare_state(state, B, x.device)
        final_states = []
        diagnostics = [] if return_diagnostics else None
        for i, block in enumerate(self.blocks):
            x, s, diag = block.step(x, reference, current[i], update_mask=update_mask)
            final_states.append(s)
            if return_diagnostics:
                diagnostics.append(diag)
        logits = self.lm_head(self.final_norm(x))
        out: Dict[str, Any] = {"logits": logits, "state": tuple(final_states)}
        if return_diagnostics:
            out["diagnostics"] = diagnostics
        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 40,
        eos_token_id: Optional[int] = None,
        forbidden_token_ids: Tuple[int, ...] = (),
        state: Optional[VPSMState] = None,
    ) -> Dict[str, Any]:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        out = self.forward(input_ids, state=state, return_state=True)
        current_state = self.detach_state(out["state"])
        logits = out["logits"][:, -1]
        generated = [input_ids]
        for _ in range(max_new_tokens):
            scores = logits.float().clone()
            if forbidden_token_ids:
                scores[:, list(forbidden_token_ids)] = -float("inf")
            if temperature <= 0:
                nxt = scores.argmax(dim=-1)
            else:
                scores = scores / temperature
                if 0 < top_k < scores.shape[-1]:
                    vals, _ = torch.topk(scores, top_k, dim=-1)
                    scores = scores.masked_fill(scores < vals[:, -1:].expand_as(scores), -float("inf"))
                nxt = torch.multinomial(F.softmax(scores, dim=-1), 1)[:, 0]
            generated.append(nxt[:, None])
            step_out = self.step(nxt, state=current_state)
            current_state = self.detach_state(step_out["state"])
            logits = step_out["logits"]
            if eos_token_id is not None and bool(torch.all(nxt.eq(eos_token_id))):
                break
        return {"sequences": torch.cat(generated, dim=1), "state": current_state}

    def persistence_summary(self) -> Dict[str, torch.Tensor]:
        nominal = []
        effective = []
        retention = []
        for block in self.blocks:
            nominal.append(block.vpsm.nominal_half_lives().detach().cpu())
            effective.append(block.vpsm.prior_effective_half_lives().detach().cpu())
            retention.append(block.vpsm.retention_factors().detach().cpu())
        return {
            "nominal_half_lives": torch.stack(nominal),
            "prior_effective_half_lives": torch.stack(effective),
            "retention_factors": torch.stack(retention),
        }


def build_uo_vpsm(
    *,
    vocab_size: int = 4096,
    training_seq_len: int = 128,
    **overrides: Any,
) -> UOVPSMModel:
    cfg = UOVPSMConfig(vocab_size=vocab_size, training_seq_len=training_seq_len, **overrides)
    return UOVPSMModel(cfg)


def build_uo_vpsm_small(*, vocab_size: int = 4096, training_seq_len: int = 128) -> UOVPSMModel:
    cfg = UOVPSMConfig(
        vocab_size=vocab_size,
        training_seq_len=training_seq_len,
        d_model=128,
        n_layers=8,
        d_ff=512,
        n_state_channels=8,
        state_dim=16,
        reference_rank=16,
        feature_read_rank=16,
        ren_groups=8,
    )
    return UOVPSMModel(cfg)


if __name__ == "__main__":
    torch.manual_seed(7)
    model = build_uo_vpsm()
    print(model)
    print(f"\nTrainable parameters: {model.num_parameters():,}")
    print(f"Recommended training chunk: {model.config.training_seq_len}")
    print(f"State values/sample total: {model.config.n_layers * model.config.total_state_dim}")
    p = model.persistence_summary()
    print("Nominal half-lives, layer 0:")
    print(p["nominal_half_lives"][0])
    print("Prior effective half-lives, layer 0:")
    print(p["prior_effective_half_lives"][0])

    # Small functional smoke test on the full architecture.
    ids = torch.randint(0, model.config.vocab_size, (2, 16))
    targets = torch.randint(0, model.config.vocab_size, (2, 16))
    out = model(ids, targets, return_state=True)
    out["loss"].backward()
    print(f"Logits shape: {tuple(out['logits'].shape)}")
    print(f"Loss: {float(out['loss']):.6f}")
    print(f"State dtype: {out['state'][0].dtype}")
    print(f"State finite: {all(bool(torch.isfinite(s).all()) for s in out['state'])}")

    # Scan path and one-token recurrent path should agree up to floating point
    # association order.
    model.eval()
    with torch.no_grad():
        full = model(ids, return_state=True)
        state = None
        pieces = []
        for t in range(ids.shape[1]):
            one = model.step(ids[:, t], state=state)
            pieces.append(one["logits"][:, None])
            state = one["state"]
        serial_logits = torch.cat(pieces, dim=1)
        max_diff = (full["logits"] - serial_logits).abs().max().item()
        state_diff = max((a - b).abs().max().item() for a, b in zip(full["state"], state))
    print(f"Parallel-scan vs serial logits max |diff|: {max_diff:.8f}")
    print(f"Parallel-scan vs serial state max |diff|:  {state_diff:.8f}")
