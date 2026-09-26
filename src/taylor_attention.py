"""
THOR-PIML — Taylor Series Linear Attention
============================================
Implementação do mecanismo de atenção linear baseado na Expansão de Séries de Taylor
de 2ª Ordem (Qin et al., 2022 / cosFormer & Taylor Expansion).

Estabilidade numérica aprimorada com normalização L2 protegida e complexidade O(N · d²).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def taylor_expansion_2nd_order(x: Tensor, remove_even_power_dups: bool = True) -> Tensor:
    """
    Aplica a expansão em série de Taylor de 2ª ordem para aproximação da exponencial:
        f(x) = 1 + x + (x^2 / 2)
    """
    x_sq = x ** 2
    if remove_even_power_dups:
        out = 1.0 + x + 0.5 * x_sq
    else:
        out = 1.0 + x + x_sq
    return F.relu(out)


class TaylorLinearAttention(nn.Module):
    """
    Módulo de Atenção Linear por Séries de Taylor com estabilidade numérica estrita.
    Aplica normalização L2 nos vetores Query (Q) e Key (K) para delimitar os elementos
    estritamente em [-1, 1], estabilizando os termos polinomiais de Taylor.
    """

    def __init__(
        self,
        dim: int,
        dim_head: int = 16,
        heads: int = 2,
        causal: bool = True,
        dropout: float = 0.0,
        remove_even_power_dups: bool = True,
        prenorm: bool = True,
        order: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.causal = causal
        self.remove_even_power_dups = remove_even_power_dups
        self.prenorm = prenorm
        self.order = order

        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5

        if prenorm:
            self.prenorm_fn = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

        # Inicialização Xavier
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

    def taylor_expansion(self, x: Tensor) -> Tensor:
        """
        Aplica a expansão em série de Taylor de ordem k=1, 2 ou 3 para aproximação da exponencial.
        Envolve com F.relu para garantir não-negatividade do mapa de features.
        """
        if self.order == 1:
            out = 1.0 + x
        elif self.order == 2:
            x_sq = x ** 2
            if self.remove_even_power_dups:
                out = 1.0 + x + 0.5 * x_sq
            else:
                out = 1.0 + x + x_sq
        elif self.order == 3:
            x_sq = x ** 2
            x_cube = x ** 3
            if self.remove_even_power_dups:
                out = 1.0 + x + 0.5 * x_sq + (1.0 / 6.0) * x_cube
            else:
                out = 1.0 + x + x_sq + x_cube
        else:
            raise ValueError(f"Ordem de Taylor não suportada: {self.order}")
        return F.relu(out)

    def forward(self, x: Tensor) -> Tensor:
        if self.prenorm:
            x_in = self.prenorm_fn(x)
        else:
            x_in = x

        b, n, d = x_in.shape
        h = self.heads
        dh = self.dim_head

        q = self.to_q(x_in).view(b, n, h, dh).transpose(1, 2)
        k = self.to_k(x_in).view(b, n, h, dh).transpose(1, 2)
        v = self.to_v(x_in).view(b, n, h, dh).transpose(1, 2)

        # Estabilização via Normalização L2 estrita (F.normalize na dimensão das heads)
        q = torch.nan_to_num(F.normalize(q, p=2, dim=-1, eps=1e-8))
        k = torch.nan_to_num(F.normalize(k, p=2, dim=-1, eps=1e-8))

        # V6 fix: scale era criado mas nunca usado → causa gradientes pequenos
        # Agora aplica scale antes da expansão para calibrar magnitude (como no Transformer original)
        q = q * self.scale

        # Expansão em série de Taylor
        q_exp = self.taylor_expansion(q)
        k_exp = self.taylor_expansion(k)

        if self.causal:
            kv = torch.einsum("b h n i, b h n j -> b h n i j", k_exp, v)
            kv_cum = torch.cumsum(kv, dim=2)
            k_cum = torch.cumsum(k_exp, dim=2)

            num = torch.einsum("b h n i, b h n i j -> b h n j", q_exp, kv_cum)
            den = torch.einsum("b h n i, b h n i -> b h n", q_exp, k_cum).unsqueeze(-1) + 1e-6
            out = num / den
        else:
            kv = torch.einsum("b h n i, b h n j -> b h i j", k_exp, v)
            k_sum = k_exp.sum(dim=2)

            num = torch.einsum("b h n i, b h i j -> b h n j", q_exp, kv)
            den = torch.einsum("b h n i, b h i -> b h n", q_exp, k_sum).unsqueeze(-1) + 1e-6
            out = num / den

        out = torch.nan_to_num(out)
        out = out.transpose(1, 2).reshape(b, n, h * dh)
        return self.to_out(out)

