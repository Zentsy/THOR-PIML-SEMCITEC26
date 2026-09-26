"""THOR-PIML V7 — arquitetura híbrida LSTM+TCN (branch hybrid-arch-test).

Módulos:
- config_v7       — features dinâmicas (V2/V3), modelo híbrido, loss V7, OneCycle
- model_v7        — THORHybridModel (ResBiLSTM || TCN multi-escala → gated fusion → SDPA → hurdle)
- physics_loss_v7 — V6d balanceada + extremes-recall + variance matching
- pipeline_v7     — zero-leakage + CV temporal bloqueada (teste cego intacto)
- train_v7        — OneCycleLR, AMP, seleção por val KGE
- run_v7          — CLI: python -m src.v7.run --model hybrid --cv 5
"""
