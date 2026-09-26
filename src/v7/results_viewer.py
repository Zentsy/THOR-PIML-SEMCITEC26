"""
THOR-PIML — Dashboard de Resultados da era híbrida
===================================================
Lê results/experiments_v7.jsonl (cada treino V7/V8 appenda 1 linha — ver
src.v7.run.log_experiment) e gera results/DASHBOARD.html:

  - tabela única de TODAS as runs (parte V7/V8/ablações diferenciadas por cor/campo)
  - linha de referência V6d (resultado histórico oficial, commit bd86455)
  - cada célula colorida vs baseline V6d (verde = melhor, vermelho = pior)
  - filtros (parte / dados / texto) e ordenação por qualquer coluna (clique no título)
  - cards de destaque: melhor KGE, melhor R20 recall, melhor QB99, última run

Uso:
    python -m src.v7.results_viewer                 # lê padrão, gera results/DASHBOARD.html
    python -m src.v7.results_viewer --open          # abre no navegador (se houver GUI)
Self-contained: um único HTML, sem CDN/dependências.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.paths import RESULTS_DIR

DEFAULT_JSONL = RESULTS_DIR / "experiments_v7.jsonl"
DEFAULT_OUT = RESULTS_DIR / "DASHBOARD.html"

# Referência histórica V6d — teste cego 2019-09→2026-06 (docs/ESTADO_ATUAL_FINAL_V6.md)
V6D_BASELINE = {
    "ts": "2026-08 (histórico)", "parte": "v6d", "variant": "resbilstm", "loss": "v6d",
    "data": "v2", "seed": "—", "ckpt": "best_model.pt (bd86455)",
    "metrics": {
        "nse": -0.18, "kge": 0.11, "rmse": 7.17, "mae": 4.69, "bias": 2.05,
        "f1_occ": 0.5469, "accuracy_occ": 0.7033, "brier_score": 0.204, "roc_auc": 0.764,
        "sdii_obs": 8.38, "sdii_pred": 8.01, "qb95_bias_pct": -10.34, "qb99_bias_pct": -36.42,
        "r10mm_obs": 287, "r10mm_pred": 610, "r20mm_obs": 100, "r20mm_pred": 13,
        "cwd_obs": 29, "cwd_pred": 210, "cdd_obs": None, "cdd_pred": None,
    },
}


def load_runs(jsonl_path: Path) -> list[dict]:
    runs = [dict(V6D_BASELINE)]  # baseline sempre presente como linha 0
    if not jsonl_path.exists():
        return runs
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[viewer] ⚠ linha inválida ignorada: {line[:80]}...")
    return runs


def _cls_good(v: float | None) -> str:
    return "" if v is None else ("good" if v else "bad")


def build_row(run: dict, base_m: dict) -> dict:
    """Deriva métricas compostas + classes de cor vs baseline."""
    m = run.get("metrics", {})
    cv = run.get("cv")
    is_base = run.get("parte") == "v6d"

    def g(k):
        return m.get(k)

    r20_recall = (g("r20mm_pred") / max(g("r20mm_obs"), 1) * 100) if g("r20mm_obs") else None
    r10_ratio = (g("r10mm_pred") / g("r10mm_obs")) if g("r10mm_obs") else None
    cwd_ratio = (g("cwd_pred") / max(g("cwd_obs"), 1)) if g("cwd_obs") else None
    sdii_d = (g("sdii_pred") - g("sdii_obs")) if g("sdii_pred") is not None else None

    # ---- classes vs baseline (verde=melhor que V6d, vermelho=pior, vazio=neutro/sem dado)
    c = {}
    if not is_base:
        c["kge"] = _cls_good(None if g("kge") is None else g("kge") - base_m["kge"] > 0.05)
        c["nse"] = _cls_good(None if g("nse") is None else g("nse") - base_m["nse"] > 0.05)
        c["rmse"] = _cls_good(None if g("rmse") is None else g("rmse") < base_m["rmse"] - 0.3)
        c["mae"] = _cls_good(None if g("mae") is None else g("mae") < base_m["mae"] - 0.3)
        c["bias"] = _cls_good(None if g("bias") is None else abs(g("bias")) < abs(base_m["bias"]) - 0.3)
        c["f1"] = _cls_good(None if g("f1_occ") is None else g("f1_occ") - base_m["f1_occ"] > 0.02)
        c["qb95"] = _cls_good(None if g("qb95_bias_pct") is None
                              else abs(g("qb95_bias_pct")) < abs(base_m["qb95_bias_pct"]) - 5)
        c["qb99"] = _cls_good(None if g("qb99_bias_pct") is None
                              else abs(g("qb99_bias_pct")) < abs(base_m["qb99_bias_pct"]) - 5)
        c["r20"] = "good" if (r20_recall is not None and r20_recall > 18) else (
            "bad" if (r20_recall is not None and r20_recall < 13) else "")
        if r20_recall is not None and r20_recall > 150:
            c["r20"] = "over"  # over-shoot (fantasma V6c): recall alto mas viciado
        c["r10"] = "good" if (r10_ratio is not None and 0.7 <= r10_ratio <= 1.4) else (
            "bad" if (r10_ratio is not None and (r10_ratio > 2.0 or r10_ratio < 0.4)) else "")
        c["cwd"] = "good" if (cwd_ratio is not None and cwd_ratio <= 1.5) else (
            "bad" if (cwd_ratio is not None and cwd_ratio >= 3.0) else "")
        c["sdii"] = "good" if (sdii_d is not None and abs(sdii_d) <= 1.0) else (
            "bad" if (sdii_d is not None and abs(sdii_d) >= 2.0) else "")

    cv_str = f"{cv['kge_mean']:.3f}±{cv['kge_std']:.3f}" if cv else "—"
    r10_str = f"{g('r10mm_obs')}/{g('r10mm_pred')}" if g("r10mm_obs") is not None else "—"
    r20_str = f"{g('r20mm_obs')}/{g('r20mm_pred')}" if g("r20mm_obs") is not None else "—"
    cwd_str = f"{g('cwd_obs')}/{g('cwd_pred')}" if g("cwd_obs") is not None else "—"
    sdii_str = f"{g('sdii_obs'):.2f}/{g('sdii_pred'):.2f}" if g("sdii_pred") is not None else "—"

    return {
        "ts": run.get("ts", "?"),
        "parte": run.get("parte", "?"),
        "variant": run.get("variant", "?"),
        "loss": run.get("loss", "?") + (f" (λphys {run['physics_lambda']})" if run.get("physics_lambda") is not None else ""),
        "data": run.get("data", "?"),
        "seed": str(run.get("seed", "—")),
        "cv": cv_str,
        "cv_sort": cv["kge_mean"] if cv else -99,
        "nse": f"{g('nse'):.3f}" if g("nse") is not None else "—",
        "kge": f"{g('kge'):.3f}" if g("kge") is not None else "—",
        "rmse": f"{g('rmse'):.2f}" if g("rmse") is not None else "—",
        "mae": f"{g('mae'):.2f}" if g("mae") is not None else "—",
        "bias": f"{g('bias'):+.2f}" if g("bias") is not None else "—",
        "f1": f"{g('f1_occ'):.3f}" if g("f1_occ") is not None else "—",
        "r20": f"{r20_recall:.0f}%" if r20_recall is not None else "—",
        "qb95": f"{g('qb95_bias_pct'):+.1f}%" if g("qb95_bias_pct") is not None else "—",
        "qb99": f"{g('qb99_bias_pct'):+.1f}%" if g("qb99_bias_pct") is not None else "—",
        "r10s": r10_str, "r20s": r20_str, "cwds": cwd_str, "sdiis": sdii_str,
        "r20_sort": r20_recall if r20_recall is not None else -99,
        "qb99_sort": -abs(g("qb99_bias_pct")) if g("qb99_bias_pct") is not None else -99,
        "ckpt": run.get("ckpt", "—"),
        "c": c,
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>THOR-PIML — Dashboard de Resultados (V7/V8)</title>
<style>
  :root{{--bg:#0b1220;--card:#151f33;--card2:#1a2740;--line:#263a5c;--ink:#e8eefc;
    --muted:#93a5c8;--v7:#14b8a6;--v8:#a78bfa;--v6d:#94a3b8;--ok:#4ade80;--bad:#f87171;
    --over:#fb923c;--blue:#60a5fa}}
  *{{box-sizing:border-box}}
  body{{margin:0;background:radial-gradient(1000px 500px at 85% -10%,rgba(139,92,246,.12),transparent 60%),
    radial-gradient(800px 400px at -10% 10%,rgba(20,184,166,.10),transparent 55%),var(--bg);
    color:var(--ink);font-family:'Segoe UI',system-ui,sans-serif;font-size:14.5px;line-height:1.5}}
  .wrap{{max-width:1280px;margin:0 auto;padding:24px 18px 70px}}
  h1{{font-size:1.45em;margin:0 0 2px}}
  .sub{{color:var(--muted);margin:0 0 18px;font-size:.9em}}
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:18px}}
  .hl{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}}
  .hl .t{{font-size:.72em;letter-spacing:1px;color:var(--muted);text-transform:uppercase}}
  .hl .v{{font-size:1.35em;font-weight:800;margin-top:2px}}
  .hl .d{{font-size:.8em;color:var(--muted)}}
  .bar{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}}
  .bar input,.bar select{{background:var(--card2);border:1px solid var(--line);color:var(--ink);
    border-radius:8px;padding:8px 12px;font-size:.92em}}
  .bar input{{min-width:240px}}
  .tblwrap{{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:12px}}
  table{{border-collapse:collapse;width:100%;font-size:.86em;white-space:nowrap}}
  th{{background:rgba(96,165,250,.10);color:#bfdbfe;padding:9px 10px;text-align:left;cursor:pointer;
    user-select:none;position:sticky;top:0}}
  th:hover{{background:rgba(96,165,250,.2)}}
  td{{border-top:1px solid var(--line);padding:7px 10px}}
  tr:hover td{{background:rgba(255,255,255,.03)}}
  .p{{font-weight:800;border-radius:5px;padding:1px 8px;font-size:.82em}}
  .p.v7{{color:var(--v7);background:rgba(20,184,166,.12)}}
  .p.v8{{color:var(--v8);background:rgba(167,139,250,.14)}}
  .p.v6d{{color:var(--v6d);background:rgba(148,163,184,.14)}}
  td.good{{color:var(--ok);font-weight:700}}
  td.bad{{color:var(--bad)}}
  td.over{{color:var(--over);font-weight:700}}
  tr.baseline td{{background:rgba(148,163,184,.07);font-style:italic}}
  .foot{{color:var(--muted);font-size:.82em;margin-top:16px}}
  .legend span{{margin-right:14px}}
  .dot{{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:middle}}
</style>
</head>
<body><div class="wrap">
<h1>THOR-PIML — Dashboard de Resultados</h1>
<p class="sub">Teste cego 2019-09→2026-06 (idêntico para todas as runs) · gerado em {gen_ts} ·
{n_runs} runs registradas · fonte: <code>results/experiments_v7.jsonl</code></p>

<div class="cards">{highlight_cards}</div>

<div class="legend" style="margin-bottom:10px;color:var(--muted);font-size:.85em">
  <span><span class="dot" style="background:var(--ok)"></span>melhor que V6d</span>
  <span><span class="dot" style="background:var(--bad)"></span>pior que V6d</span>
  <span><span class="dot" style="background:var(--over)"></span>alerta over-shoot (fantasma V6c)</span>
  <span>clique nos títulos para ordenar</span>
</div>

<div class="bar">
  <input id="q" type="text" placeholder="filtrar por texto (ex.: hybrid, seed 42, v3...)">
  <select id="fparte"><option value="">todas as partes</option><option>v7</option><option>v8</option><option>v6d</option></select>
  <select id="fdata"><option value="">todos os dados</option><option>v2</option><option>v3</option></select>
</div>

<div class="tblwrap"><table id="tbl"><thead><tr>
{head_cells}
</tr></thead><tbody id="tb"></tbody></table></div>

<p class="foot">Cada treino (V7 ou V8) appenda 1 linha automaticamente. Para atualizar este painel:
<code>python -m src.v7.results_viewer</code>. A linha <b>V6d</b> é a referência histórica oficial
(commit bd86455, docs/ESTADO_ATUAL_FINAL_V6.md). R20 recall em % ; R10/R20/CWD no formato obs/pred.</p>
</div>
<script>
const ROWS = {rows_json};
const HEAD = {head_json};
const tb = document.getElementById('tb');
function fmt(r, key) {{
  const v = r[key];
  return (v === null || v === undefined) ? '—' : v;
}}
function render() {{
  const q = document.getElementById('q').value.toLowerCase();
  const fp = document.getElementById('fparte').value;
  const fd = document.getElementById('fdata').value;
  const rows = ROWS.filter(r => {{
    if (fp && r.parte !== fp) return false;
    if (fd && r.data !== fd) return false;
    if (q) {{
      const blob = (r.ts + r.parte + r.variant + r.loss + r.data + r.seed + r.ckpt).toLowerCase();
      if (!blob.includes(q)) return false;
    }}
    return true;
  }});
  tb.innerHTML = rows.map(r => {{
    const c = r.c || {{}};
    const cls = k => c[k] ? ' class="' + c[k] + '"' : '';
    return '<tr' + (r.parte === 'v6d' ? ' class="baseline"' : '') + '>'
      + '<td>' + r.ts + '</td>'
      + '<td><span class="p ' + r.parte + '">' + r.parte + '</span></td>'
      + '<td>' + r.variant + '</td><td>' + r.loss + '</td><td>' + r.data + '</td><td>' + r.seed + '</td>'
      + '<td>' + r.cv + '</td>'
      + '<td' + cls('nse') + '>' + r.nse + '</td>'
      + '<td' + cls('kge') + '>' + r.kge + '</td>'
      + '<td' + cls('rmse') + '>' + r.rmse + '</td>'
      + '<td' + cls('mae') + '>' + r.mae + '</td>'
      + '<td' + cls('bias') + '>' + r.bias + '</td>'
      + '<td' + cls('f1') + '>' + r.f1 + '</td>'
      + '<td' + cls('r20') + '>' + r.r20 + '</td>'
      + '<td' + cls('qb95') + '>' + r.qb95 + '</td>'
      + '<td' + cls('qb99') + '>' + r.qb99 + '</td>'
      + '<td' + cls('r10') + '>' + r.r10s + '</td>'
      + '<td' + cls('r20') + '>' + r.r20s + '</td>'
      + '<td' + cls('cwd') + '>' + r.cwds + '</td>'
      + '<td' + cls('sdii') + '>' + r.sdiis + '</td>'
      + '<td style="color:var(--muted)">' + r.ckpt + '</td></tr>';
  }}).join('');
}}
let sortState = {{idx: -1, asc: true}};
document.querySelectorAll('#tbl th').forEach((th, i) => {{
  th.addEventListener('click', () => {{
    const key = HEAD[i].k;
    if (!key) return;
    const asc = sortState.idx === i ? !sortState.asc : true;
    sortState = {{idx: i, asc}};
    ROWS.sort((a, b) => {{
      const av = a[key], bv = b[key];
      const an = parseFloat(av), bn = parseFloat(bv);
      const cmp = (!isNaN(an) && !isNaN(bn)) ? an - bn : String(av).localeCompare(String(bv));
      return asc ? cmp : -cmp;
    }});
    render();
  }});
}});
document.getElementById('q').addEventListener('input', render);
document.getElementById('fparte').addEventListener('change', render);
document.getElementById('fdata').addEventListener('change', render);
render();
</script>
</body></html>
"""

COLUMNS = [
    {"t": "Data", "k": "ts"},
    {"t": "Parte", "k": "parte"},
    {"t": "Variante", "k": "variant"},
    {"t": "Loss", "k": "loss"},
    {"t": "Dados", "k": "data"},
    {"t": "Seed", "k": "seed"},
    {"t": "CV KGE", "k": "cv_sort"},
    {"t": "NSE", "k": "nse"},
    {"t": "KGE", "k": "kge"},
    {"t": "RMSE", "k": "rmse"},
    {"t": "MAE", "k": "mae"},
    {"t": "Bias", "k": "bias"},
    {"t": "F1", "k": "f1"},
    {"t": "R20 recall", "k": "r20_sort"},
    {"t": "QB95", "k": "qb95"},
    {"t": "QB99", "k": "qb99_sort"},
    {"t": "R10 o/p", "k": "r10s"},
    {"t": "R20 o/p", "k": "r20s"},
    {"t": "CWD o/p", "k": "cwds"},
    {"t": "SDII o/p", "k": "sdiis"},
    {"t": "Checkpoint", "k": "ckpt"},
]


def generate_dashboard(jsonl_path: Path, out_path: Path) -> Path:
    runs = load_runs(jsonl_path)
    base_m = V6D_BASELINE["metrics"]
    rows = [build_row(r, base_m) for r in runs]

    real = [r for r in rows if r["parte"] != "v6d"]

    def card(title, value, desc):
        return (f'<div class="hl"><div class="t">{title}</div>'
                f'<div class="v">{value}</div><div class="d">{desc}</div></div>')

    cards = []
    if real:
        best_kge = max(real, key=lambda r: float(r["kge"]) if r["kge"] != "—" else -99)
        cards.append(card("Melhor KGE (teste cego)", best_kge["kge"],
                          f"{best_kge['parte']}/{best_kge['variant']} · loss {best_kge['loss']} · seed {best_kge['seed']}"))
        best_r20 = max(real, key=lambda r: r["r20_sort"])
        cards.append(card("Melhor R20 recall", best_r20["r20"],
                          f"{best_r20['parte']}/{best_r20['variant']} · (baseline V6d: 13%)"))
        best_qb = max(real, key=lambda r: r["qb99_sort"])
        cards.append(card("QB99 mais próximo de 0", best_qb["qb99"],
                          f"{best_qb['parte']}/{best_qb['variant']} · (baseline: −36.4%)"))
        last = real[-1]
        cards.append(card("Última run", last["ts"],
                          f"{last['parte']}/{last['variant']} · {last['ckpt']}"))
    else:
        cards.append(card("Nenhuma run ainda", "—",
                          "rode um treino e depois python -m src.v7.results_viewer"))

    head_cells = "\n".join(f'<th data-k="{c["k"]}">{c["t"]}</th>' for c in COLUMNS)
    html = HTML_TEMPLATE.format(
        gen_ts=datetime.now().strftime("%Y-%m-%d %H:%M"),
        n_runs=len(real),
        highlight_cards="".join(cards),
        head_cells=head_cells,
        rows_json=json.dumps(rows, ensure_ascii=False),
        head_json=json.dumps(COLUMNS, ensure_ascii=False),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Gera DASHBOARD.html a partir do experiments_v7.jsonl")
    parser.add_argument("--jsonl", type=str, default=str(DEFAULT_JSONL))
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--open", action="store_true", help="abre no navegador (se houver GUI)")
    args = parser.parse_args()

    jsonl_path, out_path = Path(args.jsonl), Path(args.out)
    if not jsonl_path.exists():
        print(f"[viewer] ⚠ {jsonl_path} ainda não existe — o dashboard vai mostrar só o baseline V6d.")
    out = generate_dashboard(jsonl_path, out_path)
    print(f"✅ Dashboard: {out} ({out.stat().st_size/1024:.0f}KB, {len(load_runs(jsonl_path))} linhas)")
    if args.open:
        import webbrowser
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
