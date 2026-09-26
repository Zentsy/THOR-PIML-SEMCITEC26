"""
THOR-PIML — Orquestrador Mestre do Pipeline SEMCITEC 2026
==========================================================
Executa e orquestra de ponta a ponta o pipeline de projeção climática regional
e cálculo de indicadores bioclimáticos (2026–2100) em Guarulhos-SP via THOR-PIML:

Etapas do Pipeline:
  [Passo 1] Pré-processamento dos NetCDFs CORDEX SAM-22 (pr e tas para 2006–2099)
  [Passo 2] Inferência Zero-Shot THOR-PIML V8 (Lookback 30d + Física Clausius-Clapeyron)
  [Passo 3] Cálculo dos Indicadores Bioclimáticos (BIO12–BIO19) e Extremos ETCCDI nas 4 Épocas
  [Passo 4] Geração das 5 Figuras Canônicas de Publicação (300 DPI, Layout Editorial ~3:1)

Uso:
    # Execução completa com 1 comando:
    python run_semcitec_pipeline.py

    # Execução por etapa individual:
    python run_semcitec_pipeline.py --step 1
    python run_semcitec_pipeline.py --step 2
    python run_semcitec_pipeline.py --step 3
    python run_semcitec_pipeline.py --step 4

    # Teste rápido de validação (dry-run):
    python run_semcitec_pipeline.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from src.paths import CHECKPOINT_DIR, DATA_DIR, RESULTS_DIR

TEST_ARTIFACTS_DIR = ROOT_DIR / "tests" / "artifacts" / "dryrun"
TABLE_DATA_DIR = ROOT_DIR / ".agents" / "table_data"


def print_banner():
    banner = f"""
===============================================================================
     THOR-PIML : PIPELINE DE DOWNSTREAM & PROJEÇÃO CLIMÁTICA (SEMCITEC 2026)
  Taylor-Hurdle Optimized Regional Physics-Informed Machine Learning Architecture
===============================================================================
  • Diretório Raiz:   {ROOT_DIR}
  • Dispositivo PyTorch: {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}
  • Horário Local:     {time.strftime('%Y-%m-%d %H:%M:%S')}
===============================================================================
"""
    print(banner)


def check_prerequisites(step: int) -> bool:
    """Valida a presença de arquivos essenciais antes de executar cada etapa."""
    if step in (1, 0):
        # Verifica NetCDFs CORDEX
        nc_files = [
            DATA_DIR / "SAM-22" / "pr-rcp45" / "pr_SAM-20_MOHC-HadGEM2-ES_rcp45_r1i1p1_INPE-Eta_v1_day_20060101-20991230.nc",
            DATA_DIR / "SAM-22" / "pr-rcp85" / "pr_SAM-20_MOHC-HadGEM2-ES_rcp85_r1i1p1_INPE-Eta_v1_day_20060101-20991230.nc",
            DATA_DIR / "SAM-22" / "tas-rcp45" / "tas_SAM-20_MOHC-HadGEM2-ES_rcp45_r1i1p1_INPE-Eta_v1_day_20060101-20991230.nc",
            DATA_DIR / "SAM-22" / "rcp8.5-TAS" / "tas_SAM-20_MOHC-HadGEM2-ES_rcp85_r1i1p1_INPE-Eta_v1_day_20060101-20991230.nc",
        ]
        missing = [p for p in nc_files if not p.exists()]
        if missing:
            print(f"❌ Erro [Passo 1]: NetCDFs CORDEX ausentes ({len(missing)} arquivos):")
            for p in missing:
                print(f"   - {p}")
            return False

    if step in (2, 0):
        # Verifica Checkpoint e Scaler
        ckpt = CHECKPOINT_DIR / "v8_hybrid_seed42.pt"
        scaler = CHECKPOINT_DIR / "scaler_v7.json"
        if not ckpt.exists():
            print(f"❌ Erro [Passo 2]: Checkpoint não encontrado: {ckpt}")
            return False
        if not scaler.exists():
            print(f"❌ Erro [Passo 2]: Scaler não encontrado: {scaler}")
            return False

    if step in (3, 0):
        # Verifica Ground Truth histórico
        gt = DATA_DIR / "ground_truth_guarulhos_daily_v3.csv"
        if not gt.exists():
            print(f"❌ Erro [Passo 3]: Baseline histórico não encontrado: {gt}")
            return False

    return True


def run_step_1(args) -> float:
    """Executa o Passo 1: Pré-processamento CORDEX."""
    print("\n" + "#"*75)
    print("▶ [PASSO 1/4] INGESTÃO & PRÉ-PROCESSAMENTO CORDEX SAM-22")
    print("#"*75)
    t0 = time.time()
    from src.cordex_preprocessor import run_preprocessor
    norm_map = {"ssp245": "rcp45", "ssp585": "rcp85", "rcp45": "rcp45", "rcp85": "rcp85", "all": "all"}
    sc = norm_map.get(args.scenario, args.scenario)
    run_preprocessor(scenario=sc)
    dt = time.time() - t0
    print(f"\n✓ [PASSO 1 CONCLUÍDO] em {dt:.2f}s")
    return dt


def run_step_2(args) -> float:
    """Executa o Passo 2: Inferência Zero-Shot THOR-PIML V8."""
    print("\n" + "#"*75)
    print("▶ [PASSO 2/4] DOWNSCALING ZERO-SHOT COM THOR-PIML V8 (PIML)")
    print("#"*75)
    t0 = time.time()
    from src.infer_cordex_v8 import run_projections
    norm_map = {"rcp45": "ssp245", "rcp85": "ssp585", "ssp245": "ssp245", "ssp585": "ssp585", "all": "all"}
    sc = norm_map.get(args.scenario, args.scenario)
    out_dir = TEST_ARTIFACTS_DIR / "projections" if args.dry_run else RESULTS_DIR / "projections"
    run_projections(
        scenario=sc,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        out_dir=out_dir,
    )
    dt = time.time() - t0
    print(f"\n✓ [PASSO 2 CONCLUÍDO] em {dt:.2f}s")
    return dt


def run_step_3(args) -> float:
    """Executa o Passo 3: Indicadores Bioclimáticos & Extremos ETCCDI."""
    print("\n" + "#"*75)
    print("▶ [PASSO 3/4] CÁLCULO DE INDICADORES BIOCLIMÁTICOS (BIO12–BIO19) & ETCCDI")
    print("#"*75)
    t0 = time.time()
    from src.calc_bioclimatic_indices import run_pipeline
    projections_dir = TEST_ARTIFACTS_DIR / "projections" if args.dry_run else RESULTS_DIR / "projections"
    tables_dir = TEST_ARTIFACTS_DIR / "tables" if args.dry_run else TABLE_DATA_DIR
    run_pipeline(projections_dir=projections_dir, tables_dir=tables_dir)
    dt = time.time() - t0
    print(f"\n✓ [PASSO 3 CONCLUÍDO] em {dt:.2f}s")
    return dt


def run_step_4(args) -> float:
    """Executa o Passo 4: Geração das 5 Figuras Canônicas (300 DPI)."""
    print("\n" + "#"*75)
    print("▶ [PASSO 4/4] GERAÇÃO DAS 5 FIGURAS CIENTÍFICAS EDITORIAIS (300 DPI)")
    print("#"*75)
    t0 = time.time()
    from src.generate_semcitec_figures import run_all_figures
    out_dir = TEST_ARTIFACTS_DIR / "figures" if getattr(args, "dry_run", False) else RESULTS_DIR / "figures"
    run_all_figures(dpi=args.dpi, out_dir=out_dir, scenario=args.scenario, dedicated=False)
    if not getattr(args, "dry_run", False):
        from src.generate_semcitec_tables import main as generate_editorial_tables
        generate_editorial_tables()
    dt = time.time() - t0
    print(f"\n✓ [PASSO 4 CONCLUÍDO] em {dt:.2f}s")
    return dt


def print_summary(timings: dict, step_target: int = 0, dry_run: bool = False):
    """Exibe o sumário final da execução e lista apenas os arquivos da etapa executada."""
    total_time = sum(timings.values())
    print("\n" + "="*75)
    print("                RESUMO DA EXECUÇÃO DO PIPELINE SEMCITEC 2026")
    print("===============================================================================")
    for step_name, dt in timings.items():
        print(f"  • {step_name:<40}: {dt:>6.2f} s")
    print(f"  {'• Tempo Total Decorrido':<40}: {total_time:>6.2f} s")
    print("="*75)

    print("\nArquivos Gerados / Disponíveis nesta Execução:")
    key_paths = []
    if step_target in (1, 0):
        key_paths.extend([
            DATA_DIR / "cordex" / "cordex_guarulhos_rcp45_daily_2006_2099.csv",
            DATA_DIR / "cordex" / "cordex_guarulhos_rcp85_daily_2006_2099.csv",
            DATA_DIR / "cordex" / "cordex_preprocessing_metadata.json",
        ])
    if step_target in (2, 0):
        projection_dir = TEST_ARTIFACTS_DIR / "projections" if dry_run else RESULTS_DIR / "projections"
        key_paths.extend([
            projection_dir / "cordex_ssp245_downscaled_2026_2099.csv",
            projection_dir / "cordex_ssp585_downscaled_2026_2099.csv",
            projection_dir / "projections_summary.json",
        ])
    if step_target in (3, 0):
        table_dir = TEST_ARTIFACTS_DIR / "tables" if dry_run else RESULTS_DIR / "tables"
        key_paths.extend([
            table_dir / ("bioclimatic_indices_epochs_summary.csv" if dry_run else "bioclimatic_indices_epochs_summary.png"),
            table_dir / ("etccdi_extremes_epochs_summary.csv" if dry_run else "etccdi_extremes_epochs_summary.png"),
            table_dir / "tabela_resumo_semcitec_2026.md",
        ])
    if step_target in (4, 0):
        key_paths.extend([
            RESULTS_DIR / "figures" / "fig1_semcitec_prcptot_projections_300dpi.png",
            RESULTS_DIR / "figures" / "fig2_semcitec_seasonal_cycle_epochs_300dpi.png",
            RESULTS_DIR / "figures" / "fig3_semcitec_fdc_extreme_tail_300dpi.png",
            RESULTS_DIR / "figures" / "fig4_semcitec_bioclimatic_spider_panel_300dpi.png",
            RESULTS_DIR / "figures" / "fig5_semcitec_etccdi_urban_extremes_300dpi.png",
        ])

    for p in key_paths:
        if p.exists():
            status = f"✓ ({p.stat().st_size / 1024:.1f} KB)"
            print(f"  [{status:<20}] {p.relative_to(ROOT_DIR)}")
        else:
            print(f"  [{'Ainda não gerado':<20}] {p.relative_to(ROOT_DIR)}")
    print("="*75 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="THOR-PIML SEMCITEC 2026 Master Pipeline Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["1", "2", "3", "4", "all"],
        help="Etapa a executar:\n"
             "  1: Pré-processamento NetCDFs CORDEX\n"
             "  2: Inferência Zero-Shot THOR-PIML\n"
             "  3: Cálculo de Índices Bioclimáticos & ETCCDI\n"
             "  4: Geração de Figuras 300 DPI\n"
             "  all: Pipeline Completo (default)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=["ssp245", "ssp585", "rcp45", "rcp85", "all"],
        help="Cenário CORDEX a processar (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Tamanho do lote de inferência (default: 128)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolução DPI para as figuras científicas (default: 300)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execução rápida em modo de teste com amostra reduzida",
    )
    args = parser.parse_args()

    print_banner()

    step_target = 0 if args.step == "all" else int(args.step)

    # Verificação de pré-requisitos
    if not check_prerequisites(step_target):
        sys.exit(1)

    timings = {}

    try:
        if step_target in (1, 0):
            timings["Passo 1: CORDEX Preprocessor"] = run_step_1(args)

        if step_target in (2, 0):
            timings["Passo 2: THOR-PIML Inference"] = run_step_2(args)

        if step_target in (3, 0):
            timings["Passo 3: Bioclimatic Indices"] = run_step_3(args)

        if step_target in (4, 0):
            timings["Passo 4: Publication Figures"] = run_step_4(args)

        print_summary(timings, step_target, dry_run=args.dry_run)

    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrompido pelo usuário.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Erro durante a execução do pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
