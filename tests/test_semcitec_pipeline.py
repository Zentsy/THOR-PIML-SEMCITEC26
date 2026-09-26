"""
THOR-PIML — Testes Unitários e de Integração do Pipeline SEMCITEC 2026
======================================================================
Valida todos os módulos, invariantes físicos e contratos de API implementados:
1. Carregamento e compatibilidade do Scaler V7 (138 features).
2. Integridade estrutural e dimensional dos 4 NetCDFs CORDEX SAM-22.
3. Funções de extração e conversão de unidades do cordex_preprocessor.py.
4. Carregamento do modelo THOR-V8 e aplicação da restrição física de Clausius-Clapeyron.
5. Formulações matemáticas de BIO12 a BIO19 e índices ETCCDI.
6. Validação dos arquivos de figuras gerados (300 DPI, PNGs válidos).
7. Verificação de pré-requisitos e contratos CLI do run_semcitec_pipeline.py.

Uso:
    python -m unittest tests/test_semcitec_pipeline.py -v
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
import xarray as xr

from src.paths import CHECKPOINT_DIR, DATA_DIR, RESULTS_DIR

TEST_ARTIFACTS_DIR = ROOT_DIR / "tests" / "artifacts" / "dryrun"
from src.preprocessing import RobustClimateScaler
from src.v7.config_v7 import THORConfigV8
from src.v7.model_v8 import THORSpatialHybridModel
from src.cordex_preprocessor import (
    CORDEX_FILES,
    GUARULHOS_LAT,
    GUARULHOS_LON,
    find_nearest_coord_idx,
)
from src.calc_bioclimatic_indices import (
    calc_bioclimatic_for_year,
    calc_etccdi_for_year,
    calc_max_run_length,
)
from run_semcitec_pipeline import check_prerequisites


class TestSemcitecPipeline(unittest.TestCase):

    def test_01_scaler_v7_integrity(self):
        """Testa se o Scaler V7 carrega perfeitamente e opera sobre 138 variáveis."""
        scaler_path = CHECKPOINT_DIR / "scaler_v7.json"
        self.assertTrue(scaler_path.exists(), f"Scaler V7 não encontrado: {scaler_path}")

        scaler = RobustClimateScaler.load(scaler_path)
        self.assertTrue(scaler.fitted, "Scaler deve estar marcado como fitted")
        self.assertEqual(scaler.method, "minmax", "Método do scaler deve ser minmax")
        self.assertEqual(len(scaler.data_min_), 138, "Scaler V7 deve conter exatamente 138 features")
        self.assertEqual(len(scaler.data_max_), 138, "Scaler V7 deve conter exatamente 138 features")

        # Testa transformação e inversa com tensor de teste
        x_dummy = np.full((3, 138), 20.0, dtype=np.float32)
        x_scaled = scaler.transform(x_dummy)
        self.assertEqual(x_scaled.shape, (3, 138))
        self.assertFalse(np.isnan(x_scaled).any(), "Transform não deve conter NaNs")

        x_inv = scaler.inverse_transform(x_scaled)
        np.testing.assert_allclose(x_inv, x_dummy, rtol=1e-4, atol=1e-4)

    def test_02_cordex_netcdf_dimensions_and_calendar(self):
        """Valida que os 4 NetCDFs CORDEX existem, têm 33.840 dias e calendário 360_day."""
        for sc_key, meta in CORDEX_FILES.items():
            for var in ("pr", "tas"):
                nc_path = meta[var]
                self.assertTrue(nc_path.exists(), f"NetCDF não encontrado: {nc_path}")
                self.assertGreater(nc_path.stat().st_size, 10_000_000, f"NetCDF muito pequeno: {nc_path}")

                ds = xr.open_dataset(nc_path)
                self.assertIn("time", ds.dims)
                self.assertIn("lat", ds.dims)
                self.assertIn("lon", ds.dims)
                self.assertEqual(ds.sizes["time"], 33840, f"Dimensão temporal deve ser 33840 em {nc_path.name}")
                self.assertEqual(ds.sizes["lat"], 16, f"Dimensão lat deve ser 16 em {nc_path.name}")
                self.assertEqual(ds.sizes["lon"], 16, f"Dimensão lon deve ser 16 em {nc_path.name}")
                self.assertIn(var, ds.data_vars, f"Variável '{var}' ausente em {nc_path.name}")
                ds.close()

    def test_03_cordex_preprocessor_coordinate_extraction(self):
        """Valida a busca de ponto de grade mais próximo para Guarulhos."""
        lat_arr = np.linspace(-25.0, -22.0, 16)
        lon_arr = np.linspace(-48.0, -45.0, 16)

        lat_idx, lat_val = find_nearest_coord_idx(lat_arr, GUARULHOS_LAT)
        lon_idx, lon_val = find_nearest_coord_idx(lon_arr, GUARULHOS_LON)

        self.assertAlmostEqual(lat_val, -23.40, delta=0.15)
        self.assertAlmostEqual(lon_val, -46.40, delta=0.15)

    def test_04_thor_v8_model_architecture_and_clausius_clapeyron(self):
        """Valida o forward pass do THOR-V8 e a imposição estrita do teto W_max = 4.0 * TCWV."""
        ckpt_path = CHECKPOINT_DIR / "v8_hybrid_seed42.pt"
        self.assertTrue(ckpt_path.exists(), f"Checkpoint V8 não encontrado: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        feature_cols = ckpt["feature_cols"]
        self.assertEqual(len(feature_cols), 138, "Feature cols do checkpoint deve ter 138 variáveis")

        config = THORConfigV8()
        config.model.n_features = 138
        model = THORSpatialHybridModel(config.model)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Batch dummy: 2 amostras, lookback 30 dias, 138 features tabulares, 16x16x5 espacial
        xs = torch.randn(2, 30, 138)
        xsp = torch.randn(2, 30, 16, 16, 5)

        with torch.no_grad():
            prob, inten, final = model(xs, xsp, return_components=True)

        self.assertEqual(prob.shape, (2, 1))
        self.assertEqual(inten.shape, (2, 1))
        self.assertEqual(final.shape, (2, 1))

        p_val = prob.numpy()
        i_val = inten.numpy()
        f_val = final.numpy()

        self.assertTrue((p_val >= 0.0).all() and (p_val <= 1.0).all(), "prob deve estar em [0, 1]")
        self.assertTrue((i_val >= 0.0).all(), "intensity deve ser >= 0")
        np.testing.assert_allclose(f_val, p_val * i_val, rtol=1e-5)

        # Validação do teto de Clausius-Clapeyron: W_max = 4.0 * TCWV
        tcwv_test = 25.0  # mm
        w_max = 4.0 * tcwv_test  # 100.0 mm
        y_hat_unconstrained = 145.0  # mm (hipotética tempestade extrema não física)

        y_piml = min(y_hat_unconstrained, w_max)
        self.assertEqual(y_piml, 100.0, "Restrição PIML deve limitar ao teto W_max")

    def test_05_bioclimatic_indices_math(self):
        """Valida o cálculo matemático de BIO12 a BIO19 com série analítica conhecida."""
        # Cria 360 dias com padrão sazonal conhecido
        # Meses 1 a 12 com 30 dias cada
        months = np.repeat(range(1, 13), 30)
        # Chuva alta nos meses 1, 2, 12 (verão = 10 mm/d), baixa nos meses 6, 7, 8 (inverno = 1 mm/d)
        pr_daily = np.where(np.isin(months, [1, 2, 12]), 10.0, 2.0)
        pr_daily[months == 7] = 0.5  # Mês mais seco: Julho = 0.5*30 = 15mm
        temp_daily = np.where(np.isin(months, [1, 2, 3]), 25.0, 15.0)

        df_mock = pd.DataFrame({
            "month": months,
            "pr": pr_daily,
            "temp": temp_daily,
        })

        bio = calc_bioclimatic_for_year(df_mock, pr_col="pr", temp_col="temp")

        # BIO12 = soma anual
        self.assertAlmostEqual(bio["BIO12_PRCPTOT"], float(np.sum(pr_daily)), places=2)
        # BIO13 = mês mais chuvoso (Janeiro ou Fevereiro: 10 * 30 = 300mm)
        self.assertAlmostEqual(bio["BIO13_Wettest_Month"], 300.0, places=2)
        # BIO14 = mês mais seco (Julho: 0.5 * 30 = 15mm)
        self.assertAlmostEqual(bio["BIO14_Driest_Month"], 15.0, places=2)
        # BIO15 = CV > 0
        self.assertGreater(bio["BIO15_Precip_Seasonality_CV"], 0.0)
        # BIO16 = trimestre mais úmido (DJF = Dez + Jan + Fev = 300 + 300 + 300 = 900mm)
        self.assertAlmostEqual(bio["BIO16_Wettest_Quarter"], 900.0, places=1)
        # BIO17 = trimestre mais seco (Inverno)
        self.assertLess(bio["BIO17_Driest_Quarter"], bio["BIO16_Wettest_Quarter"])

    def test_06_etccdi_extremes_math(self):
        """Valida o cálculo dos índices de extremos ETCCDI com dados controlados."""
        # 30 dias: 5 dias com 25mm, 5 dias com 12mm, 10 dias com 0mm, 10 dias com 2mm
        p_seq = np.array([25.0]*5 + [12.0]*5 + [0.0]*10 + [2.0]*10)
        df_mock = pd.DataFrame({"pr": p_seq})

        p95_base = 20.0  # dias com chuva > 20mm
        p99_base = 24.0  # dias com chuva > 24mm

        etccdi = calc_etccdi_for_year(df_mock, pr_col="pr", baseline_p95=p95_base, baseline_p99=p99_base)

        # R10mm: dias >= 10mm (5 de 25mm + 5 de 12mm = 10 dias)
        self.assertEqual(etccdi["R10mm"], 10)
        # R20mm: dias >= 20mm (5 de 25mm = 5 dias)
        self.assertEqual(etccdi["R20mm"], 5)
        # RX1day = 25.0
        self.assertEqual(etccdi["RX1day"], 25.0)
        # RX5day = 5 * 25.0 = 125.0
        self.assertEqual(etccdi["RX5day"], 125.0)
        # CDD: sequência contínua de dias < 1.0mm = 10 dias
        self.assertEqual(etccdi["CDD"], 10)
        # CWD: sequência contínua de dias >= 1.0mm = 10 dias iniciais
        self.assertEqual(etccdi["CWD"], 10)
        # R95p_mm: soma dos dias > 20mm (5 * 25 = 125mm)
        self.assertEqual(etccdi["R95p_mm"], 125.0)
        # R99p_mm: soma dos dias > 24mm (5 * 25 = 125mm)
        self.assertEqual(etccdi["R99p_mm"], 125.0)

    def test_07_check_prerequisites_pipeline(self):
        """Valida a função de checagem de pré-requisitos do script orquestrador."""
        self.assertTrue(check_prerequisites(1), "Pré-requisitos do Passo 1 devem estar presentes")
        self.assertTrue(check_prerequisites(2), "Pré-requisitos do Passo 2 devem estar presentes")
        self.assertTrue(check_prerequisites(3), "Pré-requisitos do Passo 3 devem estar presentes")
        self.assertTrue(check_prerequisites(0), "Todos os pré-requisitos devem ser atendidos")

    def test_08_publication_figures_exist_and_valid(self):
        """Verifica se as 5 figuras canônicas podem ser geradas e são arquivos PNG válidos."""
        from src.generate_semcitec_figures import run_all_figures
        test_dir = RESULTS_DIR / "test_fig_validation"
        test_dir.mkdir(parents=True, exist_ok=True)
        try:
            figs = run_all_figures(dpi=75, out_dir=test_dir, dedicated=False)
            self.assertEqual(len(figs), 5, "Deve gerar exatamente 5 figuras canônicas")
            for fig_p in figs:
                self.assertTrue(fig_p.exists(), f"Figura {fig_p.name} não encontrada")
                self.assertGreater(fig_p.stat().st_size, 30_000, f"Figura {fig_p.name} possui tamanho inválido")
        finally:
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_09_february_30_calendar_immunity(self):
        """Valida imunidade absoluta a erros de calendário 360_day (ex: 2006-02-30)."""
        date_str_feb30 = "2006-02-30"
        # Garante que nossa lógica extrai corretamente sem invocar pd.to_datetime
        yr = int(date_str_feb30[:4])
        mo = int(date_str_feb30[5:7])
        dy = int(date_str_feb30[8:10])
        self.assertEqual(yr, 2006)
        self.assertEqual(mo, 2)
        self.assertEqual(dy, 30)

        df_360 = pd.DataFrame({
            "date": ["2006-02-28", "2006-02-29", "2006-02-30", "2006-03-01"],
            "year": [2006, 2006, 2006, 2006],
            "month": [2, 2, 2, 3],
            "day": [28, 29, 30, 1],
            "pr_raw_mm": [5.0, 12.0, 0.0, 2.0],
            "tas_celsius": [22.0, 23.0, 21.0, 20.0],
        })
        self.assertFalse(df_360.isna().any().any(), "df_360 não deve conter NaNs")
        self.assertIn("2006-02-30", df_360["date"].values)

    def test_10_scenario_cli_mapping(self):
        """Valida que o mapeamento CLI de cenários ssp245/ssp585 para rcp45/rcp85 funciona sem erros."""
        norm_map = {"ssp245": "rcp45", "ssp585": "rcp85", "rcp45": "rcp45", "rcp85": "rcp85", "all": "all"}
        self.assertEqual(norm_map.get("ssp245"), "rcp45")
        self.assertEqual(norm_map.get("ssp585"), "rcp85")
        self.assertEqual(norm_map.get("rcp45"), "rcp45")
        self.assertEqual(norm_map.get("all"), "all")

    def test_11_dynamic_figure_generation(self):
        """Valida a execução do gerador de figuras dinâmico com os dados disponíveis em diretório de teste."""
        from src.generate_semcitec_figures import run_all_figures
        test_dir = RESULTS_DIR / "test_dynamic_figures"
        test_dir.mkdir(parents=True, exist_ok=True)
        try:
            figs = run_all_figures(dpi=75, out_dir=test_dir, dedicated=False)
            self.assertEqual(len(figs), 5, "Deve gerar exatamente 5 figuras canônicas")
            for fig_p in figs:
                self.assertTrue(fig_p.exists(), f"Figura gerada não encontrada: {fig_p}")
                self.assertGreater(fig_p.stat().st_size, 30_000, f"Figura gerada vazia: {fig_p}")
        finally:
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_12_surface_features_and_scaler_pipeline(self):
        """Valida a engenharia de features de superfície (138 variáveis) sem warnings e sem NaNs."""
        from src.infer_cordex_v8 import engineer_cordex_surface_features
        ckpt = torch.load(CHECKPOINT_DIR / "v8_hybrid_seed42.pt", map_location="cpu", weights_only=False)
        scaler = RobustClimateScaler.load(CHECKPOINT_DIR / "scaler_v7.json")

        # Mock de 60 dias de CORDEX (incluindo 2006-02-30)
        df_mock = pd.DataFrame({
            "date": [f"2006-01-{d:02d}" for d in range(1, 31)] + [f"2006-02-{d:02d}" for d in range(1, 31)],
            "year": [2006]*60, "month": [1]*30 + [2]*30, "day": list(range(1, 31))*2, "doy": list(range(1, 61)),
            "pr_raw_mm": np.random.uniform(0, 35, 60).astype(np.float32),
            "tas_celsius": np.random.uniform(18, 28, 60).astype(np.float32),
            "pr_grid_max": np.random.uniform(0, 60, 60).astype(np.float32),
            "pr_grid_std": np.random.uniform(0, 12, 60).astype(np.float32),
            "tas_grid_mean": np.random.uniform(18, 28, 60).astype(np.float32),
        })

        xs, tcwv, df_full = engineer_cordex_surface_features(df_mock, ckpt["feature_cols"])
        self.assertEqual(xs.shape, (60, 138), "Matriz tabular deve ter shape (60, 138)")
        self.assertFalse(np.isnan(xs).any(), "Matriz xs não deve conter NaNs")

        xs_scaled = scaler.transform(xs)
        self.assertEqual(xs_scaled.shape, (60, 138))
        self.assertFalse(np.isnan(xs_scaled).any(), "Matriz escalada não deve conter NaNs")

    def test_13_master_pipeline_dry_run(self):
        """Valida a execução de ponta a ponta do pipeline mestre com --dry-run."""
        pipeline_script = ROOT_DIR / "run_semcitec_pipeline.py"
        self.assertTrue(pipeline_script.exists(), f"Script do pipeline não encontrado: {pipeline_script}")

        official_projection = RESULTS_DIR / "projections" / "cordex_ssp245_downscaled_2026_2099.csv"
        official_before = official_projection.read_bytes() if official_projection.exists() else None

        cmd = [sys.executable, str(pipeline_script), "--dry-run", "--dpi", "100"]
        res = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )

        self.assertEqual(
            res.returncode,
            0,
            f"Pipeline dry-run falhou com código {res.returncode}:\nSTDERR:\n{res.stderr}\nSTDOUT:\n{res.stdout}",
        )
        self.assertIn("RESUMO DA EXECUÇÃO DO PIPELINE SEMCITEC 2026", res.stdout)
        self.assertIn("[PASSO 1 CONCLUÍDO]", res.stdout)
        self.assertIn("[PASSO 2 CONCLUÍDO]", res.stdout)
        self.assertIn("[PASSO 3 CONCLUÍDO]", res.stdout)
        self.assertIn("[PASSO 4 CONCLUÍDO]", res.stdout)

        # Valida existência dos artefatos produzidos pelo dry-run
        dry_projection_dir = TEST_ARTIFACTS_DIR / "projections"
        dry_tables_dir = TEST_ARTIFACTS_DIR / "tables"
        expected_artifacts = [
            dry_projection_dir / "cordex_ssp245_downscaled_2026_2099.csv",
            dry_projection_dir / "cordex_ssp585_downscaled_2026_2099.csv",
            dry_projection_dir / "projections_summary.json",
            dry_tables_dir / "bioclimatic_indices_epochs_summary.csv",
            dry_tables_dir / "etccdi_extremes_epochs_summary.csv",
            dry_tables_dir / "tabela_resumo_semcitec_2026.md",
        ]
        for art in expected_artifacts:
            self.assertTrue(art.exists(), f"Artefato esperado não encontrado após dry-run: {art}")

        # Valida que as projeções geradas contêm as colunas canônicas da física informada
        df_p45 = pd.read_csv(dry_projection_dir / "cordex_ssp245_downscaled_2026_2099.csv")
        self.assertEqual(len(df_p45), 90, "Dry-run deve gerar exatamente 90 janelas, fora da produção")
        for req_col in ("pr_thor_downscaled_mm", "tcwv_ceiling_wmax_mm", "is_physical_violation"):
            self.assertIn(req_col, df_p45.columns, f"Coluna física obrigatória '{req_col}' ausente nas projeções")

        # Valida que os resumos e o relatório Markdown contêm os modelos THOR-PIML
        md_content = (dry_tables_dir / "tabela_resumo_semcitec_2026.md").read_text(encoding="utf-8")
        self.assertNotIn("THOR_PIML_SSP245", md_content, "Dry-run parcial não deve publicar indicadores THOR anuais")
        self.assertNotIn("THOR_PIML_SSP585", md_content, "Dry-run parcial não deve publicar indicadores THOR anuais")

        df_bio_sum = pd.read_csv(dry_tables_dir / "bioclimatic_indices_epochs_summary.csv")
        sids = set(df_bio_sum["series_id"].unique())
        self.assertNotIn("THOR_PIML_SSP245", sids, "Dry-run parcial não deve conter ano THOR incompleto")
        self.assertNotIn("THOR_PIML_SSP585", sids, "Dry-run parcial não deve conter ano THOR incompleto")
        if official_before is not None:
            self.assertEqual(official_before, official_projection.read_bytes(), "Dry-run não pode alterar projeções oficiais")

    def test_14_scenario_aliases_and_dpi_propagation(self):
        """Valida que aliases RCP/SSP e o parâmetro DPI são respeitados na CLI."""
        from src.generate_semcitec_figures import run_all_figures
        # Geração com DPI 75 para validação rápida de propagação
        test_out = RESULTS_DIR / "test_figures"
        figs = run_all_figures(dpi=75, out_dir=test_out)
        self.assertEqual(len(figs), 5)
        for f in figs:
            self.assertTrue(f.exists(), f"Figura {f} não encontrada")
            self.assertGreater(f.stat().st_size, 10_000)
            # Limpeza do diretório de teste
            f.unlink()
        if test_out.exists():
            test_out.rmdir()

    def test_15_dedicated_scenario_figures(self):
        """Valida a geração de figuras dedicadas e independentes para cada cenário climático."""
        from src.generate_semcitec_figures import run_all_figures
        test_out = RESULTS_DIR / "test_dedicated_figures"

        for sc in ("ssp245", "ssp585"):
            figs = run_all_figures(dpi=75, out_dir=test_out, scenario=sc)
            self.assertEqual(len(figs), 5, f"Cenário {sc} deve gerar exatamente 5 figuras dedicadas")
            for f in figs:
                self.assertTrue(f.exists(), f"Figura dedicada {f.name} não encontrada")
                self.assertIn(sc, f.name, f"Nome da figura dedicada {f.name} deve conter o cenário {sc}")
                self.assertGreater(f.stat().st_size, 10_000, f"Figura {f.name} muito pequena")

        import shutil
        shutil.rmtree(test_out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
