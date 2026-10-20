"""TreeSHAP exato do CatBoost (40 preditores) em todo o teste, refeito em CPU a partir dos mesmos dados e hiperparâmetros.

Reaproveita as células de configuração e de dados do notebook (gerado por notebooks/gera_notebook_gpu.py), reajusta o
CatBoost com os hiperparâmetros escolhidos pelo Optuna na rodada da GPU (resultados/gpu_lab_final/resultados_expandido/
resultados.md) e confere que as predições no teste reproduzem as da rodada (probs_teste_brutas.csv) antes de calcular o SHAP.
Grava resultados/gpu_lab_final/resultados_expandido/shap_catboost.npz.

uso: python 24_shap_catboost.py
"""
import ast
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
R = BASE / "resultados" / "gpu_lab_final" / "resultados_expandido"

with tempfile.TemporaryDirectory() as tmp:
    nb_path = Path(tmp) / "nb.ipynb"
    subprocess.run([sys.executable, str(BASE / "notebooks" / "gera_notebook_gpu.py"), "--conjunto", "expandido", str(nb_path)], check=True, capture_output=True)
    nb = nbformat.read(nb_path, 4)
    ambiente = {}
    os.chdir(BASE / "notebooks")
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        if "SAIDA = Path(" in c.source or "def fixa_sementes" in c.source or "df = pd.read_csv(CAMINHO_DADOS)" in c.source:
            src = c.source.replace("SAIDA.mkdir(exist_ok=True)", "")
            exec(compile(src, f"celula_{i}", "exec"), ambiente)
            if "df = pd.read_csv(CAMINHO_DADOS)" in c.source:
                break

Xtr, Xte, ytr, yte, preditores = (ambiente[k] for k in ("Xtr", "Xte", "ytr", "yte", "preditores"))
SEMENTE, N_THREADS = ambiente["SEMENTE"], ambiente["N_THREADS"]
texto = (R / "resultados.md").read_text(encoding="utf-8")
hp = ast.literal_eval(re.search(r"\| CatBoost \| [\d.]+ \| (\{.*?\}) \|", texto).group(1))

from catboost import CatBoostClassifier, Pool
ambiente["fixa_sementes"]()
modelo = CatBoostClassifier(**hp, loss_function="Logloss", random_seed=SEMENTE, verbose=0, thread_count=N_THREADS,
                            allow_writing_files=False).fit(Xtr.to_numpy(), ytr)
p = modelo.predict_proba(Xte.to_numpy())[:, 1]

ref = pd.read_csv(R / "probs_teste_brutas.csv")
ids = ambiente["df"].loc[ambiente["teste"], "id"].to_numpy()
ref = ref.set_index("id").loc[ids, "CatBoost"].to_numpy()
dif = np.abs(p - ref).max()
print(f"CatBoost: diferença máxima entre a predição refeita e a da rodada da GPU = {dif:.2e}")
assert dif < 1e-6, "o CatBoost refeito não reproduz a rodada da GPU"

contrib = modelo.get_feature_importance(Pool(Xte.to_numpy()), type="ShapValues")
sv, base = contrib[:, :-1], contrib[:, -1]
erro = np.abs(1 / (1 + np.exp(-(base + sv.sum(axis=1)))) - p).max()
print(f"TreeSHAP: erro máximo ao reconstituir a predição = {erro:.2e}")
np.savez_compressed(R / "shap_catboost.npz", shap_valores=sv, X_ref=Xte.to_numpy(), preditores=np.array(preditores))
imp = pd.DataFrame({"variavel": preditores, "shap_medio_abs": np.abs(sv).mean(axis=0)}).sort_values("shap_medio_abs", ascending=False)
print(imp.head(15).to_string(index=False))
