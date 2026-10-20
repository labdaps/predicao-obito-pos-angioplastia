"""IC95% por bootstrap das métricas por sexo e faixa etária (análise principal, 40 preditores) no teste de 2025.

Reamostra os pacientes do teste (2.000 réplicas, semente fixa) e recalcula, em cada subgrupo, AUROC, razão observado/esperado
e, no limiar de 2,64% (prevalência do treino), sensibilidade e taxa de falsos positivos. As mesmas réplicas servem para a
diferença mulheres − homens, que é pareada.
Grava resultados/gpu_lab_final/bootstrap/ic95_subgrupos.csv e ic95_subgrupos_diferencas.csv.

uso: python 25_subgrupos_ic.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).parent
G = BASE / "resultados" / "gpu_lab_final"
SEMENTE, B, LIMIAR = 20260929, 2000, 0.0264
MODELOS = ["Regressão logística", "Gradient boosting", "XGBoost", "CatBoost", "TabPFN", "Kumo Tabular"]

pr = pd.read_csv(G / "resultados_expandido" / "probs_teste_todos.csv")
ds = pd.read_csv(BASE / "dados" / "icp_dataset.csv"); ds.columns = ds.columns.str.lower()
te = ds.set_index("id").loc[pr["id"]]
y = pr["obito_30d"].to_numpy().astype(int)                     # desfecho da análise: óbito na internação, até 30 dias
idade, mulher = te["idade"].to_numpy(), te["sexo_f"].to_numpy() == 1
GRUPOS = {"Todos": np.ones(len(y), bool), "Homens": ~mulher, "Mulheres": mulher,
          "< 65 anos": idade < 65, "65–74 anos": (idade >= 65) & (idade < 75), "≥ 75 anos": idade >= 75}


def metricas(yv, p):
    s = p >= LIMIAR
    return {"AUROC": roc_auc_score(yv, p) if 0 < yv.sum() < len(yv) else np.nan, "O/E": yv.mean() / p.mean(),
            "Sensibilidade": s[yv == 1].mean() if yv.sum() else np.nan, "Falsos positivos": s[yv == 0].mean()}


rng = np.random.default_rng(SEMENTE)
idx = [rng.integers(0, len(y), len(y)) for _ in range(B)]
linhas, difs = [], []
for m in MODELOS:
    p = pr[m].to_numpy()
    reps = {g: [] for g in GRUPOS}
    for ii in idx:
        for g, mask in GRUPOS.items():
            sel = ii[mask[ii]]
            reps[g].append(metricas(y[sel], p[sel]))
    reps = {g: pd.DataFrame(v) for g, v in reps.items()}
    for g, mask in GRUPOS.items():
        r = {"Modelo": m, "Grupo": g, "N": int(mask.sum()), "Óbitos": int(y[mask].sum())}
        for met, v in metricas(y[mask], p[mask]).items():
            lo, hi = np.nanpercentile(reps[g][met], [2.5, 97.5])
            r.update({met: v, f"{met} inf": lo, f"{met} sup": hi})
        linhas.append(r)
    for met in ["AUROC", "O/E", "Sensibilidade", "Falsos positivos"]:
        d = reps["Mulheres"][met].to_numpy() - reps["Homens"][met].to_numpy()
        est = metricas(y[mulher], p[mulher])[met] - metricas(y[~mulher], p[~mulher])[met]
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        difs.append({"Modelo": m, "Métrica": met, "Mulheres − homens": est, "IC95 inf": lo, "IC95 sup": hi})

(G / "bootstrap").mkdir(exist_ok=True)
pd.DataFrame(linhas).to_csv(G / "bootstrap" / "ic95_subgrupos.csv", index=False)
pd.DataFrame(difs).to_csv(G / "bootstrap" / "ic95_subgrupos_diferencas.csv", index=False)
print(pd.DataFrame(difs).round(3).to_string(index=False))
