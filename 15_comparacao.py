"""Comparação pareada no teste (2026): modelos fundacionais (TabPFN, Kumo Tabular) contra gradient boosting e
regressão logística, e entre si.

Junta resultados/probs_teste.csv (10_modelo.py) com resultados/tabpfn/probs_teste.csv (14_tabpfn.py) e
resultados/kumo/probs_teste.csv (16_kumo.py), o que existir, pelo ID pseudonimizado de pessoa. Estima a diferença
de AUROC e AUPRC com IC95% por bootstrap PAREADO (os mesmos pacientes reamostrados para os dois modelos).
Gera resultados/comparacao_fundacionais.md.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = Path(__file__).parent
SEMENTE = 20260929
FUNDACIONAIS = {"TabPFN": "tabpfn", "Kumo Tabular": "kumo"}

df = pd.read_csv(BASE / "resultados" / "probs_teste.csv")
n_base = len(df)
incluidos = []
for nome, pasta in FUNDACIONAIS.items():
    arq = BASE / "resultados" / pasta / "probs_teste.csv"
    if not arq.exists():
        print(f"{nome}: {arq} não encontrado, fica fora da comparação")
        continue
    outro = pd.read_csv(arq)
    df = df.merge(outro, on=["id", "obito_30d"], how="inner", validate="one_to_one")
    if len(df) != n_base or len(outro) != n_base:
        raise SystemExit(f"IDs não batem com {nome}: {n_base} no 10_modelo, {len(outro)} em {nome}, {len(df)} em comum")
    incluidos.append(nome)
if not incluidos:
    raise SystemExit("Nenhum modelo fundacional encontrado em resultados/tabpfn ou resultados/kumo")

y = df["obito_30d"].to_numpy()
modelos = [c for c in df.columns if c not in ("id", "obito_30d")]
rng = np.random.default_rng(SEMENTE)
amostras = [i for i in (rng.integers(0, len(y), len(y)) for _ in range(2000)) if y[i].min() != y[i].max()]

linhas = [
    "# Comparação pareada no teste (2026)",
    "",
    f"- Pacientes no teste: {len(y)}; óbitos: {int(y.sum())}; bootstrap pareado com {len(amostras)} reamostras",
    "- Modelo principal pré-especificado: gradient boosting. TabPFN e Kumo Tabular são comparadores; "
    "Kumo Tabular é exploratório (anunciado em 29/09/2026, sem revisão por pares).",
    "",
    "| Modelo | AUROC | AUPRC |",
    "|---|---:|---:|",
]
for m in modelos:
    linhas.append(f"| {m} | {roc_auc_score(y, df[m]):.3f} | {average_precision_score(y, df[m]):.3f} |")

# Pares: cada fundacional contra os modelos clássicos, e os fundacionais entre si
pares = [(f, c) for f in incluidos for c in modelos if c not in FUNDACIONAIS] + list(combinations(incluidos, 2))
linhas += ["", "## Diferença A − B (IC95% por bootstrap pareado)", "",
           "| Comparação | ΔAUROC (IC95%) | ΔAUPRC (IC95%) |", "|---|---|---|"]
for a, b in pares:
    pa, pb = df[a].to_numpy(), df[b].to_numpy()
    d_roc = [roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]) for i in amostras]
    d_pr = [average_precision_score(y[i], pa[i]) - average_precision_score(y[i], pb[i]) for i in amostras]
    r0 = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    a0 = average_precision_score(y, pa) - average_precision_score(y, pb)
    lr, hr = np.percentile(d_roc, [2.5, 97.5])
    la, ha = np.percentile(d_pr, [2.5, 97.5])
    linhas.append(f"| {a} − {b} | {r0:+.3f} ({lr:+.3f} a {hr:+.3f}) | {a0:+.3f} ({la:+.3f} a {ha:+.3f}) |")

linhas += ["", "Intervalo que cruza zero = sem diferença detectável com este tamanho de teste."]
texto = "\n".join(linhas) + "\n"
(BASE / "resultados" / "comparacao_fundacionais.md").write_text(texto, encoding="utf-8")
print(texto)
