"""Escore de risco da Mayo Clinic para óbito hospitalar após angioplastia (Singh et al., Mayo Clin Proc 2007;82:701-8), publicado.

Pontos da Figura 1 do artigo. Idade, creatinina e fração de ejeção vêm dos nomogramas; os cortes foram lidos da figura
(cada divisória cai no meio entre dois valores inteiros):
  idade (anos)        <37,5: 2 | 37,5–48,5: 1 | 48,5–65,5: 0 | 65,5–72,5: 1 | 72,5–79,5: 2 | 79,5–86,5: 3 | ≥86,5: 4
  creatinina (mg/dL)  <0,65: 1 | 0,65–1,15: 0 | 1,15–1,35: 1 | 1,35–1,85: 2 | 1,85–3,95: 3 | ≥3,95: 4
  fração de ejeção (%) <18,5: 3 | 18,5–30,5: 2 | 30,5–44,5: 1 | ≥44,5: 0
  choque pré-procedimento 9; infarto ≤24 h 4; insuficiência cardíaca na apresentação, sem infarto agudo nem choque, 3;
  doença arterial periférica 2.
Faltantes, como o artigo recomenda: sem fração de ejeção, 1 ponto se houver insuficiência cardíaca; sem creatinina, 1 ponto
se for homem com insuficiência cardíaca. Risco: log-odds = −6,535 + 0,342 × pontos. Sem recalibração.

Grava resultados/gpu_lab_final/resultados_mayo/probs_mayo_publicado.csv (id, desfecho, probabilidade) para o teste.
uso: python 26_escore_mayo.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
R = BASE / "resultados" / "gpu_lab_final" / "resultados_mayo"

with tempfile.TemporaryDirectory() as tmp:
    nb_path = Path(tmp) / "nb.ipynb"
    subprocess.run([sys.executable, str(BASE / "notebooks" / "gera_notebook_gpu.py"), "--conjunto", "mayo", str(nb_path)], check=True, capture_output=True)
    nb = nbformat.read(nb_path, 4)
amb = {}
os.chdir(BASE / "notebooks")
for i, c in enumerate(nb.cells):
    if c.cell_type == "code" and ("SAIDA = Path(" in c.source or "def fixa_sementes" in c.source or "df = pd.read_csv(CAMINHO_DADOS)" in c.source):
        exec(compile(c.source.replace("SAIDA.mkdir(exist_ok=True)", ""), f"celula_{i}", "exec"), amb)
        if "df = pd.read_csv(CAMINHO_DADOS)" in c.source:
            break
df, teste = amb["df"], amb["teste"]


def faixa(v, cortes, pontos):
    return np.select([v < c for c in cortes], pontos[:-1], pontos[-1]).astype(float)


idade, creat, fe = df["idade"].to_numpy(float), df["creatinina"].to_numpy(float), df["fe"].to_numpy(float)
choque, iam24, dap = (df[c].fillna(0).to_numpy() == 1 for c in ("choque", "iam_24h", "dap"))
ic_apres = ((df["apres_ic"] == 1) | (df["saps3_ic"] == 1)).to_numpy()       # insuficiência cardíaca na apresentação
homem = (df["sexo_f"] == 0).to_numpy()

p_idade = np.where(np.isnan(idade), 0, faixa(idade, [37.5, 48.5, 65.5, 72.5, 79.5, 86.5], [2, 1, 0, 1, 2, 3, 4]))
p_creat = np.where(np.isnan(creat), (homem & ic_apres).astype(float), faixa(creat, [0.65, 1.15, 1.35, 1.85, 3.95], [1, 0, 1, 2, 3, 4]))
p_fe = np.where(np.isnan(fe), ic_apres.astype(float), faixa(fe, [18.5, 30.5, 44.5], [3, 2, 1, 0]))
pontos = p_idade + p_creat + p_fe + 9 * choque + 4 * iam24 + 3 * (ic_apres & ~iam24 & ~choque) + 2 * dap
risco = 1 / (1 + np.exp(-(-6.535 + 0.342 * pontos)))

out = pd.DataFrame({"id": df.loc[teste, "id"].to_numpy(), "obito_30d": df.loc[teste, amb["DESFECHO"]].to_numpy().astype(int),
                    "Mayo (escore publicado)": risco[teste], "pontos": pontos[teste]})
out.to_csv(R / "probs_mayo_publicado.csv", index=False)
from sklearn.metrics import average_precision_score, roc_auc_score
print(f"teste: {len(out)} pacientes | pontos mediana {np.median(out['pontos']):.0f} (IQR {np.percentile(out['pontos'], 25):.0f}–{np.percentile(out['pontos'], 75):.0f})")
print(f"AUROC {roc_auc_score(out['obito_30d'], out['Mayo (escore publicado)']):.3f} | AUPRC {average_precision_score(out['obito_30d'], out['Mayo (escore publicado)']):.3f} "
      f"| risco médio previsto {100*out['Mayo (escore publicado)'].mean():.2f}% contra observado {100*out['obito_30d'].mean():.2f}%")
