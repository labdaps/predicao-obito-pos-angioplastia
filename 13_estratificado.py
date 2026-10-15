"""O modelo prediz desfecho ou só o perfil eletivo/urgente?

Dois testes, no mesmo split do 10_modelo.py (treino < 2026, teste 2026, grão pessoa):
1. Modelo "só perfil": usa apenas marcadores de urgência (tem exame antes do procedimento,
   atendimento nas 48h antes, internado antes, código de angioplastia no IAM, CID de IAM/angina
   na apresentação). Se o modelo completo não passar muito dele, está só separando perfis.
2. Desempenho do modelo completo DENTRO de cada estrato de perfil. Se discrimina dentro do
   estrato, está predizendo desfecho, não só o perfil.
Gera resultados/estratificado.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = Path(__file__).parent
SEMENTE = 20260929
LAB = ["creatinina", "hemoglobina", "leucocitos", "plaquetas", "potassio", "sodio", "ureia", "troponina_pct_no_codigo"]
PERFIL = ["tem_exame", "atend_48h_antes", "internado_no_dia", "fl_codigo_iam", "apres_iam_qualquer",
          "apres_iamcsst", "apres_angina"]

df = pd.read_csv(BASE / "dados" / "icp_dataset.csv")
df.columns = [c.lower() for c in df.columns]
df = df[df["transf_externa_30d"] != 1].copy()
df["tem_exame"] = df[LAB].notna().any(axis=1).astype(int)
df["urgente"] = ((df.fl_codigo_iam == 1) | (df.apres_iam_qualquer == 1) | (df.atend_48h_antes == 1)).astype(int)

fora = ["id", "ano_icp", "mes_icp", "obito_30d", "transf_externa_30d", "tem_exame", "urgente"]
completo = [c for c in df.columns if c not in fora]
y = df.obito_30d.to_numpy()
tr, te = (df.ano_icp < 2026).to_numpy(), (df.ano_icp >= 2026).to_numpy()


def ajusta(cols):
    m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=400, l2_regularization=1.0,
                                       early_stopping=True, validation_fraction=0.2, random_state=SEMENTE)
    m.fit(df.loc[tr, cols], y[tr])
    return m.predict_proba(df.loc[te, cols])[:, 1]


def ic(yy, pp, f=roc_auc_score, n=1000):
    rng = np.random.default_rng(SEMENTE)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(yy), len(yy))
        if yy[i].min() != yy[i].max():
            v.append(f(yy[i], pp[i]))
    return np.percentile(v, [2.5, 97.5])


p_full = ajusta(completo)
p_perfil = ajusta(PERFIL)
yt = y[te]
out = ["# O modelo prediz desfecho ou só o perfil eletivo/urgente?", "",
       "## 1. Modelo completo contra modelo que só conhece o perfil (teste 2026)", "",
       "| Modelo | Variáveis | AUROC (IC95%) | AUPRC |", "|---|---:|---|---:|"]
for nome, p, k in [("Só perfil (urgência)", p_perfil, len(PERFIL)), ("Completo", p_full, len(completo))]:
    a, b = ic(yt, p)
    out.append(f"| {nome} | {k} | {roc_auc_score(yt, p):.3f} ({a:.3f}–{b:.3f}) | {average_precision_score(yt, p):.3f} |")

out += ["", "## 2. Modelo completo DENTRO de cada estrato (teste 2026)", "",
        "| Estrato | N | Óbitos | Óbito % | AUROC (IC95%) | AUPRC | AUPRC sem modelo |", "|---|---:|---:|---:|---|---:|---:|"]
tt = df[te].reset_index(drop=True)
estratos = [
    ("Com exame antes do procedimento", tt.tem_exame == 1),
    ("Sem exame antes do procedimento", tt.tem_exame == 0),
    ("Urgente (IAM, código IAM ou atendimento 48h antes)", tt.urgente == 1),
    ("Eletivo (nenhum marcador de urgência)", tt.urgente == 0),
    ("Angioplastia primária no IAM (código)", tt.fl_codigo_iam == 1),
    ("Urgente E com exame", (tt.urgente == 1) & (tt.tem_exame == 1)),
]
for nome, mask in estratos:
    mask = mask.to_numpy()
    ys, ps = yt[mask], p_full[mask]
    if ys.sum() < 5 or ys.sum() == len(ys):
        out.append(f"| {nome} | {mask.sum()} | {int(ys.sum())} | {100 * ys.mean():.2f} | óbitos insuficientes | – | – |")
        continue
    a, b = ic(ys, ps)
    out.append(f"| {nome} | {mask.sum()} | {int(ys.sum())} | {100 * ys.mean():.2f} | {roc_auc_score(ys, ps):.3f} ({a:.3f}–{b:.3f}) "
               f"| {average_precision_score(ys, ps):.3f} | {ys.mean():.3f} |")

texto = "\n".join(out) + "\n"
(BASE / "resultados" / "estratificado.md").write_text(texto, encoding="utf-8")
print(texto)
