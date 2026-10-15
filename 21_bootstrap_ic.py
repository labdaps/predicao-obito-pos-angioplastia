"""Intervalos de confiança de 95% por bootstrap no teste (rodada de referência: resultados/gpu_lab/).

Reamostra as pessoas do teste com reposição (2.000 réplicas, semente fixa) e recalcula, em cada réplica, as métricas
de todos os modelos sobre as MESMAS pessoas. Assim as diferenças entre modelos (e entre conjuntos de preditores,
que usam o mesmo teste) são pareadas.

Saída: resultados/gpu_lab/bootstrap/ic95.md, ic95_metricas.csv e ic95_diferencas.csv
uso: python 21_bootstrap_ic.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

BASE = Path(__file__).parent
G = BASE / "resultados" / "gpu_lab"
SAIDA = G / "bootstrap"
SAIDA.mkdir(exist_ok=True)
SEMENTE, B = 20260929, 2000

CONJUNTOS = {"Variáveis do CathPCI (8)": "resultados_cathpci", "Variáveis do Mayo (7)": "resultados_mayo",
             "Prontuário estruturado (30)": "resultados_modelos", "Prontuário + beira de leito (40)": "resultados_expandido"}

# limiar pré-especificado: prevalência do desfecho no treino
ds = pd.read_csv(BASE / "dados" / "icp_dataset.csv")
ds.columns = ds.columns.str.lower()
ds = ds[ds["transf_externa_30d"] != 1]
LIMIAR = float(ds.loc[ds["ano_icp"] < 2026, "obito_30d"].mean())

# probabilidades de teste (recalibradas; o escore CathPCI entra como publicado), alinhadas por ID
probs, y, ids = {}, None, None
for conj, pasta in CONJUNTOS.items():
    t = pd.read_csv(G / pasta / "probs_teste_todos.csv").sort_values("id").reset_index(drop=True)
    if ids is None:
        ids, y = t["id"].to_numpy(), t["obito_30d"].to_numpy().astype(int)
    assert (t["id"].to_numpy() == ids).all() and (t["obito_30d"].to_numpy() == y).all()
    for m in [c for c in t.columns if c not in ("id", "obito_30d")]:
        probs[(conj, m)] = t[m].to_numpy(dtype=float)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def metricas(yv, p):
    sinal = p >= LIMIAR
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), yv).coef_[0][0]
    inter = np.log(yv.mean() / (1 - yv.mean())) - np.log(p.mean() / (1 - p.mean()))
    return {"AUROC": roc_auc_score(yv, p), "AUPRC": average_precision_score(yv, p), "Brier": brier_score_loss(yv, p),
            "Intercepto": inter, "Slope": slope, "Sensibilidade": sinal[yv == 1].mean(), "Especificidade": (~sinal)[yv == 0].mean()}


pontual = {k: metricas(y, p) for k, p in probs.items()}
rng = np.random.default_rng(SEMENTE)
replicas = {k: [] for k in probs}
n = len(y)
for b in range(B):
    idx = rng.integers(0, n, n)
    while y[idx].sum() == 0:
        idx = rng.integers(0, n, n)
    for k, p in probs.items():
        replicas[k].append(metricas(y[idx], p[idx]))
replicas = {k: pd.DataFrame(v) for k, v in replicas.items()}

linhas = []
for (conj, m), d in replicas.items():
    linha = {"Conjunto": conj, "Modelo": m}
    for met in d.columns:
        lo, hi = np.percentile(d[met], [2.5, 97.5])
        linha[met] = pontual[(conj, m)][met]
        linha[f"{met} IC95 inf"], linha[f"{met} IC95 sup"] = lo, hi
    linhas.append(linha)
tab = pd.DataFrame(linhas)
tab.to_csv(SAIDA / "ic95_metricas.csv", index=False)


def delta(a, b, met):
    """Diferença pareada a − b na métrica `met`: estimativa pontual, IC95% e proporção de réplicas com sinal oposto."""
    d = replicas[a][met].to_numpy() - replicas[b][met].to_numpy()
    est = pontual[a][met] - pontual[b][met]
    lo, hi = np.percentile(d, [2.5, 97.5])
    p_bilateral = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return est, lo, hi, p_bilateral


comparacoes = []
PRINC = "Prontuário estruturado (30)"
for conj in CONJUNTOS:
    modelos = [m for (c, m) in probs if c == conj]
    for m in modelos:
        if m != "Gradient boosting":
            comparacoes.append((f"{m} − Gradient boosting", (conj, m), (conj, "Gradient boosting")))
    if conj == "Variáveis do CathPCI (8)":
        for m in modelos:
            if m != "CathPCI (escore publicado)":
                comparacoes.append((f"{m} − escore CathPCI publicado", (conj, m), (conj, "CathPCI (escore publicado)")))
for m in ["Regressão logística", "Gradient boosting", "XGBoost", "CatBoost", "TabPFN", "Kumo Tabular"]:
    comparacoes.append((f"{m}: 30 − variáveis do CathPCI", (PRINC, m), ("Variáveis do CathPCI (8)", m)))
    comparacoes.append((f"{m}: 40 − 30", ("Prontuário + beira de leito (40)", m), (PRINC, m)))
comparacoes.append(("Kumo Tabular − escore CathPCI publicado (30 preditores contra o escore)", (PRINC, "Kumo Tabular"), ("Variáveis do CathPCI (8)", "CathPCI (escore publicado)")))
comparacoes.append(("Gradient boosting − escore CathPCI publicado (30 preditores contra o escore)", (PRINC, "Gradient boosting"), ("Variáveis do CathPCI (8)", "CathPCI (escore publicado)")))

lin_d = []
for rotulo, a, b in comparacoes:
    for met in ["AUROC", "AUPRC", "Brier"]:
        est, lo, hi, pv = delta(a, b, met)
        lin_d.append({"Comparação": rotulo, "Conjunto": a[0] if a[0] == b[0] else f"{a[0]} vs {b[0]}", "Métrica": met,
                      "Diferença": est, "IC95 inf": lo, "IC95 sup": hi, "p (bootstrap)": pv})
tab_d = pd.DataFrame(lin_d)
tab_d.to_csv(SAIDA / "ic95_diferencas.csv", index=False)


def fmt(v, lo, hi, casas=3, pct=False):
    if pct:
        return f"{100*v:.1f} ({100*lo:.1f}–{100*hi:.1f})"
    return f"{v:.{casas}f} ({lo:.{casas}f}–{hi:.{casas}f})"


md = [f"# IC95% por bootstrap no teste (jan–jul/2026; {n} pessoas, {y.sum()} óbitos)", "",
      f"- {B} réplicas, pessoas reamostradas com reposição, semente {SEMENTE}; percentis 2,5 e 97,5.",
      "- Todas as métricas e diferenças são calculadas nas mesmas réplicas (comparações pareadas).",
      f"- Limiar de decisão = prevalência do treino ({100*LIMIAR:.2f}%). Probabilidades recalibradas; o escore CathPCI entra como publicado.",
      "", "## Métricas com IC95%", "",
      "| Conjunto | Modelo | AUROC | AUPRC | Brier | Slope | Sensibilidade (%) | Especificidade (%) |", "|---|---|---|---|---|---|---|---|"]
for _, r in tab.iterrows():
    md.append(f"| {r['Conjunto']} | {r['Modelo']} | {fmt(r['AUROC'], r['AUROC IC95 inf'], r['AUROC IC95 sup'])} | "
              f"{fmt(r['AUPRC'], r['AUPRC IC95 inf'], r['AUPRC IC95 sup'])} | {fmt(r['Brier'], r['Brier IC95 inf'], r['Brier IC95 sup'], 4)} | "
              f"{fmt(r['Slope'], r['Slope IC95 inf'], r['Slope IC95 sup'], 2)} | "
              f"{fmt(r['Sensibilidade'], r['Sensibilidade IC95 inf'], r['Sensibilidade IC95 sup'], pct=True)} | "
              f"{fmt(r['Especificidade'], r['Especificidade IC95 inf'], r['Especificidade IC95 sup'], pct=True)} |")
md += ["", "## Diferenças pareadas", "", "| Comparação | Conjunto | Métrica | Diferença (IC95%) | p |", "|---|---|---|---|---|"]
for _, r in tab_d.iterrows():
    md.append(f"| {r['Comparação']} | {r['Conjunto']} | {r['Métrica']} | {r['Diferença']:+.4f} ({r['IC95 inf']:+.4f} a {r['IC95 sup']:+.4f}) | {r['p (bootstrap)']:.3f} |")
(SAIDA / "ic95.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md[:40]))
