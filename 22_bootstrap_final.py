"""IC95% por bootstrap pareado para a rodada final (resultados/gpu_lab_final/).

Reamostra as pessoas do teste (2.000 réplicas, semente fixa) e recalcula, nas mesmas réplicas, as métricas de todos
os modelos de todas as partes, de modo que todas as diferenças são pareadas. Gera:
  - métricas com IC95% de cada modelo em cada parte;
  - diferenças par a par entre os seis modelos na análise principal (40 preditores);
  - análise principal contra comparadores (variáveis do CathPCI e do Mayo, escores CathPCI e Mayo publicados);
  - cada sensibilidade contra a análise principal, mesmo modelo;
  - desempenho por estrato urgente/eletivo e no subconjunto do teste com os 7 exames (análise principal e treino com casos completos).

uso: python 22_bootstrap_final.py
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

BASE = Path(__file__).parent
G = BASE / "resultados" / "gpu_lab_final"
SAIDA = G / "bootstrap"
SAIDA.mkdir(exist_ok=True)
SEMENTE, B = 20260929, 2000
PARTES = {"principal": "resultados_expandido", "cathpci": "resultados_cathpci", "mayo": "resultados_mayo",
          "sem_idade": "resultados_sem_idade", "vespera": "resultados_vespera", "casos_completos": "resultados_casos_completos"}
MODELOS = ["Regressão logística", "Gradient boosting", "XGBoost", "CatBoost", "TabPFN", "Kumo Tabular"]
DESF = "obito_internacao_30d"

# dataset (para estratos e limiar)
ds = pd.read_csv(BASE / "dados" / "icp_dataset.csv"); ds.columns = ds.columns.str.lower()
ds = ds[ds["transf_externa_30d"] != 1]
ds = ds[ds["ano_icp"].between(2023, 2025)]
LIMIAR = float(ds.loc[ds["ano_icp"] < 2025, DESF].mean())

probs, y, ids = {}, None, None
for parte, pasta in PARTES.items():
    t = pd.read_csv(G / pasta / "probs_teste_todos.csv").sort_values("id").reset_index(drop=True)
    col_y = DESF if DESF in t.columns else "obito_30d"
    if ids is None:
        ids, y = t["id"].to_numpy(), t[col_y].to_numpy().astype(int)
    assert (t["id"].to_numpy() == ids).all() and (t[col_y].to_numpy() == y).all()
    for mcol in [c for c in t.columns if c not in ("id", DESF, "obito_30d")]:
        probs[(parte, mcol)] = t[mcol].to_numpy(dtype=float)

# escore Mayo publicado (26_escore_mayo.py), mesmo teste
_mayo = pd.read_csv(G / "resultados_mayo" / "probs_mayo_publicado.csv").set_index("id").loc[ids]
assert (_mayo["obito_30d"].to_numpy() == y).all()
probs[("mayo", "Mayo (escore publicado)")] = _mayo["Mayo (escore publicado)"].to_numpy(dtype=float)

te = ds.set_index("id").loc[ids]
EXAMES = ["creatinina", "hemoglobina", "leucocitos", "plaquetas", "potassio", "sodio", "ureia"]
urgente = ((te["fl_codigo_iam"] == 1) | (te["atend_48h_antes"] == 1) | (te["apres_iam_qualquer"] == 1)).to_numpy()
completo = te[EXAMES].notna().all(axis=1).to_numpy()
SUBS = {"teste inteiro": np.ones(len(y), bool), "urgente": urgente, "eletivo": ~urgente, "com os 7 exames": completo}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def metricas(yv, p):
    if yv.sum() == 0 or yv.sum() == len(yv):
        return {k: np.nan for k in ["AUROC", "AUPRC", "Brier", "Intercepto", "Slope", "Sensibilidade", "Especificidade"]}
    sinal = p >= LIMIAR
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), yv).coef_[0][0]
    inter = np.log(yv.mean() / (1 - yv.mean())) - np.log(p.mean() / (1 - p.mean()))
    return {"AUROC": roc_auc_score(yv, p), "AUPRC": average_precision_score(yv, p), "Brier": brier_score_loss(yv, p),
            "Intercepto": inter, "Slope": slope, "Sensibilidade": sinal[yv == 1].mean(), "Especificidade": (~sinal)[yv == 0].mean()}


# o que entra em cada réplica: todas as partes no teste inteiro; a análise principal e a de casos completos também nos subgrupos
chaves = [(k, "teste inteiro") for k in probs] + [((p, m), s) for p in ("principal", "casos_completos") for m in MODELOS
                                                   for s in ("urgente", "eletivo", "com os 7 exames") if (p, m) in probs]
pontual = {(k, s): metricas(y[SUBS[s]], probs[k][SUBS[s]]) for k, s in chaves}
rng = np.random.default_rng(SEMENTE)
reps = {c: [] for c in chaves}
n = len(y)
for b in range(B):
    idx = rng.integers(0, n, n)
    for (k, s) in chaves:
        sel = idx[SUBS[s][idx]]
        reps[(k, s)].append(metricas(y[sel], probs[k][sel]))
reps = {c: pd.DataFrame(v) for c, v in reps.items()}

linhas = []
for (k, s), d in reps.items():
    r = {"Parte": k[0], "Modelo": k[1], "Avaliação": s, "N": int(SUBS[s].sum()), "Óbitos": int(y[SUBS[s]].sum())}
    for met in d.columns:
        r[met] = pontual[(k, s)][met]
        r[f"{met} inf"], r[f"{met} sup"] = np.nanpercentile(d[met], [2.5, 97.5])
    linhas.append(r)
tab = pd.DataFrame(linhas)
tab.to_csv(SAIDA / "ic95_metricas.csv", index=False)


def delta(a, b, met, s="teste inteiro"):
    d = reps[(a, s)][met].to_numpy() - reps[(b, s)][met].to_numpy()
    est = pontual[(a, s)][met] - pontual[(b, s)][met]
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return est, lo, hi, min(1.0, 2 * min(np.nanmean(d <= 0), np.nanmean(d >= 0)))


comps = []
for a, b in combinations(MODELOS, 2):                                         # par a par na análise principal
    comps.append(("principal: modelos par a par", f"{b} − {a}", ("principal", b), ("principal", a)))
for m in MODELOS:
    comps.append(("principal contra variáveis do CathPCI", m, ("principal", m), ("cathpci", m)))
    comps.append(("principal contra variáveis do Mayo", m, ("principal", m), ("mayo", m)))
    comps.append(("principal contra escore CathPCI publicado", m, ("principal", m), ("cathpci", "CathPCI (escore publicado)")))
    comps.append(("variáveis do CathPCI contra escore publicado", m, ("cathpci", m), ("cathpci", "CathPCI (escore publicado)")))
    comps.append(("principal contra escore Mayo publicado", m, ("principal", m), ("mayo", "Mayo (escore publicado)")))
    comps.append(("variáveis do Mayo contra escore publicado", m, ("mayo", m), ("mayo", "Mayo (escore publicado)")))
    for sens in ("sem_idade", "vespera", "casos_completos"):
        comps.append((f"{sens} contra principal", m, (sens, m), ("principal", m)))
lin_d = []
for grupo, rot, a, b in comps:
    for met in ["AUROC", "AUPRC", "Brier"]:
        est, lo, hi, pv = delta(a, b, met)
        lin_d.append({"Comparação": grupo, "Modelo/par": rot, "Métrica": met, "Diferença": est, "IC95 inf": lo, "IC95 sup": hi, "p": pv})
for m in MODELOS:                                                              # casos completos contra principal no subconjunto completo
    for met in ["AUROC", "AUPRC"]:
        est, lo, hi, pv = delta(("casos_completos", m), ("principal", m), met, "com os 7 exames")
        lin_d.append({"Comparação": "casos_completos contra principal (teste com os 7 exames)", "Modelo/par": m, "Métrica": met,
                      "Diferença": est, "IC95 inf": lo, "IC95 sup": hi, "p": pv})
tab_d = pd.DataFrame(lin_d)
tab_d.to_csv(SAIDA / "ic95_diferencas.csv", index=False)

f3 = lambda v, lo, hi, c=3: f"{v:.{c}f} ({lo:.{c}f}–{hi:.{c}f})"
md = [f"# IC95% por bootstrap pareado — rodada final (teste 2025; {n} pessoas, {y.sum()} óbitos)", "",
      f"- {B} réplicas, semente {SEMENTE}, percentis 2,5 e 97,5; mesmas réplicas para todos os modelos e partes.",
      f"- Limiar = prevalência do treino ({100*LIMIAR:.2f}%). Urgente = angioplastia faturada como infarto, atendimento nas 48 h anteriores ou infarto na apresentação.", "",
      "## Métricas", "", "| Parte | Modelo | Avaliação | N | Óbitos | AUROC | AUPRC | Brier | Slope | Sensibilidade | Especificidade |",
      "|---|---|---|---:|---:|---|---|---|---|---|---|"]
for _, r in tab.iterrows():
    md.append(f"| {r['Parte']} | {r['Modelo']} | {r['Avaliação']} | {r['N']} | {r['Óbitos']} | {f3(r['AUROC'], r['AUROC inf'], r['AUROC sup'])} | "
              f"{f3(r['AUPRC'], r['AUPRC inf'], r['AUPRC sup'])} | {f3(r['Brier'], r['Brier inf'], r['Brier sup'], 4)} | {f3(r['Slope'], r['Slope inf'], r['Slope sup'], 2)} | "
              f"{f3(100*r['Sensibilidade'], 100*r['Sensibilidade inf'], 100*r['Sensibilidade sup'], 1)} | {f3(100*r['Especificidade'], 100*r['Especificidade inf'], 100*r['Especificidade sup'], 1)} |")
md += ["", "## Diferenças pareadas", "", "| Comparação | Modelo/par | Métrica | Diferença (IC95%) | p |", "|---|---|---|---|---|"]
for _, r in tab_d.iterrows():
    md.append(f"| {r['Comparação']} | {r['Modelo/par']} | {r['Métrica']} | {r['Diferença']:+.4f} ({r['IC95 inf']:+.4f} a {r['IC95 sup']:+.4f}) | {r['p']:.3f} |")
(SAIDA / "ic95.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("ok", len(tab), "linhas de métricas,", len(tab_d), "diferenças")
