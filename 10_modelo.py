"""Predição de óbito hospitalar em 30 dias após angioplastia coronária.

Lê o dataset desidentificado (dados/icp_dataset.csv), treina
regressão logística (referência) e gradient boosting, com validação temporal
(treino: angioplastias de 2024-2025; teste: 2026), e grava o relatório em
resultados/metricas.md.

Só usa numpy e scikit-learn (é o que está instalado nesta máquina).
"""
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent
# uso: python 10_modelo.py [dataset.csv] [pasta_saida]
DADOS = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "dados" / "icp_dataset.csv"
SAIDA = Path(sys.argv[2]) if len(sys.argv) > 2 else BASE / "resultados"

DESFECHO = "obito_30d"
ANO = "ano_icp"
ANO_TESTE = 2026
# Colunas que nunca entram como preditor
FORA = {"id", DESFECHO, ANO, "mes_icp", "transf_externa_30d"}
# Transferido para hospital externo em 30 dias: desfecho nao observado, fica fora da analise principal
CENSURA = "transf_externa_30d"
SEMENTE = 20260929


def ler_csv(caminho):
    with open(caminho, newline="", encoding="utf-8") as f:
        linhas = [{k.strip().lower(): v for k, v in l.items()} for l in csv.DictReader(f)]
    if not linhas:
        sys.exit(f"dataset vazio: {caminho}")
    colunas = list(linhas[0].keys())

    def num(v):
        v = (v or "").strip().replace(",", ".")
        return float(v) if v not in ("", "NA", "null") else np.nan

    matriz = {c: np.array([num(l[c]) for l in linhas]) for c in colunas}
    return colunas, matriz


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibracao(y, p):
    """Intercepto (calibration-in-the-large) e slope pela recalibração logística."""
    lp = logit(p).reshape(-1, 1)
    slope = LogisticRegression(C=1e6).fit(lp, y).coef_[0][0]
    intercepto = np.log(y.mean() / (1 - y.mean())) - np.log(p.mean() / (1 - p.mean()))
    return intercepto, slope


def ic_bootstrap(y, p, metrica, n=1000):
    rng = np.random.default_rng(SEMENTE)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        vals.append(metrica(y[idx], p[idx]))
    return np.percentile(vals, [2.5, 97.5])


def tabela_decis(y, p):
    ordem = np.argsort(p)
    grupos = np.array_split(ordem, 10)
    linhas = ["| Decil | N | Previsto (%) | Observado (%) |", "|---:|---:|---:|---:|"]
    for i, g in enumerate(grupos, 1):
        linhas.append(f"| {i} | {len(g)} | {100 * p[g].mean():.2f} | {100 * y[g].mean():.2f} |")
    return "\n".join(linhas)


def main():
    colunas, m = ler_csv(DADOS)
    n_total = len(m[DESFECHO])
    if CENSURA in m:
        manter = m[CENSURA] != 1
        m = {c: v[manter] for c, v in m.items()}
    n_censurados = n_total - len(m[DESFECHO])
    preditores = [c for c in colunas if c not in FORA]
    X = np.column_stack([m[c] for c in preditores])
    y = m[DESFECHO].astype(int)
    ano = m[ANO]

    treino, teste = ano < ANO_TESTE, ano >= ANO_TESTE
    # Hold-out por pessoa: o dataset tem 1 linha por pessoa (ID pseudonimizado de pessoa),
    # entao treino e teste precisam ser disjuntos em ID.
    ids = m["id"]
    if len(np.unique(ids)) != len(ids):
        sys.exit("ID repetido: o dataset nao esta no grao de pessoa")
    sobreposicao = len(set(ids[treino]) & set(ids[teste]))
    if sobreposicao:
        sys.exit(f"{sobreposicao} pessoas no treino E no teste")
    Xtr, ytr, Xte, yte = X[treino], y[treino], X[teste], y[teste]

    modelos = {
        "Regressão logística": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(max_iter=5000, C=1.0),
        ),
        "Gradient boosting": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=400, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.2, random_state=SEMENTE,
        ),
    }

    rel = [
        "# Óbito hospitalar em 30 dias após angioplastia — métricas",
        "",
        f"- Pacientes no dataset: {n_total}; excluídos por transferência externa em 30 dias (desfecho não observado): {n_censurados}",
        f"- Preditores ({len(preditores)}): {', '.join(preditores)}",
        f"- Hold-out por pessoa e por ano: 1 linha por pessoa (primeira angioplastia); pessoas em treino e teste ao mesmo tempo: {sobreposicao}",
        f"- Treino (angioplastias < {ANO_TESTE}): {treino.sum()} pacientes, {ytr.sum()} óbitos ({100 * ytr.mean():.2f}%)",
        f"- Teste ({ANO_TESTE}): {teste.sum()} pacientes, {yte.sum()} óbitos ({100 * yte.mean():.2f}%)",
        "",
        "| Modelo | AUROC (IC95%) | AUPRC (IC95%) | Brier | Intercepto calib. | Slope calib. |",
        "|---|---|---|---:|---:|---:|",
    ]
    decis = []
    probs = {"id": ids[teste].astype(int), "obito_30d": yte}
    for nome, modelo in modelos.items():
        modelo.fit(Xtr, ytr)
        p = modelo.predict_proba(Xte)[:, 1]
        probs[nome] = p
        auc, ap = roc_auc_score(yte, p), average_precision_score(yte, p)
        auc_ic = ic_bootstrap(yte, p, roc_auc_score)
        ap_ic = ic_bootstrap(yte, p, average_precision_score)
        inter, slope = calibracao(yte, p)
        rel.append(
            f"| {nome} | {auc:.3f} ({auc_ic[0]:.3f}–{auc_ic[1]:.3f}) | {ap:.3f} ({ap_ic[0]:.3f}–{ap_ic[1]:.3f}) "
            f"| {brier_score_loss(yte, p):.4f} | {inter:.2f} | {slope:.2f} |"
        )
        decis += ["", f"## Calibração por decil — {nome}", "", tabela_decis(yte, p)]

    rel += ["", f"AUPRC de referência (sem modelo) = prevalência no teste = {yte.mean():.3f}."] + decis
    SAIDA.mkdir(exist_ok=True)
    (SAIDA / "metricas.md").write_text("\n".join(rel) + "\n", encoding="utf-8")
    # Probabilidades do teste por ID pseudonimizado, para comparacao pareada com outros modelos (14_tabpfn.py)
    with open(SAIDA / "probs_teste.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(probs.keys()))
        w.writerows(zip(*probs.values()))
    print("\n".join(rel))


if __name__ == "__main__":
    main()
