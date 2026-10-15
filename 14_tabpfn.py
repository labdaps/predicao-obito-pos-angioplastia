"""TabPFN para óbito intra-hospitalar em até 30 dias após angioplastia — mesmo desenho do 10_modelo.py.

Roda no Google Colab com GPU (a máquina local não tem GPU nem PyTorch). No Colab:

    !pip install -q tabpfn
    import os; os.environ["TABPFN_TOKEN"] = "<seu token de ux.priorlabs.ai>"   # não gravar o token no arquivo
    !python 14_tabpfn.py /content/drive/MyDrive/<pasta>/icp_dataset.csv /content/drive/MyDrive/<pasta>/tabpfn

Mesmo desenho do 10_modelo.py:
- dataset principal (icp_dataset.csv), 1 linha por pessoa, transferidos externos fora;
- treino = angioplastias < 2026, teste = 2026, checagem de pessoa nos dois lados;
- mesmos 30 preditores. O TabPFN recebe os ausentes como estão (sem imputação nem padronização) e
  usa o treino só como contexto: não há tuning, então não há escolha feita com o teste.

Saídas: <saida>/metricas.md e <saida>/probs_teste.csv (id pseudonimizado, desfecho, probabilidade),
para a comparação pareada com o gradient boosting no 15_comparacao.py.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

BASE = Path(__file__).parent
DADOS = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "dados" / "icp_dataset.csv"
SAIDA = Path(sys.argv[2]) if len(sys.argv) > 2 else BASE / "resultados" / "tabpfn"

DESFECHO, ANO, ANO_TESTE, SEMENTE = "obito_30d", "ano_icp", 2026, 20260929
FORA = ["id", DESFECHO, ANO, "mes_icp", "transf_externa_30d"]


def calibracao(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    slope = LogisticRegression(C=1e6).fit(lp, y).coef_[0][0]
    intercepto = np.log(y.mean() / (1 - y.mean())) - np.log(p.mean() / (1 - p.mean()))
    return intercepto, slope


def ic_bootstrap(y, p, metrica, n=1000):
    rng = np.random.default_rng(SEMENTE)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            vals.append(metrica(y[idx], p[idx]))
    return np.percentile(vals, [2.5, 97.5])


def tabela_decis(y, p):
    linhas = ["| Decil | N | Previsto (%) | Observado (%) |", "|---:|---:|---:|---:|"]
    for i, g in enumerate(np.array_split(np.argsort(p), 10), 1):
        linhas.append(f"| {i} | {len(g)} | {100 * p[g].mean():.2f} | {100 * y[g].mean():.2f} |")
    return "\n".join(linhas)


def main():
    from tabpfn import TabPFNClassifier
    import torch

    df = pd.read_csv(DADOS)
    df.columns = df.columns.str.lower()
    n_total = len(df)
    df = df[df["transf_externa_30d"] != 1]

    if not df["id"].is_unique:
        sys.exit("ID repetido: o dataset nao esta no grao de pessoa")
    treino, teste = df[ANO] < ANO_TESTE, df[ANO] >= ANO_TESTE
    sobreposicao = len(set(df.loc[treino, "id"]) & set(df.loc[teste, "id"]))
    if sobreposicao:
        sys.exit(f"{sobreposicao} pessoas no treino E no teste")

    preditores = [c for c in df.columns if c not in FORA]
    Xtr, ytr = df.loc[treino, preditores].astype(float), df.loc[treino, DESFECHO].to_numpy().astype(int)
    Xte, yte = df.loc[teste, preditores].astype(float), df.loc[teste, DESFECHO].to_numpy().astype(int)

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print("Dispositivo:", dispositivo, "| treino:", Xtr.shape, "| teste:", Xte.shape)

    modelo = TabPFNClassifier(random_state=SEMENTE, device=dispositivo)
    modelo.fit(Xtr, ytr)
    p = modelo.predict_proba(Xte)[:, list(modelo.classes_).index(1)]

    auc, ap = roc_auc_score(yte, p), average_precision_score(yte, p)
    auc_ic, ap_ic = ic_bootstrap(yte, p, roc_auc_score), ic_bootstrap(yte, p, average_precision_score)
    inter, slope = calibracao(yte, p)

    rel = [
        "# TabPFN — óbito intra-hospitalar em até 30 dias após angioplastia",
        "",
        f"- Pacientes no dataset: {n_total}; excluídos por transferência externa: {n_total - len(df)}",
        f"- Preditores ({len(preditores)}), ausentes mantidos (sem imputação): {', '.join(preditores)}",
        f"- Hold-out por pessoa e por ano; pessoas em treino e teste ao mesmo tempo: {sobreposicao}",
        f"- Treino (< {ANO_TESTE}): {int(treino.sum())} pacientes, {ytr.sum()} óbitos; teste ({ANO_TESTE}): {int(teste.sum())} pacientes, {yte.sum()} óbitos",
        f"- Versão do TabPFN: {__import__('tabpfn').__version__}; dispositivo: {dispositivo}; sem tuning (treino usado como contexto)",
        "",
        "| Modelo | AUROC (IC95%) | AUPRC (IC95%) | Brier | Intercepto calib. | Slope calib. |",
        "|---|---|---|---:|---:|---:|",
        f"| TabPFN | {auc:.3f} ({auc_ic[0]:.3f}–{auc_ic[1]:.3f}) | {ap:.3f} ({ap_ic[0]:.3f}–{ap_ic[1]:.3f}) "
        f"| {brier_score_loss(yte, p):.4f} | {inter:.2f} | {slope:.2f} |",
        "",
        f"AUPRC de referência (sem modelo) = prevalência no teste = {yte.mean():.3f}.",
        "",
        "## Calibração por decil — TabPFN",
        "",
        tabela_decis(yte, p),
    ]
    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / "metricas.md").write_text("\n".join(rel) + "\n", encoding="utf-8")
    with open(SAIDA / "probs_teste.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "obito_30d", "TabPFN"])
        w.writerows(zip(df.loc[teste, "id"].astype(int), yte, p))
    print("\n".join(rel))


if __name__ == "__main__":
    main()
