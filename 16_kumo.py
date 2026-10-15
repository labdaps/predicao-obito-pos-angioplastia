"""NVIDIA Kumo Tabular para óbito intra-hospitalar em até 30 dias após angioplastia — mesmo desenho do 10_modelo.py.

Comparador EXPLORATÓRIO (secundário ao TabPFN): o modelo foi anunciado em 29/09/2026, sem publicação revisada
por pares; os rankings de benchmark são do próprio fabricante. Licença dos pesos: OpenMDW-1.1.

Roda no Google Colab com GPU (CUDA). No Colab:

    !pip install -q https://github.com/NVIDIA/structured-data-models/archive/refs/heads/main.zip   # o pacote do PyPI (0.0.0a0) é placeholder vazio
    !python 16_kumo.py /content/drive/MyDrive/<pasta>/icp_dataset.csv /content/drive/MyDrive/<pasta>/kumo

Mesmo desenho do 10_modelo.py e do 14_tabpfn.py:
- dataset principal (icp_dataset.csv), 1 linha por pessoa, transferidos externos fora;
- treino (contexto) = angioplastias < 2026, teste (consulta) = 2026, checagem de pessoa nos dois lados;
- mesmos 30 preditores, ausentes mantidos (sem imputação), sem tuning.
- O desfecho do teste NUNCA entra na tabela: a coluna-alvo das linhas de teste é apagada antes de montar o tensor.

API usada = a do model card (huggingface.co/nvidia/Kumo-Tabular). O que o card não documenta (forma do retorno,
ordem das classes) é verificado em tempo de execução, sem usar o desfecho do teste.

Saídas: <saida>/metricas.md e <saida>/probs_teste.csv (id pseudonimizado, desfecho, probabilidade),
lidas pelo 15_comparacao.py.
"""
import csv
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

BASE = Path(__file__).parent
DADOS = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "dados" / "icp_dataset.csv"
SAIDA = Path(sys.argv[2]) if len(sys.argv) > 2 else BASE / "resultados" / "kumo"

DESFECHO, ANO, ANO_TESTE, SEMENTE = "obito_30d", "ano_icp", 2026, 20260929
FORA = ["id", DESFECHO, ANO, "mes_icp", "transf_externa_30d"]
NUM_ESTIMATORS = 8  # valor do exemplo oficial do model card


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


def probabilidade_obito(saida, n_esperado, prevalencia_treino):
    """O KumoTabular devolve um TableTensor com uma coluna por classe (nome = rótulo em texto;
    sdm/models/kumo/tabular/model.py). Converte com .to_pandas(), escolhe a coluna da classe 1 pelo nome e
    confere pela prevalência do TREINO (sem usar o desfecho do teste)."""
    tabela = saida.to_pandas() if hasattr(saida, "to_pandas") else pd.DataFrame(np.asarray(saida))
    print("Colunas devolvidas pelo Kumo:", list(tabela.columns))
    if tabela.shape != (n_esperado, 2):
        sys.exit(f"Formato inesperado do retorno: {tabela.shape}; esperado ({n_esperado}, 2). Mande esta mensagem.")
    if not np.allclose(tabela.sum(axis=1), 1, atol=1e-3):
        sys.exit("As linhas do retorno não somam 1: o modelo pode estar devolvendo logits. Mande esta mensagem.")
    por_nome = [c for c in tabela.columns if str(c).strip() in ("1", "1.0")]
    col = por_nome[0] if por_nome else tabela.columns[int(np.argmin(np.abs(tabela.mean().to_numpy() - prevalencia_treino)))]
    p = tabela[col].to_numpy(dtype=float)
    if abs(p.mean() - prevalencia_treino) > abs((1 - p).mean() - prevalencia_treino):
        sys.exit(f"Coluna do óbito ({col}) incoerente com a prevalência do treino. Mande esta mensagem.")
    return p


def main():
    import sdm
    import torch

    torch.manual_seed(SEMENTE)
    np.random.seed(SEMENTE)
    if not torch.cuda.is_available():
        sys.exit("Sem GPU: o Kumo Tabular é GPU-native. No Colab, use Ambiente de execução -> T4 GPU.")
    try:
        versao = version("structured-data-models")
    except PackageNotFoundError:
        versao = getattr(sdm, "__version__", "desconhecida")

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
    ytr = df.loc[treino, DESFECHO].to_numpy().astype(int)
    yte = df.loc[teste, DESFECHO].to_numpy().astype(int)
    ids_teste = df.loc[teste, "id"].astype(int).to_numpy()

    # Tabela única: primeiro as linhas de treino (contexto), depois as de teste (consulta).
    # O alvo das linhas de teste é apagado: o desfecho do teste nunca entra no tensor.
    ctx = df.loc[treino, preditores].astype(float).assign(target=ytr.astype(float))
    qry = df.loc[teste, preditores].astype(float).assign(target=np.nan)
    tabela_df = pd.concat([ctx, qry], ignore_index=True)
    n_ctx = len(ctx)

    table = sdm.TableTensor.from_pandas(
        df=tabela_df,
        stypes=sdm.infer_stypes(tabela_df, overrides={"target": "categorical"}),
        device="cuda",
    )
    modelo = sdm.models.KumoTabular(task="classification", size="large", device="cuda")
    with torch.no_grad():
        saida = modelo(
            x_context=table[:n_ctx].drop_columns("target"),
            y_context=table[:n_ctx, "target"],
            x_query=table[n_ctx:].drop_columns("target"),
            num_estimators=NUM_ESTIMATORS,
        )
    p = probabilidade_obito(saida, len(yte), ytr.mean())

    auc, ap = roc_auc_score(yte, p), average_precision_score(yte, p)
    auc_ic, ap_ic = ic_bootstrap(yte, p, roc_auc_score), ic_bootstrap(yte, p, average_precision_score)
    inter, slope = calibracao(yte, p)

    rel = [
        "# NVIDIA Kumo Tabular — óbito intra-hospitalar em até 30 dias após angioplastia (exploratório)",
        "",
        f"- Pacientes no dataset: {n_total}; excluídos por transferência externa: {n_total - len(df)}",
        f"- Preditores ({len(preditores)}), ausentes mantidos (sem imputação): {', '.join(preditores)}",
        f"- Hold-out por pessoa e por ano; pessoas em treino e teste ao mesmo tempo: {sobreposicao}",
        f"- Contexto (< {ANO_TESTE}): {n_ctx} pacientes, {ytr.sum()} óbitos; consulta ({ANO_TESTE}): {len(yte)} pacientes, {yte.sum()} óbitos",
        f"- structured-data-models {versao}; KumoTabular (size=large); num_estimators={NUM_ESTIMATORS}; "
        f"GPU {torch.cuda.get_device_name(0)}; sem tuning",
        "- Modelo anunciado em 29/09/2026, sem publicação revisada por pares; licença dos pesos OpenMDW-1.1",
        "",
        "| Modelo | AUROC (IC95%) | AUPRC (IC95%) | Brier | Intercepto calib. | Slope calib. |",
        "|---|---|---|---:|---:|---:|",
        f"| Kumo Tabular | {auc:.3f} ({auc_ic[0]:.3f}–{auc_ic[1]:.3f}) | {ap:.3f} ({ap_ic[0]:.3f}–{ap_ic[1]:.3f}) "
        f"| {brier_score_loss(yte, p):.4f} | {inter:.2f} | {slope:.2f} |",
        "",
        f"AUPRC de referência (sem modelo) = prevalência no teste = {yte.mean():.3f}.",
        "",
        "## Calibração por decil — Kumo Tabular",
        "",
        tabela_decis(yte, p),
    ]
    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / "metricas.md").write_text("\n".join(rel) + "\n", encoding="utf-8")
    with open(SAIDA / "probs_teste.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "obito_30d", "Kumo Tabular"])
        w.writerows(zip(ids_teste, yte, p))
    print("\n".join(rel))


if __name__ == "__main__":
    main()
