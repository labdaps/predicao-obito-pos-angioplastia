"""Gera notebooks/ICP_notebook_unico.ipynb: TODAS as análises de modelos do paper num notebook só.

Junta, em sequência, os quatro notebooks que o gera_notebook_gpu.py produz (mesmo código, sem duplicar lógica):
  Parte A  expandido  30 preditores originais + 10 novas
  Parte B  cathpci    8 variáveis do NCDR CathPCI + o escore publicado
  Parte C  mayo       7 variáveis do Mayo Clinic Risk Score
  Parte D  originais  30 preditores (análise principal do paper)
  Comparação entre os conjuntos (tabela e figura)
  SHAP aproximado do Kumo nos 30 preditores (por último: é o trecho longo e só roda com GPU)

Cada parte repete o roteiro completo: Optuna, TabPFN, Kumo, recalibração, métricas, ROC, calibração, SHAP de
árvore, ponto de operação, curva de decisão, capacidade, equidade e tabela de características.

uso: python gera_notebook_unico.py [saida.ipynb]
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat as nbf

AQUI = Path(__file__).parent
saida = Path(sys.argv[1]) if len(sys.argv) > 1 else AQUI / "ICP_notebook_unico.ipynb"

PARTES = [("A", "cathpci", "8 variáveis do NCDR CathPCI de beira de leito + o escore publicado (comparador)"),
          ("B", "mayo", "7 variáveis do Mayo Clinic Risk Score (comparador)"),
          ("C", "sem_idade", "Sensibilidade: 40 preditores sem a idade"),
          ("D", "vespera", "Sensibilidade: 40 preditores com exames só até a véspera"),
          ("E", "casos_completos", "Sensibilidade: 40 preditores, treino só com os 7 exames presentes"),
          ("F", "expandido", "Análise principal: 40 preditores (prontuário estruturado + dados de beira de leito), com SHAP")]


def gera(conjunto, destino):
    args = ["--conjunto", conjunto]
    subprocess.run([sys.executable, str(AQUI / "gera_notebook_gpu.py"), *args, str(destino)], check=True, capture_output=True)
    return nbf.read(destino, 4).cells


def indice(cells, comeco):
    return next(i for i, c in enumerate(cells) if c.cell_type == "markdown" and c.source.startswith(comeco))


md = lambda s: nbf.v4.new_markdown_cell(s.strip("\n"))
code = lambda s: nbf.v4.new_code_cell(s.strip("\n"))

with tempfile.TemporaryDirectory() as tmp:
    fontes = {conj: gera(conj, Path(tmp) / f"{conj}.ipynb") for _, conj, _ in PARTES}

cells = [md("""
# Óbito na internação da angioplastia (até 30 dias) — notebook único

- **Dados:** `icp_dataset.csv` e `icp_extra.csv`, 1 linha por pessoa (primeira angioplastia), **sem identificadores** (ID pseudonimizado).
- **Desfecho:** óbito na internação da angioplastia (atendimentos separados por até 12 horas contam como a mesma internação), em até 30 dias.
- **Desenho:** treino = angioplastias de 2023–2024; teste = angioplastias de 2025; nenhuma pessoa nos dois conjuntos; transferidos para hospital externo fora.
- **Partes, mesmo roteiro em cada uma:** A) variáveis do CathPCI (8) com o escore publicado; B) variáveis do Mayo (7); C–E) sensibilidades da análise principal (sem idade; exames só até a véspera; treino com casos completos); F) **análise principal, 40 preditores**.
- **Seis modelos por conjunto:** regressão logística, gradient boosting, XGBoost e CatBoost (hiperparâmetros por Optuna, validação cruzada de 5 folds só no treino), TabPFN e Kumo Tabular (sem ajuste).
- **Calibração e limiar:** recalibração logística com predições fora da amostra do treino; limiar de decisão = prevalência do desfecho no treino.
- **Métricas:** valores pontuais no teste (sem bootstrap).
- **Na análise principal:** SHAP do modelo com maior AUROC no teste. **Ao final:** comparação entre as partes.
""")]

# ---- preparação comum: configuração (da parte D), sementes, ambiente e instalação
base = fontes["expandido"]
i0, i2 = indice(base, "## 0) Configuração"), indice(base, "## 2) Dados")
cells += [md("# Preparação")] + base[i0:i2]
cells.append(code("""
#guarda as probabilidades de cada parte para a comparação final
PARTES = {}
"""))

kumo_shap = []
for letra, conj, descricao in PARTES:
    fonte = fontes[conj]
    cfg = fonte[indice(fonte, "## 0) Configuração") + 1]                     # célula de configuração da variante (pasta de saída)
    corpo = fonte[indice(fonte, "## 2) Dados"):]
    cells.append(md(f"# Parte {letra} — {descricao}"))
    cells.append(code(f"""
#começa a parte {letra} do zero: as probabilidades originais da parte anterior não podem ser reaproveitadas
globals().pop("probs_brutas", None)
print("Parte {letra}: conjunto '{conj}'")
"""))
    cells.append(nbf.v4.new_code_cell(cfg.source))
    for c in corpo:
        novo = nbf.v4.new_markdown_cell(c.source) if c.cell_type == "markdown" else nbf.v4.new_code_cell(c.source)
        if c.cell_type == "markdown" and c.source.startswith("## "):
            novo.source = f"## {letra} · " + c.source[3:]
        cells.append(novo)
    cells.append(code(f"""
PARTES["{conj}"] = {{"probs": dict(probs), "erros": dict(erros), "saida": SAIDA, "n_preditores": len(preditores)}}
print("Parte {letra} concluída |", len(probs), "modelos | erros:", erros or "nenhum")
"""))

cells += [md("# Comparação entre os conjuntos de preditores"),
          md("→ Mesmo teste (2025) e mesmo desenho em todas as partes; só mudam os preditores. Valores pontuais: diferenças pequenas entre modelos ou conjuntos não permitem afirmar superioridade."),
          code("""
ROTULOS = {"cathpci": "CathPCI (8)", "mayo": "Mayo (7)", "expandido": "Principal (40)", "sem_idade": "Sem idade (39)",
           "vespera": "Exames até a véspera (40)", "casos_completos": "Treino com casos completos (40)"}
ORDEM = [c for c in ["cathpci", "mayo", "expandido", "sem_idade", "vespera", "casos_completos"] if c in PARTES]
SAIDA_GERAL = Path("../resultados/comparacao_conjuntos") if Path("../resultados").exists() else Path("resultados_comparacao")
SAIDA_GERAL.mkdir(exist_ok=True)

linhas_cmp = []
for conj in ORDEM:
  for nome, p in PARTES[conj]["probs"].items():
    inter, slope = calibracao(yte, p)
    sinal = p >= LIMIAR
    linhas_cmp.append({"Conjunto": ROTULOS[conj], "Modelo": nome, "AUROC": roc_auc_score(yte, p), "AUPRC": average_precision_score(yte, p),
                       "Brier": brier_score_loss(yte, p), "Intercepto": inter, "Slope": slope,
                       "Sensibilidade (%)": 100 * sinal[yte == 1].mean(), "Especificidade (%)": 100 * (~sinal)[yte == 0].mean()})
tab_cmp = pd.DataFrame(linhas_cmp)
tab_cmp.to_csv(SAIDA_GERAL / "comparacao_conjuntos.csv", index=False)

pd.set_option("display.width", 250)
for metrica in ["AUROC", "AUPRC", "Brier"]:
  print(f"\\n{metrica} no teste, por modelo (linhas) e conjunto de preditores (colunas)")
  print(tab_cmp.pivot(index="Modelo", columns="Conjunto", values=metrica)[[ROTULOS[c] for c in ORDEM]].round(4).to_string())
print("\\nErros por parte:", {c: PARTES[c]["erros"] for c in ORDEM if PARTES[c]["erros"]} or "nenhum")
"""),
          code("""
fig, eixos = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, metrica in zip(eixos, ["AUROC", "AUPRC"]):
  for nome in tab_cmp["Modelo"].unique():
    t = tab_cmp[tab_cmp["Modelo"] == nome].set_index("Conjunto").reindex([ROTULOS[c] for c in ORDEM])
    ax.plot(t.index, t[metrica], marker="o", lw=1.8, color=cores.get(nome), label=nome)
  ax.set_ylabel(metrica + " no teste"); ax.set_xlabel("Conjunto de preditores"); ax.grid(alpha=0.3)
  ax.set_title(f"{metrica} por conjunto de preditores")
eixos[1].axhline(yte.mean(), color="gray", ls="--", lw=1, label="Prevalência (referência)")
eixos[0].legend(frameon=False, fontsize=9); eixos[1].legend(frameon=False, fontsize=9)
fig.suptitle(f"Teste: 2025 (n = {len(yte)}, {yte.sum()} óbitos)")
fig.tight_layout()
fig.savefig(SAIDA_GERAL / "comparacao_conjuntos.png", dpi=300)
plt.show()
""")]

cells.append(code("""
print("Notebook único concluído. Erros por parte:", {c: PARTES[c]["erros"] for c in PARTES if PARTES[c]["erros"]} or "nenhum")
"""))

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
nbf.write(nb, str(saida))
print("ok", saida, len(cells), "celulas")
