# Predição de óbito intra-hospitalar após angioplastia coronária

Código de análise de um estudo de desenvolvimento e validação temporal de modelos de predição
(regressão logística, gradient boosting, XGBoost, CatBoost, TabPFN e Kumo Tabular) para óbito
intra-hospitalar após angioplastia coronária, a partir de dados estruturados de prontuário
registrados antes do procedimento. Reporte segundo o TRIPOD+AI.

**Este repositório não contém dados.** Os dados de pacientes não podem ser compartilhados. Os
scripts esperam, na pasta `dados/` (não versionada), um arquivo por pessoa com os preditores, o
ano do procedimento e o desfecho (`icp_dataset.csv`) e, para os conjuntos ampliados,
`icp_extra.csv`, ligados por um identificador pseudonimizado.

## Conteúdo
- `notebooks/gera_notebook_gpu.py` — gera o notebook de análise (um conjunto de preditores);
  `--conjunto expandido|cathpci|mayo` gera as variantes; `--shap-kumo` acrescenta o SHAP do Kumo.
- `notebooks/gera_notebook_unico.py` — junta as variantes num notebook único
  (`notebooks/ICP_notebook_unico.ipynb`, sem saídas).
- `10_modelo.py` a `19_treino_casos_completos.py` — análises auxiliares e de sensibilidade.
- `21_bootstrap_ic.py` — intervalos de confiança por bootstrap pareado.

## Desenho, em resumo
Treino e teste separados por período (validação temporal), uma linha por pessoa e nenhuma pessoa
nos dois conjuntos; hiperparâmetros por Optuna em validação cruzada só no treino; recalibração
logística com predições fora da amostra do treino; limiar de decisão igual à prevalência do
treino; semente única para todos os processos aleatórios.
