"""Gera notebooks/ICP_modelos_fundacionais_GPU.ipynb: pipeline completo do paper para rodar numa máquina com GPU.

Reúne, num só notebook, o que está em 10_modelo.py (regressão logística e gradient boosting), 14_tabpfn.py (TabPFN),
16_kumo.py (Kumo Tabular) e 15_comparacao.py (comparação pareada). Mesmo desenho e mesmas sementes.
"""
import sys
from pathlib import Path

import nbformat as nbf

# uso: python gera_notebook_gpu.py [--shap-kumo] [--conjunto NOME] [saida.ipynb]
#   sem opção         -> notebook principal (30 preditores; roda em CPU ou GPU; SHAP do melhor modelo de árvore)
#   --shap-kumo       -> com uma seção a mais (só GPU): SHAP aproximado do Kumo Tabular
#   --conjunto NOME   -> outro conjunto de preditores, no mesmo desenho:
#        expandido  30 originais + 10 novas (sinais vitais, SAPS3, fração de ejeção), de icp_extra.csv
#        cathpci    8 variáveis do NCDR CathPCI de beira de leito + o escore publicado como comparador
#        mayo       7 variáveis do Mayo Clinic Risk Score
SHAP_KUMO = "--shap-kumo" in sys.argv
CONJUNTO = sys.argv[sys.argv.index("--conjunto") + 1] if "--conjunto" in sys.argv else "originais"
assert CONJUNTO in ("originais", "expandido", "cathpci", "mayo", "sem_idade", "vespera", "casos_completos"), CONJUNTO
_args = [a for i, a in enumerate(sys.argv[1:], 1) if a not in ("--shap-kumo", "--conjunto") and sys.argv[i - 1] != "--conjunto"]
if CONJUNTO != "originais":
    _padrao = f"ICP_modelos_{CONJUNTO}.ipynb"
else:
    _padrao = "ICP_modelos_GPU_shap_kumo.ipynb" if SHAP_KUMO else "ICP_modelos_fundacionais_GPU.ipynb"
saida = Path(_args[0]) if _args else Path(__file__).parent / _padrao

DESCRICAO_CONJUNTO = {
    "expandido": "os 30 preditores originais mais 10 variáveis novas, todas registradas antes de t0: frequência cardíaca, pressão sistólica e diastólica, Glasgow, qSOFA, NEWS2, insuficiência cardíaca e droga vasoativa do formulário SAPS3, fração de ejeção (regex no laudo de ecocardiograma) e dias desde o eco",
    "cathpci": "as 8 variáveis do NCDR CathPCI de beira de leito (Peterson et al., JACC 2010): idade, choque, insuficiência cardíaca, doença arterial periférica, doença pulmonar crônica, filtração glomerular, infarto com supra e status do procedimento. NYHA IV não existe no dado. O escore publicado entra como comparador, sem treino",
    "sem_idade": "análise de sensibilidade: o conjunto principal de 40 preditores sem a idade",
    "vespera": "análise de sensibilidade: o conjunto principal de 40 preditores, com os exames laboratoriais restritos aos coletados até a véspera da angioplastia",
    "casos_completos": "análise de sensibilidade: o conjunto principal de 40 preditores, com o treino restrito às pessoas com os 7 exames laboratoriais",
    "mayo": "as 7 variáveis do Mayo Clinic Risk Score (Singh et al., Mayo Clin Proc 2007): idade, creatinina, fração de ejeção, choque, infarto nas 24 h anteriores, insuficiência cardíaca e doença arterial periférica. O escore em pontos não é calculado (tabela de pontuação não conferida)",
}
BLOCO_PREDITORES = '''
#--- conjunto de preditores: __CONJUNTO__
_extra = pd.read_csv("icp_extra.csv" if Path("icp_extra.csv").exists() else "../dados/icp_extra.csv")
_extra.columns = _extra.columns.str.lower()
_originais = [c for c in df.columns if c not in FORA]
df = df.merge(_extra, on="id", how="left", validate="one_to_one")       # mantém a ordem das linhas
_mulher = df["sexo_f"] == 1
_k, _a = np.where(_mulher, 0.7, 0.9), np.where(_mulher, -0.241, -0.302)
#filtração glomerular: CKD-EPI 2021 (sem raça), a partir de creatinina, idade e sexo
df["tfg"] = 142 * np.minimum(df["creatinina"] / _k, 1) ** _a * np.maximum(df["creatinina"] / _k, 1) ** -1.200 * 0.9938 ** df["idade"] * np.where(_mulher, 1.012, 1.0)
#choque: CID de choque ou parada antes de t0, ou droga vasoativa "Sim" no SAPS3 preenchido antes de t0
df["choque"] = ((df["apres_choque_pcr"] == 1) | (df["saps3_dva"] == 1)).astype(float)
#insuficiência cardíaca: CID no ano anterior, CID na apresentação ou "Sim" no SAPS3
df["ic"] = ((df["hx_ic"] == 1) | (df["apres_ic"] == 1) | (df["saps3_ic"] == 1)).astype(float)
df["stemi"] = (df["apres_iamcsst"] == 1).astype(float)
#status do procedimento: 2 emergência (código de angioplastia primária), 1 urgente (atendimento nas 48 h ou infarto), 0 eletivo
_emerg = df["fl_codigo_iam"] == 1
df["status_icp"] = np.select([_emerg, (df["atend_48h_antes"] == 1) | (df["apres_iam_qualquer"] == 1)], [2.0, 1.0], 0.0)
#infarto nas 24 h anteriores: angioplastia primária, ou infarto com até 1 dia de internação
df["iam_24h"] = (_emerg | ((df["apres_iam_qualquer"] == 1) & (df["dias_internacao_ate_icp"].fillna(0) <= 1))).astype(float)
_novas = ["fc", "pas", "pad", "glasgow", "qsofa_max", "news2_max", "saps3_ic", "saps3_dva", "fe", "fe_dias"]
_base40 = _originais + _novas
EXAMES = ["creatinina", "hemoglobina", "leucocitos", "plaquetas", "potassio", "sodio", "ureia"]
if "__CONJUNTO__" == "vespera":
  #exames só até a véspera: troca os valores laboratoriais pelos da versão restrita (mesmo ID) e refaz o percentil da troponina
  _vesp = pd.read_csv("icp_dataset_vespera.csv" if Path("icp_dataset_vespera.csv").exists() else "../dados/icp_dataset_vespera.csv")
  _vesp.columns = _vesp.columns.str.lower(); _vesp = _vesp.set_index("id")
  for _c in EXAMES + ["troponina_valor", "troponina_codigo"]:
    df[_c] = df["id"].map(_vesp[_c])
  troponina_pct_treino(df, treino)
if "__CONJUNTO__" == "casos_completos":
  #treino só com quem tem os 7 exames; o teste continua inteiro (e o subconjunto completo é avaliado à parte)
  teste_completo = df.loc[teste, EXAMES].notna().all(axis=1).to_numpy()
  treino = treino & df[EXAMES].notna().all(axis=1).to_numpy()
preditores = {"cathpci": ["idade", "choque", "ic", "dap", "dpoc", "tfg", "stemi", "status_icp"],
              "mayo": ["idade", "creatinina", "fe", "choque", "iam_24h", "ic", "dap"],
              "expandido": _base40, "vespera": _base40, "casos_completos": _base40,
              "sem_idade": [c for c in _base40 if c != "idade"]}["__CONJUNTO__"]
print("Conjunto: __CONJUNTO__ |", len(preditores), "preditores:", preditores)
'''.replace("__CONJUNTO__", CONJUNTO)
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


md("""
# Óbito na internação da angioplastia (até 30 dias) — modelos clássicos e fundacionais

- **Dado:** `icp_dataset.csv`, 1 linha por pessoa (primeira angioplastia), **sem identificadores** (ID pseudonimizado).
- **Desfecho:** óbito na internação da angioplastia (atendimentos a até 12 horas um do outro = mesma internação), em até 30 dias.
- **Desenho:** treino = 2023–2024; teste = 2025; nenhuma pessoa nos dois conjuntos; transferidos externos fora.
- **Modelos, em pé de igualdade:** regressão logística, gradient boosting, XGBoost, CatBoost, TabPFN e Kumo Tabular (este último sem descrição revisada por pares).
- **Sem vazamento:** nenhum ajuste usa o teste. Regressão logística, gradient boosting, XGBoost e CatBoost têm hiperparâmetros otimizados por **Optuna** com validação cruzada estratificada de 5 folds **só no treino**. Os três de boosting aceitam ausentes nativamente (sem imputação); a regressão logística usa imputação pela mediana com indicador de ausente e padronização, **ajustadas dentro de cada fold** (pipeline). Os modelos fundacionais usam o treino como contexto e não têm tuning.
- **Métricas:** valores pontuais no teste (sem bootstrap).
- **Calibração:** todos os modelos são recalibrados (intercepto e inclinação no logit) com predições **fora da amostra do treino** (os mesmos 5 folds). O teste não entra.
- **Limiar de decisão:** pré-especificado como a **prevalência do desfecho no treino**; paciente com risco calibrado igual ou acima dela é sinalizado.
- **Token do TabPFN:** definir a variável de ambiente `TABPFN_TOKEN` na máquina **antes** de abrir o Jupyter. Nunca grave o token neste notebook.
""")

if CONJUNTO != "originais":
    md(f"**Conjunto de preditores deste notebook (`{CONJUNTO}`):** {DESCRICAO_CONJUNTO[CONJUNTO]}. Precisa de `icp_extra.csv` ao lado de `icp_dataset.csv`.")

md("## 0) Configuração")
code("""
from pathlib import Path

#arquivo de dados e pasta de saída: ao lado do notebook ou, dentro do projeto, em ../dados e ../resultados
CAMINHO_DADOS = Path("icp_dataset.csv")
if not CAMINHO_DADOS.exists():
  CAMINHO_DADOS = Path("../dados/icp_dataset.csv")
SAIDA = Path("../resultados/modelos_paper") if Path("../resultados").exists() else Path("resultados_modelos")
SAIDA.mkdir(exist_ok=True)

#mesmas constantes dos scripts do projeto
DESFECHO, ANO, ANO_TESTE, SEMENTE = "obito_internacao_30d", "ano_icp", 2025, 20260929
#análise de sensibilidade: variáveis retiradas dos preditores (ex.: ["idade"]); vazio = análise principal
VARIAVEIS_EXCLUIDAS = []
#fora dos preditores: identificador, desfecho, período, a definição antiga de desfecho e a troponina bruta (vira percentil só do treino)
FORA = ["id", DESFECHO, ANO, "mes_icp", "transf_externa_30d", "obito_30d", "troponina_pct_no_codigo", "troponina_valor", "troponina_codigo"] + VARIAVEIS_EXCLUIDAS
NUM_ESTIMATORS_KUMO = 8
N_TRIALS_OPTUNA = 50      # tentativas por modelo clássico
N_FOLDS = 5               # validação cruzada estratificada, só no treino
N_THREADS = 8             # fixo: o número de threads entra na reprodutibilidade
""")

md("## 0.1) Sementes e determinismo")
md("""
→ **Uma semente só (`SEMENTE`)** para tudo: Python, NumPy, PyTorch (CPU e CUDA), Optuna, folds e cada modelo.

→ `fixa_sementes()` é chamada no início e **de novo antes de cada modelo**, para o resultado de um não depender da ordem de execução dos outros.

→ XGBoost e CatBoost rodam em **CPU com número fixo de threads**: o treino em GPU não garante o mesmo resultado entre placas diferentes. TabPFN e Kumo usam a GPU se houver (senão CPU), com algoritmos determinísticos do PyTorch e gerador próprio com a mesma semente. Entre máquinas ou versões diferentes pode haver diferença nas últimas casas decimais — por isso as versões exatas ficam gravadas em `versoes.txt`.
""")
code("""
import os, random
#precisam existir ANTES de importar o torch / iniciar a CUDA
os.environ["PYTHONHASHSEED"] = str(SEMENTE)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np

def fixa_sementes(semente=SEMENTE):
  random.seed(semente)
  np.random.seed(semente)
  try:
    import torch
    torch.manual_seed(semente)
    torch.cuda.manual_seed_all(semente)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(N_THREADS)
  except ImportError:
    pass

fixa_sementes()
print("Semente:", SEMENTE, "| threads:", N_THREADS)
""")

md("## 1) Ambiente: GPU e pacotes")
code("""
import importlib, subprocess, sys

def versao(pacote):
  try:
    from importlib.metadata import version
    return version(pacote)
  except Exception:
    return None

print("Python:", sys.version.split()[0])
try:
  import torch
  print("torch:", torch.__version__, "| CUDA disponível:", torch.cuda.is_available())
  if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
except ImportError:
  print("torch NÃO instalado: instale a versão com CUDA compatível com a GPU (pytorch.org) antes de seguir")

for p in ["tabpfn", "structured-data-models", "optuna", "xgboost", "catboost", "scikit-learn", "pandas", "numpy"]:
  print(f"{p}: {versao(p)}")

import os
print("TABPFN_TOKEN definido:", bool(os.environ.get("TABPFN_TOKEN")))
""")
code("""
#instala só o que falta (torch não é instalado aqui: a versão depende da CUDA da máquina)
#o structured-data-models do PyPI (0.0.0a0) é placeholder vazio: a biblioteca real vem do GitHub
FONTES = {"tabpfn": "tabpfn", "optuna": "optuna", "xgboost": "xgboost", "catboost": "catboost", "matplotlib": "matplotlib",
          "structured-data-models": "https://github.com/NVIDIA/structured-data-models/archive/refs/heads/main.zip"}
faltando = [p for p in FONTES if versao(p) in (None, "0.0.0a0")]
if faltando:
  import shutil
  alvos = [FONTES[p] for p in faltando]
  #ambientes criados pelo uv não têm pip: usa o uv quando o pip não existir
  tem_pip = subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True).returncode == 0
  uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / ("uv.exe" if os.name == "nt" else "uv"))
  comando = [sys.executable, "-m", "pip", "install", "-q", *alvos] if tem_pip else [uv, "pip", "install", "--python", sys.executable, *alvos]
  subprocess.run(comando, check=False)
  print("Instalados:", faltando, "-> se algum import falhar, reinicie o kernel e rode de novo")
else:
  print("Nada a instalar")
""")

md("## 2) Dados e separação treino/teste (por ano e por pessoa)")
code("""
import numpy as np
import pandas as pd

df = pd.read_csv(CAMINHO_DADOS)
df.columns = df.columns.str.lower()
n_total = len(df)

#transferidos para hospital externo: desfecho não observado
df = df[df["transf_externa_30d"] != 1].reset_index(drop=True)

#coorte: angioplastias de 2023 a 2025
df = df[df[ANO].between(2023, ANO_TESTE)].reset_index(drop=True)

#hold-out por pessoa e por período: treino 2023–2024; teste 2025
assert df["id"].is_unique, "ID repetido: o dataset não está no grão de pessoa"
treino = (df[ANO] < ANO_TESTE).to_numpy()
teste = (df[ANO] == ANO_TESTE).to_numpy()
sobreposicao = len(set(df.loc[treino, "id"]) & set(df.loc[teste, "id"]))
assert sobreposicao == 0

def troponina_pct_treino(d, mascara_treino):
  '''Percentil da troponina dentro do código do exame, contra a distribuição do TREINO (o teste não entra).'''
  d["troponina_pct"] = np.nan
  ref_tr = d.loc[mascara_treino & d["troponina_valor"].notna().to_numpy()]
  for cod, g in ref_tr.groupby("troponina_codigo"):
    ref = np.sort(g["troponina_valor"].to_numpy(dtype=float))
    m = ((d["troponina_codigo"] == cod) & d["troponina_valor"].notna()).to_numpy()
    d.loc[m, "troponina_pct"] = np.searchsorted(ref, d.loc[m, "troponina_valor"].to_numpy(dtype=float), side="right") / len(ref)

troponina_pct_treino(df, treino)

preditores = [c for c in df.columns if c not in FORA]
X = df[preditores].astype(float)
y = df[DESFECHO].to_numpy().astype(int)
Xtr, ytr, Xte, yte = X[treino], y[treino], X[teste], y[teste]
ids_teste = df.loc[teste, "id"].astype(int).to_numpy()

print(f"Dataset: {n_total} | transferidos excluídos: {n_total - len(df)} | preditores: {len(preditores)}")
print(f"Treino: {len(ytr)} pacientes, {ytr.sum()} óbitos ({100*ytr.mean():.2f}%)")
print(f"Teste:  {len(yte)} pacientes, {yte.sum()} óbitos ({100*yte.mean():.2f}%)")
print("Pessoas em treino e teste ao mesmo tempo:", sobreposicao)

#limiar de decisão pré-especificado: prevalência do desfecho no TREINO (o teste não participa)
LIMIAR = float(ytr.mean())
print(f"Limiar de decisão (prevalência do treino): {100*LIMIAR:.2f}%")
""")

md("## 3) Funções de avaliação (iguais às do projeto)")
code("""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

def calibracao(y, p):
  p = np.clip(p, 1e-6, 1 - 1e-6)
  lp = np.log(p / (1 - p)).reshape(-1, 1)
  slope = LogisticRegression(C=1e6).fit(lp, y).coef_[0][0]
  intercepto = np.log(y.mean() / (1 - y.mean())) - np.log(p.mean() / (1 - p.mean()))
  return intercepto, slope

def tabela_decis(y, p):
  linhas = ["| Decil | N | Previsto (%) | Observado (%) |", "|---:|---:|---:|---:|"]
  for i, g in enumerate(np.array_split(np.argsort(p), 10), 1):
    linhas.append(f"| {i} | {len(g)} | {100*p[g].mean():.2f} | {100*y[g].mean():.2f} |")
  return "\\n".join(linhas)

#probabilidades no teste de cada modelo e erros de execução
probs = {}
erros = {}
""")

md("## 4) Regressão logística, gradient boosting, XGBoost e CatBoost com hiperparâmetros otimizados por Optuna")
md("""
→ Busca com **Optuna (TPE, semente fixa)**, maximizando a **AUROC média da validação cruzada estratificada de 5 folds no treino**. O teste não participa da busca.

→ Depois da busca, cada modelo é retreinado com os melhores hiperparâmetros em **todo o treino** e avaliado **uma única vez** no teste.
""")
code("""
import optuna
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMENTE)
Xtr_np, Xte_np = Xtr.to_numpy(), Xte.to_numpy()
hiperparametros = {}
modelos = {}                 # modelos ajustados em todo o treino (usados no SHAP)
fabricas = {}                # como recriar cada modelo (usado na recalibração dentro dos folds)
DISPOSITIVO_XGB = "cpu"     # CPU de propósito: reprodutível entre máquinas

def cria_lr(p):
  #imputação e padronização ficam dentro do pipeline: são ajustadas só no treino de cada fold
  return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                       LogisticRegression(C=p["C"], max_iter=5000, random_state=SEMENTE))

def objetivo_lr(trial):
  p = {"C": trial.suggest_float("C", 1e-3, 100, log=True)}   # força da regularização L2
  return cross_val_score(cria_lr(p), Xtr_np, ytr, cv=cv, scoring="roc_auc", n_jobs=1).mean()

def cria_xgb(p):
  return XGBClassifier(**p, objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                       device=DISPOSITIVO_XGB, random_state=SEMENTE, n_jobs=N_THREADS)

def objetivo_xgb(trial):
  p = {
    "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
    "max_depth": trial.suggest_int("max_depth", 2, 8),
    "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    "gamma": trial.suggest_float("gamma", 1e-3, 5, log=True),
    "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
  }
  return cross_val_score(cria_xgb(p), Xtr_np, ytr, cv=cv, scoring="roc_auc", n_jobs=1).mean()

def cria_cat(p):
  return CatBoostClassifier(**p, loss_function="Logloss", random_seed=SEMENTE, verbose=0, thread_count=N_THREADS,
                            allow_writing_files=False)

def objetivo_cat(trial):
  p = {
    "iterations": trial.suggest_int("iterations", 200, 1000, step=100),
    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
    "depth": trial.suggest_int("depth", 3, 8),
    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 20, log=True),
    "random_strength": trial.suggest_float("random_strength", 1e-3, 10, log=True),
    "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
  }
  return cross_val_score(cria_cat(p), Xtr_np, ytr, cv=cv, scoring="roc_auc", n_jobs=1).mean()

def cria_gb(p):
  return HistGradientBoostingClassifier(**p, early_stopping=False, random_state=SEMENTE)

def objetivo_gb(trial):
  p = {
    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
    "max_iter": trial.suggest_int("max_iter", 100, 800, step=50),
    "max_depth": trial.suggest_int("max_depth", 2, 8),
    "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 64, log=True),
    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200, log=True),
    "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10, log=True),
    "max_features": trial.suggest_float("max_features", 0.5, 1.0),
  }
  return cross_val_score(cria_gb(p), Xtr_np, ytr, cv=cv, scoring="roc_auc", n_jobs=1).mean()

for nome, objetivo, cria in [("Regressão logística", objetivo_lr, cria_lr), ("Gradient boosting", objetivo_gb, cria_gb),
                             ("XGBoost", objetivo_xgb, cria_xgb), ("CatBoost", objetivo_cat, cria_cat)]:
  fixa_sementes()
  estudo = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEMENTE))
  estudo.optimize(objetivo, n_trials=N_TRIALS_OPTUNA)
  hiperparametros[nome] = {"melhores": estudo.best_params, "auroc_cv_treino": estudo.best_value}
  modelo = cria(estudo.best_params).fit(Xtr_np, ytr)          # retreino em todo o treino
  modelos[nome] = modelo
  fabricas[nome] = cria
  probs[nome] = modelo.predict_proba(Xte_np)[:, 1]
  print(f"{nome}: AUROC CV treino {estudo.best_value:.4f} | melhores {estudo.best_params}")
  print(f"   teste: AUROC {roc_auc_score(yte, probs[nome]):.4f} | AUPRC {average_precision_score(yte, probs[nome]):.4f}")
""")

md("## 5) TabPFN")
md("→ Ausentes mantidos (sem imputação), sem tuning: o treino é usado só como contexto.")
code("""
try:
  import torch
  from tabpfn import TabPFNClassifier
  dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
  fixa_sementes()
  #em CPU o TabPFN recusa mais de 1.000 linhas, a menos que o limite seja liberado explicitamente
  tabpfn = TabPFNClassifier(random_state=SEMENTE, device=dispositivo, ignore_pretraining_limits=(dispositivo == "cpu"))
  tabpfn.fit(Xtr, ytr)
  p = tabpfn.predict_proba(Xte)[:, list(tabpfn.classes_).index(1)]
  probs["TabPFN"] = p
  print(f"TabPFN ({versao('tabpfn')}, {dispositivo}): AUROC {roc_auc_score(yte, p):.3f} | AUPRC {average_precision_score(yte, p):.3f}")
except Exception as e:
  erros["TabPFN"] = repr(e)
  print("TabPFN falhou:", repr(e))
""")

md("## 6) Kumo Tabular")
md("""
→ Anunciado em 29/09/2026, sem revisão por pares; pesos sob OpenMDW-1.1. API do model card do Hugging Face.

→ A API recebe uma tabela única: o **alvo das linhas de teste é apagado** antes de montar o tensor, então o desfecho do teste nunca entra no modelo.

→ O que o model card não documenta é verificado aqui: forma do retorno, soma 1 por linha e coluna do óbito (pela prevalência do **treino**).
""")
code("""
def probabilidade_obito(saida, prevalencia_treino, n_esperado=None, verboso=True):
  '''O Kumo devolve um TableTensor com uma coluna por classe (nome = rótulo em texto).
  Converte com .to_pandas() e escolhe a coluna da classe 1; confere pela prevalência do TREINO.'''
  tabela = saida.to_pandas() if hasattr(saida, "to_pandas") else pd.DataFrame(np.asarray(saida))
  if verboso:
    print("Colunas devolvidas pelo Kumo:", list(tabela.columns))
  assert tabela.shape == (n_esperado or len(yte), 2), f"formato inesperado do retorno: {tabela.shape}"
  assert np.allclose(tabela.sum(axis=1), 1, atol=1e-3), "linhas não somam 1 (logits?)"
  por_nome = [c for c in tabela.columns if str(c).strip() in ("1", "1.0")]
  col = por_nome[0] if por_nome else tabela.columns[int(np.argmin(np.abs(tabela.mean().to_numpy() - prevalencia_treino)))]
  p = tabela[col].to_numpy(dtype=float)
  if verboso:
    print("Coluna do óbito:", col, "| média:", round(p.mean(), 4), "| prevalência no treino:", round(prevalencia_treino, 4))
  assert abs(p.mean() - prevalencia_treino) < abs((1 - p).mean() - prevalencia_treino), "coluna do óbito incoerente com a prevalência do treino"
  return p

try:
  import torch, sdm
  dispositivo_kumo = "cuda" if torch.cuda.is_available() else "cpu"

  ctx = Xtr.assign(target=ytr.astype(float))
  qry = Xte.assign(target=np.nan)          # alvo do teste apagado
  tabela_df = pd.concat([ctx, qry], ignore_index=True)
  n_ctx = len(ctx)

  table = sdm.TableTensor.from_pandas(df=tabela_df,
                                      stypes=sdm.infer_stypes(tabela_df, overrides={"target": "categorical"}),
                                      device=dispositivo_kumo)
  kumo = sdm.models.KumoTabular(task="classification", size="large", device=dispositivo_kumo)
  fixa_sementes()          # de novo, logo antes da inferência
  gerador = torch.Generator(device=dispositivo_kumo).manual_seed(SEMENTE)
  with torch.no_grad():
    saida_kumo = kumo(x_context=table[:n_ctx].drop_columns("target"),
                      y_context=table[:n_ctx, "target"],
                      x_query=table[n_ctx:].drop_columns("target"),
                      num_estimators=NUM_ESTIMATORS_KUMO, generator=gerador)
  pk = probabilidade_obito(saida_kumo, ytr.mean())
  probs["Kumo Tabular"] = pk

  def kumo_prever(X_contexto, y_contexto, X_consulta):
    \'\'\'Saída do Kumo para X_consulta usando (X_contexto, y_contexto) como contexto; mesma semente da predição principal.\'\'\'
    tb = pd.concat([X_contexto.assign(target=np.asarray(y_contexto, dtype=float)), X_consulta.assign(target=np.nan)], ignore_index=True)
    t = sdm.TableTensor.from_pandas(df=tb, stypes=sdm.infer_stypes(tb, overrides={"target": "categorical"}), device=dispositivo_kumo)
    fixa_sementes()
    g = torch.Generator(device=dispositivo_kumo).manual_seed(SEMENTE)
    n = len(X_contexto)
    with torch.no_grad():
      return kumo(x_context=t[:n].drop_columns("target"), y_context=t[:n, "target"],
                  x_query=t[n:].drop_columns("target"), num_estimators=NUM_ESTIMATORS_KUMO, generator=g)
  print(f"Kumo Tabular ({versao('structured-data-models')}, large): AUROC {roc_auc_score(yte, pk):.3f} | "
        f"AUPRC {average_precision_score(yte, pk):.3f}")
except Exception as e:
  erros["Kumo Tabular"] = repr(e)
  print("Kumo Tabular falhou:", repr(e))
""")

if CONJUNTO == "cathpci":
    md("## 6.0) Escore NCDR CathPCI publicado (versão aproximada, sem treino)")
    md("""
→ Pontos da **Tabela 4 de Peterson et al. (JACC 2010)**, somados para cada pessoa do teste; o risco é o publicado para o total de pontos (interpolação entre os degraus de 5 pontos).

→ **Não é o escore validado:** NYHA IV não existe no dado (0 ponto); filtração glomerular ausente conta 0 ponto (como se fosse > 90); doença arterial periférica e pulmonar vêm só de CID; "salvamento" não tem equivalente. Onde o dado falta, o risco sai subestimado.

→ Entra na comparação **sem recalibração**: é o escore como publicado.
""")
    code("""
def pontos_cathpci(d):
  p = np.select([d["idade"] >= 80, d["idade"] >= 70, d["idade"] >= 60], [14, 8, 4], 0).astype(float)
  p += 25 * d["choque"] + 5 * d["ic"] + 5 * (d["dap"] == 1) + 4 * (d["dpoc"] == 1)
  p += np.select([d["tfg"] < 30, d["tfg"] < 60, d["tfg"] <= 90], [18, 10, 6], 0)          # ausente cai em 0
  com_supra = np.select([d["status_icp"] == 2, d["status_icp"] == 1], [20, 15], 12)
  sem_supra = np.select([d["status_icp"] == 2, d["status_icp"] == 1], [20, 8], 0)
  return (p + np.where(d["stemi"] == 1, com_supra, sem_supra)).to_numpy()

PONTOS_TAB = np.arange(0, 101, 5)
RISCO_TAB = np.array([0.0, 0.1, 0.1, 0.2, 0.3, 0.6, 1.1, 2.0, 3.6, 6.3, 10.9, 18.3, 29.0, 42.7, 57.6, 71.2, 81.0, 89.2, 93.8, 96.5, 98.0]) / 100

pontos_teste = pontos_cathpci(df.loc[teste])
probs["CathPCI (escore publicado)"] = np.clip(np.interp(pontos_teste, PONTOS_TAB, RISCO_TAB), 1e-4, 1 - 1e-4)
print("Pontos no teste:", pd.Series(pontos_teste).describe()[["min", "25%", "50%", "75%", "max"]].round(0).to_dict())
print(f"AUROC dos pontos: {roc_auc_score(yte, pontos_teste):.4f} | AUPRC: {average_precision_score(yte, pontos_teste):.4f}")
""")

md("## 6.1) Recalibração com o treino (o teste não entra)")
md("""
→ **Problema:** a probabilidade que sai de cada modelo pode estar sistematicamente alta ou baixa. Como o limiar de decisão é um valor de risco, a calibração precisa estar resolvida antes.

→ **Método:** recalibração logística (intercepto e inclinação no logit da probabilidade). Só dois parâmetros, o que é estável com poucos eventos; isotônica sobreajustaria.

→ **Sem vazamento:** os dois parâmetros são estimados com predições **fora da amostra do treino**: nos mesmos 5 folds, cada pessoa do treino é predita por um modelo que não a viu. Para os modelos com Optuna, usam-se os hiperparâmetros já escolhidos. O teste só recebe a transformação pronta.

→ A recalibração é monotônica: **não muda AUROC nem AUPRC**, só o valor da probabilidade. O SHAP continua explicando a saída original do modelo.

→ **O que ela não resolve:** a prevalência do desfecho pode mudar entre o treino (2023–2024) e o teste (2025). Nenhuma recalibração feita só com o treino corrige essa mudança no tempo.
""")
code("""
from sklearn.model_selection import cross_val_predict

def logit(p):
  p = np.clip(p, 1e-6, 1 - 1e-6)
  return np.log(p / (1 - p))

if "probs_brutas" not in globals():
  probs_brutas = dict(probs)       # probabilidades originais (a célula pode ser reexecutada sem recalibrar duas vezes)

#predições fora da amostra no treino (OOF), nos mesmos folds do Optuna
oof = {}
for nome in fabricas:
  fixa_sementes()
  oof[nome] = cross_val_predict(fabricas[nome](hiperparametros[nome]["melhores"]), Xtr_np, ytr, cv=cv,
                                method="predict_proba", n_jobs=1)[:, 1]

if "TabPFN" in probs_brutas:
  try:
    o = np.zeros(len(ytr))
    for i_tr, i_va in cv.split(Xtr_np, ytr):
      fixa_sementes()
      m = TabPFNClassifier(random_state=SEMENTE, device=dispositivo, ignore_pretraining_limits=(dispositivo == "cpu"))
      m.fit(Xtr.iloc[i_tr], ytr[i_tr])
      o[i_va] = m.predict_proba(Xtr.iloc[i_va])[:, list(m.classes_).index(1)]
    oof["TabPFN"] = o
  except Exception as e:
    erros["Recalibração do TabPFN"] = repr(e)

if "Kumo Tabular" in probs_brutas:
  try:
    o = np.zeros(len(ytr))
    for i_tr, i_va in cv.split(Xtr_np, ytr):
      o[i_va] = probabilidade_obito(kumo_prever(Xtr.iloc[i_tr], ytr[i_tr], Xtr.iloc[i_va]), ytr[i_tr].mean(),
                                    n_esperado=len(i_va), verboso=False)
    oof["Kumo Tabular"] = o
  except Exception as e:
    erros["Recalibração do Kumo"] = repr(e)

probs, linhas_rec, recal_params = {}, [], {}
for nome, p_bruta in probs_brutas.items():
  if nome not in oof:
    probs[nome] = p_bruta          # sem OOF: fica sem recalibrar (o motivo está em `erros`)
    continue
  rec = LogisticRegression(C=1e6, max_iter=1000).fit(logit(oof[nome]).reshape(-1, 1), ytr)
  a, b = float(rec.intercept_[0]), float(rec.coef_[0][0])
  recal_params[nome] = (a, b)
  probs[nome] = 1 / (1 + np.exp(-(a + b * logit(p_bruta))))
  i0, s0 = calibracao(yte, p_bruta)
  i1, s1 = calibracao(yte, probs[nome])
  linhas_rec.append({"Modelo": nome, "AUROC fora da amostra (treino)": f"{roc_auc_score(ytr, oof[nome]):.4f}",
                     "a": f"{a:+.3f}", "b": f"{b:.3f}",
                     "Intercepto teste (antes → depois)": f"{i0:+.3f} → {i1:+.3f}",
                     "Slope teste (antes → depois)": f"{s0:.3f} → {s1:.3f}",
                     "Brier teste (antes → depois)": f"{brier_score_loss(yte, p_bruta):.4f} → {brier_score_loss(yte, probs[nome]):.4f}",
                     "Previsto médio teste (antes → depois)": f"{100*p_bruta.mean():.2f}% → {100*probs[nome].mean():.2f}%"})
tab_rec = pd.DataFrame(linhas_rec)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(f"Observado no teste: {100*yte.mean():.2f}% | prevalência do treino: {100*ytr.mean():.2f}%")
print(tab_rec.to_string(index=False))
""")

md("## 7) Métricas no teste (valores pontuais) e calibração")
md("→ Daqui em diante, `probs` são as probabilidades **recalibradas**; as originais ficam em `probs_brutas`.")
code("""
linhas = ["| Modelo | AUROC | AUPRC | Brier | Intercepto calib. | Slope calib. |",
          "|---|---:|---:|---:|---:|---:|"]
decis = []
for nome, p in probs.items():
  inter, slope = calibracao(yte, p)
  linhas.append(f"| {nome} | {roc_auc_score(yte, p):.4f} | {average_precision_score(yte, p):.4f} | "
                f"{brier_score_loss(yte, p):.4f} | {inter:+.3f} | {slope:.3f} |")
  decis += ["", f"### Calibração por decil — {nome}", "", tabela_decis(yte, p)]

metricas = "\\n".join(linhas) + f"\\n\\nAUPRC de referência (prevalência no teste) = {yte.mean():.4f}\\n"
print(metricas)
""")

md("## 7.1) Curvas ROC no teste")
code("""
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

cores = {"Regressão logística": "#7f7f7f", "Gradient boosting": "#1f77b4", "XGBoost": "#ff7f0e", "CatBoost": "#2ca02c",
         "TabPFN": "#9467bd", "Kumo Tabular": "#d62728", "CathPCI (escore publicado)": "#000000"}

fig, ax = plt.subplots(figsize=(7, 7))
for nome, p in probs.items():
  fpr, tpr, _ = roc_curve(yte, p)
  ax.plot(fpr, tpr, lw=2, color=cores.get(nome), label=f"{nome} (AUROC = {roc_auc_score(yte, p):.3f})")
ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="Aleatório")
ax.set_xlabel("1 − Especificidade (taxa de falsos positivos)")
ax.set_ylabel("Sensibilidade (taxa de verdadeiros positivos)")
ax.set_title(f"Curvas ROC — teste: 2025 (n = {len(yte)}, {yte.sum()} óbitos)")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.set_aspect("equal")
ax.legend(loc="lower right", frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(SAIDA / "roc_teste.png", dpi=300)
plt.show()
""")

md("## 7.1.1) Calibração no teste (decis de risco previsto)")
md("→ Cada ponto é um décimo dos pacientes do teste, ordenados pelo risco previsto. Sobre a diagonal = calibrado; abaixo = o modelo superestima; acima = subestima.")
code("""
fig, ax = plt.subplots(figsize=(7, 7))
for nome, p in probs.items():
  grupos = np.array_split(np.argsort(p), 10)
  ax.plot([100 * p[g].mean() for g in grupos], [100 * yte[g].mean() for g in grupos], marker="o", ms=4, lw=1.5,
          color=cores.get(nome), label=nome)
limite = 100 * max(0.30, max(p[np.argsort(p)[-len(p) // 10:]].mean() for p in probs.values())) * 1.05
ax.plot([0, limite], [0, limite], ls="--", color="gray", lw=1, label="Calibração perfeita")
ax.set_xlim(0, limite); ax.set_ylim(0, limite); ax.set_aspect("equal")
ax.set_xlabel("Risco previsto médio no decil (%)"); ax.set_ylabel("Óbito observado no decil (%)")
ax.set_title(f"Calibração — teste: 2025 (n = {len(yte)}, {yte.sum()} óbitos)")
ax.legend(loc="upper left", frameon=False); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig(SAIDA / "calibracao.png", dpi=300)
plt.show()
""")

if CONJUNTO == "expandido":
    md("## 7.2) SHAP do modelo com maior AUROC no teste")
    md("""
→ **Qual modelo:** o de maior AUROC no teste entre os seis. A escolha é só para a explicação; não altera nenhuma métrica.

→ **Como:** XGBoost e CatBoost têm TreeSHAP exato (todas as pessoas do teste). Para os demais, o valor de Shapley é **estimado por amostragem de permutações** (Štrumbelj e Kononenko, 2014) em 200 pessoas do teste, 20 permutações cada, com pessoas de referência sorteadas do treino.

→ Escala em **log-odds** da saída original do modelo (antes da recalibração). Positivo = empurra a predição para óbito.
""")
    code("""
import time
from catboost import Pool
import xgboost as xgb

N_SHAP, K_PERM, LOTE = 200, 20, 2500
candidatos = [m for m in probs_brutas if m != "CathPCI (escore publicado)"]
MODELO_SHAP = max(candidatos, key=lambda m: roc_auc_score(yte, probs_brutas[m]))
print("Modelo do SHAP (maior AUROC no teste):", MODELO_SHAP, f"| AUROC {roc_auc_score(yte, probs_brutas[MODELO_SHAP]):.4f}")

def f_logodds(Xn):
  #log-odds da saída original do modelo escolhido, em lotes
  partes = []
  for ini in range(0, len(Xn), LOTE):
    bloco = Xn[ini:ini + LOTE]
    if MODELO_SHAP == "TabPFN":
      p = tabpfn.predict_proba(pd.DataFrame(bloco, columns=preditores))[:, list(tabpfn.classes_).index(1)]
    elif MODELO_SHAP == "Kumo Tabular":
      p = probabilidade_obito(kumo_prever(Xtr, ytr, pd.DataFrame(bloco, columns=preditores)), ytr.mean(), n_esperado=len(bloco), verboso=False)
    else:
      p = modelos[MODELO_SHAP].predict_proba(bloco)[:, 1]
    partes.append(p)
  return logit(np.concatenate(partes))

if MODELO_SHAP in ("XGBoost", "CatBoost"):
  if MODELO_SHAP == "CatBoost":
    contrib = modelos[MODELO_SHAP].get_feature_importance(Pool(Xte_np), type="ShapValues")
  else:
    contrib = modelos[MODELO_SHAP].get_booster().predict(xgb.DMatrix(Xte_np), pred_contribs=True)
  shap_valores, X_ref = contrib[:, :-1], Xte_np
  p_shap = 1 / (1 + np.exp(-(contrib[:, -1] + shap_valores.sum(axis=1))))
  assert np.allclose(p_shap, probs_brutas[MODELO_SHAP], atol=1e-4), "SHAP não reconstitui a predição"
else:
  rng_shap = np.random.default_rng(SEMENTE)
  idx_shap = np.sort(rng_shap.choice(len(Xte_np), size=min(N_SHAP, len(Xte_np)), replace=False))
  X_ref = Xte_np[idx_shap]
  n_exp, n_var = X_ref.shape
  f_exp = f_logodds(X_ref)
  print(f"Diferença máxima (log-odds) entre predizer em lote e no teste inteiro: {np.abs(f_exp - logit(probs_brutas[MODELO_SHAP][idx_shap])).max():.4f}")
  shap_valores, f_fundo = np.zeros((n_exp, n_var)), np.zeros(n_exp)
  for k in range(K_PERM):
    t_ini = time.time()
    ordem = rng_shap.permutation(n_var)
    atual = Xtr_np[rng_shap.integers(0, len(Xtr_np), size=n_exp)].copy()     # referências sorteadas do TREINO
    passos = [atual.copy()]
    for j in ordem:
      atual[:, j] = X_ref[:, j]
      passos.append(atual.copy())
    f = f_logodds(np.vstack(passos)).reshape(n_var + 1, n_exp)
    for pos, j in enumerate(ordem):
      shap_valores[:, j] += f[pos + 1] - f[pos]
    f_fundo += f[0]
    if k == 0:
      print(f"1ª permutação: {time.time() - t_ini:.0f}s -> estimativa total: {(time.time() - t_ini) * K_PERM / 60:.0f} min")
  shap_valores /= K_PERM; f_fundo /= K_PERM
  print("Erro máximo de aditividade (log-odds):", np.abs(shap_valores.sum(axis=1) - (f_exp - f_fundo)).max())

importancia = pd.DataFrame({"variavel": preditores, "shap_medio_abs": np.abs(shap_valores).mean(axis=0)})
importancia = importancia.sort_values("shap_medio_abs", ascending=False).reset_index(drop=True)
importancia.to_csv(SAIDA / "shap_importancia.csv", index=False)
print(importancia.head(15).to_string(index=False))
""")
    code("""
TOP = 15
topo = importancia.head(TOP)

fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(topo["variavel"][::-1], topo["shap_medio_abs"][::-1], color=cores.get(MODELO_SHAP, "#1f77b4"))
ax.set_xlabel("Média de |SHAP| (log-odds)")
ax.set_title(f"Importância por SHAP — {MODELO_SHAP} (teste, n = {len(X_ref)})")
fig.tight_layout(); fig.savefig(SAIDA / "shap_importancia.png", dpi=300); plt.show()

#beeswarm: cada ponto é uma pessoa; cor = valor da variável (percentil); cinza = ausente
rng_plot = np.random.default_rng(SEMENTE)
fig, ax = plt.subplots(figsize=(8, 7))
for linha, var in enumerate(topo["variavel"][::-1]):
  j = preditores.index(var)
  valor, sv = X_ref[:, j], shap_valores[:, j]
  ausente = np.isnan(valor)
  jitter = linha + rng_plot.uniform(-0.3, 0.3, size=len(sv))
  ax.scatter(sv[ausente], jitter[ausente], s=6, color="lightgray", alpha=0.6, linewidths=0)
  percentil = pd.Series(valor[~ausente]).rank(pct=True).to_numpy()
  pontos = ax.scatter(sv[~ausente], jitter[~ausente], s=6, c=percentil, cmap="coolwarm", vmin=0, vmax=1, alpha=0.7, linewidths=0)
ax.axvline(0, color="gray", lw=1)
ax.set_yticks(range(len(topo))); ax.set_yticklabels(topo["variavel"][::-1])
ax.set_xlabel("SHAP (log-odds; positivo = maior risco de óbito)")
ax.set_title(f"SHAP — {MODELO_SHAP} (teste, n = {len(X_ref)})")
barra = fig.colorbar(pontos, ax=ax, ticks=[0, 1]); barra.ax.set_yticklabels(["baixo", "alto"]); barra.set_label("Valor da variável (cinza = ausente)")
fig.tight_layout(); fig.savefig(SAIDA / "shap_beeswarm.png", dpi=300); plt.show()
""")

md("## 7.4) Ponto de operação: limiar = prevalência do treino")
md("""
→ **Regra pré-especificada:** sinalizar o paciente cujo risco calibrado é igual ou maior que a prevalência do desfecho no treino, isto é, quem tem risco acima do paciente médio. O número vem só do treino.

→ Nesse limiar, aceita-se cerca de 1/prevalência − 1 alarmes falsos por óbito identificado (perto de 28). É um limiar **sensível**: sinaliza muita gente.
""")
code("""
linhas_op = []
for nome, p in probs.items():
  sinal = p >= LIMIAR
  vp, fp = int((sinal & (yte == 1)).sum()), int((sinal & (yte == 0)).sum())
  fn, vn = int((~sinal & (yte == 1)).sum()), int((~sinal & (yte == 0)).sum())
  linhas_op.append({"Modelo": nome, "Sinalizados": f"{vp + fp} ({100*(vp + fp)/len(yte):.1f}%)",
                    "Sensibilidade (%)": f"{100*vp/(vp + fn):.1f}", "Especificidade (%)": f"{100*vn/(vn + fp):.1f}",
                    "VPP (%)": f"{100*vp/(vp + fp):.1f}" if vp + fp else "—", "VPN (%)": f"{100*vn/(vn + fn):.1f}" if vn + fn else "—",
                    "Óbitos não sinalizados": fn,
                    "Nº necessário para rastrear": f"{(vp + fp)/vp:.1f}" if vp else "—"})
tab_op = pd.DataFrame(linhas_op)
print(f"Limiar = {100*LIMIAR:.2f}% (prevalência do treino) | teste: {len(yte)} pessoas, {yte.sum()} óbitos")
print(tab_op.to_string(index=False))
""")

md("## 7.5) Utilidade clínica: curva de decisão (benefício líquido)")
md("""
→ **Par desfecho–ação:** a predição sai ao fim da angioplastia; a ação é o destino e a intensidade de vigilância do paciente depois dela (UTI ou unidade coronariana, monitorização, reavaliação precoce).

→ **Benefício líquido** num limiar de risco `pt`: `VP/n − FP/n × pt/(1 − pt)`. O limiar traduz quantos alarmes falsos se aceita por óbito identificado (pt = 10% → até 9 falsos por verdadeiro).

→ O modelo só é útil nos limiares em que fica acima de **vigiar todos** e de **não vigiar ninguém**. Depende de discriminação **e** de calibração: maior AUROC não garante maior benefício.

→ A curva não mede se a ação reduz óbito; isso este estudo não avalia.
""")
code("""
def para_md(tabela):
  '''DataFrame -> tabela em markdown (sem depender do pacote tabulate).'''
  linhas = ["| " + " | ".join(map(str, tabela.columns)) + " |", "|" + "---|" * len(tabela.columns)]
  linhas += ["| " + " | ".join(map(str, linha)) + " |" for linha in tabela.itertuples(index=False)]
  return "\\n".join(linhas) + "\\n"

def beneficio_liquido(y, p, pt):
  sinalizado = p >= pt
  vp, fp = (sinalizado & (y == 1)).sum(), (sinalizado & (y == 0)).sum()
  return vp / len(y) - fp / len(y) * pt / (1 - pt)

LIMIARES = np.arange(0.01, 0.301, 0.005)
prevalencia = yte.mean()
nb_todos = [prevalencia - (1 - prevalencia) * pt / (1 - pt) for pt in LIMIARES]

fig, ax = plt.subplots(figsize=(8, 6))
for nome, p in probs.items():
  ax.plot(100 * LIMIARES, [beneficio_liquido(yte, p, pt) for pt in LIMIARES], lw=2, color=cores.get(nome), label=nome)
ax.plot(100 * LIMIARES, nb_todos, color="black", lw=1, ls=":", label="Vigiar todos")
ax.axhline(0, color="gray", lw=1, ls="--", label="Não vigiar ninguém")
ax.axvline(100 * LIMIAR, color="gray", lw=1, ls="-.", label=f"Limiar do estudo ({100*LIMIAR:.1f}%)")
ax.set_ylim(-0.005, prevalencia * 1.05)
ax.set_xlabel("Limiar de risco (%)"); ax.set_ylabel("Benefício líquido")
ax.set_title(f"Curva de decisão — teste: 2025 (n = {len(yte)}, {yte.sum()} óbitos)")
ax.legend(frameon=False); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig(SAIDA / "curva_decisao.png", dpi=300)
plt.show()

LIMIARES_TABELA = [LIMIAR, 0.05, 0.10, 0.20]
tab_nb = pd.DataFrame({"Modelo": list(probs) + ["Vigiar todos"]})
for pt in LIMIARES_TABELA:
  valores = [beneficio_liquido(yte, p, pt) for p in probs.values()] + [prevalencia - (1 - prevalencia) * pt / (1 - pt)]
  tab_nb[f"pt = {100*pt:.1f}%"] = [f"{1000*v:.1f}" for v in valores]
print("Benefício líquido por 1.000 pacientes (não vigiar ninguém = 0):")
print(tab_nb.to_string(index=False))
""")

md("## 7.6) Capacidade de trabalho: quantos pacientes vigiar para achar um óbito")
md("""
→ Em vez de um limiar de risco, fixa-se a **capacidade da equipe**: vigiar de perto os 5%, 10% ou 20% de maior risco previsto.

→ **Número necessário para rastrear** = pacientes sinalizados para cada óbito na lista (1 / valor preditivo positivo). Sem modelo, seria 1 / prevalência.
""")
code("""
linhas_cap = []
for nome, p in probs.items():
  ordem_risco = np.argsort(-p, kind="stable")
  for fracao in [0.05, 0.10, 0.20]:
    k = int(np.ceil(fracao * len(yte)))
    obitos_lista = int(yte[ordem_risco[:k]].sum())
    linhas_cap.append({"Modelo": nome, "Lista": f"{100*fracao:.0f}% de maior risco", "Sinalizados": k,
                       "Óbitos na lista": obitos_lista, "Óbitos capturados (%)": f"{100*obitos_lista/yte.sum():.1f}",
                       "VPP (%)": f"{100*obitos_lista/k:.1f}",
                       "Nº necessário para rastrear": f"{k/obitos_lista:.1f}" if obitos_lista else "—"})
tab_cap = pd.DataFrame(linhas_cap)
print(f"Sem modelo: {1/yte.mean():.1f} pacientes por óbito (prevalência {100*yte.mean():.2f}%)")
print(tab_cap.to_string(index=False))
""")

md("## 7.7) Equidade: desempenho por sexo e por faixa etária")
md("""
→ **Paridade de classificação:** AUROC, sensibilidade e taxa de falsos positivos por subgrupo, no limiar do estudo (prevalência do treino), igual para todos os grupos.

→ **Calibração por subgrupo:** óbito observado contra previsto médio (razão O/E; 1 = calibrado).

→ **Leitura descritiva:** os subgrupos têm poucas dezenas de óbitos e não há intervalo de confiança. Diferenças pequenas não permitem concluir que o modelo é melhor ou pior num grupo. Raça/cor não está no dataset.
""")
code("""
df_teste = df.loc[teste].reset_index(drop=True)
faixa = pd.cut(df_teste["idade"], bins=[0, 65, 75, 200], right=False, labels=["< 65 anos", "65–74 anos", "≥ 75 anos"]).astype(str)
grupos = {"Todos": np.ones(len(yte), dtype=bool),
          "Homens": (df_teste["sexo_f"] == 0).to_numpy(), "Mulheres": (df_teste["sexo_f"] == 1).to_numpy()}
grupos.update({f: (faixa == f).to_numpy() for f in ["< 65 anos", "65–74 anos", "≥ 75 anos"]})

linhas_eq = []
for nome, p in probs.items():
  for g, m in grupos.items():
    yg, pg = yte[m], p[m]
    sinal = pg >= LIMIAR
    linhas_eq.append({"Modelo": nome, "Grupo": g, "N": int(m.sum()), "Óbitos": int(yg.sum()),
                      "Observado (%)": f"{100*yg.mean():.2f}", "Previsto (%)": f"{100*pg.mean():.2f}",
                      "O/E": f"{yg.mean()/pg.mean():.2f}", "AUROC": f"{roc_auc_score(yg, pg):.3f}",
                      "Sensibilidade (%)": f"{100*sinal[yg == 1].mean():.1f}",
                      "Falsos positivos (%)": f"{100*sinal[yg == 0].mean():.1f}"})
tab_eq = pd.DataFrame(linhas_eq)
pd.set_option("display.width", 250)
print(tab_eq.to_string(index=False))
""")

md("## 7.8) Características de treino e teste (reporte mínimo — MINIMAR)")
md("→ Demografia e preditores dos dois conjuntos lado a lado, para o leitor comparar com a população em que pretende aplicar o modelo. Variáveis 0/1 em %; contínuas em mediana [intervalo interquartil] e % de ausentes.")
code("""
def resumo(serie):
  valores = serie.dropna()
  if set(valores.unique()) <= {0, 1}:
    return f"{100*valores.mean():.1f}%"
  q1, med, q3 = valores.quantile([0.25, 0.5, 0.75])
  ausente = f"; ausente {100*serie.isna().mean():.1f}%" if serie.isna().any() else ""
  return f"{med:.1f} [{q1:.1f}–{q3:.1f}]{ausente}"

df_treino = df.loc[treino]
linhas_t1 = [{"Característica": "Pessoas", "Treino (2023–2024)": len(df_treino), "Teste (2025)": len(df_teste)},
             {"Característica": "Óbito intra-hospitalar em 30 dias", "Treino (2023–2024)": f"{ytr.sum()} ({100*ytr.mean():.2f}%)",
              "Teste (2025)": f"{yte.sum()} ({100*yte.mean():.2f}%)"}]
for var in [c for c in df.columns if c not in ["id", DESFECHO, ANO, "mes_icp", "transf_externa_30d"]]:
  linhas_t1.append({"Característica": var, "Treino (2023–2024)": resumo(df_treino[var]), "Teste (2025)": resumo(df_teste[var])})
tab_t1 = pd.DataFrame(linhas_t1)
print(tab_t1.to_string(index=False))
""")

md("## 8) Diferenças entre modelos (valores pontuais)")
code("""
nomes = list(probs)
comp = ["| Comparação | ΔAUROC | ΔAUPRC |", "|---|---:|---:|"]
for i in range(len(nomes)):
  for j in range(i + 1, len(nomes)):
    a, b = nomes[j], nomes[i]
    comp.append(f"| {a} − {b} | {roc_auc_score(yte, probs[a]) - roc_auc_score(yte, probs[b]):+.4f} | "
                f"{average_precision_score(yte, probs[a]) - average_precision_score(yte, probs[b]):+.4f} |")
comparacao = "\\n".join(comp) + "\\n"

otim = ["| Modelo | AUROC CV treino | Melhores hiperparâmetros |", "|---|---:|---|"]
for nome, h in hiperparametros.items():
  otim.append(f"| {nome} | {h['auroc_cv_treino']:.4f} | {h['melhores']} |")
otimizacao = "\\n".join(otim) + "\\n"
print(comparacao); print(otimizacao)
""")

md("## 9) Salvando os resultados")
code("""
import platform

cabecalho = [
  "# Óbito na internação da angioplastia (até 30 dias) — resultados",
  "",
  f"- Dataset: {n_total}; transferidos excluídos: {n_total - len(df)}; preditores: {len(preditores)}"
  f"; variáveis excluídas (sensibilidade): {VARIAVEIS_EXCLUIDAS or 'nenhuma'}",
  f"- Treino: {len(ytr)} pacientes, {ytr.sum()} óbitos; teste: {len(yte)} pacientes, {yte.sum()} óbitos; pessoas nos dois lados: {sobreposicao}",
  f"- Python {platform.python_version()}; tabpfn {versao('tabpfn')}; structured-data-models {versao('structured-data-models')}; "
  f"scikit-learn {versao('scikit-learn')}; optuna {versao('optuna')}",
  f"- Otimização: Optuna TPE (semente {SEMENTE}), {N_TRIALS_OPTUNA} tentativas por modelo, AUROC média em CV estratificada "
  f"de {N_FOLDS} folds no treino; métricas no teste são valores pontuais (sem bootstrap)",
  f"- Reprodutibilidade: semente única {SEMENTE} (Python, NumPy, PyTorch, Optuna, folds, modelos); boosting em CPU com "
  f"{N_THREADS} threads; PyTorch determinístico; Kumo com {NUM_ESTIMATORS_KUMO} estimadores; versões em versoes.txt",
  f"- Calibração: recalibração logística com predições fora da amostra do treino ({N_FOLDS} folds); limiar de decisão = "
  f"prevalência do treino ({100*LIMIAR:.2f}%); as métricas abaixo usam as probabilidades recalibradas",
  f"- xgboost {versao('xgboost')}; catboost {versao('catboost')}; numpy {versao('numpy')}; pandas {versao('pandas')}; {platform.platform()}",
]
try:
  import torch
  cabecalho.append(f"- torch {torch.__version__}; GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'nenhuma'}")
except ImportError:
  pass
if erros:
  cabecalho += ["", "## Modelos que falharam", ""] + [f"- {k}: `{v}`" for k, v in erros.items()]

texto = "\\n".join(cabecalho) + "\\n\\n## Métricas no teste (2025)\\n\\n" + metricas + \\
        "\\n## Diferenças entre modelos\\n\\n" + comparacao + "\\n## Otimização de hiperparâmetros (Optuna)\\n\\n" + otimizacao + \\
        "\\n## Recalibração com o treino (a e b da recalibração logística; efeito no teste)\\n\\n" + para_md(tab_rec) + \\
        f"\\n## Ponto de operação: limiar = prevalência do treino ({100*LIMIAR:.2f}%)\\n\\n" + para_md(tab_op) + \\
        "\\n## Benefício líquido por 1.000 pacientes (curva de decisão)\\n\\n" + para_md(tab_nb) + \\
        "\\n## Capacidade de trabalho (lista dos de maior risco)\\n\\n" + f"Sem modelo: {1/yte.mean():.1f} pacientes por óbito.\\n\\n" + para_md(tab_cap) + \\
        "\\n## Desempenho por sexo e faixa etária (limiar = prevalência do treino; descritivo, sem IC)\\n\\n" + para_md(tab_eq) + \\
        "\\n## Características de treino e teste\\n\\n" + para_md(tab_t1) + \\
        "\\n## Calibração por decil\\n" + "\\n".join(decis) + "\\n"
(SAIDA / "resultados.md").write_text(texto, encoding="utf-8")

#probabilidades do teste por ID pseudonimizado (para o 15_comparacao.py do projeto)
pd.DataFrame({"id": ids_teste, "obito_30d": yte, **probs}).to_csv(SAIDA / "probs_teste_todos.csv", index=False)
pd.DataFrame({"id": ids_teste, "obito_30d": yte, **probs_brutas}).to_csv(SAIDA / "probs_teste_brutas.csv", index=False)
#versões exatas de todos os pacotes, para replicar o ambiente
from importlib.metadata import distributions
(SAIDA / "versoes.txt").write_text("\\n".join(sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions()
                                                       if d.metadata["Name"]})), encoding="utf-8")
print("Salvo em", SAIDA.resolve())
print("Erros:", erros or "nenhum")
""")

if CONJUNTO != "originais":
    _trocas = [
        ("preditores = [c for c in df.columns if c not in FORA]", BLOCO_PREDITORES),
        ('SAIDA = Path("../resultados/modelos_paper") if Path("../resultados").exists() else Path("resultados_modelos")',
         f'SAIDA = Path("../resultados/modelos_{CONJUNTO}") if Path("../resultados").exists() else Path("resultados_{CONJUNTO}")'),
        ('for var in [c for c in df.columns if c not in ["id", DESFECHO, ANO, "mes_icp", "transf_externa_30d"]]:', "for var in preditores:"),
    ]
    _fonte = "\\n".join(c.source for c in cells)
    for _antigo, _novo in _trocas:
        assert _fonte.count(_antigo) == 1, _antigo
    for c in cells:
        for _antigo, _novo in _trocas:
            c.source = c.source.replace(_antigo, _novo)

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, str(saida))
print("ok", saida, len(cells), "celulas")
