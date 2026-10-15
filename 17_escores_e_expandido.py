"""Escore clássico x aprendizado de máquina com as mesmas variáveis x versão expandida.

Compara, no mesmo teste temporal (jan-jul/2026) do paper:
  1. NCDR CathPCI de beira de leito APROXIMADO (pontos de Peterson et al., JACC 2010, Tabela 4);
  2. regressão logística e gradient boosting treinados só com as variáveis do CathPCI;
  3. idem, só com as variáveis do Mayo Clinic Risk Score (Singh et al., Mayo Clin Proc 2007);
  4. idem, com os 30 preditores originais do paper;
  5. idem, com os 30 + variáveis novas (sinais vitais, SAPS3, fração de ejeção), todas anteriores a t0.

Mesmo desenho do notebook: Optuna TPE (semente fixa, 50 tentativas, AUROC em CV estratificada de 5 folds só no
treino), recalibração logística com predições fora da amostra do treino, limiar = prevalência do treino.
O escore inteiro do Mayo NÃO é calculado: a tabela de pontos está em figura do artigo e não foi conferida.

uso: python 17_escores_e_expandido.py
"""
import os
import random
from pathlib import Path

SEMENTE = 20260929
os.environ["PYTHONHASHSEED"] = str(SEMENTE)

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent
SAIDA = BASE / "resultados" / "escores_expandido"
SAIDA.mkdir(parents=True, exist_ok=True)
DESFECHO, ANO, ANO_TESTE = "obito_30d", "ano_icp", 2026
N_TRIALS, N_FOLDS = 50, 5
optuna.logging.set_verbosity(optuna.logging.WARNING)


def fixa_sementes():
    random.seed(SEMENTE)
    np.random.seed(SEMENTE)


# ---------------------------------------------------------------- dados
df = pd.read_csv(BASE / "dados" / "icp_dataset.csv")
df.columns = df.columns.str.lower()
extra = pd.read_csv(BASE / "dados" / "icp_extra.csv")
extra.columns = extra.columns.str.lower()
assert df["id"].is_unique and extra["id"].is_unique and set(df["id"]) == set(extra["id"])
df = df.merge(extra, on="id", validate="one_to_one")
df = df[df["transf_externa_30d"] != 1].reset_index(drop=True)

ORIGINAIS = [c for c in pd.read_csv(BASE / "dados" / "icp_dataset.csv", nrows=0).columns.str.lower()
             if c not in ["id", DESFECHO, ANO, "mes_icp", "transf_externa_30d"]]
NOVAS = ["fc", "pas", "pad", "glasgow", "qsofa_max", "news2_max", "saps3_ic", "saps3_dva", "fe", "fe_dias"]

# --- variáveis derivadas para os escores (definições aproximadas, declaradas no resultado)
# filtração glomerular: CKD-EPI 2021 (sem raça), a partir de creatinina, idade e sexo
cr, idade, mulher = df["creatinina"], df["idade"], df["sexo_f"] == 1
kappa, alfa = np.where(mulher, 0.7, 0.9), np.where(mulher, -0.241, -0.302)
df["tfg"] = 142 * np.minimum(cr / kappa, 1) ** alfa * np.maximum(cr / kappa, 1) ** -1.200 * 0.9938 ** idade * np.where(mulher, 1.012, 1.0)
# choque: CID de choque/parada antes de t0 OU droga vasoativa "Sim" no SAPS3 preenchido antes de t0
df["choque"] = ((df["apres_choque_pcr"] == 1) | (df["saps3_dva"] == 1)).astype(float)
# insuficiência cardíaca: CID no ano anterior, CID na apresentação ou "Sim" no SAPS3
df["ic"] = ((df["hx_ic"] == 1) | (df["apres_ic"] == 1) | (df["saps3_ic"] == 1)).astype(float)
df["stemi"] = (df["apres_iamcsst"] == 1).astype(float)
# status do procedimento (0 eletivo, 1 urgente, 2 emergência); "salvamento" não tem equivalente no dado
emergencia = df["fl_codigo_iam"] == 1                                   # código de angioplastia primária no infarto
urgente = ~emergencia & ((df["atend_48h_antes"] == 1) | (df["apres_iam_qualquer"] == 1))
df["status_icp"] = np.select([emergencia, urgente], [2.0, 1.0], 0.0)
# infarto nas 24 h anteriores (Mayo): angioplastia primária, ou infarto com até 1 dia de internação
df["iam_24h"] = (emergencia | ((df["apres_iam_qualquer"] == 1) & (df["dias_internacao_ate_icp"].fillna(0) <= 1))).astype(float)

CONJUNTOS = {
    "Variáveis do CathPCI": ["idade", "choque", "ic", "dap", "dpoc", "tfg", "stemi", "status_icp"],
    "Variáveis do Mayo": ["idade", "creatinina", "fe", "choque", "iam_24h", "ic", "dap"],
    "30 preditores originais": ORIGINAIS,
    "Expandido (30 + 10 novas)": ORIGINAIS + NOVAS,
}

treino, teste = (df[ANO] < ANO_TESTE).to_numpy(), (df[ANO] >= ANO_TESTE).to_numpy()
y = df[DESFECHO].to_numpy().astype(int)
ytr, yte = y[treino], y[teste]
LIMIAR = float(ytr.mean())
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMENTE)
print(f"Treino {treino.sum()} ({ytr.sum()} óbitos) | teste {teste.sum()} ({yte.sum()} óbitos) | limiar {100*LIMIAR:.2f}%", flush=True)


# ---------------------------------------------------------------- escore CathPCI aproximado
def pontos_cathpci(d):
    """Pontos da Tabela 4 de Peterson et al. (JACC 2010). NYHA IV indisponível = 0; TFG ausente = 0 (assume > 90)."""
    p = np.select([d["idade"] >= 80, d["idade"] >= 70, d["idade"] >= 60], [14, 8, 4], 0).astype(float)
    p += 25 * d["choque"] + 5 * d["ic"] + 5 * (d["dap"] == 1) + 4 * (d["dpoc"] == 1)
    p += np.select([d["tfg"] < 30, d["tfg"] < 60, d["tfg"] <= 90], [18, 10, 6], 0)        # NaN cai em 0
    com_supra = np.select([d["status_icp"] == 2, d["status_icp"] == 1], [20, 15], 12)
    sem_supra = np.select([d["status_icp"] == 2, d["status_icp"] == 1], [20, 8], 0)
    return (p + np.where(d["stemi"] == 1, com_supra, sem_supra)).to_numpy()


PONTOS_TAB = np.arange(0, 101, 5)
RISCO_TAB = np.array([0.0, 0.1, 0.1, 0.2, 0.3, 0.6, 1.1, 2.0, 3.6, 6.3, 10.9, 18.3, 29.0, 42.7, 57.6, 71.2, 81.0, 89.2,
                      93.8, 96.5, 98.0]) / 100


def risco_cathpci(pontos):
    """Risco publicado para o total de pontos (interpolação linear entre os degraus de 5 pontos da tabela)."""
    return np.clip(np.interp(pontos, PONTOS_TAB, RISCO_TAB), 1e-4, 1 - 1e-4)


# ---------------------------------------------------------------- modelos
def cria_lr(p):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(C=p["C"], max_iter=5000, random_state=SEMENTE))


def espaco_lr(t):
    return {"C": t.suggest_float("C", 1e-3, 100, log=True)}


def cria_gb(p):
    return HistGradientBoostingClassifier(**p, early_stopping=False, random_state=SEMENTE)


def espaco_gb(t):
    return {"learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_iter": t.suggest_int("max_iter", 100, 800, step=50),
            "max_depth": t.suggest_int("max_depth", 2, 8),
            "max_leaf_nodes": t.suggest_int("max_leaf_nodes", 8, 64, log=True),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 10, 200, log=True),
            "l2_regularization": t.suggest_float("l2_regularization", 1e-3, 10, log=True),
            "max_features": t.suggest_float("max_features", 0.5, 1.0)}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ajusta(nome_modelo, cria, espaco, Xtr, Xte):
    """Optuna no treino, retreino em todo o treino, recalibração com predições fora da amostra do treino."""
    fixa_sementes()
    estudo = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEMENTE))
    estudo.optimize(lambda t: cross_val_score(cria(espaco(t)), Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=1).mean(),
                    n_trials=N_TRIALS)
    bruta = cria(estudo.best_params).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    fixa_sementes()
    oof = cross_val_predict(cria(estudo.best_params), Xtr, ytr, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    rec = LogisticRegression(C=1e6, max_iter=1000).fit(logit(oof).reshape(-1, 1), ytr)
    p = 1 / (1 + np.exp(-(rec.intercept_[0] + rec.coef_[0][0] * logit(bruta))))
    return p, estudo.best_value


def calibracao(yv, p):
    slope = LogisticRegression(C=1e6).fit(logit(p).reshape(-1, 1), yv).coef_[0][0]
    return np.log(yv.mean() / (1 - yv.mean())) - np.log(p.mean() / (1 - p.mean())), slope


def linha(rotulo, yv, p, auroc_cv=None, pontuacao=None):
    """Métricas de uma predição. `pontuacao` (ex.: pontos do escore) é usada para AUROC/AUPRC quando informada."""
    s = p if pontuacao is None else pontuacao
    sinal = p >= LIMIAR
    inter, slope = calibracao(yv, p)
    return {"Modelo": rotulo, "AUROC CV treino": "—" if auroc_cv is None else f"{auroc_cv:.4f}",
            "AUROC": f"{roc_auc_score(yv, s):.4f}", "AUPRC": f"{average_precision_score(yv, s):.4f}",
            "Brier": f"{brier_score_loss(yv, p):.4f}", "Intercepto": f"{inter:+.3f}", "Slope": f"{slope:.3f}",
            "Sinalizados (%)": f"{100*sinal.mean():.1f}", "Sensibilidade (%)": f"{100*sinal[yv == 1].mean():.1f}",
            "Especificidade (%)": f"{100*(~sinal)[yv == 0].mean():.1f}"}


def para_md(t):
    linhas = ["| " + " | ".join(map(str, t.columns)) + " |", "|" + "---|" * len(t.columns)]
    return "\n".join(linhas + ["| " + " | ".join(map(str, r)) + " |" for r in t.itertuples(index=False)]) + "\n"


# ---------------------------------------------------------------- execução
probs = {}
linhas = []
pts_te = pontos_cathpci(df[teste])
probs["CathPCI aproximado (escore publicado)"] = risco_cathpci(pts_te)
linhas.append(linha("CathPCI aproximado (escore publicado)", yte, probs["CathPCI aproximado (escore publicado)"], pontuacao=pts_te))
print(linhas[-1], flush=True)

for conjunto, colunas in CONJUNTOS.items():
    X = df[colunas].astype(float)
    Xtr, Xte = X[treino].to_numpy(), X[teste].to_numpy()
    for nome_modelo, cria, espaco in [("Regressão logística", cria_lr, espaco_lr), ("Gradient boosting", cria_gb, espaco_gb)]:
        rotulo = f"{nome_modelo} — {conjunto}"
        probs[rotulo], auroc_cv = ajusta(nome_modelo, cria, espaco, Xtr, Xte)
        linhas.append(linha(rotulo, yte, probs[rotulo], auroc_cv))
        print(linhas[-1], flush=True)

tab = pd.DataFrame(linhas)

# subgrupo com fração de ejeção conhecida (onde o Mayo seria calculável)
com_fe = df.loc[teste, "fe"].notna().to_numpy()
tab_fe = pd.DataFrame([linha(r, yte[com_fe], p[com_fe], pontuacao=pts_te[com_fe] if r.startswith("CathPCI") else None)
                       for r, p in probs.items()]).drop(columns=["AUROC CV treino"])

# cobertura das variáveis novas e derivadas no treino e no teste
cob = pd.DataFrame([{"Variável": v, "Treino com valor (%)": f"{100*df.loc[treino, v].notna().mean():.1f}",
                     "Teste com valor (%)": f"{100*df.loc[teste, v].notna().mean():.1f}",
                     "Média no teste": f"{df.loc[teste, v].mean():.2f}"} for v in NOVAS + ["tfg", "choque", "ic", "stemi", "status_icp", "iam_24h"]])
dist_pts = pd.Series(pts_te).describe()[["min", "25%", "50%", "75%", "max"]].round(0).to_dict()

texto = f"""# Escore clássico x aprendizado de máquina com as mesmas variáveis x versão expandida

- Treino: {treino.sum()} pessoas, {ytr.sum()} óbitos (2024–2025). Teste: {teste.sum()} pessoas, {yte.sum()} óbitos (jan–jul/2026).
- Optuna TPE (semente {SEMENTE}), {N_TRIALS} tentativas, AUROC em CV estratificada de {N_FOLDS} folds só no treino; recalibração logística com
  predições fora da amostra do treino; limiar = prevalência do treino ({100*LIMIAR:.2f}%). Valores pontuais, sem intervalo de confiança.
- **CathPCI aproximado:** pontos da Tabela 4 de Peterson et al. (JACC 2010). NYHA IV indisponível (0 ponto); filtração glomerular ausente
  conta 0 ponto; choque = CID antes de t0 ou droga vasoativa no SAPS3; insuficiência cardíaca = CID ou SAPS3; doença arterial periférica e
  pulmonar só por CID; status = emergência (código de angioplastia primária), urgente (atendimento nas 48 h anteriores ou infarto) ou eletivo;
  "salvamento" não existe no dado. AUROC e AUPRC usam os pontos; Brier, calibração e limiar usam o risco publicado para o total de pontos.
  Não é o escore validado: onde o dado falta, o risco é subestimado.
- **Mayo:** o escore inteiro não foi calculado (tabela de pontos não conferida). As linhas "Variáveis do Mayo" são modelos reajustados com as
  sete variáveis; a fração de ejeção vem de regex no laudo de eco, não validada em amostra manual.
- Distribuição dos pontos do CathPCI no teste: {dist_pts}

## Teste inteiro

{para_md(tab)}
## Subgrupo do teste com fração de ejeção conhecida ({com_fe.sum()} pessoas, {yte[com_fe].sum()} óbitos)

{para_md(tab_fe)}
## Cobertura das variáveis novas e derivadas

{para_md(cob)}"""
(SAIDA / "resultados.md").write_text(texto, encoding="utf-8")
pd.DataFrame({"id": df.loc[teste, "id"].to_numpy(), "obito_30d": yte, "pontos_cathpci": pts_te, **probs}).to_csv(SAIDA / "probs_teste.csv", index=False)
print(texto)
