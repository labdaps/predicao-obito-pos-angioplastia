"""Sensibilidade: treinar só nas linhas em que as variáveis-chave do conjunto estão preenchidas.

Para cada conjunto de preditores, o treino é restrito aos casos completos pela regra do conjunto e o modelo é
avaliado (a) no teste inteiro e (b) no teste restrito pela mesma regra. A versão treinada no treino inteiro vem
das pastas resultados/modelos_* (mesmo desenho), para comparação na hora do relatório.

Regras:
  cathpci   -> creatinina presente (filtração glomerular calculável)
  mayo      -> creatinina e fração de ejeção presentes
  originais -> os 7 exames presentes (creatinina, hemoglobina, leucócitos, plaquetas, potássio, sódio, ureia)

Modelos: gradient boosting, XGBoost e CatBoost, com o mesmo Optuna do notebook (TPE, semente fixa, 50 tentativas,
AUROC em CV estratificada de 5 folds no treino) e a mesma recalibração logística fora da amostra.

uso: python 19_treino_casos_completos.py
"""
import os
import random
from pathlib import Path

SEMENTE = 20260929
os.environ["PYTHONHASHSEED"] = str(SEMENTE)

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from xgboost import XGBClassifier

BASE = Path(__file__).parent
SAIDA = BASE / "resultados" / "treino_casos_completos"
SAIDA.mkdir(parents=True, exist_ok=True)
N_TRIALS, N_FOLDS, N_THREADS = 50, 5, 8
optuna.logging.set_verbosity(optuna.logging.WARNING)


def fixa_sementes():
    random.seed(SEMENTE)
    np.random.seed(SEMENTE)


# ---------------------------------------------------------------- dados (mesmas definições do 18)
df = pd.read_csv(BASE / "dados" / "icp_dataset.csv")
df.columns = df.columns.str.lower()
ORIGINAIS = [c for c in df.columns if c not in ["id", "obito_30d", "ano_icp", "mes_icp", "transf_externa_30d"]]
extra = pd.read_csv(BASE / "dados" / "icp_extra.csv")
extra.columns = extra.columns.str.lower()
df = df.merge(extra, on="id", validate="one_to_one")
df = df[df["transf_externa_30d"] != 1].reset_index(drop=True)

mulher = df["sexo_f"] == 1
k, a = np.where(mulher, 0.7, 0.9), np.where(mulher, -0.241, -0.302)
df["tfg"] = 142 * np.minimum(df["creatinina"] / k, 1) ** a * np.maximum(df["creatinina"] / k, 1) ** -1.200 \
            * 0.9938 ** df["idade"] * np.where(mulher, 1.012, 1.0)
df["choque"] = ((df["apres_choque_pcr"] == 1) | (df["saps3_dva"] == 1)).astype(float)
df["ic"] = ((df["hx_ic"] == 1) | (df["apres_ic"] == 1) | (df["saps3_ic"] == 1)).astype(float)
df["stemi"] = (df["apres_iamcsst"] == 1).astype(float)
emerg = df["fl_codigo_iam"] == 1
df["status_icp"] = np.select([emerg, (df["atend_48h_antes"] == 1) | (df["apres_iam_qualquer"] == 1)], [2.0, 1.0], 0.0)
df["iam_24h"] = (emerg | ((df["apres_iam_qualquer"] == 1) & (df["dias_internacao_ate_icp"].fillna(0) <= 1))).astype(float)

EXAMES = ["creatinina", "hemoglobina", "leucocitos", "plaquetas", "potassio", "sodio", "ureia"]
CONJUNTOS = {
    "cathpci": (["idade", "choque", "ic", "dap", "dpoc", "tfg", "stemi", "status_icp"], df["creatinina"].notna()),
    "mayo": (["idade", "creatinina", "fe", "choque", "iam_24h", "ic", "dap"], df["creatinina"].notna() & df["fe"].notna()),
    "originais": (ORIGINAIS, df[EXAMES].notna().all(axis=1)),
}
treino, teste = (df["ano_icp"] < 2026).to_numpy(), (df["ano_icp"] >= 2026).to_numpy()
y = df["obito_30d"].to_numpy().astype(int)


# ---------------------------------------------------------------- modelos (mesmos espaços de busca do notebook)
def cria_gb(p):
    return HistGradientBoostingClassifier(**p, early_stopping=False, random_state=SEMENTE)


def espaco_gb(t):
    return {"learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_iter": t.suggest_int("max_iter", 100, 800, step=50), "max_depth": t.suggest_int("max_depth", 2, 8),
            "max_leaf_nodes": t.suggest_int("max_leaf_nodes", 8, 64, log=True),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 10, 200, log=True),
            "l2_regularization": t.suggest_float("l2_regularization", 1e-3, 10, log=True),
            "max_features": t.suggest_float("max_features", 0.5, 1.0)}


def cria_xgb(p):
    return XGBClassifier(**p, objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cpu",
                         random_state=SEMENTE, n_jobs=N_THREADS)


def espaco_xgb(t):
    return {"n_estimators": t.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True), "max_depth": t.suggest_int("max_depth", 2, 8),
            "min_child_weight": t.suggest_float("min_child_weight", 1, 20, log=True),
            "subsample": t.suggest_float("subsample", 0.6, 1.0), "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": t.suggest_float("gamma", 1e-3, 5, log=True), "reg_alpha": t.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True)}


def cria_cat(p):
    return CatBoostClassifier(**p, loss_function="Logloss", random_seed=SEMENTE, verbose=0, thread_count=N_THREADS,
                              allow_writing_files=False)


def espaco_cat(t):
    return {"iterations": t.suggest_int("iterations", 200, 1000, step=100),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True), "depth": t.suggest_int("depth", 3, 8),
            "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1, 20, log=True),
            "random_strength": t.suggest_float("random_strength", 1e-3, 10, log=True),
            "bagging_temperature": t.suggest_float("bagging_temperature", 0, 1)}


MODELOS = [("Gradient boosting", cria_gb, espaco_gb), ("XGBoost", cria_xgb, espaco_xgb), ("CatBoost", cria_cat, espaco_cat)]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def metricas(yv, p):
    slope = LogisticRegression(C=1e6).fit(logit(p).reshape(-1, 1), yv).coef_[0][0]
    inter = np.log(yv.mean() / (1 - yv.mean())) - np.log(p.mean() / (1 - p.mean()))
    return {"N": len(yv), "Óbitos": int(yv.sum()), "AUROC": f"{roc_auc_score(yv, p):.4f}", "AUPRC": f"{average_precision_score(yv, p):.4f}",
            "Brier": f"{brier_score_loss(yv, p):.4f}", "Intercepto": f"{inter:+.3f}", "Slope": f"{slope:.3f}"}


def para_md(t):
    linhas = ["| " + " | ".join(map(str, t.columns)) + " |", "|" + "---|" * len(t.columns)]
    return "\n".join(linhas + ["| " + " | ".join(map(str, r)) + " |" for r in t.itertuples(index=False)]) + "\n"


# ---------------------------------------------------------------- execução
linhas, probs = [], {"id": df.loc[teste, "id"].to_numpy(), "obito_30d": y[teste]}
yte = y[teste]
for conjunto, (colunas, completo) in CONJUNTOS.items():
    completo = completo.to_numpy()
    X = df[colunas].astype(float).to_numpy()
    tr = treino & completo
    te_completo = completo[teste]
    probs[f"completo_{conjunto}"] = te_completo.astype(int)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMENTE)
    print(f"== {conjunto}: treino restrito {tr.sum()} ({y[tr].sum()} óbitos) de {treino.sum()}; teste completo {te_completo.sum()} ({yte[te_completo].sum()} óbitos)", flush=True)
    for nome, cria, espaco in MODELOS:
        fixa_sementes()
        estudo = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEMENTE))
        estudo.optimize(lambda t: cross_val_score(cria(espaco(t)), X[tr], y[tr], cv=cv, scoring="roc_auc", n_jobs=1).mean(), n_trials=N_TRIALS)
        bruta = cria(estudo.best_params).fit(X[tr], y[tr]).predict_proba(X[teste])[:, 1]
        fixa_sementes()
        oof = cross_val_predict(cria(estudo.best_params), X[tr], y[tr], cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        rec = LogisticRegression(C=1e6, max_iter=1000).fit(logit(oof).reshape(-1, 1), y[tr])
        p = 1 / (1 + np.exp(-(rec.intercept_[0] + rec.coef_[0][0] * logit(bruta))))
        probs[f"{nome} | {conjunto}"] = p
        for alvo, m in [("teste inteiro", np.ones(len(yte), bool)), ("teste com variáveis preenchidas", te_completo)]:
            linhas.append({"Conjunto": conjunto, "Modelo": nome, "Avaliado em": alvo, "AUROC CV treino": f"{estudo.best_value:.4f}", **metricas(yte[m], p[m])})
            print(linhas[-1], flush=True)

tab = pd.DataFrame(linhas)
texto = f"""# Sensibilidade: treino restrito às linhas com as variáveis-chave preenchidas

- Treino restrito pela regra do conjunto; avaliação no teste inteiro e no teste restrito pela mesma regra (jan–jul/2026).
- Regras: cathpci = creatinina presente; mayo = creatinina e fração de ejeção presentes; originais = os 7 exames presentes.
- Optuna TPE (semente {SEMENTE}), {N_TRIALS} tentativas, CV de {N_FOLDS} folds no treino restrito; recalibração logística fora da amostra.
- Valores pontuais, sem intervalo de confiança. O treino restrito é uma população diferente (mais grave e mais urgente).

{para_md(tab)}"""
(SAIDA / "resultados.md").write_text(texto, encoding="utf-8")
pd.DataFrame(probs).to_csv(SAIDA / "probs_teste.csv", index=False)
print(texto)
