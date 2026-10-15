"""SHAP (TreeSHAP exato do XGBoost) para o modelo de óbito em 30 dias pós-angioplastia.

Mesmo recorte do 10_modelo.py: exclui transferidos externos, treina em < 2026 e explica o
conjunto de teste (2026). Gera resultados/shap_beeswarm.png e resultados/shap_importancia.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).parent
SAIDA = BASE / "resultados"
SEMENTE = 20260929

df = pd.read_csv(BASE / "dados" / "icp_dataset.csv")
df.columns = [c.lower() for c in df.columns]
df = df[df["transf_externa_30d"] != 1]
fora = ["id", "ano_icp", "mes_icp", "obito_30d", "transf_externa_30d"]
X = df.drop(columns=fora)
y = df["obito_30d"].astype(int)
treino, teste = df["ano_icp"] < 2026, df["ano_icp"] >= 2026

modelo = xgb.XGBClassifier(
    n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, eval_metric="logloss", random_state=SEMENTE,
)
modelo.fit(X[treino], y[treino])
p = modelo.predict_proba(X[teste])[:, 1]
print(f"AUROC XGBoost (teste 2026): {roc_auc_score(y[teste], p):.3f}")

contrib = modelo.get_booster().predict(xgb.DMatrix(X[teste]), pred_contribs=True)
shap = contrib[:, :-1]  # última coluna é o viés
media_abs = np.abs(shap).mean(axis=0)
ordem = np.argsort(media_abs)[::-1]
nomes = X.columns.to_numpy()
SAIDA.mkdir(exist_ok=True)

# 1. Importância média |SHAP|
fig, ax = plt.subplots(figsize=(8, 9))
ax.barh(nomes[ordem][::-1], media_abs[ordem][::-1], color="#3b6ea5")
ax.set_xlabel("média |SHAP| (log-odds)")
ax.set_title("Importância das variáveis — óbito em 30 dias pós-angioplastia\n(XGBoost, teste 2026)")
fig.tight_layout()
fig.savefig(SAIDA / "shap_importancia.png", dpi=150)
plt.close(fig)

# 2. Beeswarm (cor = valor da variável, normalizado por percentil; cinza = faltante)
top = ordem[:20]
rng = np.random.default_rng(SEMENTE)
fig, ax = plt.subplots(figsize=(9, 10))
Xt = X[teste].to_numpy(dtype=float)
for pos, j in enumerate(top[::-1]):
    v = Xt[:, j]
    s = shap[:, j]
    falta = np.isnan(v)
    cor = np.full(len(v), np.nan)
    if (~falta).any():
        cor[~falta] = pd.Series(v[~falta]).rank(pct=True).to_numpy()
    jitter = rng.normal(0, 0.12, len(s))
    ax.scatter(s[falta], pos + jitter[falta], c="#bbbbbb", s=6, alpha=0.6)
    sc = ax.scatter(s[~falta], pos + jitter[~falta], c=cor[~falta], cmap="coolwarm", s=6, alpha=0.7, vmin=0, vmax=1)
ax.set_yticks(range(len(top)))
ax.set_yticklabels(nomes[top][::-1])
ax.axvline(0, color="#555", lw=0.8)
ax.set_xlabel("valor SHAP (impacto no log-odds de óbito)")
ax.set_title("SHAP por paciente — 20 variáveis mais importantes\n(vermelho = valor alto, azul = baixo, cinza = faltante)")
cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("percentil do valor")
fig.tight_layout()
fig.savefig(SAIDA / "shap_beeswarm.png", dpi=150)
plt.close(fig)

# 3. Tabela para checar vazamento: importância e prevalência entre óbitos vs vivos
tab = pd.DataFrame({"variavel": nomes[ordem], "media_abs_shap": media_abs[ordem].round(4)})
bin_cols = [c for c in X.columns if set(X[c].dropna().unique()) <= {0, 1}]
tt = df[teste]
for c in bin_cols:
    tab.loc[tab.variavel == c, "pct_obitos"] = round(100 * tt.loc[tt.obito_30d == 1, c].mean(), 1)
    tab.loc[tab.variavel == c, "pct_vivos"] = round(100 * tt.loc[tt.obito_30d == 0, c].mean(), 1)
tab.to_csv(SAIDA / "shap_importancia.csv", index=False)
print(tab.head(15).to_string(index=False))
