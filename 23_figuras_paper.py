"""Figuras do manuscrito, em inglês, a partir das probabilidades do teste (resultados/gpu_lab_final/).

Gera em manuscrito/figuras/:
  fig1_fluxo.png          fluxo de pacientes
  fig2_roc_pr.png         (A) curvas ROC e (B) precision-recall dos seis modelos e do escore CathPCI publicado
  fig3_calibracao.png     calibração suavizada (LOWESS) com banda de 95% por bootstrap, decis e distribuição das predições
  fig4_conjuntos.png      AUROC e AUPRC com IC95% por conjunto de preditores (variáveis do CathPCI, do Mayo, 40 preditores)
  fig5_curva_decisao.png  benefício líquido (curva de decisão)
  fig6_shap.png           SHAP do TabPFN: (A) importância e (B) beeswarm (shap_tabpfn.npz, rodada da GPU)
  figS1_shap_catboost.png TreeSHAP exato do CatBoost em todo o teste (24_shap_catboost.py)

Paleta categórica validada para daltonismo (skill dataviz); cada modelo tem também um estilo de linha próprio.
uso: python 23_figuras_paper.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from statsmodels.nonparametric.smoothers_lowess import lowess

BASE = Path(__file__).parent
G = BASE / "resultados" / "gpu_lab_final"
OUT = BASE / "manuscrito" / "figuras"
OUT.mkdir(exist_ok=True)
SEMENTE, LIMIAR = 20260929, 0.0264

EN = {"Regressão logística": "Logistic regression", "Gradient boosting": "Gradient boosting", "XGBoost": "XGBoost",
      "CatBoost": "CatBoost", "TabPFN": "TabPFN", "Kumo Tabular": "Kumo Tabular", "CathPCI (escore publicado)": "Published CathPCI score"}
ORDEM = ["Regressão logística", "Gradient boosting", "XGBoost", "CatBoost", "TabPFN", "Kumo Tabular"]
COR = dict(zip(ORDEM, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]))
COR["CathPCI (escore publicado)"] = "#3a3a38"
ESTILO = dict(zip(ORDEM, ["-", "--", "-.", ":", "-", "--"]))
ESTILO["CathPCI (escore publicado)"] = (0, (1, 1.5))
TINTA, TINTA2, GRADE = "#1f1f1e", "#5f5e58", "#e6e5e0"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA,
                     "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.6, "legend.frameon": False, "savefig.dpi": 300})

pr = pd.read_csv(G / "resultados_expandido" / "probs_teste_todos.csv").sort_values("id").reset_index(drop=True)
cp = pd.read_csv(G / "resultados_cathpci" / "probs_teste_todos.csv").sort_values("id").reset_index(drop=True)
assert (pr["id"].to_numpy() == cp["id"].to_numpy()).all()
y = pr["obito_30d"].to_numpy().astype(int)          # coluna com o desfecho da análise (óbito na internação, até 30 dias)
P = {m: pr[m].to_numpy() for m in ORDEM}
P["CathPCI (escore publicado)"] = cp["CathPCI (escore publicado)"].to_numpy()
n, ev = len(y), int(y.sum())
IC = pd.read_csv(G / "bootstrap" / "ic95_metricas.csv")
IC = IC[IC["Avaliação"] == "teste inteiro"]

# ---------------------------------------------------------------- Figura 1: fluxo
fig, ax = plt.subplots(figsize=(7, 6.2)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
def caixa(x, yy, txt, w=5.6, h=1.0):
    ax.add_patch(plt.Rectangle((x - w / 2, yy - h / 2), w, h, fill=False, ec=TINTA, lw=1))
    ax.text(x, yy, txt, ha="center", va="center", fontsize=8.5, color=TINTA)
def seta(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=TINTA2, lw=1))
caixa(4.2, 9.2, "9,267 patient registrations with a first PCI\nin the owned hospital network, 2023–2025")
caixa(4.2, 7.4, "9,243 persons\n(registrations linked to a unified medical record)")
caixa(8.2, 5.9, "Excluded: 345\ntransferred to an external\nhospital within 30 days", w=3.2, h=1.3)
caixa(4.2, 4.4, "8,898 patients analysed")
caixa(2.2, 2.0, "Development set\nPCI in 2023–2024\n4,961 patients\n131 in-hospital deaths (2.64%)", w=3.6, h=1.8)
caixa(6.6, 2.0, "Temporal test set\nPCI in 2025\n3,937 patients\n140 in-hospital deaths (3.56%)", w=3.6, h=1.8)
seta(4.2, 8.7, 4.2, 7.9); seta(4.2, 6.9, 4.2, 4.9); seta(4.2, 5.9, 6.6, 5.9)
seta(4.2, 3.9, 2.2, 2.9); seta(4.2, 3.9, 6.6, 2.9)
fig.savefig(OUT / "fig1_fluxo.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Figura 2: ROC e precision-recall
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.9))
for m in ORDEM + ["CathPCI (escore publicado)"]:
    fpr, tpr, _ = roc_curve(y, P[m])
    a1.plot(fpr, tpr, color=COR[m], ls=ESTILO[m], lw=1.6 if m in ORDEM else 1.4, label=f"{EN[m]} ({roc_auc_score(y, P[m]):.3f})")
    prec, rec, _ = precision_recall_curve(y, P[m])
    a2.plot(rec, prec, color=COR[m], ls=ESTILO[m], lw=1.6 if m in ORDEM else 1.4, label=f"{EN[m]} ({average_precision_score(y, P[m]):.3f})",
            drawstyle="steps-post")
a1.plot([0, 1], [0, 1], color=TINTA2, lw=0.8, ls=(0, (4, 3)))
a1.set(xlabel="1 − specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1.01)); a1.set_aspect("equal")
a1.legend(title="AUROC", loc="lower right", fontsize=7.5, title_fontsize=8)
a1.set_title("A  Receiver operating characteristic", loc="left", fontsize=10, color=TINTA)
a2.axhline(y.mean(), color=TINTA2, lw=0.8, ls=(0, (4, 3)), label=f"Prevalence ({y.mean():.3f})")
a2.set(xlabel="Sensitivity (recall)", ylabel="Positive predictive value (precision)", xlim=(0, 1), ylim=(0, 1.01))
a2.legend(title="AUPRC", loc="upper right", fontsize=7.5, title_fontsize=8)
a2.set_title("B  Precision–recall", loc="left", fontsize=10, color=TINTA)
fig.tight_layout(); fig.savefig(OUT / "fig2_roc_pr.png"); plt.close(fig)

# ---------------------------------------------------------------- Figura 3: calibração
rng = np.random.default_rng(SEMENTE)
grade = np.linspace(0, 0.30, 121)
fig, eixos = plt.subplots(2, 3, figsize=(10, 6.6), sharex=True, sharey=True)
for ax, m in zip(eixos.ravel(), ORDEM):
    p = P[m]
    def curva(pp, yy):
        s = lowess(yy, pp, frac=0.6, it=0, delta=0.002, return_sorted=True)
        return np.interp(grade, s[:, 0], s[:, 1], left=np.nan, right=np.nan)
    base = curva(p, y)
    boots = np.array([curva(p[i], y[i]) for i in (rng.integers(0, n, n) for _ in range(200))])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    ax.fill_between(grade, lo, hi, color=COR[m], alpha=0.18, lw=0)
    ax.plot(grade, base, color=COR[m], lw=1.8)
    ordem_p = np.argsort(p)
    for g in np.array_split(ordem_p, 10):
        ax.plot(p[g].mean(), y[g].mean(), "o", ms=4, mfc="white", mec=COR[m], mew=1.2)
    ax.plot([0, 0.30], [0, 0.30], color=TINTA2, lw=0.8, ls=(0, (4, 3)))
    cont, bordas = np.histogram(np.clip(p, 0, 0.30), bins=60, range=(0, 0.30))
    ax.bar(bordas[:-1], 0.04 * cont / cont.max(), width=bordas[1] - bordas[0], bottom=0, color=TINTA2, alpha=0.35, align="edge", lw=0)
    r = IC[(IC["Parte"] == "principal") & (IC["Modelo"] == m)].iloc[0]
    ax.text(0.015, 0.285, f"Intercept {r['Intercepto']:+.2f}\nSlope {r['Slope']:.2f}", fontsize=7.5, va="top", color=TINTA)
    ax.set_title(EN[m], fontsize=9.5, loc="left", color=TINTA)
    ax.set_xlim(0, 0.30); ax.set_ylim(0, 0.30)
for ax in eixos[1]: ax.set_xlabel("Predicted risk")
for ax in eixos[:, 0]: ax.set_ylabel("Observed risk")
fig.tight_layout(); fig.savefig(OUT / "fig3_calibracao.png"); plt.close(fig)

# ---------------------------------------------------------------- Figura 4: conjuntos de preditores
CONJ = [("cathpci", "CathPCI\nvariables (8)"), ("mayo", "Mayo\nvariables (7)"), ("principal", "40\npredictors")]
fig, eixos = plt.subplots(1, 2, figsize=(10, 4.4))
desloc = np.linspace(-0.3, 0.3, len(ORDEM))
for ax, met in zip(eixos, ["AUROC", "AUPRC"]):
    for k, m in enumerate(ORDEM):
        xs, vs, los, his = [], [], [], []
        for i, (parte, _) in enumerate(CONJ):
            r = IC[(IC["Parte"] == parte) & (IC["Modelo"] == m)].iloc[0]
            xs.append(i + desloc[k]); vs.append(r[met]); los.append(r[met] - r[f"{met} inf"]); his.append(r[f"{met} sup"] - r[met])
        ax.errorbar(xs, vs, yerr=[los, his], fmt="o", ms=5, color=COR[m], ecolor=COR[m], elinewidth=1.2, capsize=0, label=EN[m])
    r = IC[(IC["Parte"] == "cathpci") & (IC["Modelo"] == "CathPCI (escore publicado)")].iloc[0]
    ax.axhline(r[met], color=COR["CathPCI (escore publicado)"], lw=1, ls=ESTILO["CathPCI (escore publicado)"])
    ax.text(2.45, r[met], " Published\n CathPCI score", fontsize=7.5, va="center", color=TINTA2)
    if met == "AUPRC":
        ax.axhline(y.mean(), color=TINTA2, lw=0.8, ls=(0, (4, 3))); ax.text(2.45, y.mean(), " Prevalence", fontsize=7.5, va="center", color=TINTA2)
    ax.set_xticks(range(len(CONJ))); ax.set_xticklabels([c[1] for c in CONJ]); ax.set_xlim(-0.5, 3.25)
    ax.set_ylabel(f"{met} (95% CI)"); ax.grid(axis="x", visible=False)
    ax.set_title(("A  " if met == "AUROC" else "B  ") + met, loc="left", fontsize=10, color=TINTA)
eixos[0].legend(loc="upper left", fontsize=7.5)
fig.tight_layout(); fig.savefig(OUT / "fig4_conjuntos.png"); plt.close(fig)

# ---------------------------------------------------------------- Figura 5: curva de decisão
pts = np.linspace(0.005, 0.20, 196)
def nb(p, t):
    s = p >= t
    return (np.sum(s & (y == 1)) - np.sum(s & (y == 0)) * t / (1 - t)) / n
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.plot(pts * 100, [1000 * (y.mean() - (1 - y.mean()) * t / (1 - t)) for t in pts], color=TINTA2, lw=1, label="Monitor all")
ax.axhline(0, color=TINTA, lw=0.8, label="Monitor none")
for m in ORDEM + ["CathPCI (escore publicado)"]:
    ax.plot(pts * 100, [1000 * nb(P[m], t) for t in pts], color=COR[m], ls=ESTILO[m], lw=1.6 if m in ORDEM else 1.4, label=EN[m])
ax.axvline(LIMIAR * 100, color=TINTA2, lw=0.8, ls=(0, (4, 3)))
ax.text(LIMIAR * 100 + 0.2, 33, f"{100*LIMIAR:.2f}%", fontsize=7.5, color=TINTA2)
ax.set(xlabel="Threshold probability (%)", ylabel="Net benefit per 1,000 patients", xlim=(0, 20), ylim=(-5, 36))
ax.legend(loc="upper right", fontsize=7.5, ncol=2)
fig.tight_layout(); fig.savefig(OUT / "fig5_curva_decisao.png"); plt.close(fig)

# ---------------------------------------------------------------- Figura 6: SHAP do TabPFN
NOME = {"idade": "Age", "apres_angina": "Angina at presentation", "leucocitos": "Leukocytes", "fl_codigo_iam": "PCI billed as AMI procedure",
        "atend_48h_antes": "Encounter in 48 h before PCI", "fe": "LVEF", "sodio": "Sodium", "creatinina": "Creatinine", "ureia": "Urea",
        "potassio": "Potassium", "hemoglobina": "Haemoglobin", "fe_dias": "Days since echocardiogram", "plaquetas": "Platelets",
        "dias_internacao_ate_icp": "Days from admission to PCI", "internado_no_dia": "Inpatient on day of PCI", "troponina_pct": "Troponin (percentile)",
        "fc": "Heart rate", "pas": "Systolic blood pressure", "pad": "Diastolic blood pressure", "glasgow": "Glasgow coma scale",
        "apres_iam_qualquer": "Any AMI at presentation", "sexo_f": "Female sex", "fl_multiarterial_tronco": "Multivessel or left main PCI code"}
NOME.update({"saps3_ic": "Heart failure (SAPS 3)", "saps3_dva": "Vasoactive drug (SAPS 3)", "qsofa_max": "qSOFA", "news2_max": "NEWS2",
             "hx_iam": "Prior myocardial infarction", "n_intern_365d": "Hospitalisations in prior year", "apres_iamcsst": "STEMI at presentation"})
def beeswarm(ax, sv, X, nomes, tam):
    imp = np.abs(sv).mean(axis=0); topo = np.argsort(imp)[::-1][:15]
    rng_p = np.random.default_rng(SEMENTE)
    for linha, j in enumerate(topo[::-1]):
        v, s_ = X[:, j].astype(float), sv[:, j]; aus = np.isnan(v)
        jit = linha + rng_p.uniform(-0.3, 0.3, len(s_))
        ax.scatter(s_[aus], jit[aus], s=tam, color="#c9c8c1", lw=0, alpha=0.8)
        pts_ = ax.scatter(s_[~aus], jit[~aus], s=tam, c=pd.Series(v[~aus]).rank(pct=True), cmap="coolwarm", vmin=0, vmax=1, lw=0, alpha=0.85)
    ax.axvline(0, color=TINTA2, lw=0.8); ax.grid(axis="y", visible=False)
    ax.set_yticks(range(len(topo))); ax.set_yticklabels([NOME.get(nomes[j], nomes[j]) for j in topo[::-1]])
    ax.set_xlabel("Shapley value (log-odds; positive = higher predicted risk)")
    return pts_

# Figura 6: TabPFN (melhor AUROC no teste), Shapley por permutação refeito na GPU com a mesma semente (shap_tabpfn.npz)
zt = np.load(G / "resultados_expandido" / "shap_tabpfn.npz", allow_pickle=False)
svt, Xt, nt = zt["shap_valores"], zt["X_ref"], list(zt["preditores"])
impt = np.abs(svt).mean(axis=0); topot = np.argsort(impt)[::-1][:15]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 6.2), gridspec_kw={"width_ratios": [1, 1.35]})
a1.barh(range(len(topot)), impt[topot][::-1], color=COR["TabPFN"], height=0.62)
a1.set_yticks(range(len(topot))); a1.set_yticklabels([NOME.get(nt[j], nt[j]) for j in topot[::-1]])
a1.set_xlabel("Mean |Shapley value| (log-odds)"); a1.grid(axis="y", visible=False)
a1.set_title("A  Mean absolute Shapley value", loc="left", fontsize=9.5, color=TINTA)
pts_ = beeswarm(a2, svt, Xt, nt, 9)
a2.set_title(f"B  Shapley values per patient (n = {len(svt)})", loc="left", fontsize=9.5, color=TINTA)
cb = fig.colorbar(pts_, ax=a2, ticks=[0, 1], fraction=0.035); cb.ax.set_yticklabels(["Low", "High"]); cb.set_label("Predictor value (grey = missing)")
fig.tight_layout(); fig.savefig(OUT / "fig6_shap.png"); plt.close(fig)

# Figura suplementar S1: TreeSHAP exato do CatBoost em todo o teste (24_shap_catboost.py)
zc = np.load(G / "resultados_expandido" / "shap_catboost.npz", allow_pickle=False)
fig, ax = plt.subplots(figsize=(8, 6.2))
pts_ = beeswarm(ax, zc["shap_valores"], zc["X_ref"], list(zc["preditores"]), 4)
cb = fig.colorbar(pts_, ax=ax, ticks=[0, 1], fraction=0.035); cb.ax.set_yticklabels(["Low", "High"]); cb.set_label("Predictor value (grey = missing)")
fig.tight_layout(); fig.savefig(OUT / "figS1_shap_catboost.png"); plt.close(fig)
print("figuras em", OUT)
