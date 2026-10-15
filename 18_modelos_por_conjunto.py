"""Roda TODOS os modelos do notebook (Optuna para regressão logística, gradient boosting, XGBoost e CatBoost;
TabPFN; Kumo Tabular; recalibração; limiar; curva de decisão; equidade; SHAP) num outro conjunto de preditores.

Não duplica o pipeline: executa as células de código de notebooks/ICP_modelos_fundacionais_GPU.ipynb, trocando
só (a) a lista de preditores e (b) a pasta de saída. Assim o desenho é idêntico ao da análise principal.

Conjuntos:
  cathpci    8 variáveis do NCDR CathPCI de beira de leito (NYHA IV não existe no dado)
  mayo       7 variáveis do Mayo Clinic Risk Score
  expandido  30 preditores originais + 10 novas (sinais vitais, SAPS3, fração de ejeção), todas anteriores a t0

uso: python 18_modelos_por_conjunto.py <cathpci|mayo|expandido>
Saída: resultados/modelos_<conjunto>/
"""
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import nbformat

BASE = Path(__file__).parent
CONJUNTO = sys.argv[1]
assert CONJUNTO in ("cathpci", "mayo", "expandido"), CONJUNTO

# bloco que substitui a linha `preditores = [...]` do notebook
BLOCO = '''
#--- conjunto de preditores alternativo (18_modelos_por_conjunto.py)
_extra = pd.read_csv("../dados/icp_extra.csv"); _extra.columns = _extra.columns.str.lower()
_originais = [c for c in df.columns if c not in FORA]
df = df.merge(_extra, on="id", how="left", validate="one_to_one")       # mantém a ordem das linhas
_mulher = df["sexo_f"] == 1
_k, _a = np.where(_mulher, 0.7, 0.9), np.where(_mulher, -0.241, -0.302)
#filtração glomerular: CKD-EPI 2021
df["tfg"] = 142 * np.minimum(df["creatinina"] / _k, 1) ** _a * np.maximum(df["creatinina"] / _k, 1) ** -1.200 \\
            * 0.9938 ** df["idade"] * np.where(_mulher, 1.012, 1.0)
#choque: CID antes de t0 ou droga vasoativa "Sim" no SAPS3; insuficiência cardíaca: CID ou SAPS3
df["choque"] = ((df["apres_choque_pcr"] == 1) | (df["saps3_dva"] == 1)).astype(float)
df["ic"] = ((df["hx_ic"] == 1) | (df["apres_ic"] == 1) | (df["saps3_ic"] == 1)).astype(float)
df["stemi"] = (df["apres_iamcsst"] == 1).astype(float)
#status: 2 emergência (código de angioplastia primária), 1 urgente (atendimento nas 48 h ou infarto), 0 eletivo
_emerg = df["fl_codigo_iam"] == 1
df["status_icp"] = np.select([_emerg, (df["atend_48h_antes"] == 1) | (df["apres_iam_qualquer"] == 1)], [2.0, 1.0], 0.0)
df["iam_24h"] = (_emerg | ((df["apres_iam_qualquer"] == 1) & (df["dias_internacao_ate_icp"].fillna(0) <= 1))).astype(float)
_novas = ["fc", "pas", "pad", "glasgow", "qsofa_max", "news2_max", "saps3_ic", "saps3_dva", "fe", "fe_dias"]
preditores = {"cathpci": ["idade", "choque", "ic", "dap", "dpoc", "tfg", "stemi", "status_icp"],
              "mayo": ["idade", "creatinina", "fe", "choque", "iam_24h", "ic", "dap"],
              "expandido": _originais + _novas}["%s"]
print("Conjunto:", "%s", "|", len(preditores), "preditores:", preditores)
''' % (CONJUNTO, CONJUNTO)

nb = nbformat.read(BASE / "notebooks" / "ICP_modelos_fundacionais_GPU.ipynb", 4)
celulas = [c.source for c in nb.cells if c.cell_type == "code" and "FONTES = {" not in c.source]   # sem a célula de instalação
codigo = "\n\n".join(celulas)

LINHA_PRED = "preditores = [c for c in df.columns if c not in FORA]"
LINHA_SAIDA = 'SAIDA = Path("../resultados/modelos_paper") if Path("../resultados").exists() else Path("resultados_modelos")'
assert codigo.count(LINHA_PRED) == 1 and codigo.count(LINHA_SAIDA) == 1, "o notebook mudou: rever os pontos de troca"
codigo = codigo.replace(LINHA_PRED, BLOCO).replace(LINHA_SAIDA, f'SAIDA = Path("../resultados/modelos_{CONJUNTO}")')
#a tabela de características lista só os preditores do conjunto
codigo = codigo.replace('for var in [c for c in df.columns if c not in ["id", DESFECHO, ANO, "mes_icp", "transf_externa_30d"]]:',
                        "for var in preditores:")

os.chdir(BASE / "notebooks")          # os caminhos do notebook são relativos à pasta dele
exec(compile(codigo, f"notebook[{CONJUNTO}]", "exec"), {"__name__": "__main__"})
