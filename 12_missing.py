"""Distribuição de faltantes: por variável, por ano do evento e por desfecho.

Faltante informativo (mais faltante em quem morre) muda a leitura do modelo: o boosting
aprende o próprio "não colheu exame" como sinal. Gera resultados/missing.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
arq = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "dados" / "icp_dataset.csv"
saida = Path(sys.argv[2]) if len(sys.argv) > 2 else BASE / "resultados" / "missing.md"

df = pd.read_csv(arq)
df.columns = [c.lower() for c in df.columns]
df = df[df["transf_externa_30d"] != 1]
cols = [c for c in df.columns if c not in {"id", "ano_icp", "mes_icp", "obito_30d", "transf_externa_30d"}]
cols = [c for c in cols if df[c].isna().any()]

linhas = ["| Variável | Faltante total | 2024 | 2025 | 2026 | Óbitos | Vivos | Óbito se faltante | Óbito se presente |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for c in sorted(cols, key=lambda c: -df[c].isna().mean()):
    f = df[c].isna()
    pa = [100 * f[df.ano_icp == a].mean() for a in (2024, 2025, 2026)]
    linhas.append(
        f"| {c} | {100 * f.mean():.1f}% | {pa[0]:.1f}% | {pa[1]:.1f}% | {pa[2]:.1f}% "
        f"| {100 * f[df.obito_30d == 1].mean():.1f}% | {100 * f[df.obito_30d == 0].mean():.1f}% "
        f"| {100 * df.obito_30d[f].mean():.2f}% | {100 * df.obito_30d[~f].mean():.2f}% |"
    )

lab = ["creatinina", "hemoglobina", "leucocitos", "plaquetas", "potassio", "sodio", "ureia", "troponina_pct_no_codigo"]
lab = [c for c in lab if c in df.columns]
sem_lab = df[lab].isna().all(axis=1)
padrao = [
    "",
    f"- Pessoas analisadas: {len(df)}; óbitos: {int(df.obito_30d.sum())} ({100 * df.obito_30d.mean():.2f}%)",
    f"- Sem **nenhum** exame: {100 * sem_lab.mean():.1f}% "
    f"(óbito {100 * df.obito_30d[sem_lab].mean():.2f}% contra {100 * df.obito_30d[~sem_lab].mean():.2f}% em quem tem exame)",
    f"- Sem exame, por ano: " + ", ".join(f"{a}: {100 * sem_lab[df.ano_icp == a].mean():.1f}%" for a in (2024, 2025, 2026)),
]
texto = f"# Faltantes — {arq.name}\n" + "\n".join(padrao) + "\n\n" + "\n".join(linhas) + "\n"
saida.parent.mkdir(exist_ok=True)
saida.write_text(texto, encoding="utf-8")
print(texto)
