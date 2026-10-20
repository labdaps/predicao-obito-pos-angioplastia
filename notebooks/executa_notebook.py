"""Executa o notebook único na máquina da GPU e grava as saídas em ICP_notebook_unico_executado.ipynb; marca o fim em unico.done."""
import asyncio, os, sys, time, traceback
from pathlib import Path
import nbformat
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from nbclient import NotebookClient
aqui = Path(__file__).parent
os.chdir(aqui)
marca = aqui / "unico.done"
marca.unlink(missing_ok=True)
nb = nbformat.read("ICP_notebook_unico.ipynb", 4)
t0, estado = time.time(), "OK"
try:
    NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
except Exception:
    estado = "ERRO\n" + traceback.format_exc()[-3000:]
finally:
    nbformat.write(nb, "ICP_notebook_unico_executado.ipynb")
    n = sum(1 for c in nb.cells if c.cell_type == "code" and c.outputs)
    marca.write_text(f"{estado}\ncelulas com saida: {n}\nminutos: {(time.time() - t0) / 60:.0f}\n", encoding="utf-8")
