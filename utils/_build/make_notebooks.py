"""Build .ipynb (nbformat) from the percent-format templates and write the nbconverted .py next to them.
Usage: python make_notebooks.py <template.py> <out_notebook.ipynb> [KEY=VALUE ...]   (placeholders {KEY})"""
import sys, pathlib, subprocess, nbformat

tpl, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
subs = dict(a.split("=", 1) for a in sys.argv[3:])
text = tpl.read_text()
for k, v in subs.items():
    text = text.replace("{" + k + "}", v)
text = text.replace("{{", "{").replace("}}", "}")   # unescape doubled braces
cells, kind, buf = [], None, []
def flush():
    global buf, kind
    if kind is None:
        return
    body = "\n".join(buf).strip("\n")
    if kind == "markdown":
        body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.split("\n"))
        cells.append(nbformat.v4.new_markdown_cell(body))
    elif body.strip():
        cells.append(nbformat.v4.new_code_cell(body))
    buf = []
for line in text.split("\n"):
    if line.startswith("# %% [markdown]"):
        flush(); kind = "markdown"
    elif line.startswith("# %%"):
        flush(); kind = "code"
    else:
        buf.append(line)
flush()
nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                                    "language_info": {"name": "python", "file_extension": ".py", "mimetype": "text/x-python"}})
nbformat.write(nb, out)
nbc = out.parent / "nbconverted"
nbc.mkdir(exist_ok=True)
subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "script", "--output-dir", str(nbc), str(out)], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("wrote", out, "and", nbc / (out.stem + ".py"))
