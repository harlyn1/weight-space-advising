"""Read API keys from wsp/.env so they never have to be exported or pasted: KEY=VALUE lines, # comments, optional quotes.
Values already in the environment win. Copy .env.example to .env and chmod 600 it."""
import os


def load(path):
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k.startswith("export "):
            k = k[7:].strip()
        v = v.split(" #", 1)[0].split("\t#", 1)[0].strip().strip('"').strip("'")   # inline comments, as the shell reads them
        if v:
            os.environ.setdefault(k, v)
