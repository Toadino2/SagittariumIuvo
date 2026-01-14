import pickle
from pathlib import Path
import os
from letturascrittura.percorsi import cartellasessioni, filesessione
from datetime import date


def elencasessioni():
    cartella = cartellasessioni()
    sessioni = []
    for nomefile in os.listdir(cartella):
        percorso = os.path.join(cartella, nomefile)
        if not os.path.isfile(percorso):
            continue
        try:
            sessione = leggifile(percorso)
            sid = sessione.id
            data = sessione.data
            if not isinstance(data, date):
                continue
            sessioni.append({"id": sid, "data": data, "path": percorso})
        except Exception:
            continue
    sessioni.sort(key=lambda s: s["data"])
    return sessioni


def cancellasessione(id_: int):
    os.remove(filesessione(id_))


def creafile(file: str | Path, oggetto):
    with open(file, "xb") as f:
        pickle.dump(oggetto, f)


def leggifile(file: str | Path):
    with open(file, "rb") as f:
        oggetto = pickle.load(f)
    return oggetto


def scrivifile(file: str | Path, oggetto):
    with open(file, "wb") as f:
        pickle.dump(oggetto, f)