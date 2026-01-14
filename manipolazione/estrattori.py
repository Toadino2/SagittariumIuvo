import math
import scipy.stats as st
import numpy as np
from analisi.descrittive import varianzaangoli
from classi.dati import REGISTRO, registra, monitora, Libro, SessioneGrezza


@registra("crf", set(), "correlazioni tra ascisse e ordinate per ogni freccia", REGISTRO)
@monitora("crf")
def correlazionefrecce(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> dict[int, float] | None:
    coordinate = dati.dati.xy
    frecce = dati.dati.idfrecce
    if frecce is None or dati.dati.frecce < 2:
        return None
    return {f: np.corrcoef(coordinate[frecce == f, :])[0, 1] for f in range(dati.dati.frecce)}


@registra("mpf", set(), "media dei punteggi per ogni freccia", REGISTRO)
@monitora("mpf")
def mpf(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> dict[int, float] | None:
    punteggi = dati.dati.punteggi
    frecce = dati.dati.idfrecce
    if frecce is None or dati.dati.frecce < 2:
        return None
    return {f: np.mean(punteggi[frecce == f]) for f in range(dati.dati.frecce)}


@registra("mcf", set(), "media delle coordinate per ogni freccia", REGISTRO)
@monitora("mcf")
def mcf(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> dict[int, tuple[float, float]] | None:
    coordinate = dati.dati.xy
    frecce = dati.dati.idfrecce
    if frecce is None or dati.dati.frecce < 2:
        return None
    return {f: np.mean(coordinate[frecce == f, :], axis=1) for f in range(dati.dati.frecce)}


@registra("maf", set(), "angoli medi per ogni freccia", REGISTRO)
@monitora("maf")
def maf(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> dict[int, float] | None:
    angoli = dati.dati.angoli
    frecce = dati.dati.idfrecce
    if frecce is None or dati.dati.frecce < 2:
        return None
    return {f: st.circmean(angoli[frecce == f], high=math.pi, low=-math.pi) for f in range(dati.dati.frecce)}


@registra("vpf", set(), "varianza dei punteggi per ogni freccia", REGISTRO)
@monitora("vpf")
def vpf(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> dict[int, float] | None:
    punteggi = dati.dati.punteggi
    frecce = dati.dati.idfrecce
    if frecce is None or dati.dati.frecce < 2:
        return None
    return {f: np.var(punteggi[frecce == f], ddof=1) for f in range(dati.dati.frecce)}


@registra("vcf", set(), "varianze delle coordinate per ogni freccia", REGISTRO)
@monitora("vcf")
def vcf(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> dict[int, tuple[float, float]] | None:
    coordinate = dati.dati.xy
    frecce = dati.dati.idfrecce
    if frecce is None or dati.dati.frecce < 2:
        return None
    return {f: np.var(coordinate[frecce == f, :], axis=1, ddof=1) for f in range(dati.dati.frecce)}


@registra("vaf", set(), "lunghezze risultanti medie per ogni freccia", REGISTRO)
@monitora("vaf")
def vaf(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> dict[int, float] | None:
    angoli = dati.dati.angoli
    frecce = dati.dati.idfrecce
    if frecce is None or dati.dati.frecce < 2:
        return None
    return {f: varianzaangoli(angoli[frecce == f]) for f in range(dati.dati.frecce)}


def intervallocircolare(fittato: np.ndarray, confidenza: float) -> tuple[float, float]:
    mappato = np.sort((fittato + 2 * np.pi) % (2 * np.pi))
    m = np.floor(confidenza * len(mappato))
    larghezzaminima = 2 * np.pi
    iniziomigliore = 0
    finemigliore = 0
    for i in range(len(mappato)):
        j = int((i + m) % len(mappato))
        inizio = mappato[i]
        fine = mappato[j] if j > i else mappato[j] + 2 * np.pi
        larghezza = fine - inizio
        if larghezza < larghezzaminima:
            larghezzaminima = larghezza
            iniziomigliore = inizio
            finemigliore = fine
    if finemigliore < iniziomigliore:
        return iniziomigliore, finemigliore + 2 * np.pi
    else:
        return iniziomigliore, finemigliore
