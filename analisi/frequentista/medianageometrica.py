from classi.dati import registra, monitora, REGISTRO, Libro, SessioneGrezza
import numpy as np
import math


@registra("mg", {"mc"}, "mediana geometrica delle coordinate", REGISTRO)
@monitora("mg")
def mg(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> np.ndarray | None:
    if dati.dati.taglia < 2:
        return None
    media = libro.contenuto["mc"].valore
    guess = libro.contenuto["mc"].valore
    for _ in range(impostazioni["img"]):
        guessprecedente = guess
        distanze = np.linalg.norm(dati.dati.xy-media, axis=1)
        distanze[distanze == 0] = 1e-10
        pesi = 1/distanze
        pesi /= np.sum(pesi)
        guess = np.sum(dati.dati.xy*pesi[:, np.newaxis], axis=0)
        if np.sum(np.abs(guessprecedente-guess)) < impostazioni["smg"]:
            break
    return guess


@registra("mi", {"mg"}, "raccomandazioni per lo spostamento del mirino", REGISTRO)
@monitora("mi")
def mi(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> tuple[float, float]:
    coccamirino = impostazioni["cmmi"]
    mediane = libro.contenuto["mg"].valore
    gamma = impostazioni["gmi"]
    raggiobersaglio = impostazioni["rbmi"]
    p = math.pi
    seno = math.sin
    coseno = math.cos
    arcoseno = math.asin
    cm = coccamirino
    a = abs(mediane[0])*raggiobersaglio
    o = abs(mediane[1])*raggiobersaglio
    g1 = math.radians(gamma[0])
    g2 = math.radians(gamma[1])
    d1 = float(dati.metadati.distanza)
    d2 = float(impostazioni["dvmi"])
    i1 = (d1**2+a**2-2*d1*a*math.cos(g1))**0.5
    i2 = (d2**2+o**2-2*d2*o*math.cos(g2))**0.5
    s1 = (cm*a*seno(g1))/((seno(p-g1)*coseno(arcoseno(a/i1))-a*seno(g1)*coseno(p-g1)/i1)*i1) if mediane[0] != 0 else 0
    s2 = (cm*o*seno(g2))/((seno(p-g2)*coseno(arcoseno(o/i2))-o*seno(g2)*coseno(p-g2)/i2)*i2) if mediane[1] != 0 else 0
    if mediane[0] < 0:
        s1 *= -1
    if mediane[1] < 0:
        s2 *= -1
    return s1, s2
