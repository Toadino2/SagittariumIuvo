# Dovresti mantenere solo le funzioni strettamente grafiche
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import patches, colormaps
import math
from scipy.special import i1, i0
import scipy.stats as st
import colorsys
from manipolazione.appiattitori import appiattisci
from classi.dati import Sessione
matplotlib.use("Agg")


def intervallohotelling(intervallo: tuple) -> tuple | None:
    autovalori, autovettori, media, costante, _ = intervallo
    lunghezzeassi = np.sqrt(costante * autovalori)
    theta = np.linspace(0, 2 * np.pi, 100)
    ellisse = np.array([lunghezzeassi[0] * np.cos(theta), lunghezzeassi[1] * np.sin(theta)])
    ellisseruotato = autovettori @ ellisse
    return ellisseruotato[0] + media[0], ellisseruotato[1] + media[1]


def graficocluster(sessione: Sessione):
    allenamento = sessione.dati.dati.xy
    colori = sessione.libro.contenuto["cl"]
    figura, asse = plt.subplots()
    bersagliosugrafico(asse)
    asse.scatter(allenamento[:, 0], allenamento[:, 1], c=colori, cmap="viridis", edgecolors="k")
    asse.set_aspect("equal")
    plt.title("Grafico dei cluster")
    unici = np.unique(colori)
    mappa = plt.get_cmap("viridis")
    maniglie = [patches.Patch(color=mappa(etichetta / max(unici)), label=f"Cluster {etichetta}") for etichetta in unici]
    asse.legend(handles=maniglie, title="Cluster", loc="upper left", bbox_to_anchor=(1, 1))
    return figura


def graficointervallohotelling(sessione: Sessione):
    ellissex, ellissey = intervallohotelling(sessione.libro.contenuto["ih"].valore.intervallo.valore)
    indipendenza = sessione.impostazioni["cdih"]
    tagliamenouno = sessione.dati.dati.taglia-1
    regressione = sessione.libro.contenuto["rc"].valore
    figura, asse = plt.subplots()
    bersagliosugrafico(asse)
    asse.plot(ellissex, ellissey, color="purple")
    if indipendenza == "d":
        asse.add_patch(patches.Arrow(regressione[0][0]+regressione[0][1],
                                     regressione[1][0]+regressione[1][1],
                                     regressione[0][0]*tagliamenouno+regressione[0][1],
                                     regressione[1][0]*tagliamenouno+regressione[1][1],
                                     width=2.0, color="green"))
    asse.set_aspect("equal")
    titolo = "Intervallo di confidenza bivariato"
    if indipendenza == "s":
        titolo += " con bootstrap stazionario"
    elif indipendenza == "m":
        titolo += " con bootstrap a blocchi mobili"
    elif indipendenza == "d":
        titolo += " con detrending"
    plt.title(titolo)
    return figura


def graficohotellingbayesiano(sessione: Sessione):
    intervallo = sessione.libro.contenuto["i_hb"].valore.intervallo.valore
    figura, asse = plt.subplots()
    bersagliosugrafico(asse)
    asse.contour(intervallo[0], intervallo[1], intervallo[2], levels=[intervallo[3]], colors="purple")
    asse.set_aspect("equal")
    plt.title("Intervallo di credibilità bayesiano bivariato")
    return figura


def bersagliosugrafico(asse: plt.Axes):
    for raggio, colore in [(1.0, "white"), (0.8, "black"), (0.6, "blue"), (0.4, "red"), (0.2, "yellow")]:
        asse.add_patch(patches.Circle((0, 0), raggio, facecolor=colore, edgecolor="black"))


def graficodispersione(sessione: Sessione):
    ascisse = sessione.dati.dati.xy[:, 0].tolist()
    ordinate = sessione.dati.dati.xy[:, 1].tolist()
    figura, asse = plt.subplots()
    bersagliosugrafico(asse)
    n_gruppi = len(ascisse)
    cmap = colormaps["viridis"]
    norm = matplotlib.colors.Normalize(vmin=0, vmax=n_gruppi - 1)
    for i, (x, y) in enumerate(zip(ascisse, ordinate)):
        colore = cmap(norm(i))
        asse.scatter(x, y, color=colore, edgecolors="white", linewidths=0.5)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # required for older matplotlib versions
    cbar = plt.colorbar(sm, ax=asse)
    cbar.set_label("Indice volée")
    cbar.set_ticks(range(n_gruppi))
    cbar.set_ticklabels(range(n_gruppi))
    asse.set_aspect("equal")
    asse.set_xlim(-1.1, 1.1)
    asse.set_ylim(-1.1, 1.1)
    plt.title("Grafico di dispersione")
    return figura


def graficomediesubersaglio(sessione: Sessione):
    medie = sessione.libro.contenuto["mc"].valore
    medievolée = sessione.libro.contenuto["mcv"].valore
    mediefrecce = sessione.libro.contenuto["mcf"].valore
    regressione = sessione.libro.contenuto["rc"].valore
    frecce = sessione.dati.dati.taglia
    figura, asse = plt.subplots()
    bersagliosugrafico(asse)
    if medie is not None:
        asse.scatter(medie[0], medie[1], color="green")
    if medievolée is not None:
        medievoléeascisse = [media[0] for media in medievolée]
        medievoléeordinate = [media[1] for media in medievolée]
        asse.scatter(medievoléeascisse, medievoléeordinate, color="yellow", edgecolors="black")
        for x, y, etichetta in zip(medievoléeascisse, medievoléeordinate, [str(i + 1) for i in range(len(medievolée))]):
            asse.text(x, y, etichetta, color="grey")
    if mediefrecce is not None:
        mediefrecceascisse = list()
        mediefrecceordinate = list()
        etichette = list()
        for freccia in mediefrecce:
            mediefrecceascisse.append(mediefrecce[freccia][0])
            mediefrecceordinate.append(mediefrecce[freccia][1])
            etichette.append(str(int(freccia)+1))
        asse.scatter(mediefrecceascisse, mediefrecceordinate, color="red", edgecolors="white")
        for x, y, etichetta in zip(mediefrecceascisse, mediefrecceordinate, etichette):
            asse.text(x, y, etichetta, color="orange")
    if regressione is not None:
        asse.add_patch(patches.Arrow(float(regressione[0][1] + regressione[0][0]),
                                     float(regressione[1][1] + regressione[1][0]),
                                     float(regressione[0][0] * (frecce - 1)),
                                     float(regressione[1][0] * (frecce - 1)), color="purple", width=0.1))
    asse.set_aspect("equal")
    asse.set_xlim(-1.1, 1.1)
    asse.set_ylim(-1.1, 1.1)
    plt.title("Medie e regressione delle coordinate")
    plt.legend(handles=[patches.Patch(color="green",
                                      label="Media di tutto l'allenamento" if medie is not None else "Assente"),
                        patches.Patch(color="yellow",
                                      label="Media delle volée" if medievolée is not None else "Assente"),
                        patches.Patch(color="red",
                                      label="Media delle specifiche frecce" if mediefrecce is not None else "Assente"),
                        patches.Patch(color="purple",
                                      label="Regressione" if regressione is not None else "Assente")],
               loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    return figura


def graficopunteggi(sessione: Sessione):
    figura, asse = plt.subplots()
    punti = sessione.dati.dati.punteggi
    media = sessione.libro.contenuto["mp"].valore
    regressione = sessione.libro.contenuto["rp"].valore
    ordine = sessione.dati.dati.ordine
    if ordine:
        asse.plot([i for i in range(len(punti))], punti, color="purple")
        if media is not None:
            asse.axhline(media, color="green")
        if regressione is not None:
            asse.plot([i for i in range(len(punti))],
                      [regressione[1] + regressione[0] * i for i in range(len(punti))], color="blue")
    else:
        conteggi = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for punteggio in punti:
            conteggi[punteggio] += 1
        asse.bar([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], conteggi, color="purple")
        if media is not None:
            asse.axvline(media, color="green")
    plt.title("Grafico dei punteggi")
    plt.legend(handles=[patches.Patch(color="purple", label="Punteggi"),
                        patches.Patch(color="green", label="Media" if media is not None else "Assente"),
                        patches.Patch(color="blue",
                                      label="Regressione" if ordine and regressione is not None else "Assente")])
    return figura


def graficovolée(sessione: Sessione):
    medievolée = sessione.libro.contenuto["mpv"].valore
    ordine = sessione.dati.dati.ordine
    regressione = sessione.libro.contenuto["rp"].valore
    mediefrecce = sessione.libro.contenuto["mpf"].valore
    figura, asse = plt.subplots()
    if medievolée is not None:
        asse.plot([i for i in range(len(medievolée))], medievolée, color="yellow")
    if not ordine and regressione is not None:
        asse.plot([i for i in range(len(medievolée))],
                  [regressione[1] + regressione[0] * i for i in range(len(medievolée))],
                  color="blue")
    if mediefrecce is not None:
        for freccia in mediefrecce:
            asse.axhline(mediefrecce[freccia], color="red")
            asse.text(1 + freccia, mediefrecce[freccia], str(1 + freccia))
    plt.title("Grafico delle volée")
    plt.legend(handles=[patches.Patch(color="yellow",
                                      label="Volée" if medievolée is not None else "Assente"),
                        patches.Patch(color="blue",
                                      label="Regressione" if not ordine and regressione is not None else "Assente"),
                        patches.Patch(color="red",
                                      label="Frecce specifiche" if mediefrecce is not None else "Assente")])
    return figura


def graficoangoli(sessione: Sessione):
    angolisessione = sessione.dati.dati.angoli
    angolomedio = sessione.libro.contenuto["ma"].valore
    medievolée = sessione.libro.contenuto["mav"].valore
    mediefrecce = sessione.libro.contenuto["maf"].valore
    figura, asse = plt.subplots(subplot_kw={"projection": "polar"})
    intervalli = np.linspace(0, 2 * math.pi, 12)
    istogramma, estremi = np.histogram(np.mod(angolisessione, 2*np.pi), intervalli)
    centri = (estremi[:-1] + estremi[1:]) / 2
    asse.bar(centri, istogramma, color="purple")
    altezzamedie = np.max(istogramma)
    if angolomedio is not None:
        asse.bar([angolomedio % (2*np.pi)], [altezzamedie], width=0.05, color="green")
    if medievolée is not None:
        for media in range(len(medievolée)):
            asse.bar([medievolée[media] % (2*np.pi)], [altezzamedie], width=0.05, color="yellow")
            asse.text(medievolée[media] % (2*np.pi), altezzamedie, str(media+1), color="grey")
    if mediefrecce is not None:
        for media in mediefrecce:
            asse.bar([mediefrecce[media] % (2*np.pi)], [altezzamedie], width=0.05, color="red")
            asse.text(mediefrecce[media] % (2*np.pi), altezzamedie, str(media+1), color="orange")
    asse.set_aspect("equal")
    plt.title("Grafico degli angoli")
    plt.legend(handles=[patches.Patch(color="purple", label="Frequenze"),
                        patches.Patch(color="green", label="Media complessiva" if angolomedio is not None else "Assente"),
                        patches.Patch(color="yellow", label="Medie delle volée" if medievolée is not None else "Assente"),
                        patches.Patch(color="red", label="Medie delle frecce" if mediefrecce is not None else "Assente")])
    return figura


def graficiautocorrelazioni(sessione: Sessione):
    autocorrelazionipunteggi = sessione.libro.contenuto["ap"].valore
    autocorrelazioniirp = sessione.libro.contenuto["airp"].valore
    autocorrelazioniascisse, autocorrelazioniordinate = sessione.libro.contenuto["ac"].valore
    autocorrelazioniangolipositive, autocorrelazioniangolinegative = sessione.libro.contenuto["aa"].valore
    figura, asse = plt.subplots(2, 3)
    if autocorrelazionipunteggi is not None:
        asse[0, 0].bar([i for i in range(len(autocorrelazionipunteggi))], autocorrelazionipunteggi, color="black")
        asse.title("Autocorrelazioni dei punteggi")
    if autocorrelazioniirp is not None:
        asse[0, 1].bar([i for i in range(len(autocorrelazioniirp))], autocorrelazioniirp, color="black")
        asse.title("Autocorrelazioni degli IRP")
    if autocorrelazioniascisse is not None:
        asse[1, 0].bar([i for i in range(len(autocorrelazioniascisse))], autocorrelazioniascisse, color="black")
        asse.title("Autocorrelazioni delle ascisse")
    if autocorrelazioniordinate is not None:
        asse[1, 1].bar([i for i in range(len(autocorrelazioniordinate))], autocorrelazioniordinate, color="black")
        asse.title("Autocorrelazioni delle ordinate")
    if autocorrelazioniangolipositive is not None:
        asse[2, 0].bar([i for i in range(len(autocorrelazioniangolipositive))], autocorrelazioniangolipositive, color="black")
        asse.title("Autocorrelazioni degli angoli (positive)")
    if autocorrelazioniangolinegative is not None:
        asse[2, 1].bar([i for i in range(len(autocorrelazioniangolinegative))], autocorrelazioniangolinegative, color="black")
        asse.title("Autocorrelazioni degli angoli (negative)")
    return figura


def graficomedianageometrica(sessione: Sessione):
    medianageom = sessione.libro.contenuto["mg"].valore
    figura, asse = plt.subplots()
    bersagliosugrafico(asse)
    asse.scatter(medianageom[0], medianageom[1], color="green")
    asse.set_aspect("equal")
    plt.title("Mediana geometrica")
    return figura


def graficointervallicluster(sessione: Sessione):
    intervalli = sessione.libro.contenuto["ihc"].valore
    figura, asse = plt.subplots()
    bersagliosugrafico(asse)
    for numero, intervallo in enumerate(intervalli):
        if intervallo.intervallo.valore is not None:
            ellissi = intervallohotelling(intervallo)
            asse.plot(ellissi[0], ellissi[1], color="purple")
            asse.text(ellissi[0][0], ellissi[1][0], str(numero), color="green")
    asse.set_aspect("equal")
    plt.title("Intervalli di confidenza bivariati per ogni cluster di frecce")
    return figura


def graficointervallonorme(sessione: Sessione):
    intervallo = sessione.libro.contenuto["iirp"].valore
    figura, asse = plt.subplots()
    bersagliosugrafico(asse)
    asse.plot(intervallo[0] / 10 * np.cos(np.linspace(0, 2 * np.pi, 200)),
              intervallo[0] / 10 * np.sin(np.linspace(0, 2 * np.pi, 200)), color="purple")
    asse.plot(intervallo[1] / 10 * np.cos(np.linspace(0, 2 * np.pi, 200)),
              intervallo[1] / 10 * np.sin(np.linspace(0, 2 * np.pi, 200)), color="purple")
    asse.set_aspect("equal")
    plt.title("Grafico dell'intervallo di confidenza delle norme")
    return figura


def graficointervallivarianze(sessione: Sessione):
    medie = sessione.libro.contenuto["mc"].valore
    intervalli = sessione.libro.contenuto["_iv"].valore[0].intervallo.valore, sessione.libro.contenuto["_iv"].valore[1].intervallo.valore
    figura, asse = plt.subplots()
    bersagliosugrafico(asse)
    asse.plot(medie[0] + intervalli[0][0]**0.5 * np.cos(np.linspace(0, 2 * np.pi, 200)),
              medie[1] + intervalli[1][0]**0.5 * np.sin(np.linspace(0, 2 * np.pi, 200)), color="green")
    asse.plot(medie[0] + intervalli[0][1]**0.5 * np.cos(np.linspace(0, 2 * np.pi, 200)),
              medie[1] + intervalli[1][1]**0.5 * np.sin(np.linspace(0, 2 * np.pi, 200)), color="purple")
    asse.set_aspect("equal")
    plt.title("Grafico degli intervalli di confidenza delle varianze")
    return figura


def graficointervallovarianzebayesiano(sessione: Sessione):
    medie = sessione.libro.contenuto["mc"].valore
    intervalli = sessione.libro.contenuto["ivb"].valore[0].intervallo.valore, sessione.libro.contenuto["ivb"].valore[1].intervallo.valore
    figura, asse = plt.subplots()
    bersagliosugrafico(asse)
    asse.plot(medie[0] + intervalli[0][0]**0.5 * np.cos(np.linspace(0, 2 * np.pi, 200)),
              medie[1] + intervalli[1][0]**0.5 * np.sin(np.linspace(0, 2 * np.pi, 200)), color="green")
    asse.plot(medie[0] + intervalli[0][1]**0.5 * np.cos(np.linspace(0, 2 * np.pi, 200)),
              medie[1] + intervalli[1][1]**0.5 * np.sin(np.linspace(0, 2 * np.pi, 200)), color="purple")
    asse.set_aspect("equal")
    plt.title("Grafico degli intervalli di credibilità bayesiani delle varianze")
    return figura


def graficointervalloangolomedio(sessione: Sessione):
    intervallo = sessione.libro.contenuto["iam"].valore.intervallo.valore
    a = (intervallo[0] + 2 * np.pi) % (2 * np.pi)
    b = (intervallo[1] + 2 * np.pi) % (2 * np.pi)
    if b >= a:
        arco = b - a
        inizio = a
    else:
        arco = (2 * np.pi - a) + b
        inizio = a
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.bar([inizio], [1.0], width=[arco], color='green', alpha=0.5, align='edge')
    ax.set_aspect('equal')
    plt.title("Intervallo di confidenza per l'angolo medio")
    return fig


def graficointervallokappa(sessione: Sessione):
    media = sessione.libro.contenuto["ma"].valore
    intervallo = sessione.libro.contenuto["ik"].valore.intervallo.valore
    figura, asse = plt.subplots(subplot_kw={"projection": "polar"})
    sigma1 = (-2 * math.log(i1(intervallo[1]) / i0(intervallo[1]))) ** 0.5
    sigma2 = (-2 * math.log(i1(intervallo[0]) / i0(intervallo[0]))) ** 0.5
    asse.bar([float(media)], [2.0], color="blue", width=[float(4 * sigma2)])
    asse.bar([float(media)], [1.0], color="red", width=[float(4 * sigma1)])
    asse.set_aspect("equal")
    plt.title("Grafico dell'intervallo di confidenza del parametro di concentrazione degli angoli")
    return figura


def graficoangolomediobayesiano(sessione: Sessione):
    intervallo = sessione.libro.contenuto["iamb"].valore.intervallo.valore
    a = (intervallo[0] + 2 * np.pi) % (2 * np.pi)
    b = (intervallo[1] + 2 * np.pi) % (2 * np.pi)
    if b >= a:
        arco = b - a
        inizio = a
    else:
        arco = (2 * np.pi - a) + b
        inizio = a
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.bar([inizio], [1.0], width=[arco], color='red', alpha=0.5, align='edge')
    ax.set_aspect('equal')
    plt.title("Intervallo di credibilità per l'angolo medio")
    return fig


def avvolgi(theta: float) -> float:
    return (theta + np.pi) % (2 * np.pi) - np.pi


def proteggisigma(kappa: float) -> float:
    tiny = 1e-8
    if not np.isfinite(kappa) or kappa <= tiny:
        return 1e-6
    try:
        frazione = i1(kappa) / i0(kappa)
    except Exception:
        return 1e-6
    frazione = min(max(frazione, tiny), 1 - tiny)
    val = -2.0 * math.log(frazione)
    if val <= 0:
        return 1e-6
    sigma = math.sqrt(val)
    if sigma > 10 * np.pi:
        sigma = 10 * np.pi
    return sigma


def _raw_interval_to_wrapped_segments(xb_raw: float, xa_raw: float):
    segments = []
    k_start = math.floor((xb_raw + np.pi) / (2 * np.pi))
    k_end = math.floor((xa_raw + np.pi) / (2 * np.pi))
    for k in range(k_start, k_end + 1):
        block_start = k * 2 * np.pi - np.pi
        block_end = (k + 1) * 2 * np.pi - np.pi
        seg_start = max(xb_raw, block_start)
        seg_end = min(xa_raw, block_end)
        if seg_end > seg_start:
            segments.append((seg_start, seg_end))
    return segments


def riavvolgi(ax, angolo, sigma, proporzione, colore, alpha):
    xb_raw = angolo - 2.0 * sigma
    xa_raw = angolo + 2.0 * sigma
    if xa_raw < xb_raw:
        xa_raw += 2.0 * np.pi
    width_raw = xa_raw - xb_raw
    if width_raw >= 2.0 * np.pi:
        xb_raw = -np.pi
        xa_raw = np.pi
        width_raw = 2.0 * np.pi
    segments = _raw_interval_to_wrapped_segments(xb_raw, xa_raw)
    for seg_start, seg_end in segments:
        mid_raw = 0.5 * (seg_start + seg_end)
        width = seg_end - seg_start
        mid_wrapped = avvolgi(mid_raw)
        ax.bar([float(mid_wrapped)], [float(proporzione)], width=[float(width)], color=colore, alpha=alpha, edgecolor="none")


def schiarisci(colore, quanto):
    c = colorsys.rgb_to_hls(*colore[:3])
    return colorsys.hls_to_rgb(c[0], 1 - quanto * (1 - c[1]), c[2])


def graficomisturevonmises(sessione: Sessione):
    intervallimedie, intervallikappa, componenti, assegnazioni = sessione.libro.contenuto["vmv"].valore
    angoli = sessione.dati.dati.angoli
    figura, asse = plt.subplots(subplot_kw={"projection": "polar"})
    mappacolori = colormaps["tab20"]
    promemoria = [(componente[1], i) for i, componente in enumerate(componenti)]
    cose = sorted(
        zip(intervallimedie, intervallikappa, componenti),
        key=lambda x: x[2][1],
        reverse=True
    )
    mappa = np.array([i for _, i in sorted(promemoria, key=lambda x: x[0], reverse=True)])
    for numero, ([angolobasso, angoloalto],
                 (kappabasso, kappaalto),
                 (angolo, proporzione)) in enumerate(cose):
        colorechiaro = mappacolori(numero / len(componenti))
        sigma2 = proteggisigma(kappabasso)
        if sigma2 > 0:
            riavvolgi(asse, angolo, sigma2, proporzione, schiarisci(colorechiaro, 0.1), alpha=0.9)
        sigma1 = proteggisigma(kappaalto)
        if sigma1 > 0:
            riavvolgi(asse, angolo, sigma1, proporzione, schiarisci(colorechiaro, 0.4), alpha=0.9)
        xb_raw, xa_raw = angolobasso, angoloalto
        if xa_raw < xb_raw:
            xa_raw += 2.0 * np.pi
        if xa_raw - xb_raw >= 2.0 * np.pi:
            xb_raw = -np.pi
            xa_raw = np.pi
        for seg_start, seg_end in _raw_interval_to_wrapped_segments(xb_raw, xa_raw):
            mid = avvolgi(0.5 * (seg_start + seg_end))
            width = seg_end - seg_start
            asse.bar([float(mid)], [float(proporzione)], width=[float(width)],
                     color=schiarisci(colorechiaro, 0.7), alpha=0.9, edgecolor="none")
        asse.bar([float(avvolgi(angolo))], [float(proporzione)], width=0.05, color=colorechiaro, edgecolor="black")
    raggi = np.ones_like(angoli) * (max(cose, key=lambda x: x[2][1])[2][1] + 0.02)
    nuoveassegnazioni = mappa[np.array(assegnazioni).astype(int)]
    colori = mappacolori(np.array([int(cluster)/len(componenti) for cluster in nuoveassegnazioni]))
    asse.scatter(np.array(angoli), raggi, c=colori, alpha=0.9, edgecolor="black")
    asse.set_aspect("equal")
    plt.title("Grafico dei diversi gruppi di angoli trovati")
    return figura
