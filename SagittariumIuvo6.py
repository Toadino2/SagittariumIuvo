import pickle
'''
with open("Impostazioni.txt", "rb") as fileimpostazioni:
    if not pickle.load(fileimpostazioni)["consoledikivy"]:
        from os import environ
        environ["KIVY_NO_CONSOLELOG"] = "1"
'''
from os import environ
environ["KIVY_NO_CONSOLELOG"] = "1"
import matplotlib
matplotlib.use("Agg")

import cmdstanpy
import cv2
import datetime
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Ellipse, InstructionGroup, Line, Rectangle
from kivy.graphics.texture import Texture
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget
import math
from matplotlib import colormaps
from matplotlib import patches
import matplotlib.pyplot as plt
import miomodulo
import numpy as np
from os import makedirs, remove, rename
from os.path import exists as esiste
import pandas as pd
from plyer import filechooser
from pmdarima import auto_arima
import pycircstat
from requests.exceptions import ConnectionError
from roboflow import Roboflow
from scipy.special import i0, i1, iv
import scipy.stats as st
from statsmodels.regression.linear_model import GLSAR
from statsmodels.tsa.holtwinters import Holt
from time import time


class ArcoOlimpico:
    def __init__(self, dizionario: dict):
        self.nomesetup = dizionario["Nomesetup"]
        self.nomemirino = dizionario["Nomemirino"]
        self.modelloarco = dizionario["Modelloarco"]
        self.libbraggio = dizionario["Libbraggio"]
        self.pivotcorda = dizionario["Pivotcorda"]
        self.allungo = dizionario["Allungo"]
        self.tipocorda = dizionario["Tipocorda"]
        self.estensione = dizionario["Estensione"]
        self.puntoincocco = dizionario["Puntoincocco"]
        self.tiller = dizionario["Tiller"]
        self.occhiopin = dizionario["Occhiopin"]
        self.occhiocorda = dizionario["Occhiocorda"]
        self.molla = dizionario["Molla"]
        self.centershotbottone = dizionario["Centershotbottone"]
        self.settaggiobottone = dizionario["Settaggiobottone"]
        self.note = dizionario["Note"]


class ArcoNudo:
    def __init__(self, dizionario: dict):
        self.nomesetup = dizionario["Nomesetup"]
        self.modelloarco = dizionario["Modelloarco"]
        self.libbraggio = dizionario["Libbraggio"]
        self.pivotcorda = dizionario["Pivotcorda"]
        self.allungo = dizionario["Allungo"]
        self.tipocorda = dizionario["Tipocorda"]
        self.puntoincocco = dizionario["Puntoincocco"]
        self.tiller = dizionario["Tiller"]
        self.occhiopin = dizionario["Occhiopin"]
        self.occhiocorda = dizionario["Occhiocorda"]
        self.molla = dizionario["Molla"]
        self.centershotbottone = dizionario["Centershotbottone"]
        self.settaggiobottone = dizionario["Settaggiobottone"]
        self.note = dizionario["Note"]


class ArcoCompound:
    def __init__(self, dizionario: dict):
        self.nomesetup = dizionario["Nomesetup"]
        self.nomemirino = dizionario["Nomemirino"]
        self.modelloarco = dizionario["Modelloarco"]
        self.libbraggio = dizionario["Libbraggio"]
        self.pivotcorda = dizionario["Pivotcorda"]
        self.allungo = dizionario["Allungo"]
        self.tipocorda = dizionario["Tipocorda"]
        self.estensione = dizionario["Estensione"]
        self.puntoincocco = dizionario["Puntoincocco"]
        self.tiller = dizionario["Tiller"]
        self.visettediottra = dizionario["Visettediottra"]
        self.caricovalle = dizionario["Caricovalle"]
        self.release = dizionario["Release"]
        self.sincronizzazione = dizionario["Sincronizzazione"]
        self.orizzontalerest = dizionario["Orizzontalerest"]
        self.verticalerest = dizionario["Verticalerest"]
        self.rigidezzarest = dizionario["Rigidezzarest"]
        self.tiporest = dizionario["Tiporest"]
        self.ingrandimentodiottra = dizionario["Ingrandimentodiottra"]
        self.visettecocca = dizionario["Visettecocca"]
        self.note = dizionario["Note"]


def creafile(file: str, oggetto):
    with open(file, "xb") as f:
        pickle.dump(oggetto, f)


def leggifile(file: str):
    with open(file, "rb") as f:
        oggetto = pickle.load(f)
    return oggetto


def scrivifile(oggetto, file: str):
    with open(file, "wb") as f:
        pickle.dump(oggetto, f)


def mettibottoneindietro(funzione):
    Window.unbind(on_back_button=None)
    Window.bind(on_back_button=funzione)


def aggiungioggetti(aggiungendo: RelativeLayout | GridLayout | ScrollView, oggetti: tuple | list):
    for oggetto in oggetti:
        aggiungendo.add_widget(oggetto)


def rimuovioggetti(rimuovendo: RelativeLayout | GridLayout | ScrollView, oggetti: tuple | list):
    for oggetto in oggetti:
        rimuovendo.remove_widget(oggetto)


def taglia(allenamento: np.ndarray | list, identificazione: bool) -> int:
    if isinstance(allenamento, np.ndarray):
        return int(allenamento.size / 3) if identificazione else int(allenamento.size / 2)
    else:
        return sum([len(volée) for volée in allenamento])


def punteggi(allenamento: np.ndarray | list, identificazione: bool, ordine: bool):
    def calcolapunteggio(freccia2):
        if (ipotenusa := math.hypot(freccia2[0], freccia2[1])) < 1:
            return 10 - math.floor(10 * ipotenusa)
        else:
            return 0

    if isinstance(allenamento, np.ndarray):
        if identificazione:
            listapunteggi = list()
            listapunteggiidentificati = list()
            for volée in allenamento:
                listavolée = list()
                listavoléeidentificata = list()
                for freccia in volée:
                    punteggio = calcolapunteggio(freccia)
                    listavolée.append(punteggio)
                    listavoléeidentificata.append([punteggio, freccia[2]])
                listapunteggi.append(listavolée)
                listapunteggiidentificati.append(listavoléeidentificata)
            return np.array(listapunteggi), listapunteggiidentificati
        else:
            return np.array([[calcolapunteggio(freccia) for freccia in volée]
                             for volée in allenamento])
    else:
        if identificazione:
            listapunteggi = list()
            listapunteggiidentificati = list()
            for volée in allenamento:
                if ordine:
                    listavolée = list()
                    listavoléeidentificata = list()
                    for freccia in volée:
                        punteggio = calcolapunteggio(freccia)
                        listavolée.append(punteggio)
                        listavoléeidentificata.append((punteggio, freccia[2]))
                else:
                    listavolée = list()
                    listavoléeidentificata = list()
                    for freccia in volée:
                        punteggio = calcolapunteggio(freccia)
                        listavolée.append(punteggio)
                        listavoléeidentificata.append((punteggio, freccia[2]))
                listapunteggi.append(listavolée)
                listapunteggiidentificati.append(listavoléeidentificata)
            return listapunteggi, listapunteggiidentificati
        else:
            if ordine:
                return [[calcolapunteggio(freccia) for freccia in volée] for volée in allenamento]
            else:
                return [[calcolapunteggio(freccia) for freccia in volée] for volée in allenamento]


def norme(allenamento: np.ndarray | list, identificazione: bool, ordine: bool):
    def calcolanorma(coppia):
        return 10.0 * (1.0 - math.hypot(coppia[0], coppia[1]))

    if isinstance(allenamento, np.ndarray):
        listapunteggi = list()
        if identificazione:
            listapunteggiidentificati = list()
            for volée in allenamento:
                listavolée = list()
                listavoléeidentificata = list()
                for freccia in volée:
                    punteggio = calcolanorma(freccia)
                    listavolée.append(punteggio)
                    listavoléeidentificata.append([punteggio, freccia[2]])
                listapunteggi.append(listavolée)
                listapunteggiidentificati.append(listavoléeidentificata)
            return np.array(listapunteggi), listapunteggiidentificati
        else:
            return np.array([[calcolanorma(freccia) for freccia in volée] for volée in allenamento])
    else:
        listapunteggi = list()
        if identificazione:
            listapunteggiidentificati = list()
            for volée in allenamento:
                if ordine:
                    listavolée = list()
                    listavoléeidentificata = list()
                    for freccia in volée:
                        punteggio = calcolanorma(freccia)
                        listavolée.append(punteggio)
                        listavoléeidentificata.append((punteggio, freccia[2]))
                else:
                    listavolée = list()
                    listavoléeidentificata = list()
                    for freccia in volée:
                        punteggio = calcolanorma(freccia)
                        listavolée.append(punteggio)
                        listavoléeidentificata.append((punteggio, freccia[2]))
                listapunteggi.append(listavolée)
                listapunteggiidentificati.append(listavoléeidentificata)
            return listapunteggi, listapunteggiidentificati
        else:
            if ordine:
                return [[calcolanorma(freccia) for freccia in volée] for volée in allenamento]
            else:
                return [[calcolanorma(freccia) for freccia in volée] for volée in allenamento]


# noinspection PyUnboundLocalVariable
def coordinate(allenamento: np.ndarray | list, identificazione: bool, ordine: bool):
    if isinstance(allenamento, np.ndarray):
        if identificazione:
            ascisseidentificate = list()
            ordinateidentificate = list()
            for riga in allenamento:
                ascisseidentificate.append([(ascissa, numero) for ascissa, numero in zip(riga[:, 0], riga[:, 2])])
                ordinateidentificate.append([(ordinata, numero) for ordinata, numero in zip(riga[:, 1], riga[:, 2])])
            return allenamento[:, :, 0], allenamento[:, :, 1], ascisseidentificate, ordinateidentificate
        else:
            return allenamento[:, :, 0], allenamento[:, :, 1]
    else:
        ascisse = list()
        ordinate = list()
        if identificazione:
            ascisseidentificate = list()
            ordinateidentificate = list()
        for riga in allenamento:
            if ordine:
                ascisse.append([freccia[0] for freccia in riga])
                ordinate.append([freccia[1] for freccia in riga])
                if identificazione:
                    ascisseidentificate.append([(freccia[0], freccia[2]) for freccia in riga])
                    ordinateidentificate.append([(freccia[1], freccia[2]) for freccia in riga])
            else:
                ascisse.append([freccia[0] for freccia in riga])
                ordinate.append([freccia[1] for freccia in riga])
                if identificazione:
                    ascisseidentificate.append([(freccia[0], freccia[2]) for freccia in riga])
                    ordinateidentificate.append([(freccia[1], freccia[2]) for freccia in riga])
        if identificazione:
            return ascisse, ordinate, ascisseidentificate, ordinateidentificate
        else:
            return ascisse, ordinate


def correlazionefrecce(ascisse, ordinate, numerofrecce) -> dict:
    frecce = [[] for _ in range(numerofrecce)]
    for voléeascisse, voléeordinate in zip(ascisse, ordinate):
        for ascissafreccia, ordinatafreccia in zip(voléeascisse, voléeordinate):
            if ascissafreccia[1] != ordinatafreccia[1]:
                raise IndexError("Gli indici delle frecce nel dataset sono messi male")
            else:
                frecce[ascissafreccia[1]].append([ascissafreccia[0], ordinatafreccia[0]])
    if any(len(lista) == 0 for lista in frecce):
        raise IndexError("C'è una freccia segnata a cui non corrisponde nessun'osservazione")
    return {freccia: np.corrcoef(np.array(lista).T)[0, 1] for freccia, lista in enumerate(frecce)}


def angoli(allenamento: np.ndarray | list, identificazione: bool, ordine: bool):
    if isinstance(allenamento, np.ndarray):
        listaangoli = list()
        if identificazione:
            listaangoliidentificati = list()
            for volée in allenamento:
                listavolée = list()
                listavoléeidentificata = list()
                for freccia in volée:
                    angolo = math.atan2(freccia[1], freccia[0])
                    listavolée.append(angolo)
                    listavoléeidentificata.append((angolo, freccia[2]))
                listaangoli.append(listavolée)
                listaangoliidentificati.append(listavoléeidentificata)
            return np.array(listaangoli), listaangoliidentificati
        else:
            return np.array([[math.atan2(freccia[1], freccia[0]) for freccia in volée] for volée in allenamento])
    else:
        listaangoli = list()
        if identificazione:
            listaangoliidentificati = list()
            for volée in allenamento:
                if ordine:
                    listavolée = list()
                    listavoléeidentificata = list()
                    for freccia in volée:
                        angolo = math.atan2(freccia[1], freccia[0])
                        listavolée.append(angolo)
                        listavoléeidentificata.append((angolo, freccia[2]))
                else:
                    listavolée = list()
                    listavoléeidentificata = list()
                    for freccia in volée:
                        angolo = math.atan2(freccia[1], freccia[0])
                        listavolée.append(angolo)
                        listavoléeidentificata.append((angolo, freccia[2]))
                listaangoli.append(listavolée)
                listaangoliidentificati.append(listavoléeidentificata)
            return listaangoli, listaangoliidentificati
        else:
            if ordine:
                return [[math.atan2(freccia[1], freccia[0]) for freccia in volée] for volée in allenamento]
            else:
                return [[math.atan2(freccia[1], freccia[0]) for freccia in volée] for volée in allenamento]


def mediapunteggifrecce(listapunteggi: np.ndarray | list) -> dict[int, float]:
    medie = dict()
    taglie = dict()
    for volée in listapunteggi:
        for freccia in volée:
            if freccia[1] not in medie:
                medie[freccia[1]] = 0
                taglie[freccia[1]] = 0
    for volée in listapunteggi:
        for freccia in volée:
            medie[freccia[1]] += freccia[0]
            taglie[freccia[1]] += 1
    for lettera in medie:
        try:
            medie[lettera] /= taglie[lettera]
        except ZeroDivisionError:
            raise ValueError("C'è segnata una freccia che non corrisponde a nessun'osservazione")
    return medie


def mediacoordinatefrecce(ascisse: np.ndarray | list, ordinate: np.ndarray | list) -> dict[int, tuple[float, float]]:
    medieascisse = dict()
    medieordinate = dict()
    taglie = dict()
    for volée in ascisse:
        for freccia in volée:
            if freccia[1] not in medieascisse:
                medieascisse[freccia[1]] = 0
                medieordinate[freccia[1]] = 0
                taglie[freccia[1]] = 0
    for volée in ascisse:
        for freccia in volée:
            medieascisse[freccia[1]] += freccia[0]
            taglie[freccia[1]] += 1
    if any(taglietta == 0 for taglietta in taglie.items()):
        raise ValueError("C'è una freccia segnata a cui non corrisponde nessun'osservazione")
    for volée in ordinate:
        for freccia in volée:
            medieordinate[freccia[1]] += freccia[0]
    return {lettera: (medieascisse[lettera] / taglie[lettera],
                      medieordinate[lettera] / taglie[lettera]) for lettera in taglie}


def mediaangolifrecce(angoliidentificati: np.ndarray | list) -> dict:
    angolifrecce = dict()
    for volée in angoliidentificati:
        for angolo in volée:
            if angolo[1] not in angolifrecce:
                angolifrecce[angolo[1]] = list()
    for volée in angoliidentificati:
        for angolo in volée:
            angolifrecce[angolo[1]].append(angolo[0])
    return {etichetta: st.circmean(angolifrecce[etichetta],
                                   high=math.pi, low=-math.pi) for etichetta in angolifrecce}


def varianzapunteggifrecce(punteggiidentificati: np.ndarray | list) -> dict[int, float]:
    punteggifrecce = dict()
    for volée in punteggiidentificati:
        for freccia in volée:
            if freccia[1] not in punteggifrecce:
                punteggifrecce[freccia[1]] = list()
    for volée in punteggiidentificati:
        for freccia in volée:
            punteggifrecce[freccia[1]].append(freccia[0])
    return {etichetta: np.var(punteggifrecce[etichetta], ddof=1) if len(punteggifrecce[etichetta]) > 1 else 0
            for etichetta in punteggifrecce}


def varianzacoordinatefrecce(ascisseidentificate: np.ndarray | list,
                             ordinateidentificate: np.ndarray | list) -> dict[int, tuple[float, float]]:
    ascisse = dict()
    ordinate = dict()
    for volée in ascisseidentificate:
        for freccia in volée:
            if freccia[1] not in ascisse:
                ascisse[freccia[1]] = list()
                ordinate[freccia[1]] = list()
    for volée in ascisseidentificate:
        for freccia in volée:
            ascisse[freccia[1]].append(freccia[0])
    for volée in ordinateidentificate:
        for freccia in volée:
            ordinate[freccia[1]].append(freccia[0])
    return {etichetta: (np.var(ascisse[etichetta], ddof=1) if len(ascisse[etichetta]) > 1 else 0,
                        np.var(ordinate[etichetta], ddof=1) if len(ordinate[etichetta]) > 1 else 0)
            for etichetta in ascisse}


def varianzaangoli(dati: np.ndarray | list) -> float:
    if len(appiattisci(dati)) == 0:
        raise IndexError("Non ci sono dati su cui calcolare \\bar{R}")
    else:
        return (np.mean(np.cos(appiattisci(dati))) ** 2 + np.mean(np.sin(appiattisci(dati))) ** 2) ** 0.5


def varianzaangolifrecce(angoliidentificati: np.ndarray | list) -> dict:
    angolifrecce = dict()
    for volée in angoliidentificati:
        for freccia in volée:
            if freccia[1] not in angolifrecce:
                angolifrecce[freccia[1]] = list()
    for volée in angoliidentificati:
        for freccia in volée:
            angolifrecce[freccia[1]].append(freccia[0])
    return {etichetta: varianzaangoli(angolifrecce[etichetta]) for etichetta in angolifrecce}


def autocorrelazioni(osservazioni: np.ndarray | list, varianza: float):
    autocovarianze = np.correlate(osservazioni - np.mean(osservazioni), osservazioni - np.mean(osservazioni),
                        mode="full")
    return autocovarianze[autocovarianze.size//2+1:]/autocovarianze[autocovarianze.size//2]


def testnormalità(allenamento: np.ndarray, frecce: int, alfa: float, test: str) -> bool:
    if test == "henze-zirkler":
        statistica, media, varianza = miomodulo.testhenzezirkler(allenamento[:, 0].tolist(),
                                                                 allenamento[:, 1].tolist(), frecce)
        return st.lognorm.sf(statistica, media, scale=varianza) > alfa
    else:
        return miomodulo.testmardia(allenamento[:, 0], allenamento[:, 1], frecce,
                                    st.chi2.ppf((1 - alfa) ** 0.5, df=4),
                                    st.norm.ppf(0.5+((1 - alfa)**0.5)/2))


def betamardia(frecce: int, distribuzione: str, iterazioni: int, gradit: int, alfa: float, tipotest: str,
               asimmetriaascisse: float, asimmetriaordinate: float, distanzacomponenti: float) -> float:
    if tipotest == "henze-zirkler":
        accettazioni = 0
        statistiche = miomodulo.betatesthenzezirkler(distribuzione,
                                                     frecce, iterazioni,
                                                     gradit, asimmetriaascisse,
                                                     asimmetriaordinate,
                                                     distanzacomponenti)
        for statistica in statistiche:
            if st.lognorm.sf(statistica[0], statistica[1], scale=statistica[2]) > alfa:
                accettazioni += 1
        return accettazioni / iterazioni
    else:
        return miomodulo.betatestmardia(iterazioni, frecce,
                                        distribuzione, asimmetriaascisse,
                                        asimmetriaordinate, distanzacomponenti,
                                        gradit, st.chi2.ppf((1 - alfa) ** 0.5, df=4),
                                        st.norm.ppf(0.5+((1 - alfa)**0.5)/2))


def alfaveromardia(frecce: int, iterazioni: int, alfa: float, tipotest: str) -> float:
    if tipotest == "henze-zirkler":
        accettazioni = 0
        statistiche = miomodulo.alfaverohenzezirkler(iterazioni, frecce)
        for statistica in statistiche:
            if any([elemento <= 0.0 for elemento in statistica]):
                print(f"\033[35mStatistiche di Henze-Zirkler brutte: {statistica}\033[0m")
            if st.lognorm.sf(statistica[0], statistica[1], scale=statistica[2]) > alfa:
                accettazioni += 1
        return 1 - accettazioni / iterazioni
    else:
        return miomodulo.alfaveromardia(iterazioni, frecce,
                                        st.chi2.ppf((1 - alfa) ** 0.5, df=4),
                                        st.norm.ppf(0.5 + ((1 - alfa) ** 0.5) / 2))


def graficocluster(allenamento: np.ndarray, colori: list[int], data: str):
    figura, asse = plt.subplots()
    bersagliosugrafico(asse)
    asse.scatter(allenamento[:, 0], allenamento[:, 1], c=colori, cmap="viridis", edgecolors="k")
    asse.set_aspect("equal")
    plt.title("Grafico dei cluster")
    unici = np.unique(colori)
    mappa = plt.get_cmap("viridis")
    maniglie = [patches.Patch(color=mappa(etichetta / max(unici)), label=f"Cluster {etichetta}") for etichetta in unici]
    asse.legend(handles=maniglie, title="Cluster", loc="upper left", bbox_to_anchor=(1, 1))
    plt.savefig(f"./Grafici del {data}/graficocluster.png")
    plt.close()


def suddividicampioni(etichette: list[int], allenamento: np.ndarray) -> list[np.ndarray]:
    sottocampioni = list()
    for numero in range(max(etichette) + 1):
        sottocampione = list()
        for unità in range(len(etichette)):
            if etichette[unità] == numero:
                sottocampione.append(allenamento[unità])
        sottocampioni.append(np.array(sottocampione))
    return sottocampioni


def testcluster(etichette: list[int], allenamento: np.ndarray, alfa: float, frecce: int) -> list[bool]:
    sottocampioni = suddividicampioni(etichette, allenamento)
    return [miomodulo.testhotelling(len(elemento[:, 0]), elemento[:, 0], elemento[:, 1], st.f.ppf(1 - alfa, 2, frecce - 2)) for elemento in sottocampioni]


def intervallohotelling(n: int, confidenza: float, varianza: np.ndarray, media: tuple, costante=None) -> tuple | None:
    if n <= 2:
        return None
    if costante is None:
        costante = ((2 * (n - 1) / (n * (n - 2))) * st.f.ppf(confidenza, 2, n - 2))
    autovalori, autovettori = np.linalg.eigh(varianza)
    lunghezzeassi = np.sqrt(costante * autovalori)
    theta = np.linspace(0, 2 * np.pi, 100)
    ellisse = np.array([lunghezzeassi[0] * np.cos(theta), lunghezzeassi[1] * np.sin(theta)])
    ellisseruotato = autovettori @ ellisse
    return ellisseruotato[0] + media[0], ellisseruotato[1] + media[1]


def graficointervallohotelling(ellissex: np.ndarray, ellissey: np.ndarray, data: str, indipendenza: str,
                               regressione: tuple|None=None, tagliamenouno: int|None=None):
    figura, asse = plt.subplots()
    bersagliosugrafico(asse)
    asse.plot(ellissex, ellissey, color="purple")
    if indipendenza == "detrending":
        asse.add_patch(patches.Arrow(regressione[0][0]+regressione[0][1],
                                     regressione[1][0]+regressione[1][1],
                                     regressione[0][0]*tagliamenouno+regressione[0][1],
                                     regressione[1][0]*tagliamenouno+regressione[1][1],
                                     width=2.0, color="green"))
    asse.set_aspect("equal")
    titolo = "Intervallo di confidenza bivariato"
    if indipendenza == "stazionario":
        titolo += " con bootstrap stazionario"
    elif indipendenza == "mobile":
        titolo += " con bootstrap a blocchi mobili"
    elif indipendenza == "detrending":
        titolo += " con detrending"
    plt.title(titolo)
    plt.savefig(f"./Grafici del {data}/graficointervallohotelling{indipendenza if indipendenza != 'regolare' else ''}.png")
    plt.close()


def intervallicluster(etichette: list[int], allenamento: np.ndarray, alfa: float) -> list:
    sottocampioni = suddividicampioni(etichette, allenamento)
    intervalli = list()
    for sottocampione in sottocampioni:
        try:
            intervalli.append(intervallohotelling(len(sottocampione), alfa, np.cov(sottocampione.T),
                                                  np.mean(sottocampione, axis=0)))
        except Exception as e:
            print(f"\033[35mSottocampione sbagliato: {sottocampione}\033[0m")
            intervalli.append(None)
    return intervalli


def parametripriori(noninformativa: bool, allenamenti: int, apiacere: bool, mu0: list[float],
                    lambda0: list[list[float]], a0: float, b0: float, eta: float):
    if noninformativa:
        return [0, 0], [[10, 0], [0, 10]], 0.3, 1, 1
    elif apiacere:
        return mu0, lambda0, a0, b0, eta
    else:
        listadate = leggifile("Date.txt")
        mediepriori = list()
        varianzepriori = list()
        correlazionipriori = list()
        for allenamento in range(allenamenti):
            allenamentopriore: Allenamento = leggifile(f"Sessione del {listadate[-allenamento-1]}")
            if allenamentopriore.mediacoordinate is None:
                mediepriori.append([0, 0])
            else:
                mediepriori.append(allenamentopriore.mediacoordinate)
            if allenamentopriore.varianzacoordinate is None:
                varianzepriori.append(1)
                varianzepriori.append(1)
            else:
                varianzepriori.append(allenamentopriore.varianzacoordinate[0])
                varianzepriori.append(allenamentopriore.varianzacoordinate[1])
            if allenamentopriore.correlazione is None:
                correlazionipriori.append(1)
            else:
                correlazionipriori.append(allenamentopriore.correlazione)
        mu0 = np.mean(np.array(mediepriori), axis=0)
        lambda0 = np.cov(np.array(mediepriori).T)
        v = np.mean(varianzepriori)
        s = np.var(varianzepriori)
        a0 = v ** 2 / s ** 2
        b0 = v / s ** 2
        if all([correlazione < 0.2 for correlazione in correlazionipriori]):
            eta = 2
        elif any([0.2 < correlazione < 0.7 for correlazione in correlazionipriori]):
            eta = 1
        else:
            eta = 0.5
        return mu0, lambda0, a0, b0, eta


def hotellingbayesiano(allenamento: np.ndarray, numerocatene: int, iterazioni: int, distanza: float, confidenza: float,
                       noninformativa: bool, allenamentipriori: int, gerarchica: bool, intervallopreciso: bool,
                       test: bool, intervallo: bool, apiacere: bool, muapiacere: list[float],
                       lambdaapiacere: list[list[float]], aapiacere: float | int,
                       bapiacere: float | int, etaapiacere: float | int, console: bool):
    if gerarchica and allenamentipriori > 0:
        numerositàpriori = list()
        osservazionipriori = list()
        indicipriori = list()
        date = leggifile("Date.txt")
        numero = 1
        for data in date[-allenamentipriori:]:
            allenamentopriore = leggifile(f"Sessione del {data}")
            numerositàpriori.append(allenamentopriore.taglia)
            for osservazione in allenamentopriore.depura():
                osservazionipriori.append(osservazione)
                indicipriori.append(numero)
            numero += 1
        modello = cmdstanpy.CmdStanModel(exe_file="Hotellingbayesianogerarchico.exe")
        fittato = modello.sample(data={"G": allenamentipriori, "N": numerositàpriori, "y_prior": osservazionipriori,
                                       "prior_group": indicipriori, "N_current": len(allenamento),
                                       "y": allenamento.tolist()},
                                 chains=numerocatene, iter_sampling=iterazioni, show_console=console).stan_variable("mu")
    else:
        mu0, lambda0, a0, b0, eta = parametripriori(noninformativa, allenamentipriori, apiacere, muapiacere,
                                                    lambdaapiacere, aapiacere, bapiacere, etaapiacere)
        modello = cmdstanpy.CmdStanModel(exe_file="Hotellingbayesiano.exe")
        fittato = modello.sample(data={"N": len(allenamento), "y": allenamento.tolist(), "mu0": mu0, "lambda0": lambda0,
                                       "a0": a0, "b0": b0, "eta": eta},
                                 chains=numerocatene, iter_sampling=iterazioni, show_console=console).stan_variable("mu")
    if test:
        maschera = (fittato[:, 0] ** 2 + fittato[:, 1] ** 2) <= distanza ** 2
        probabilitànulla = np.sum(maschera) / fittato.shape[0]
    else:
        probabilitànulla = None
    if intervallo:
        if intervallopreciso:
            kernel = st.gaussian_kde(fittato.T)
            grigliax = np.linspace(-2, 2, 1000)
            grigliay = np.linspace(-2, 2, 1000)
            x, y = np.meshgrid(grigliax, grigliay)
            posizioni = np.vstack([x.ravel(), y.ravel()])
            z = kernel(posizioni).reshape(x.shape)
            zordinata = np.sort(z.ravel())[::-1]
            cumulata = np.cumsum(zordinata)
            cumulata /= cumulata[-1]
            livello = zordinata[np.searchsorted(cumulata, confidenza)]
        else:
            mediafittato = np.mean(fittato, axis=0)
            covarianzefittato = np.cov(fittato, rowvar=False, bias=False)
            covarianzeinverse = np.linalg.inv(covarianzefittato)
            x, y = np.meshgrid(np.linspace(-2, 2, 1000), np.linspace(-2, 2, 1000))
            differenze = np.stack([x.ravel(), y.ravel()], axis=1) - mediafittato
            z = np.einsum("ij,jk,ik->i", differenze, covarianzeinverse, differenze).reshape(x.shape)
            livello = st.chi2.ppf(confidenza, df=2)
    else:
        x = None
        y = None
        z = None
        livello = None
    return probabilitànulla, (x, y, z, livello)


def graficohotellingbayesiano(intervallo: tuple, data: str):
    figura, asse = plt.subplots()
    bersagliosugrafico(asse)
    asse.contour(intervallo[0], intervallo[1], intervallo[2], levels=[intervallo[3]], colors="purple")
    asse.set_aspect("equal")
    plt.title("Intervallo di credibilità bayesiano bivariato")
    plt.savefig(f"./Grafici del {data}/graficohotellingbayesiano.png")
    plt.close()


def medianageometrica(iterazioni: int, media: np.ndarray, allenamento: np.ndarray, soglia: float):
    guess = media.copy()
    for _ in range(iterazioni):
        guessprecedente = guess
        distanze = np.linalg.norm(allenamento - media, axis=1)
        distanze[distanze == 0] = 1e-10
        pesi = 1 / distanze
        pesi /= np.sum(pesi)
        guess = np.sum(allenamento * pesi[:, np.newaxis], axis=0)
        if np.sum(np.abs(guessprecedente - guess)) < soglia:
            break
    return guess


def betatestnorme(frecce: int, alternativa: str, distanza: float, iterazioni: int, media: float, varianza: float,
                  alfa: float) -> float:
    if alternativa == "maggiore":
        return miomodulo.betatestnorme(frecce, True, False, iterazioni, media, distanza, varianza,
                                       st.t.ppf(1 - alfa, df=frecce-1))
    elif alternativa == "disuguale":
        return miomodulo.betatestnorme(frecce, False, True, iterazioni, media, distanza, varianza,
                                       st.t.ppf(1 - alfa / 2, df=frecce-1))
    else:
        return miomodulo.betatestnorme(frecce, False, False, iterazioni, media, distanza, varianza,
                                       st.t.ppf(alfa, df=frecce-1))


def parametrinorme(noninformativa: bool, allenamentipriori: int, apiacere: bool, alfaapiacere: float | int,
                   betaapiacere: float | int, alfa2apiacere: float | int, beta2apiacere: float | int) -> tuple:
    if noninformativa:
        return 1.0, 1.0, 4.0, 3.0
    elif apiacere:
        return alfaapiacere, betaapiacere, alfa2apiacere, beta2apiacere
    else:
        listadate = leggifile("Date.txt")[-allenamentipriori:]
        listanorme = [leggifile(f"Sessione del {data}").norme for data in listadate]
        theta = np.array([np.mean(appiattisci(elemento)) / 10 for elemento in listanorme])
        tau = np.array([100 / np.var(appiattisci(elemento)) for elemento in listanorme])
        mediatheta = np.mean(theta)
        varianzatheta = np.var(theta)
        mediatau = np.mean(tau)
        alfa = mediatheta * (mediatheta * (1 - mediatheta) / varianzatheta - 1)
        beta = (1 - mediatheta) * (mediatheta * (1 - mediatheta) / varianzatheta - 1)
        alfa2 = mediatau ** 2 / varianzatheta
        beta2 = mediatau / varianzatheta
        return alfa, beta, alfa2, beta2


def normebayesiane(allenamento: np.ndarray, catene: int, iterazioni: int, alternativa: str, media: float,
                   credibilità: float, rope: float, noninformativa: bool, allenamentipriori: int,
                   gerarchica: bool, apiacere: bool, alfaapiacere: float | int, betaapiacere: float | int,
                   alfa2apiacere: float | int, beta2apiacere: float | int, console: bool) -> tuple:
    if gerarchica:
        arraypriori = [leggifile(f"Sessione del {data}").norme for data in leggifile("Date.txt")[-allenamentipriori:]]
        modello = cmdstanpy.CmdStanModel(exe_file="Normebayesianegerarchiche.exe")
        fittato = modello.sample(data={"G": allenamentipriori, "N": [len(appiattisci(pregresso)) for pregresso in arraypriori],
                                       "y": appiattisci(appiattisci(arraypriori)),
                                       "group_id": appiattisci([[g + 1 for _ in range(len(appiattisci(arraypriori[g])))]
                                                    for g in range(len(arraypriori))]),
                                       "Ncorrente": len(allenamento),
                                       "ycorrente": allenamento.tolist() if isinstance(allenamento, np.ndarray) else allenamento},
                                 chains=catene, iter_sampling=iterazioni, show_console=console).stan_variable("mucorrente")
    else:
        alfa, beta, alfa2, beta2 = parametrinorme(noninformativa, allenamentipriori, apiacere, alfaapiacere,
                                                  betaapiacere, alfa2apiacere, beta2apiacere)
        modello = cmdstanpy.CmdStanModel(exe_file="Normebayesiane.exe")
        fittato = modello.sample(
            data={"N": len(allenamento), "y": allenamento.tolist() if isinstance(allenamento, np.ndarray) else allenamento,
                  "alpha": alfa, "beta": beta, "alpha2": alfa2,
                  "beta2": beta2}, chains=catene, iter_sampling=iterazioni, show_console=console).stan_variable("mu")
    if alternativa == "greater":
        return len(fittato[fittato <= media]) / len(fittato), [np.quantile(fittato, (1 - credibilità) / 2),
                                                               np.quantile(fittato, 1-(1-credibilità) / 2)]
    elif alternativa == "two-sided":
        return len(fittato[media - rope <= fittato <= media + rope]) / len(fittato), [
            np.quantile(fittato, (1 - credibilità) / 2), np.quantile(fittato, 1-(1-credibilità) / 2)]
    else:
        return len(fittato[fittato >= media]) / len(fittato), [np.quantile(fittato, (1 - credibilità) / 2),
                                                               np.quantile(fittato, 1-(1-credibilità) / 2)]


def testvarianze(alternativa: str, varianza: float, ipotesinulla: float, frecce: int):
    statistica = (frecce - 1) * varianza / ipotesinulla
    if alternativa == "maggiore":
        return 1 - st.chi2.cdf(statistica, df=frecce - 1)
    elif alternativa == "disuguale":
        return 2 * min([st.chi2.cdf(statistica, df=frecce - 1), 1 - st.chi2.cdf(statistica, df=frecce - 1)])
    else:
        return st.chi2.cdf(statistica, df=frecce - 1)


def betatestvarianze(iterazioni: int, alternativa: str, ipotesinulla: float, distanza: float,
                     frecce: int, alfa: float):
    if alternativa == "maggiore":
        return miomodulo.betatestvarianze(iterazioni, frecce, True, False, distanza,
                                          st.chi2.ppf(1 - alfa, df=frecce - 1),
                                          0, ipotesinulla)
    elif alternativa == "disuguale":
        return miomodulo.betatestvarianze(iterazioni, frecce, False, True, distanza,
                                          st.chi2.ppf(alfa / 2, df=frecce - 1),
                                          st.chi2.ppf(1 - alfa / 2, df=frecce - 1), ipotesinulla)
    else:
        return miomodulo.betatestvarianze(iterazioni, frecce, False, False, distanza, st.chi2.ppf(alfa, df=frecce - 1),
                                          0, ipotesinulla)


def varianzebayesiane(allenamento: np.ndarray, catene: int, iterazioni: int,
                      alternativaascisse: str,
                      alternativaordinate: str, ipotesinullaascisse: float,
                      ipotesinullaordinate: float, ropeascisse: float, ropeordinate: float,
                      noninformativa: bool,
                      allenamentipriori: int,
                      alfaascisse: float, alfaordinate: float, gerarchica: bool, apiacere: bool,
                      muapiacere, sigmaapiacere, aapiacere: float | int, bapiacere: float | int,
                      etaapiacere: float | int, console: bool) -> tuple:
    if gerarchica:
        sessionipriori = [leggifile(f"Sessione del {data}") for data in leggifile("Date.txt")[-allenamentipriori:]]
        lunghezzepriori = [allenamento.taglia for allenamento in sessionipriori]
        datipriori = [punto for sessione in sessionipriori for punto in sessione.depura()]
        modello = cmdstanpy.CmdStanModel(exe_file="Varianzebayesianegerarchiche.exe")
        campione = modello.sample(data={"G": allenamentipriori, "Npriori": lunghezzepriori, "datipriori": datipriori,
                                       "group_id": appiattisci([[g + 1 for _ in range(lunghezzepriori[g])]
                                                                for g in range(allenamentipriori)]),
                                       "Ncorrente": len(allenamento), "daticorrenti": allenamento},
                                 chains=catene, iter_sampling=iterazioni, show_console=console)
        fittato = np.square(campione.stan_variable("sigmacorrente"))
        correlazioni = campione.stan_variable("Lcorrente")[:, 0, 1]
    else:
        mu0, sigma0, a, b, eta = parametripriori(noninformativa, allenamentipriori, apiacere, muapiacere, sigmaapiacere,
                                                 aapiacere, bapiacere, etaapiacere)
        modello = cmdstanpy.CmdStanModel(exe_file="Varianzebayesiane.exe")
        campione = modello.sample(
            data={"N": len(allenamento), "dati": allenamento.tolist(), "mu0": mu0, "Sigma0": sigma0, "a": a, "b": b,
                  "eta": eta}, chains=catene, iter_sampling=iterazioni, show_console=console)
        fittato = np.square(campione.stan_variable("sigma"))
        correlazioni = campione.stan_variable("L")[:, 0, 1]
    ascisse = fittato[:, 0]
    ordinate = fittato[:, 1]
    print([np.quantile(correlazioni, 0.025), np.quantile(correlazioni, 0.975)])
    if alternativaascisse == "maggiore":
        testascisse = len(ascisse[ascisse <= ipotesinullaascisse]) / len(ascisse)
    elif alternativaascisse == "disuguale":
        testascisse = (len(ascisse[ipotesinullaascisse - ropeascisse <= ascisse <= ipotesinullaascisse + ropeascisse])
                       / len(ascisse))
    else:
        testascisse = len(ascisse[ascisse >= ipotesinullaascisse]) / len(ascisse)
    intervalloascisse = [np.quantile(ascisse, alfaascisse / 2), np.quantile(ascisse, 1 - alfaascisse / 2)]
    if alternativaordinate == "maggiore":
        testordinate = len(ordinate[ordinate <= ipotesinullaordinate]) / len(ordinate)
    elif alternativaordinate == "disuguale":
        testordinate = (len(ordinate[
                                ipotesinullaordinate - ropeordinate <= ordinate <= ipotesinullaordinate + ropeordinate])
                        / len(ordinate))
    else:
        testordinate = len(ordinate[ordinate >= ipotesinullaordinate]) / len(ordinate)
    intervalloordinate = [np.quantile(ordinate, alfaordinate / 2), np.quantile(ordinate, 1 - alfaordinate / 2)]
    return (testascisse, testordinate), (intervalloascisse, intervalloordinate)


def varianzedifettose(allenamento: np.ndarray | list, alfa: float, numerofrecce: int):
    risultati = list()
    for numero in range(numerofrecce):
        difettanda = list()
        altre = list()
        for volée in allenamento:
            for freccia in volée:
                if freccia[2] == numero:
                    difettanda.append([freccia[0], freccia[1]])
                else:
                    altre.append([freccia[0], freccia[1]])
        vdifettanda = np.cov(np.array(difettanda), rowvar=False, bias=False)
        valtre = np.cov(np.array(altre), rowvar=False, bias=False)
        statisticaascisse = vdifettanda[0, 0] / valtre[0, 0]
        statisticaordinate = vdifettanda[1, 1] / valtre[1, 1]
        risultati.append((statisticaascisse > st.f.ppf(1 - alfa, len(difettanda) - 1, len(altre) - 1),
                          statisticaordinate > st.f.ppf(1 - alfa, len(difettanda) - 1, len(altre) - 1)))
    return risultati


def intervalloangolomedio(facile: bool, angolomedio: float, confidenza: float, kappa: float, r: float,
                          frecce: int) -> list[float]:
    quantile = st.chi2.ppf(1 - confidenza, df=1)
    if facile:
        distanza = math.acos((1 - quantile) / (2 * kappa * r * frecce))
        return [angolomedio - distanza, angolomedio + distanza]
    elif r < 2 / 3:
        n2r2 = frecce ** 2 * r ** 2
        distanza = math.acos((2 * frecce * (2 * n2r2 - frecce * quantile) / (n2r2 * (4 * frecce - quantile))) ** 0.5)
        return [angolomedio - distanza, angolomedio + distanza]
    else:
        n2r2 = frecce ** 2 * r ** 2
        distanza = math.acos((frecce ** 2 - (frecce ** 2 - n2r2) * math.exp(quantile)) ** 0.5 / n2r2)
        return [angolomedio - distanza, angolomedio + distanza]


def prioricircolari(noninformativa: bool, allenamentipriori: int, apiacere: bool, mu: float,
                    kappa: float, alfa: float, beta: float) -> tuple:
    if noninformativa:
        return 0, 0.001, 2, 0.1
    elif apiacere:
        return mu, kappa, alfa, beta
    else:
        listadate = leggifile("Date.txt")[-allenamentipriori:]
        listamu = [leggifile(f"Sessione del {data}").mediaangoli for data in listadate]
        listakappa = np.array([leggifile(f"Sessione del {data}").kappa for data in listadate])
        if any(mu is None for mu in listamu) or any(kappa is None for kappa in listakappa):
            return 0, 0.001, 2, 0.1
        else:
            return pycircstat.mean(np.array(listamu)), pycircstat.kappa(np.array(listamu)), np.mean(
                listakappa) ** 2 / np.var(listakappa), np.mean(listakappa) / np.var(listakappa)


def intervallocircolare(fittato: np.ndarray, confidenza: float) -> list[float]:
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
        return [iniziomigliore, finemigliore + 2 * np.pi]
    else:
        return [iniziomigliore, finemigliore]


def angolomediobayesiano(allenamento: np.ndarray, catene: int, iterazioni: int, credibilitàmu: float,
                         credibilitàkappa: float,
                         noninformativa: bool, allenamentipriori: int, variazionale: bool, gerarchica: bool,
                         apiacere: bool, mu: float, kappa: float, alfa: float, beta1: float, console: bool) -> tuple:
    if variazionale:
        mu0, _, a0, b0 = prioricircolari(noninformativa, allenamentipriori, apiacere, mu, kappa, alfa, beta1)
        risultati = miomodulo.angolomediobayesiano(mu0, a0, b0, allenamento.tolist() if isinstance(allenamento, np.ndarray) else allenamento,
                                                   len(allenamento))
        emme = risultati[0]
        beta = risultati[1]
        a = risultati[2]
        b = risultati[3]
        lambdasimulati = np.random.gamma(a, 1/b, size=10000)
        musimulati = np.random.vonmises(emme, beta * lambdasimulati, size=10000)
        return (intervallocircolare(musimulati, credibilitàmu), np.mean(lambdasimulati),
                [np.quantile(lambdasimulati, (1 - credibilitàkappa) / 2),
                 np.quantile(lambdasimulati, 1 - (1 - credibilitàkappa) / 2)])
    else:
        if gerarchica:
            listaallenamentipriori = [appiattisci(leggifile(f"Sessione del {data}").angoli)
                                      for data in leggifile("Date.txt")[-allenamentipriori:]]
            modello = cmdstanpy.CmdStanModel(exe_file="Angolomediogerarchico.exe")
            fittato = modello.sample(data={"Ncorrente": len(allenamento), "angolicorrenti": allenamento.tolist() if isinstance(allenamento, np.ndarray) else allenamento,
                                           "priori": allenamentipriori,
                                           "Npriori": [len(priore) for priore in listaallenamentipriori],
                                           "gruppipriori": appiattisci(
                                               [[numero for _ in range(len(listaallenamentipriori[numero - 1]))]
                                                for numero in range(1, allenamentipriori + 1)]),
                                           "angolipriori": appiattisci(listaallenamentipriori), "c": 1, "d": 1,
                                           "xi": 1, "zeta": 1}, show_console=console)
        else:
            mu0, kappa0, alfa, beta = prioricircolari(noninformativa, allenamentipriori, apiacere,
                                                      mu, kappa, alfa, beta1)
            modello = cmdstanpy.CmdStanModel(exe_file="Angolomedio.exe")
            fittato = (modello.sample(
                data={"N": len(allenamento), "y": allenamento.tolist() if isinstance(allenamento, np.ndarray) else allenamento,
                      "mu0": mu0, "kappa0": kappa0, "alpha": alfa,
                      "beta": beta}, chains=catene, iter_sampling=iterazioni, show_console=console))
        fittatomu = fittato.stan_variable("mu")
        fittatokappa = fittato.stan_variable("kappa")
        return (intervallocircolare(fittatomu, credibilitàmu), np.mean(fittatokappa),
                [np.quantile(fittatokappa, (1 - credibilitàkappa) / 2),
                 np.quantile(fittatokappa, 1 - (1 - credibilitàkappa) / 2)])


def priorimisturevonmises(noninformativa: bool, allenamentipriori: int, componenti: int) -> tuple:
    if noninformativa:
        return ([1 / componenti for _ in range(componenti)], [1 for _ in range(componenti)],
                np.linspace(-np.pi, np.pi, componenti, endpoint=False).tolist(), [2 for _ in range(componenti)],
                [0.1 for _ in range(componenti)], None)
    else:
        listadate = leggifile("Date.txt")[-allenamentipriori:]
        tuttiangoli = appiattisci([leggifile(f"Sessione del {data}").angoli for data in listadate])
        return miomodulo.misturevonmisesvariazionali([1 / componenti for _ in range(componenti)],
                                                     [1 for _ in range(componenti)],
                                                     np.linspace(-np.pi, np.pi, componenti, endpoint=False).tolist(),
                                                     [2 for _ in range(componenti)],
                                                     [0.1 for _ in range(componenti)], componenti,
                                                     tuttiangoli.tolist() if isinstance(tuttiangoli,
                                                                                        np.ndarray) else tuttiangoli,
                                                     len(tuttiangoli))


def misturevonmises(componentifissate: bool, noninformativa: bool,
                    allenamentipriori: int, componentimassime: int, allenamento: np.ndarray,
                    credibilitàmu: float, credibilitàkappa: float) -> tuple:
    if componentifissate:
        alfa0, beta0, m0, a0, b0, _ = priorimisturevonmises(noninformativa, allenamentipriori, componentimassime)
        m, beta, a, b, alfa, _, assegnazioni = miomodulo.misturevonmisesvariazionali(alfa0, beta0, m0, a0, b0, componentimassime,
                                                                       allenamento.tolist() if isinstance(allenamento, np.ndarray) else allenamento,
                                                                       len(allenamento))
        lambdasimulati = [np.random.gamma(shape, rate, size=10000) for shape, rate in zip(a, b)]
        musimulati = [np.random.vonmises(media, fattore * varianza, size=10000)
                      for media, fattore, varianza in zip(m, beta, lambdasimulati)]
        return ([intervallocircolare(campione, credibilitàmu) for campione in musimulati],
                [(np.quantile(campione, (1 - credibilitàkappa) / 2),
                  np.quantile(campione, 1 - (1 - credibilitàkappa) / 2))
                 for campione in lambdasimulati], list(zip(m, alfa)), assegnazioni)
    else:
        bic = list()
        parametri = list()
        for componenti in range(2, componentimassime + 1):
            alfa0, beta0, m0, a0, b0, _ = priorimisturevonmises(noninformativa, allenamentipriori, componenti)
            m, beta, a, b, alfa, bicsingolo, assegnazioni = miomodulo.misturevonmisesvariazionali(alfa0, beta0, m0, a0, b0,
                                                                                    componenti,
                                                                                    allenamento.tolist() if isinstance(allenamento, np.ndarray) else allenamento,
                                                                                    len(allenamento))
            parametri.append([m, beta, a, b, alfa])
            bic.append(bicsingolo[0])
        scelti = parametri[bic.index(min(bic))]
        lambdasimulati = [np.random.gamma(shape, 1/rate, size=10000) for shape, rate in zip(scelti[2], scelti[3])]
        musimulati = [np.random.vonmises(media, fattore * varianza, size=10000)
                      for media, fattore, varianza in zip(scelti[0], scelti[1], lambdasimulati)]
        return ([intervallocircolare(campione, credibilitàmu) for campione in musimulati],
                [(np.quantile(campione, (1 - credibilitàkappa) / 2),
                  np.quantile(campione, 1 - (1 - credibilitàkappa) / 2))
                 for campione in lambdasimulati], [(mediasingola, alfasingolo/sum(scelti[4])) for mediasingola,
                alfasingolo in zip(scelti[0], scelti[4])], assegnazioni)
    # Hai scelto di non implementare la possibilità di un metodo MCMC


def bersagliosugrafico(asse: plt.Axes):
    for raggio, colore in [(1.0, "white"), (0.8, "black"), (0.6, "blue"), (0.4, "red"), (0.2, "yellow")]:
        asse.add_patch(patches.Circle((0, 0), raggio, facecolor=colore, edgecolor="black"))


def graficodispersione(ascisse: np.ndarray | list, ordinate: np.ndarray | list, data: str):
    figura, asse = plt.subplots()
    bersagliosugrafico(asse)
    asse.scatter(appiattisci(ascisse), appiattisci(ordinate), color="purple", edgecolors="white")
    asse.set_aspect("equal")
    asse.set_xlim(-1.1, 1.1)
    asse.set_ylim(-1.1, 1.1)
    plt.title("Grafico di dispersione")
    plt.savefig(f"./Grafici del {data}/graficodispersione.png")
    plt.close()


def graficomediesubersaglio(medie: tuple[float, float] | None, medievolée: list[tuple[float, float]] | None,
                            mediefrecce: dict | None, regressione: tuple[np.ndarray, np.ndarray] | None,
                            frecce: int, data: str):
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
    plt.savefig(f"./Grafici del {data}/graficomedie.png")
    plt.close()


def graficopunteggi(ordine: bool, punti: np.ndarray | list, media: float | None, regressione: tuple | np.ndarray | None,
                    data: str):
    figura, asse = plt.subplots()
    punti = appiattisci(punti)
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
    plt.savefig(f"./Grafici del {data}/graficopunteggi.png")
    plt.close()


def graficovolée(medievolée: np.ndarray | None, ordine: bool, regressione: tuple | np.ndarray | None, mediefrecce: dict | None,
                 data: str):
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
    plt.savefig(f"./Grafici del {data}/graficovolée.png")
    plt.close()


def graficoangoli(angolisessione: np.ndarray | list, angolomedio: float | None, medievolée: list | None | np.ndarray,
                  mediefrecce: dict | None, data: str):
    figura, asse = plt.subplots(subplot_kw={"projection": "polar"})
    intervalli = np.linspace(0, 2 * math.pi, 12)
    istogramma, estremi = np.histogram(np.mod(np.array(appiattisci(angolisessione)), 2*np.pi), intervalli)
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
    plt.savefig(f"./Grafici del {data}/graficoangoli.png")
    plt.close()


def graficiautocorrelazioni(autocorrelazionipunteggi: np.ndarray | None, autocorrelazioniascisse: np.ndarray | None,
                            autocorrelazioniordinate: np.ndarray | None,
                            autocorrelazioniangolipositive: np.ndarray | None,
                            autocorrelazioniangolinegative: np.ndarray | None, data: str):
    if autocorrelazionipunteggi is not None:
        figura, asse = plt.subplots()
        asse.bar([i for i in range(len(autocorrelazionipunteggi))], autocorrelazionipunteggi, color="black")
        plt.title("Autocorrelazioni dei punteggi")
        plt.savefig(f"./Grafici del {data}/graficoautocorrelazionipunteggi.png")
        plt.close()
    if autocorrelazioniascisse is not None:
        figura, asse = plt.subplots()
        asse.bar([i for i in range(len(autocorrelazioniascisse))], autocorrelazioniascisse, color="black")
        plt.title("Autocorrelazioni delle ascisse")
        plt.savefig(f"./Grafici del {data}/graficoautocorrelazioniascisse.png")
        plt.close()
    if autocorrelazioniordinate is not None:
        figura, asse = plt.subplots()
        asse.bar([i for i in range(len(autocorrelazioniordinate))], autocorrelazioniordinate, color="black")
        plt.title("Autocorrelazioni delle ordinate")
        plt.savefig(f"./Grafici del {data}/graficoautocorrelazioniordinate.png")
        plt.close()
    if autocorrelazioniangolipositive is not None:
        figura, asse = plt.subplots()
        asse.bar([i for i in range(len(autocorrelazioniangolipositive))], autocorrelazioniangolipositive, color="black")
        plt.title("Autocorrelazioni degli angoli (positive)")
        plt.savefig(f"./Grafici del {data}/graficoautocorrelazioniangolipositive.png")
        plt.close()
    if autocorrelazioniangolinegative is not None:
        figura, asse = plt.subplots()
        asse.bar([i for i in range(len(autocorrelazioniangolinegative))], autocorrelazioniangolinegative, color="black")
        plt.title("Autocorrelazioni degli angoli (negative)")
        plt.savefig(f"./Grafici del {data}/graficoautocorrelazioniangolinegative.png")
        plt.close()


def graficomedianageometrica(medianageom: np.ndarray, data: str):
    figura, asse = plt.subplots()
    bersagliosugrafico(asse)
    asse.scatter(medianageom[0], medianageom[1], color="green")
    asse.set_aspect("equal")
    plt.title("Mediana geometrica")
    plt.savefig(f"./Grafici del {data}/graficomedianageometrica.png")
    plt.close()


def appiattisci(array: np.ndarray | list):
    if isinstance(array, np.ndarray):
        return array.flatten()
    else:
        if isinstance(array[0], list) or isinstance(array[0], set):
            return [elemento for lista in array for elemento in lista]
        else:
            return array


def graficointervallicluster(intervalli: list[tuple | None], data: str):
    figura, asse = plt.subplots()
    bersagliosugrafico(asse)
    for numero, ellissi in enumerate(intervalli):
        if ellissi is not None:
            asse.plot(ellissi[0], ellissi[1], color="purple")
            asse.text(ellissi[0][0], ellissi[1][0], str(numero), color="green")
    asse.set_aspect("equal")
    plt.title("Intervalli di confidenza bivariati per ogni cluster di frecce")
    plt.savefig(f"./Grafici del {data}/graficointervallicluster.png")
    plt.close()


def betanormenormali(alternativa: str, iterazioni: int, alfa: float, frecce: int, asimmetria: float,
                     distanzacomponenti: float, gradit: int) -> float:
    accettazioni = 0
    if alternativa == "laplace":
        for _ in range(iterazioni):
            if st.shapiro(np.random.laplace(size=frecce)).pvalue > alfa:
                accettazioni += 1
    elif alternativa == "normaleasimmetrica":
        for _ in range(iterazioni):
            if st.shapiro(st.skewnorm.rvs(asimmetria, size=frecce)).pvalue > alfa:
                accettazioni += 1
    elif alternativa == "uniforme":
        for _ in range(iterazioni):
            if st.shapiro(np.random.uniform(size=frecce)).pvalue > alfa:
                accettazioni += 1
    elif alternativa == "mistura":
        for _ in range(iterazioni):
            if st.shapiro(np.array([np.random.normal(-distanzacomponenti/2, 1.0) if np.random.uniform(0, 1, 1) >= 0.5
                                    else np.random.normal(distanzacomponenti/2, 1.0) for _ in range(frecce)])).pvalue > alfa:
                accettazioni += 1
    elif alternativa == "lognormale":
        for _ in range(iterazioni):
            if st.shapiro(np.random.lognormal(size=frecce)).pvalue > alfa:
                accettazioni += 1
    elif alternativa == "t":
        for _ in range(iterazioni):
            if st.shapiro(np.random.standard_t(df=gradit, size=frecce)).pvalue > alfa:
                accettazioni += 1
    return accettazioni / iterazioni


def alfaveronormenormali(iterazioni: int, frecce: int, alfa: float) -> float:
    accettazioni = 0
    for _ in range(iterazioni):
        if st.shapiro(np.random.normal(size=frecce)).pvalue > alfa:
            accettazioni += 1
    return 1 - accettazioni / iterazioni


def graficointervallonorme(intervallo: list[float], data: str):
    figura, asse = plt.subplots()
    bersagliosugrafico(asse)
    asse.plot(intervallo[0] / 10 * np.cos(np.linspace(0, 2 * np.pi, 200)),
              intervallo[0] / 10 * np.sin(np.linspace(0, 2 * np.pi, 200)), color="purple")
    asse.plot(intervallo[1] / 10 * np.cos(np.linspace(0, 2 * np.pi, 200)),
              intervallo[1] / 10 * np.sin(np.linspace(0, 2 * np.pi, 200)), color="purple")
    asse.set_aspect("equal")
    plt.title("Grafico dell'intervallo di confidenza delle norme")
    plt.savefig(f"./Grafici del {data}/graficointervallonorme.png")
    plt.close()


def graficointervallivarianze(intervalli: tuple, medie: tuple, data: str):
    figura, asse = plt.subplots()
    bersagliosugrafico(asse)
    asse.plot(medie[0] + intervalli[0][0]**0.5 * np.cos(np.linspace(0, 2 * np.pi, 200)),
              medie[1] + intervalli[1][0]**0.5 * np.sin(np.linspace(0, 2 * np.pi, 200)), color="green")
    asse.plot(medie[0] + intervalli[0][1]**0.5 * np.cos(np.linspace(0, 2 * np.pi, 200)),
              medie[1] + intervalli[1][1]**0.5 * np.sin(np.linspace(0, 2 * np.pi, 200)), color="purple")
    asse.set_aspect("equal")
    plt.title("Grafico degli intervalli di confidenza delle varianze")
    plt.savefig(f"./Grafici del {data}/graficointervallivarianze.png")
    plt.close()


def graficointervallovarianzebayesiano(intervalli: tuple, medie: tuple, data: str):
    figura, asse = plt.subplots()
    bersagliosugrafico(asse)
    asse.plot(medie[0] + intervalli[0][0]**0.5 * np.cos(np.linspace(0, 2 * np.pi, 200)),
              medie[1] + intervalli[1][0]**0.5 * np.sin(np.linspace(0, 2 * np.pi, 200)), color="green")
    asse.plot(medie[0] + intervalli[0][1]**0.5 * np.cos(np.linspace(0, 2 * np.pi, 200)),
              medie[1] + intervalli[1][1]**0.5 * np.sin(np.linspace(0, 2 * np.pi, 200)), color="purple")
    asse.set_aspect("equal")
    plt.title("Grafico degli intervalli di credibilità bayesiani delle varianze")
    plt.savefig(f"./Grafici del {data}/graficointervallivarianzebayesiani.png")
    plt.close()


def graficointervalloangolomedio(intervallo: list[float], data: str):
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
    plt.savefig(f"./Grafici del {data}/graficointervalloangolomedio.png")
    plt.close()


def graficointervallokappa(intervallo: list[float], media: float, data: str):
    figura, asse = plt.subplots(subplot_kw={"projection": "polar"})
    sigma1 = (-2 * math.log(i1(intervallo[1]) / i0(intervallo[1]))) ** 0.5
    sigma2 = (-2 * math.log(i1(intervallo[0]) / i0(intervallo[0]))) ** 0.5
    asse.bar([float(media)], [2.0], color="blue", width=[float(4 * sigma2)])
    asse.bar([float(media)], [1.0], color="red", width=[float(4 * sigma1)])
    asse.set_aspect("equal")
    plt.title("Grafico dell'intervallo di confidenza del parametro di concentrazione degli angoli")
    plt.savefig(f"./Grafici del {data}/graficointervallokappa.png")
    plt.close()


def graficoangolomediobayesiano(intervallo: list[float], data: str):
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
    plt.title("Credible interval for mean angle")
    plt.savefig(f"./Grafici del {data}/graficoangolomediobayesiano.png")
    plt.close()


def avvolgi(theta: float) -> float:
    return (theta + np.pi) % (2 * np.pi) - np.pi


def proteggisigma(kappa: float) -> float:
    if not np.isfinite(kappa) or kappa <= 1e-8:
        return 0.0
    frazione = i1(kappa) / i0(kappa)
    if frazione <= 0 or frazione >= 1:
        return 0.0
    return (-2 * math.log(frazione)) ** 0.5


def riavvolgi(ax, angolo, sigma, proporzione, colore, alpha):
    xmin = avvolgi(angolo - 2 * sigma)
    xmax = avvolgi(angolo + 2 * sigma)
    if xmin < xmax:
        ax.bar([float((xmax+xmin)/2)], [float(proporzione)], width=[float(xmax-xmin)], color=colore, alpha=alpha)
    else:
        ax.bar([float((np.pi+xmin)/2)], [float(proporzione)], width=[float(np.pi-xmin)], color=colore, alpha=alpha)
        ax.bar([float((xmax-np.pi)/2)], [float(proporzione)], width=[float(xmax+np.pi)], color=colore, alpha=alpha)


def graficomisturevonmises(intervallimedie: list, intervallikappa: list,
                           componenti: list, angoli: list, assegnazioni: list, data: str):
    figura, asse = plt.subplots(subplot_kw={"projection": "polar"})
    mappacolori = colormaps["tab10"]
    cose = sorted(
        zip(intervallimedie, intervallikappa, componenti),
        key=lambda x: x[2][1],
        reverse=True
    )
    for numero, ([angolobasso, angoloalto],
                 (kappabasso, kappaalto),
                 (angolo, proporzione)) in enumerate(cose):
        colore = mappacolori(numero / len(componenti))
        sigma1 = proteggisigma(kappaalto)
        sigma2 = proteggisigma(kappabasso)
        if sigma2 > 0:
            riavvolgi(asse, angolo, sigma2, proporzione, colore, alpha=0.2)
        if sigma1 > 0:
            riavvolgi(asse, angolo, sigma1, proporzione, colore, alpha=0.4)
        xb, xa = avvolgi(angolobasso), avvolgi(angoloalto)
        if xb < xa:
            asse.bar([float((xa+xb)/2)], [float(proporzione)], width=[float(xa-xb)], color=colore, alpha=0.6)
        else:
            asse.bar([float((np.pi+xb)/2)], [float(proporzione)], width=[float(np.pi-xb)], color=colore, alpha=0.6)
            asse.bar([float((xa-np.pi)/2)], [float(proporzione)], width=[float(xa+np.pi)], color=colore, alpha=0.6)
        asse.bar([float(avvolgi(angolo))], [float(proporzione)], width=0.05, color=colore)
    raggi = np.ones_like(angoli)*(max(cose, key=lambda x: x[2][1])[2][1]+0.02)
    colori = mappacolori[np.array(assegnazioni)]
    asse.scatter(np.array(angoli), raggi, c=colori, alpha=0.9, edgecolor="black")
    asse.set_aspect("equal")
    plt.title("Grafico dei diversi gruppi di angoli trovati")
    plt.savefig(f"./Grafici del {data}/graficomisturevonmises.png")
    plt.close()


def mirinoideale(gamma: list[float], distanza: list[float|int], coccamirino: float, mediane: list[float],
                 raggiobersaglio=0.2):
    p = math.pi
    seno = math.sin
    coseno = math.cos
    arcoseno = math.asin
    cm = coccamirino
    a = abs(mediane[0])*raggiobersaglio
    o = abs(mediane[1])*raggiobersaglio
    g1 = math.radians(gamma[0])
    g2 = math.radians(gamma[1])
    d1 = float(distanza[0])
    d2 = float(distanza[1])
    i1 = (d1**2+a**2-2*d1*a*math.cos(g1))**0.5
    i2 = (d2**2+o**2-2*d2*o*math.cos(g2))**0.5
    s1 = (cm*a*seno(g1))/((seno(p-g1)*coseno(arcoseno(a/i1))-a*seno(g1)*coseno(p-g1)/i1)*i1) if mediane[0] != 0 else 0
    s2 = (cm*o*seno(g2))/((seno(p-g2)*coseno(arcoseno(o/i2))-o*seno(g2)*coseno(p-g2)/i2)*i2) if mediane[1] != 0 else 0
    if mediane[0] < 0:
        s1 *= -1
    if mediane[1] < 0:
        s2 *= -1
    return s1, s2


class Allenamento:
    def cambiatag(self, nuovitag):
        self.tag = nuovitag

    def cambiadistanza(self, nuovadistanza: int):
        self.distanza = nuovadistanza

    def cambiaarco(self, nuovoarco):
        self.arco = nuovoarco

    def cambiadata(self, nuovadata: str):
        rename(f"./Grafici del {self.data}", f"./Grafici del {nuovadata}")
        self.data = nuovadata

    def cambiaimpostazioni(self, nuoveimpostazioni: dict):
        self.impostazioni = nuoveimpostazioni

    def calcolamediapunteggifrecce(self):
        if self.identificazione and self.frecce > 1:
            try:
                tempo = time()
                self.mediapunteggifrecce = mediapunteggifrecce(self.punteggiidentificati)
                self.tempomediapunteggifrecce = time() - tempo
                self.impostazioni["mediapunteggifrecce"] = True
                self.tempototale += self.tempomediapunteggifrecce
                if self.tempograficovolée is not None:
                    self.tempototale -= self.tempograficovolée
                    remove(f"./Grafici del {self.data}/graficovolée.png")
                tempo = time()
                graficovolée(self.mediapunteggivolée, self.ordine, self.regressionepunteggi,
                             self.mediapunteggifrecce, self.data)
                self.tempograficovolée = time()-tempo
                self.tempototale += self.tempograficovolée
            except Exception as e:
                self.mediapunteggifrecce = None
                self.tempomediapunteggifrecce = None
                print(f"\033[35mNon ho potuto calcolare mediapunteggifrecce perché {e}\033[0m")

    def calcolamediacoordinatefrecce(self):
        if self.identificazione and self.frecce > 1:
            try:
                tempo = time()
                self.mediacoordinatefrecce = mediacoordinatefrecce(self.ascisseidentificate,
                                                                   self.ordinateidentificate)
                self.tempomediacoordinatefrecce = time() - tempo
                self.impostazioni["mediacoordinatefrecce"] = True
                self.tempototale += self.tempomediacoordinatefrecce
                if self.tempograficomedie is not None:
                    self.tempototale -= self.tempograficomedie
                    remove(f"./Grafici del {self.data}/graficomedie.png")
                tempo = time()
                graficomediesubersaglio(self.mediacoordinate, self.mediacoordinatevolée, self.mediacoordinatefrecce,
                                        self.regressionecoordinate, self.taglia if self.ordine else self.volée, self.data)
                self.tempograficomedie = time() - tempo
                self.tempototale += self.tempograficomedie
            except Exception as e:
                self.mediacoordinatefrecce = None
                self.tempomediacoordinatefrecce = None
                print(f"\033[35mNon ho potuto calcolare mediacoordinatefrecce perché {e}\033[0m")

    def calcolamediaangolifrecce(self):
        if self.identificazione and self.frecce > 1:
            try:
                tempo = time()
                self.mediaangolifrecce = mediaangolifrecce(self.angoliidentificati)
                self.tempomediaangolifrecce = time() - tempo
                self.impostazioni["mediaangolifrecce"] = True
                self.tempototale += self.tempomediaangolifrecce
            except Exception as e:
                self.mediaangolifrecce = None
                self.tempomediacoordinatefrecce = None
                print(f"\033[35mNon ho potuto calcolare mediaangolifrecce perché {e}\033[0m")

    def calcolavarianzapunteggifrecce(self):
        if self.identificazione and self.frecce > 1:
            try:
                tempo = time()
                self.varianzapunteggifrecce = varianzapunteggifrecce(self.punteggiidentificati)
                self.tempovarianzapunteggifrecce = time() - tempo
                self.impostazioni["varianzapunteggifrecce"] = True
                self.tempototale += self.tempovarianzapunteggifrecce
            except Exception as e:
                self.varianzapunteggifrecce = None
                self.tempovarianzapunteggifrecce = None
                print(f"\033[35mNon ho potuto calcolare varianzapunteggifrecce perché {e}\033[0m")

    def calcolavarianzacoordinatefrecce(self):
        if self.identificazione and self.frecce > 1:
            try:
                tempo = time()
                self.varianzacoordinatefrecce = varianzacoordinatefrecce(self.ascisseidentificate,
                                                                         self.ordinateidentificate)
                self.tempovarianzacoordinatefrecce = time() - tempo
                self.impostazioni["varianzacoordinatefrecce"] = True
                self.tempototale += self.tempovarianzacoordinatefrecce
            except Exception as e:
                self.varianzacoordinatefrecce = None
                self.tempovarianzacoordinatefrecce = None
                print(f"\033[35mNon ho potuto calcolare varianzacoordinatefrecce perché {e}\033[0m")

    def calcolavarianzaangolifrecce(self):
        if self.identificazione and self.frecce > 1:
            try:
                tempo = time()
                self.varianzaangolifrecce = varianzaangolifrecce(self.angoliidentificati)
                self.tempovarianzaangolifrecce = time() - tempo
                self.impostazioni["varianzaangolifrecce"] = True
                self.tempototale += self.tempovarianzaangolifrecce
            except Exception as e:
                self.varianzaangolifrecce = None
                self.tempovarianzaangolifrecce = None
                print(f"\033[35mNon ho potuto calcolare varianzaangolifrecce perché {e}\033[0m")

    def calcolacorrelazionefrecce(self):
        if self.identificazione and self.frecce > 1:
            try:
                tempo = time()
                self.correlazionefrecce = correlazionefrecce(self.ascisseidentificate,
                                                             self.ordinateidentificate,
                                                             self.frecce)
                self.tempocorrelazionefrecce = time() - tempo
                self.impostazioni["correlazionefrecce"] = True
                self.tempototale += self.tempocorrelazionefrecce
            except Exception as e:
                self.correlazionefrecce = None
                self.tempocorrelazionefrecce = None
                print(f"\033[35mNon ho potuto calcolare correlazionefrecce perché {e}\033[0m")

    def calcolamediapunteggi(self):
        try:
            tempo = time()
            self.mediapunteggi = np.mean(appiattisci(self.punteggi))
            self.tempomediapunteggi = time() - tempo
            self.impostazioni["mediapunteggi"] = True
            self.tempototale += self.tempomediapunteggi
            if self.tempograficopunteggi is not None:
                self.tempototale -= self.tempograficopunteggi
                remove(f"./Grafici del {self.data}/graficopunteggi.png")
            tempo = time()
            graficopunteggi(self.ordine, self.punteggi, self.mediapunteggi, self.regressionepunteggi,
                            self.data)
            self.tempograficopunteggi = time()-tempo
            self.tempototale += self.tempograficopunteggi
        except Exception as e:
            self.mediapunteggi = None
            self.tempomediapunteggi = None
            print(f"\033[35mNon ho potuto calcolare mediapunteggi perché {e}\033[0m")

    def calcolamediapunteggivolée(self):
        if self.volée > 1:
            try:
                tempo = time()
                self.mediapunteggivolée = np.array([np.mean(np.array(volée)) for volée in self.punteggi])
                self.tempomediapunteggivolée = time() - tempo
                self.impostazioni["mediapunteggivolée"] = True
                self.tempototale += self.tempomediapunteggivolée
                if self.tempograficovolée is not None:
                    self.tempototale -= self.tempograficovolée
                    remove(f"./Grafici del {self.data}/graficovolée.png")
                tempo = time()
                graficovolée(self.mediapunteggivolée, self.ordine, self.regressionepunteggi,
                             self.mediapunteggifrecce, self.data)
                self.tempograficovolée = time()-tempo
                self.tempototale += self.tempograficovolée
            except Exception as e:
                self.mediapunteggivolée = None
                self.tempomediapunteggivolée = None
                print(f"\033[35mNon ho potuto calcolare mediapunteggivolée perché {e}\033[0m")

    def calcolamediacoordinate(self):
        try:
            tempo = time()
            self.mediacoordinate = (np.mean(appiattisci(self.ascisse)),
                                    np.mean(appiattisci(self.ordinate)))
            self.tempomediacoordinate = time() - tempo
            self.impostazioni["mediacoordinate"] = True
            self.tempototale += self.tempomediacoordinate
            if self.tempograficomedie is not None:
                self.tempototale -= self.tempograficomedie
                remove(f"./Grafici del {self.data}/graficomedie.png")
            tempo = time()
            graficomediesubersaglio(self.mediacoordinate, self.mediacoordinatevolée, self.mediacoordinatefrecce,
                                    self.regressionecoordinate, self.taglia if self.ordine else self.volée, self.data)
            self.tempograficomedie = time()-tempo
            self.tempototale += self.tempograficomedie
        except Exception as e:
            self.mediacoordinate = None
            self.tempomediacoordinate = None
            print(f"\033[35mNon ho potuto calcolare mediacoordinate perché {e}\033[0m")

    def calcolamediacoordinatevolée(self):
        if self.volée > 1:
            try:
                tempo = time()
                self.mediacoordinatevolée = [(np.mean(np.array(self.ascisse[i])),
                                              np.mean(np.array(self.ordinate[i]))) for i in range(len(self.ascisse))]
                self.tempomediacoordinatevolée = time() - tempo
                self.impostazioni["mediacoordinatevolée"] = True
                self.tempototale += self.tempomediacoordinatevolée
                if self.tempograficomedie is not None:
                    self.tempototale -= self.tempograficomedie
                    remove(f"./Grafici del {self.data}/graficomedie.png")
                tempo = time()
                graficomediesubersaglio(self.mediacoordinate, self.mediacoordinatevolée, self.mediacoordinatefrecce,
                                        self.regressionecoordinate, self.taglia if self.ordine else self.volée,
                                        self.data)
                self.tempograficomedie = time()-tempo
                self.tempototale += self.tempograficomedie
            except Exception as e:
                self.mediacoordinatevolée = None
                self.tempomediacoordinatevolée = None
                print(f"\033[35mNon ho potuto calcolare mediacoordinatevolée perché {e}\033[0m")

    def calcolamediaangoli(self):
        try:
            tempo = time()
            self.mediaangoli = st.circmean(appiattisci(self.angoli))
            self.tempomediaangoli = time() - tempo
            self.impostazioni["mediaangoli"] = True
            self.tempototale += self.tempomediaangoli
        except Exception as e:
            self.mediaangoli = None
            self.tempomediaangoli = None
            print(f"\033[35mNon ho potuto calcolare mediaangoli perché {e}\033[0m")

    def calcolamediaangolivolée(self):
        if self.volée > 1:
            try:
                tempo = time()
                self.mediaangolivolée = np.array([st.circmean(volée) for volée in self.angoli])
                self.tempomediaangolivolée = time() - tempo
                self.impostazioni["mediaangolivolée"] = True
                self.tempototale += self.tempomediaangolivolée
            except Exception as e:
                self.mediaangolivolée = None
                self.tempomediaangolivolée = None
                print(f"\033[35mNon ho potuto calcolare mediaangolivolée perché {e}\033[0m")

    def calcolavarianzapunteggi(self):
        if self.taglia > 1:
            try:
                tempo = time()
                self.varianzapunteggi = np.var(appiattisci(self.punteggi), ddof=1)
                self.tempovarianzapunteggi = time() - tempo
                self.impostazioni["varianzapunteggi"] = True
                self.tempototale += self.tempovarianzapunteggi
            except Exception as e:
                self.varianzapunteggi = None
                self.tempovarianzapunteggi = None
                print(f"\033[35mNon ho potuto calcolare varianzapunteggi perché {e}\033[0m")

    def calcolavarianzapunteggivolée(self):
        if self.volée > 1:
            try:
                tempo = time()
                self.varianzapunteggivolée = np.array([(np.var(np.array(volée), ddof=1) if len(volée) > 1 else 0)
                                                       for volée in self.punteggi])
                self.tempovarianzapunteggivolée = time() - tempo
                self.impostazioni["varianzapunteggivolée"] = True
                self.tempototale += self.tempovarianzapunteggivolée
            except Exception as e:
                self.varianzapunteggivolée = None
                self.tempovarianzapunteggivolée = None
                print(f"\033[35mNon ho potuto calcolare varianzapunteggivolée perché {e}\033[0m")

    def calcolavarianzacoordinate(self):
        if self.taglia > 1:
            try:
                tempo = time()
                self.varianzacoordinate = (np.var(appiattisci(self.ascisse), ddof=1),
                                           np.var(appiattisci(self.ordinate), ddof=1))
                self.tempovarianzacoordinate = time() - tempo
                self.impostazioni["varianzacoordinate"] = True
                self.tempototale += self.tempovarianzacoordinate
            except Exception as e:
                self.varianzacoordinate = None
                self.tempovarianzacoordinate = None
                print(f"\033[35mNon ho potuto calcolare varianzacoordinate perché {e}\033[0m")

    def calcolavarianzacoordinatevolée(self):
        if self.volée > 1:
            try:
                tempo = time()
                self.varianzacoordinatevolée = [((np.var(np.array(self.ascisse[i]), ddof=1),
                                                  np.var(np.array(self.ordinate[i]), ddof=1))
                                                 if len(self.ascisse[i]) > 1 else 0)
                                                for i in range(len(self.ascisse))]
                self.tempovarianzacoordinatevolée = time() - tempo
                self.impostazioni["varianzacoordinatevolée"] = True
                self.tempototale += self.tempovarianzacoordinatevolée
            except Exception as e:
                self.varianzacoordinatevolée = None
                self.tempovarianzacoordinatevolée = None
                print(f"\033[35mNon ho potuto calcolare varianzacoordinatevolée perché {e}\033[0m")

    def calcolavarianzaangoli(self):
        if self.taglia > 1:
            try:
                tempo = time()
                self.varianzaangoli = varianzaangoli(appiattisci(self.angoli))
                self.tempovarianzaangoli = time() - tempo
                self.impostazioni["varianzaangoli"] = True
                self.tempototale += self.tempovarianzaangoli
            except Exception as e:
                self.varianzaangoli = None
                self.tempovarianzaangoli = None
                print(f"\033[35mNon ho potuto calcolare varianzaangoli perché {e}\033[0m")

    def calcolavarianzaangolivolée(self):
        if self.volée > 1:
            try:
                tempo = time()
                self.mediaangolivolée = np.array([st.circmean(volée) for volée in self.angoli])
                self.tempomediaangolivolée = time() - tempo
                self.impostazioni["varianzaangolivolée"] = True
                self.tempototale += self.tempomediaangolivolée
            except Exception as e:
                self.mediaangolivolée = None
                self.tempomediaangolivolée = None
                print(f"\033[35mNon ho potuto calcolare mediaangolivolée perché {e}\033[0m")

    def calcolacorrelazione(self):
        if self.taglia > 1:
            try:
                tempo = time()
                self.correlazione = np.corrcoef(appiattisci(self.ascisse),
                                                appiattisci(self.ordinate))[0, 1]
                self.tempocorrelazione = time() - tempo
                self.impostazioni["correlazione"] = True
                self.tempototale += self.tempocorrelazione
            except Exception as e:
                self.correlazione = None
                self.tempocorrelazione = None
                print(f"\033[35mNon ho potuto calcolare correlazione perché {e}\033[0m")

    def calcolacorrelazionevolée(self):
        if self.volée > 1:
            try:
                tempo = time()
                self.correlazionevolée = np.array([(np.corrcoef(self.ascisse[i],
                                                                self.ordinate[i])[0, 1] if len(
                    self.ascisse[i]) > 1 else 0)
                                                   for i in range(len(self.ascisse))])
                self.tempocorrelazionevolée = time() - tempo
                self.impostazioni["correlazionevolée"] = True
                self.tempototale += self.tempocorrelazionevolée
            except Exception as e:
                self.correlazionevolée = None
                self.tempocorrelazionevolée = None
                print(f"\033[35mNon ho potuto calcolare correlazionevolée perché {e}\033[0m")

    def calcolaregressionepunteggi(self):
        if self.ordine and self.taglia > 1:
            try:
                tempo = time()
                self.regressionepunteggi = np.polyfit([i for i in range(self.taglia)],
                                                      appiattisci(self.punteggi),
                                                      1)
                self.temporegressionepunteggi = time() - tempo
                self.impostazioni["regressionepunteggi"] = True
                self.tempototale += self.temporegressionepunteggi
                if self.tempograficopunteggi is not None:
                    self.tempototale -= self.tempograficopunteggi
                tempo = time()
                graficopunteggi(self.ordine, self.punteggi, self.mediapunteggi, self.regressionepunteggi,
                                self.data)
                self.tempograficopunteggi = time()-tempo
                self.tempototale += self.tempograficopunteggi
            except Exception as e:
                self.regressionepunteggi = None
                self.temporegressionepunteggi = None
                print(f"\033[35mNon ho potuto calcolare regressionepunteggi perché {e}\033[0m")
        elif self.volée > 1:
            try:
                tempo = time()
                self.regressionepunteggi = np.polyfit([i for i in range(len(self.punteggi))],
                                                      self.mediapunteggivolée if self.mediapunteggivolée is not None
                                                      else np.array([np.mean(volée) for volée in self.punteggi]),
                                                      1)
                self.temporegressionepunteggi = time() - tempo
                self.impostazioni["regressionepunteggi"] = True
                self.tempototale += self.temporegressionepunteggi
                if self.tempograficovolée is not None:
                    self.tempototale -= self.tempograficovolée
                tempo = time()
                graficovolée(self.mediapunteggivolée, self.ordine, self.regressionepunteggi, self.mediapunteggifrecce,
                             self.data)
                self.tempograficovolée = time()-tempo
                self.tempototale += self.tempograficovolée
            except Exception as e:
                self.regressionepunteggi = None
                self.temporegressionepunteggi = None
                print(f"\033[35mNon ho potuto calcolare regressionepunteggi perché {e}\033[0m")

    def calcolaregressionecoordinate(self):
        if self.ordine and self.taglia > 1:
            try:
                tempo = time()
                self.regressionecoordinate = (np.polyfit([i for i in range(self.taglia)],
                                                         appiattisci(self.ascisse),
                                                         1),
                                              np.polyfit([i for i in range(self.taglia)],
                                                         appiattisci(self.ordinate),
                                                         1))
                self.temporegressionecoordinate = time() - tempo
                self.impostazioni["regressionecoordinate"] = True
                self.tempototale += self.temporegressionecoordinate
                if self.tempograficomedie is not None:
                    self.tempototale -= self.tempograficomedie
                tempo = time()
                graficomediesubersaglio(self.mediacoordinate, self.mediacoordinatevolée, self.mediacoordinatefrecce,
                                        self.regressionecoordinate, self.taglia, self.data)
                self.tempograficomedie = time()-tempo
                self.tempototale += self.tempograficomedie
            except Exception as e:
                self.regressionecoordinate = None
                self.temporegressionecoordinate = None
                print(f"\033[35mNon ho potuto calcolare regressionecoordinate perché {e}\033[0m")
        elif self.volée > 1:
            try:
                tempo = time()
                self.regressionecoordinate = (np.polyfit([i for i in range(len(self.ascisse))],
                                                         ([media[0] for media in self.mediacoordinatevolée])
                                                         if self.mediacoordinatevolée is not None
                                                         else np.mean(self.ascisse, axis=0), 1),
                                              np.polyfit([i for i in range(len(self.ordinate))],
                                                         ([media[1] for media in self.mediacoordinatevolée])
                                                         if self.mediacoordinatevolée is not None
                                                         else np.mean(self.ordinate, axis=0), 1))
                self.temporegressionecoordinate = time() - tempo
                self.impostazioni["regressionecoordinate"] = True
                self.tempototale += self.temporegressionecoordinate
                if self.tempograficomedie is not None:
                    self.tempototale -= self.tempograficomedie
                tempo = time()
                graficomediesubersaglio(self.mediacoordinate, self.mediacoordinatevolée, self.mediacoordinatefrecce,
                                        self.regressionecoordinate, self.volée, self.data)
                self.tempograficomedie = time()-tempo
                self.tempototale += self.tempograficomedie
            except Exception as e:
                self.regressionecoordinate = None
                self.temporegressionecoordinate = None
                print(f"\033[35mNon ho potuto calcolare regressionecoordiante perché {e}\033[0m")

    def calcolaautocorrelazionepunteggi(self):
        if self.ordine and self.taglia > 1:
            try:
                tempo = time()
                self.autocorrelazionepunteggi = autocorrelazioni(appiattisci(self.punteggi),
                                                                 self.varianzapunteggi
                                                                 if self.varianzapunteggi is not None
                                                                 else np.var(appiattisci(self.punteggi)))
                self.tempoautocorrelazionepunteggi = time() - tempo
                self.impostazioni["autocorrelazionepunteggi"] = True
                self.tempototale += self.tempoautocorrelazionepunteggi
                if self.tempograficoautocorrelazioni is not None:
                    self.tempototale -= self.tempograficoautocorrelazioni
                tempo = time()
                graficiautocorrelazioni(self.autocorrelazionepunteggi, self.autocorrelazioneascisse,
                                        self.autocorrelazioneordinate, self.autocorrelazioniangolipositive,
                                        self.autocorrelazioniangolinegative, self.data)
                self.tempograficoautocorrelazioni = time()-tempo
                self.tempototale += self.tempograficoautocorrelazioni
            except Exception as e:
                self.autocorrelazionepunteggi = None
                self.tempoautocorrelazionepunteggi = None
                print(f"\033[35mNon ho potuto calcolare autocorrelazionepunteggi perché {e}\033[0m")
        elif self.volée > 1:
            try:
                tempo = time()
                self.autocorrelazionepunteggi = autocorrelazioni(self.mediapunteggivolée
                                                                 if self.mediapunteggivolée is not None
                                                                 else np.array([np.mean(volée) for volée in self.punteggi]),
                                                                 np.var(self.mediapunteggivolée
                                                                        if self.mediapunteggivolée is not None
                                                                        else np.array([np.mean(volée) for volée in self.punteggi])))
                self.tempoautocorrelazionepunteggi = time() - tempo
                self.impostazioni["autocorrelazionepunteggi"] = True
                self.tempototale += self.tempoautocorrelazionepunteggi
                if self.tempograficoautocorrelazioni is not None:
                    self.tempototale -= self.tempograficoautocorrelazioni
                tempo = time()
                graficiautocorrelazioni(self.autocorrelazionepunteggi, self.autocorrelazioneascisse,
                                        self.autocorrelazioneordinate, self.autocorrelazioniangolipositive,
                                        self.autocorrelazioniangolinegative, self.data)
                self.tempograficoautocorrelazioni = time()-tempo
                self.tempototale += self.tempograficoautocorrelazioni
            except Exception as e:
                self.autocorrelazionepunteggi = None
                self.tempoautocorrelazionepunteggi = None
                print(f"\033[35mNon ho potuto calcolare autocorrelazionepunteggi perché {e}\033[0m")

    def calcolaautocorrelazioneascisse(self):
        if self.ordine and self.taglia > 1:
            try:
                tempo = time()
                self.autocorrelazioneascisse = autocorrelazioni(appiattisci(self.ascisse),
                                                                self.varianzacoordinate[0]
                                                                if self.varianzacoordinate is not None
                                                                else np.var(appiattisci(self.ascisse)))
                self.tempoautocorrelazioneascisse = time() - tempo
                self.impostazioni["autocorrelazioneascisse"] = True
                self.tempototale += self.tempoautocorrelazioneascisse
                if self.tempograficoautocorrelazioni is not None:
                    self.tempototale -= self.tempograficoautocorrelazioni
                tempo = time()
                graficiautocorrelazioni(self.autocorrelazionepunteggi, self.autocorrelazioneascisse,
                                        self.autocorrelazioneordinate, self.autocorrelazioniangolipositive,
                                        self.autocorrelazioniangolinegative, self.data)
                self.tempograficoautocorrelazioni = time()-tempo
                self.tempototale += self.tempograficoautocorrelazioni
            except Exception as e:
                self.autocorrelazioneascisse = None
                self.tempoautocorrelazioneascisse = None
                print(f"\033[35mNon ho potuto calcolare autocorrelazioneascisse perché {e}\033[0m")
        elif self.volée > 1:
            try:
                tempo = time()
                medieascisse = (np.array([media[0] for media in self.mediacoordinatevolée])
                                if self.mediacoordinatevolée is not None
                                else np.mean(self.ascisse, axis=0))
                self.autocorrelazioneascisse = autocorrelazioni(medieascisse, np.var(medieascisse))
                self.tempoautocorrelazioneascisse = time() - tempo
                self.impostazioni["autocorrelazioneascisse"] = True
                self.tempototale += self.tempoautocorrelazioneascisse
                if self.tempograficoautocorrelazioni is not None:
                    self.tempototale -= self.tempograficoautocorrelazioni
                tempo = time()
                graficiautocorrelazioni(self.autocorrelazionepunteggi, self.autocorrelazioneascisse,
                                        self.autocorrelazioneordinate, self.autocorrelazioniangolipositive,
                                        self.autocorrelazioniangolinegative, self.data)
                self.tempograficoautocorrelazioni = time()-tempo
                self.tempototale += self.tempograficoautocorrelazioni
            except Exception as e:
                self.autocorrelazioneascisse = None
                self.tempoautocorrelazioneascisse = None
                print(f"\033[35mNon ho potuto calcolare autocorrelazioneascisse perché {e}\033[0m")

    def calcolaautocorrelazioneordinate(self):
        if self.ordine and self.taglia > 1:
            try:
                tempo = time()
                self.autocorrelazioneordinate = autocorrelazioni(appiattisci(self.ordinate),
                                                                 self.varianzacoordinate[1]
                                                                 if self.varianzacoordinate is not None
                                                                 else np.var(appiattisci(self.ordinate)))
                self.tempoautocorrelazioneordinate = time() - tempo
                self.impostazioni["autocorrelazioneordinate"] = True
                self.tempototale += self.tempoautocorrelazioneordinate
                if self.tempograficoautocorrelazioni is not None:
                    self.tempototale -= self.tempograficoautocorrelazioni
                tempo = time()
                graficiautocorrelazioni(self.autocorrelazionepunteggi, self.autocorrelazioneascisse,
                                        self.autocorrelazioneordinate, self.autocorrelazioniangolipositive,
                                        self.autocorrelazioniangolinegative, self.data)
                self.tempograficoautocorrelazioni = time()-tempo
                self.tempototale += self.tempograficoautocorrelazioni
            except Exception as e:
                self.autocorrelazioneordinate = None
                self.tempoautocorrelazioneordinate = None
                print(f"\033[35mNon ho potuto calcolare autocorrelazioneordinate perché {e}\033[0m")
        elif self.volée > 1:
            try:
                tempo = time()
                medieordinate = (np.array([media[1] for media in self.mediacoordinatevolée])
                                 if self.mediacoordinatevolée is not None
                                 else np.mean(self.ordinate, axis=0))
                self.autocorrelazioneordinate = autocorrelazioni(medieordinate, np.var(medieordinate))
                self.tempoautocorrelazioneordinate = time() - tempo
                self.impostazioni["autocorrelazioneordinate"] = True
                self.tempototale += self.tempoautocorrelazioneordinate
                if self.tempograficoautocorrelazioni is not None:
                    self.tempototale -= self.tempograficoautocorrelazioni
                tempo = time()
                graficiautocorrelazioni(self.autocorrelazionepunteggi, self.autocorrelazioneascisse,
                                        self.autocorrelazioneordinate, self.autocorrelazioniangolipositive,
                                        self.autocorrelazioniangolinegative, self.data)
                self.tempograficoautocorrelazioni = time()-tempo
                self.tempototale += self.tempograficoautocorrelazioni
            except Exception as e:
                self.autocorrelazioneordinate = None
                self.tempoautocorrelazioneordinate = None
                print(f"\033[35mNon ho potuto calcolare autocorrelazioneordinate perché {e}\033[0m")

    def calcolaautocorrelazioniangoli(self):
        if self.ordine and self.taglia > 1:
            try:
                tempo = time()
                self.autocorrelazioniangolipositive, self.autocorrelazioniangolinegative = miomodulo.autocorrelazioniangolari(
                    appiattisci(self.angoli), self.taglia)
                self.tempoautocorrelazioniangoli = time() - tempo
                self.impostazioni["autocorrelazioniangoli"] = True
                self.tempototale += self.tempoautocorrelazioniangoli
                if self.tempograficoautocorrelazioni is not None:
                    self.tempototale -= self.tempograficoautocorrelazioni
                tempo = time()
                graficiautocorrelazioni(self.autocorrelazionepunteggi, self.autocorrelazioneascisse,
                                        self.autocorrelazioneordinate, self.autocorrelazioniangolipositive,
                                        self.autocorrelazioniangolinegative, self.data)
                self.tempograficoautocorrelazioni = time()-tempo
                self.tempototale += self.tempograficoautocorrelazioni
            except Exception as e:
                self.autocorrelazioniangolipositive = None
                self.autocorrelazioniangolinegative = None
                self.tempoautocorrelazioniangoli = None
                print(f"\033[35mNon ho potuto calcolare autocorrelazioniangoli perché {e}\033[0m")
        elif self.volée > 1:
            try:
                tempo = time()
                self.autocorrelazioniangolipositive, self.autocorrelazioniangolinegative = miomodulo.autocorrelazioniangolari(
                    self.mediaangolivolée.tolist() if self.mediaangolivolée is not None else [st.circmean(volée) for volée in self.angoli],
                    self.volée)
                self.tempoautocorrelazioniangoli = time() - tempo
                self.impostazioni["autocorrelazioniangoli"] = True
                self.tempototale += self.tempoautocorrelazioniangoli
                if self.tempograficoautocorrelazioni is not None:
                    self.tempototale -= self.tempograficoautocorrelazioni
                tempo = time()
                graficiautocorrelazioni(self.autocorrelazionepunteggi, self.autocorrelazioneascisse,
                                        self.autocorrelazioneordinate, self.autocorrelazioniangolipositive,
                                        self.autocorrelazioniangolinegative, self.data)
                self.tempograficoautocorrelazioni = time()-tempo
                self.tempototale += self.tempograficoautocorrelazioni
            except Exception as e:
                self.autocorrelazioniangolipositive = None
                self.autocorrelazioniangolinegative = None
                self.tempoautocorrelazioniangoli = None
                print(f"\033[35mNon ho potuto calcolare autocorrelazioniangoli perché {e}\033[0m")

    # Se "congiunto" è True, questa funzione DEVE essere eseguita prima di rieseguiljungboxordinate.
    def rieseguiljungboxascisse(self, alfa: float, h: int, congiunto: bool):
        if alfa <= 0 or alfa >= 1 or h <= 0:
            raise Exception("Valori dei parametri sbagliati")
        if (self.impostazioni["testhotelling"] or self.impostazioni["intervallohotelling"] or self.impostazioni["hotellingbayesiano"]
        or self.impostazioni["intervallobayesiano"]) and self.taglia > 2:
            if self.betaljungboxascisse is not None and self.probabilitàtuttogiusto is not None:
                self.probabilitàtuttogiusto /= 1-self.betaljungboxascisse
                self.betaljungboxascisse = None
            elif self.alfaveroljungboxascisse is not None and self.probabilitàtuttogiusto is not None:
                self.probabilitàtuttogiusto /= 1-self.alfaveroljungboxascisse
                self.alfaveroljungboxascisse = None
            elif self.probabilitàtuttogiusto is not None and (not self.ljungboxascisse if self.ljungboxascisse is not None else False):
                self.probabilitàtuttogiusto /= 1-self.impostazioni["alfaljungboxascisse"]
            if self.tempobetaljungboxascisse is not None:
                self.tempototale -= self.tempobetaljungboxascisse
                self.tempobetaljungboxascisse = None
            elif self.tempoalfaveroljungboxascisse is not None:
                self.tempototale -= self.tempoalfaveroljungboxascisse
                self.tempoalfaveroljungboxascisse = None
            self.impostazioni["alfaljungboxascisse"] = alfa
            self.impostazioni["hljungboxascisse"] = h
            autasc = None
            cutoff1 = None
            soglie1 = None
            try:
                tempo = time()
                if self.autocorrelazioneascisse is not None:
                    autasc = self.autocorrelazioneascisse
                else:
                    if self.ordine:
                        autasc = autocorrelazioni(appiattisci(self.ascisse), np.var(appiattisci(self.ascisse)))
                    else:
                        medieascisse = ([media[0] for media in self.mediacoordinatevolée]
                                        if self.mediacoordinatevolée is not None
                                        else [np.mean(np.array(self.ascisse[i]))
                                              for i in range(len(self.ascisse))])
                        autasc = autocorrelazioni(medieascisse, np.var(medieascisse))
                if not isinstance(autasc, np.ndarray):
                    autasc = None
                    raise IndexError("Le liste di autocorrelazioni delle ascisse e delle ordinate sono mal formattate")
                cutoff1 = list()
                soglie1 = list()
                if h < len(autasc):
                    cutoffattuale1 = h
                    while cutoffattuale1 < len(autasc):
                        cutoff1.append(cutoffattuale1)
                        soglie1.append(st.chi2.ppf(1 - alfa, df=cutoffattuale1))
                        cutoffattuale1 += h
                cutoff1.append(len(autasc))
                soglie1.append(st.chi2.ppf(1 - alfa, df=len(autasc)))
                print(autasc)
                print(cutoff1)
                print(len(autasc)+1)
                print(soglie1)
                print(len(cutoff1))
                self.ljungboxascisse = miomodulo.ljungbox(autasc.tolist(), cutoff1, len(autasc) + 1,
                                                          soglie1, len(cutoff1))
                if congiunto:
                    self.tempoljungbox = time()-tempo
                elif self.tempoljungbox is not None:
                    self.tempototale -= self.tempoljungbox
                    self.tempoljungbox = self.tempoljungbox/2 + time() - tempo
                    self.tempototale += self.tempoljungbox
                else:
                    raise Exception("Qualcosa non va")
            except Exception as e:
                if self.tempoljungbox is not None:
                    self.tempototale -= self.tempoljungbox
                self.ljungboxascisse = None
                self.ljungboxordinate = None
                self.tempoljungbox = None
                print(f"\033[35mNon ho potuto eseguire il test di Ljung-Box perché {e}\033[0m")
            if self.ljungboxascisse is None or self.ljungboxordinate is None or self.mardia is None:
                self.affidabilitàhotelling = False
            elif self.ljungboxascisse and self.ljungboxordinate and self.mardia:
                self.affidabilitàhotelling = True
            else:
                self.affidabilitàhotelling = False
            if self.ljungboxascisse is not None and self.ljungboxordinate is not None:
                if autasc is not None and cutoff1 is not None and soglie1 is not None:
                    if self.ljungboxascisse and self.impostazioni["calcolaerrori"]:
                        try:
                            tempo = time()
                            self.betaljungboxascisse = miomodulo.betaljungbox(len(autasc)+1, cutoff1, soglie1,
                                                                              len(cutoff1),
                                                                              self.impostazioni["voltebetaljungboxascisse"])
                            self.tempobetaljungboxascisse = time() - tempo
                            self.tempototale += self.tempobetaljungboxascisse
                            self.probabilitàtuttogiusto *= (1 - self.betaljungboxascisse)
                            self.alfaveroljungboxascisse = None
                            self.tempoalfaveroljungboxascisse = None
                        except Exception as e:
                            self.betaljungboxascisse = None
                            self.tempobetaljungboxascisse = None
                            self.alfaveroljungboxascisse = None
                            self.tempoalfaveroljungboxascisse = None
                            self.probabilitàtuttogiusto = None
                            self.impostazioni["calcolaerrori"] = False
                            print(
                                f"\033[35mNon ho potuto calcolare il beta del test di Ljung-Box perché {e}; smetteremo di calcolare le probabilità di errore\033[0m")
                    elif len(autasc) < 50 and self.impostazioni["calcolaerrori"]:
                        try:
                            tempo = time()
                            self.alfaveroljungboxascisse = miomodulo.alfaveroljungbox(len(autasc) + 1, cutoff1, soglie1,
                                                                                      len(cutoff1),
                                                                                      self.impostazioni[
                                                                                          "voltealfaljungboxascisse"])
                            self.tempoalfaveroljungboxascisse = time() - tempo
                            self.tempototale += self.tempoalfaveroljungboxascisse
                            self.probabilitàtuttogiusto *= (1 - self.alfaveroljungboxascisse)
                            self.betaljungboxascisse = None
                            self.tempobetaljungboxascisse = None
                        except Exception as e:
                            self.alfaveroljungboxascisse = None
                            self.tempoalfaveroljungboxascisse = None
                            self.betaljungboxascisse = None
                            self.tempobetaljungboxascisse = None
                            self.probabilitàtuttogiusto = None
                            self.impostazioni["calcolaerrori"] = False
                            print(
                                f"\033[35mNon ho potuto calcolare l'alfa effettivo per il test di Ljung-Box perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
                    else:
                        if self.impostazioni["calcolaerrori"]:
                            self.probabilitàtuttogiusto *= (1 - alfa) ** len(cutoff1)
                        self.alfaveroljungboxascisse = None
                        self.betaljungboxascisse = None
                        self.tempoalfaveroljungboxascisse = None
                        self.tempobetaljungboxascisse = None
                else:
                    self.alfaveroljungboxascisse = None
                    self.betaljungboxascisse = None
                    self.tempoalfaveroljungboxascisse = None
                    self.tempobetaljungboxascisse = None
                    self.probabilitàtuttogiusto = None
                    self.impostazioni["calcolaerrori"] = False
                    print(
                        f"\033[35mNon ho potuto calcolare la probabilità di errore per il test di Ljung-Box per un errore precedente; smetteremo di calcolare le probabilità degli errori\033[0m")
            else:
                self.alfaveroljungboxascisse = None
                self.betaljungboxascisse = None
                self.tempoalfaveroljungboxascisse = None
                self.tempobetaljungboxascisse = None
                self.alfaveroljungboxordinate = None
                self.betaljungboxordinate = None
                self.tempoalfaveroljungboxordinate = None
                self.tempobetaljungboxordinate = None

    def rieseguiljungboxordinate(self, alfa: float, h: int, congiunto: bool):
        if alfa <= 0 or alfa >= 1 or h <= 0:
            raise Exception("Valori dei parametri sbagliati")
        if self.betaljungboxordinate is not None and self.probabilitàtuttogiusto is not None:
            self.probabilitàtuttogiusto /= 1-self.betaljungboxordinate
            self.betaljungboxordinate = None
        elif self.alfaveroljungboxordinate is not None and self.probabilitàtuttogiusto is not None:
            self.probabilitàtuttogiusto /= 1-self.alfaveroljungboxordinate
            self.alfaveroljungboxordinate = None
        elif self.probabilitàtuttogiusto is not None and (not self.ljungboxascisse if self.ljungboxascisse is not None else False):
            self.probabilitàtuttogiusto /= 1-self.impostazioni["alfaljungboxordinate"]
        if self.tempobetaljungboxordinate is not None:
            self.tempototale -= self.tempobetaljungboxordinate
            self.tempobetaljungboxordinate = None
        elif self.tempoalfaveroljungboxordinate is not None:
            self.tempototale -= self.tempoalfaveroljungboxordinate
            self.tempoalfaveroljungboxordinate = None
        self.impostazioni["alfaljungboxordinate"] = alfa
        self.impostazioni["hljungboxordinate"] = h
        autord = None
        cutoff2 = None
        soglie2 = None
        try:
            tempo = time()
            if self.autocorrelazioneordinate is not None:
                autord = self.autocorrelazioneordinate
            else:
                if self.ordine:
                    autord = autocorrelazioni(appiattisci(self.ordinate), np.var(appiattisci(self.ordinate)))
                else:
                    medieordinate = ([media[1] for media in self.mediacoordinatevolée]
                                     if self.mediacoordinatevolée is not None
                                     else [np.mean(np.array(self.ordinate[i]))
                                           for i in range(len(self.ordinate))])
                    autord = autocorrelazioni(medieordinate, np.var(medieordinate))
            if not isinstance(autord, np.ndarray):
                autord = None
                raise IndexError("Le liste di autocorrelazioni delle ascisse e delle ordinate sono mal formattate")
            cutoff2 = list()
            soglie2 = list()
            if h < len(autord):
                cutoffattuale2 = h
                while cutoffattuale2 < len(autord):
                    cutoff2.append(cutoffattuale2)
                    soglie2.append(st.chi2.ppf(1 - alfa, df=cutoffattuale2))
                    cutoffattuale2 += h
            cutoff2.append(len(autord) - 1)
            soglie2.append(st.chi2.ppf(1 - alfa, df=len(autord)))
            self.ljungboxordinate = miomodulo.ljungbox(autord.tolist(), cutoff2, len(autord) + 1,
                                                       soglie2, len(cutoff2))
            if congiunto:
                self.tempoljungbox += time()-tempo
                self.tempototale += self.tempoljungbox
            elif self.tempoljungbox is not None:
                self.tempototale -= self.tempoljungbox
                self.tempoljungbox = self.tempoljungbox/2+time()-tempo
                self.tempototale += self.tempoljungbox
            else:
                raise Exception("Qualcosa non va")
        except Exception as e:
            if self.tempoljungbox is not None:
                self.tempototale -= self.tempoljungbox
            self.ljungboxascisse = None
            self.ljungboxordinate = None
            self.tempoljungbox = None
            print(f"\033[35mNon ho potuto eseguire il test di Ljung-Box perché {e}\033[0m")
        if self.ljungboxascisse is None or self.ljungboxordinate is None or self.mardia is None:
            self.affidabilitàhotelling = False
        elif self.ljungboxascisse and self.ljungboxordinate and self.mardia:
            self.affidabilitàhotelling = True
        else:
            self.affidabilitàhotelling = False
        if self.ljungboxascisse is not None and self.ljungboxordinate is not None:
            if autord is not None and cutoff2 is not None and soglie2 is not None:
                if self.ljungboxordinate and self.impostazioni["calcolaerrori"]:
                    try:
                        tempo = time()
                        self.betaljungboxordinate = miomodulo.betaljungbox(len(autord)+1, cutoff2, soglie2,
                                                                          len(cutoff2),
                                                                          self.impostazioni["voltebetaljungboxordinate"])
                        self.tempobetaljungboxordinate = time() - tempo
                        self.tempototale += self.tempobetaljungboxordinate
                        self.probabilitàtuttogiusto *= (1 - self.betaljungboxordinate)
                        self.alfaveroljungboxordinate = None
                        self.tempoalfaveroljungboxordinate = None
                    except Exception as e:
                        self.betaljungboxordinate = None
                        self.tempobetaljungboxordinate = None
                        self.alfaveroljungboxordinate = None
                        self.tempoalfaveroljungboxordinate = None
                        self.probabilitàtuttogiusto = None
                        self.impostazioni["calcolaerrori"] = False
                        print(
                            f"\033[35mNon ho potuto calcolare il beta del test di Ljung-Box perché {e}; smetteremo di calcolare le probabilità di errore\033[0m")
                elif len(autord) < 50 and self.impostazioni["calcolaerrori"]:
                    try:
                        tempo = time()
                        self.alfaveroljungboxordinate = miomodulo.alfaveroljungbox(len(autord) + 1, cutoff2, soglie2,
                                                                                  len(cutoff2),
                                                                                  self.impostazioni[
                                                                                      "voltealfaljungboxordinate"])
                        self.tempoalfaveroljungboxordinate = time() - tempo
                        self.tempototale += self.tempoalfaveroljungboxordinate
                        self.probabilitàtuttogiusto *= (1 - self.alfaveroljungboxordinate)
                        self.betaljungboxordinate = None
                        self.tempobetaljungboxordinate = None
                    except Exception as e:
                        self.alfaveroljungboxordinate = None
                        self.tempoalfaveroljungboxordinate = None
                        self.betaljungboxordinate = None
                        self.tempobetaljungboxordinate = None
                        self.probabilitàtuttogiusto = None
                        self.impostazioni["calcolaerrori"] = False
                        print(
                            f"\033[35mNon ho potuto calcolare l'alfa effettivo per il test di Ljung-Box perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
                else:
                    if self.impostazioni["calcolaerrori"]:
                        self.probabilitàtuttogiusto *= (1 - alfa) ** len(cutoff2)
                    self.alfaveroljungboxordinate = None
                    self.betaljungboxordinate = None
                    self.tempoalfaveroljungboxordinate = None
                    self.tempobetaljungboxordinate = None
            else:
                self.alfaveroljungboxordinate = None
                self.betaljungboxordinate = None
                self.tempoalfaveroljungboxordinate = None
                self.tempobetaljungboxordinate = None
                self.probabilitàtuttogiusto = None
                self.impostazioni["calcolaerrori"] = False
                print(
                    f"\033[35mNon ho potuto calcolare la probabilità di errore per il test di Ljung-Box per un errore precedente; smetteremo di calcolare le probabilità degli errori\033[0m")
        else:
            self.alfaveroljungboxascisse = None
            self.betaljungboxascisse = None
            self.tempoalfaveroljungboxascisse = None
            self.tempobetaljungboxascisse = None
            self.alfaveroljungboxordinate = None
            self.betaljungboxordinate = None
            self.tempoalfaveroljungboxordinate = None
            self.tempobetaljungboxordinate = None

    def depura(self) -> np.ndarray:
        return np.array([[ascissa, ordinata] for ascissa, ordinata in zip(appiattisci(self.ascisse), appiattisci(self.ordinate))])

    def rieseguimardia(self, alfa: float, tipo: str):
        if alfa <= 0 or alfa >= 0.5 or (tipo != "henze-zirkler" and tipo != "mardia"):
            raise Exception("Valori dei parametri sbagliati")
        if self.betamardia is not None and self.probabilitàtuttogiusto is not None:
            self.probabilitàtuttogiusto /= 1-self.betamardia
            self.betamardia = None
        elif self.alfaveromardia is not None and self.probabilitàtuttogiusto is not None:
            self.probabilitàtuttogiusto /= 1-self.alfaveromardia
            self.alfaveromardia = None
        elif self.probabilitàtuttogiusto is not None and (not self.mardia if self.mardia is not None else False):
            self.probabilitàtuttogiusto /= 1-self.impostazioni["alfamardia"]
        if self.tempobetamardia is not None:
            self.tempototale -= self.tempobetamardia
            self.tempobetamardia = None
        elif self.tempoalfaveromardia is not None:
            self.tempototale -= self.tempoalfaveromardia
            self.tempoalfaveromardia = None
        self.impostazioni["alfamardia"] = alfa
        self.impostazioni["tipotestnormalità"] = tipo
        if self.tempomardia is not None:
            self.tempototale -= self.tempomardia
        try:
            tempo = time()
            self.mardia = testnormalità(self.depura(), self.taglia, alfa, tipo)
            self.tempomardia = time() - tempo
            self.tempototale += self.tempomardia
        except Exception as e:
            self.mardia = None
            self.tempomardia = None
            print(f"\033[35mNon ho potuto eseguire il test di normalità multivariata perché {e}\033[0m")
        if self.mardia is not None:
            if self.mardia and self.impostazioni["calcolaerrori"]:
                try:
                    tempo = time()
                    self.betamardia = betamardia(frecce=self.taglia,
                                                 distribuzione=self.impostazioni["distribuzionebetamardia"],
                                                 iterazioni=self.impostazioni["voltebetamardia"],
                                                 gradit=self.impostazioni["graditbetamardia"],
                                                 alfa=alfa,
                                                 tipotest=tipo,
                                                 asimmetriaascisse=self.impostazioni["asimmetriaascissebetamardia"],
                                                 asimmetriaordinate=self.impostazioni["asimmetriaordinatebetamardia"],
                                                 distanzacomponenti=self.impostazioni["distanzacomponentibetamardia"])
                    self.tempobetamardia = time() - tempo
                    self.tempototale += self.tempobetamardia
                    self.probabilitàtuttogiusto *= 1 - self.betamardia
                    self.alfaveromardia = None
                    self.tempoalfaveromardia = None
                except Exception as e:
                    self.betamardia = None
                    self.tempobetamardia = None
                    self.alfaveromardia = None
                    self.tempoalfaveromardia = None
                    self.probabilitàtuttogiusto = None
                    self.impostazioni["calcolaerrori"] = False
                    print(
                        f"\033[35mNon ho potuto calcolare il beta del test di multinormalità perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
            elif self.taglia < 50 and self.impostazioni["calcolaerrori"]:
                try:
                    tempo = time()
                    self.alfaveromardia = alfaveromardia(frecce=self.taglia,
                                                         iterazioni=self.impostazioni["voltealfamardia"],
                                                         alfa=alfa,
                                                         tipotest=tipo)
                    self.tempoalfaveromardia = time() - tempo
                    self.tempototale += self.tempoalfaveromardia
                    self.probabilitàtuttogiusto *= 1 - self.alfaveromardia
                    self.betamardia = None
                    self.tempobetamardia = None
                except Exception as e:
                    self.alfaveromardia = None
                    self.tempoalfaveromardia = None
                    self.betamardia = None
                    self.tempobetamardia = None
                    self.probabilitàtuttogiusto = None
                    self.impostazioni["calcolaerrori"] = False
                    print(
                        f"\033[35mNon ho potuto calcolare l'alfa effettivo del test di multinormalità perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
            else:
                if self.impostazioni["calcolaerrori"]:
                    self.probabilitàtuttogiusto *= 1 - alfa
                self.alfaveromardia = None
                self.tempoveromardia = None
                self.betamardia = None
                self.tempobetamardia = None
        else:
            self.alfaveromardia = None
            self.tempoalfaveromardia = None
            self.betamardia = None
            self.tempobetamardia = None

    def rieseguibetaljungboxascisse(self, iterazioni: int):
        if iterazioni <= 0:
            raise Exception("Valori dei parametri sbagliati")
        if (self.impostazioni["testhotelling"] or self.impostazioni["intervallohotelling"]
        or self.impostazioni["hotellingbayesiano"] or self.impostazioni["intervallobayesiano"]) and self.taglia > 2:
            if self.ljungboxascisse and self.impostazioni["calcolaerrori"]:
                self.impostazioni["voltebetaljungboxascisse"] = iterazioni
                if self.betaljungboxascisse is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1-self.betaljungboxascisse
                if self.tempobetaljungboxascisse is not None:
                    self.tempototale -= self.tempobetaljungboxascisse
                if self.autocorrelazioneascisse is not None:
                    autasc = self.autocorrelazioneascisse
                else:
                    if self.ordine:
                        autasc = autocorrelazioni(appiattisci(self.ascisse), np.var(appiattisci(self.ascisse)))
                    else:
                        medieascisse = ([media[0] for media in self.mediacoordinatevolée]
                                         if self.mediacoordinatevolée is not None
                                         else [np.mean(np.array(self.ascisse[i]))
                                               for i in range(len(self.ascisse))])
                        autasc = autocorrelazioni(medieascisse, np.var(medieascisse))
                if not isinstance(autasc, np.ndarray):
                    autasc = None
                    raise IndexError("Le liste di autocorrelazioni delle ascisse e delle ordinate sono mal formattate")
                cutoff1 = list()
                soglie1 = list()
                if self.impostazioni["hljungboxascisse"] < len(autasc):
                    cutoffattuale1 = self.impostazioni["hljungboxascisse"]
                    while cutoffattuale1 < len(autasc):
                        cutoff1.append(cutoffattuale1)
                        soglie1.append(st.chi2.ppf(1 - self.impostazioni["alfaljungboxascisse"], df=cutoffattuale1))
                        cutoffattuale1 += self.impostazioni["hljungboxascisse"]
                cutoff1.append(len(autasc))
                soglie1.append(st.chi2.ppf(1 - self.impostazioni["alfaljungboxascisse"], df=len(autasc)))
                try:
                    tempo = time()
                    self.betaljungboxascisse = miomodulo.betaljungbox(len(autasc) + 1, cutoff1, soglie1,
                                                                       len(cutoff1),
                                                                       iterazioni)
                    self.tempobetaljungboxascisse = time() - tempo
                    self.tempototale += self.tempobetaljungboxascisse
                    self.probabilitàtuttogiusto *= (1 - self.betaljungboxascisse)
                    self.alfaveroljungboxascisse = None
                    self.tempoalfaveroljungboxascisse = None
                except Exception as e:
                    self.betaljungboxascisse = None
                    self.tempobetaljungboxascisse = None
                    self.alfaveroljungboxascisse = None
                    self.tempoalfaveroljungboxascisse = None
                    self.probabilitàtuttogiusto = None
                    self.impostazioni["calcolaerrori"] = False
                    print(
                        f"\033[35mNon ho potuto calcolare il beta del test di Ljung-Box perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
        else:
            self.betaljungboxascisse = None
            self.tempobetaljungboxascisse = None
            self.alfaveroljungboxascisse = None
            self.tempoalfaveroljungboxascisse = None

    def rieseguibetaljungboxordinate(self, iterazioni: int):
        if iterazioni <= 0:
            raise Exception("Valori dei parametri sbagliati")
        if (self.impostazioni["testhotelling"] or self.impostazioni["intervallohotelling"]
        or self.impostazioni["hotellingbayesiano"] or self.impostazioni["intervallobayesiano"]) and self.taglia > 2:
            if self.ljungboxordinate and self.impostazioni["calcolaerrori"]:
                self.impostazioni["voltebetaljungboxordinate"] = iterazioni
                if self.betaljungboxordinate is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1 - self.betaljungboxordinate
                if self.tempobetaljungboxordinate is not None:
                    self.tempototale -= self.tempobetaljungboxordinate
                if self.autocorrelazioneordinate is not None:
                    autord = self.autocorrelazioneordinate
                else:
                    if self.ordine:
                        autord = autocorrelazioni(appiattisci(self.ordinate), np.var(appiattisci(self.ordinate)))
                    else:
                        medieordinate = ([media[1] for media in self.mediacoordinatevolée]
                                        if self.mediacoordinatevolée is not None
                                        else [np.mean(np.array(self.ordinate[i]))
                                              for i in range(len(self.ordinate))])
                        autord = autocorrelazioni(medieordinate, np.var(medieordinate))
                if not isinstance(autord, np.ndarray):
                    autord = None
                    raise IndexError("Le liste di autocorrelazioni delle ascisse e delle ordinate sono mal formattate")
                cutoff2 = list()
                soglie2 = list()
                if self.impostazioni["hljungboxordinate"] < self.taglia - 1:
                    cutoffattuale2 = self.impostazioni["hljungboxordinate"]
                    while cutoffattuale2 < self.taglia - 1:
                        cutoff2.append(cutoffattuale2)
                        soglie2.append(st.chi2.ppf(1 - self.impostazioni["alfaljungboxordinate"], df=cutoffattuale2))
                        cutoffattuale2 += self.impostazioni["hljungboxordinate"]
                cutoff2.append(self.taglia - 1)
                soglie2.append(st.chi2.ppf(1 - self.impostazioni["alfaljungboxordinate"], df=self.taglia - 1))
                try:
                    tempo = time()
                    self.betaljungboxordinate = miomodulo.betaljungbox(len(autord) + 1, cutoff2, soglie2,
                                                                      len(cutoff2),
                                                                      iterazioni)
                    self.tempobetaljungboxordinate = time() - tempo
                    self.tempototale += self.tempobetaljungboxordinate
                    self.probabilitàtuttogiusto *= (1 - self.betaljungboxordinate)
                    self.alfaveroljungboxordinate = None
                    self.tempoalfaveroljungboxordinate = None
                except Exception as e:
                    self.betaljungboxordinate = None
                    self.tempobetaljungboxordinate = None
                    self.alfaveroljungboxordinate = None
                    self.tempoalfaveroljungboxordinate = None
                    self.probabilitàtuttogiusto = None
                    self.impostazioni["calcolaerrori"] = False
                    print(
                        f"\033[35mNon ho potuto calcolare il beta del test di Ljung-Box perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
        else:
            self.betaljungboxordinate = None
            self.tempobetaljungboxordinate = None
            self.alfaveroljungboxordinate = None
            self.tempoalfaveroljungboxordinate = None

    def rieseguialfaveroljungboxascisse(self, iterazioni: int):
        if iterazioni <= 0:
            raise Exception("Valori dei parametri sbagliati")
        if (self.impostazioni["testhotelling"] or self.impostazioni["intervallohotelling"]
             or self.impostazioni["hotellingbayesiano"] or self.impostazioni["intervallobayesiano"]) and self.taglia > 2:
            if not self.ljungboxascisse and (self.taglia if self.ordine else self.volée) < 51 and self.impostazioni["calcolaerrori"]:
                self.impostazioni["voltealfaveroljungboxascisse"] = iterazioni
                if self.alfaveroljungboxascisse is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1 - self.alfaveroljungboxascisse
                if self.tempoalfaveroljungboxascisse is not None:
                    self.tempototale -= self.tempoalfaveroljungboxascisse
                if self.autocorrelazioneascisse is not None:
                    autasc = self.autocorrelazioneascisse
                else:
                    if self.ordine:
                        autasc = autocorrelazioni(appiattisci(self.ascisse), np.var(appiattisci(self.ascisse)))
                    else:
                        medieascisse = ([media[0] for media in self.mediacoordinatevolée]
                                        if self.mediacoordinatevolée is not None
                                        else [np.mean(np.array(self.ascisse[i]))
                                              for i in range(len(self.ascisse))])
                        autasc = autocorrelazioni(medieascisse, np.var(medieascisse))
                if not isinstance(autasc, np.ndarray):
                    autasc = None
                    raise IndexError("Le liste di autocorrelazioni delle ascisse e delle ordinate sono mal formattate")
                cutoff1 = list()
                soglie1 = list()
                if self.impostazioni["hljungboxascisse"] < self.taglia - 1:
                    cutoffattuale1 = self.impostazioni["hljungboxascisse"]
                    while cutoffattuale1 < self.taglia - 1:
                        cutoff1.append(cutoffattuale1)
                        soglie1.append(st.chi2.ppf(1 - self.impostazioni["alfaljungboxascisse"], df=cutoffattuale1))
                        cutoffattuale1 += self.impostazioni["hljungboxascisse"]
                cutoff1.append(self.taglia - 1)
                soglie1.append(st.chi2.ppf(1 - self.impostazioni["alfaljungboxascisse"], df=self.taglia - 1))
                try:
                    tempo = time()
                    self.alfaveroljungboxascisse = miomodulo.alfaveroljungbox(len(autasc) + 1, cutoff1, soglie1,
                                                                              len(cutoff1),
                                                                              iterazioni)
                    self.tempoalfaveroljungboxascisse = time() - tempo
                    self.tempototale += self.tempoalfaveroljungboxascisse
                    self.probabilitàtuttogiusto *= (1 - self.alfaveroljungboxascisse)
                    self.betaljungboxascisse = None
                    self.tempobetaljungboxascisse = None
                except Exception as e:
                    self.alfaveroljungboxascisse = None
                    self.tempoalfaveroljungboxascisse = None
                    self.betaljungboxascisse = None
                    self.tempobetaljungboxascisse = None
                    self.probabilitàtuttogiusto = None
                    self.impostazioni["calcolaerrori"] = False
                    print(
                        f"\033[35mNon ho potuto calcolare l'alfa effettivo per il test di Ljung-Box perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
        else:
            self.betaljungboxascisse = None
            self.tempobetaljungboxascisse = None
            self.alfaveroljungboxascisse = None
            self.tempoalfaveroljungboxascisse = None

    def rieseguialfaveroljungboxordinate(self, iterazioni: int):
        if iterazioni <= 0:
            raise Exception("Valori dei parametri sbagliati")
        if (self.impostazioni["testhotelling"] or self.impostazioni["intervallohotelling"]
             or self.impostazioni["hotellingbayesiano"] or self.impostazioni["intervallobayesiano"]) and self.taglia > 2:
            if not self.ljungboxordinate and (self.taglia if self.ordine else self.volée) < 51 and self.impostazioni["calcolaerrori"]:
                self.impostazioni["voltealfaveroljungboxordinate"] = iterazioni
                if self.alfaveroljungboxordinate is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1 - self.alfaveroljungboxordinate
                if self.tempoalfaveroljungboxordinate is not None:
                    self.tempototale -= self.tempoalfaveroljungboxordinate
                if self.autocorrelazioneordinate is not None:
                    autord = self.autocorrelazioneordinate
                else:
                    if self.ordine:
                        autord = autocorrelazioni(appiattisci(self.ordinate), np.var(appiattisci(self.ordinate)))
                    else:
                        medieordinate = ([media[1] for media in self.mediacoordinatevolée]
                                        if self.mediacoordinatevolée is not None
                                        else [np.mean(np.array(self.ordinate[i]))
                                              for i in range(len(self.ordinate))])
                        autord = autocorrelazioni(medieordinate, np.var(medieordinate))
                if not isinstance(autord, np.ndarray):
                    autord = None
                    raise IndexError("Le liste di autocorrelazioni delle ascisse e delle ordinate sono mal formattate")
                cutoff2 = list()
                soglie2 = list()
                if self.impostazioni["hljungboxordinate"] < self.taglia - 1:
                    cutoffattuale2 = self.impostazioni["hljungboxordinate"]
                    while cutoffattuale2 < self.taglia - 1:
                        cutoff2.append(cutoffattuale2)
                        soglie2.append(st.chi2.ppf(1 - self.impostazioni["alfaljungboxordinate"], df=cutoffattuale2))
                        cutoffattuale2 += self.impostazioni["hljungboxordinate"]
                cutoff2.append(self.taglia - 1)
                soglie2.append(st.chi2.ppf(1 - self.impostazioni["alfaljungboxordinate"], df=self.taglia - 1))
                try:
                    tempo = time()
                    self.alfaveroljungboxordinate = miomodulo.alfaveroljungbox(len(autord) + 1, cutoff2, soglie2,
                                                                              len(cutoff2),
                                                                              iterazioni)
                    self.tempoalfaveroljungboxordinate = time() - tempo
                    self.tempototale += self.tempoalfaveroljungboxordinate
                    self.probabilitàtuttogiusto *= (1 - self.alfaveroljungboxordinate)
                    self.betaljungboxordinate = None
                    self.tempobetaljungboxordinate = None
                except Exception as e:
                    self.alfaveroljungboxordinate = None
                    self.tempoalfaveroljungboxordinate = None
                    self.betaljungboxordinate = None
                    self.tempobetaljungboxordinate = None
                    self.probabilitàtuttogiusto = None
                    self.impostazioni["calcolaerrori"] = False
                    print(
                        f"\033[35mNon ho potuto calcolare l'alfa effettivo per il test di Ljung-Box perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
        else:
            self.betaljungboxordinate = None
            self.tempobetaljungboxordinate = None
            self.alfaveroljungboxordinate = None
            self.tempoalfaveroljungboxordinate = None

    def rieseguibetamardia(self, distribuzione: str, iterazioni: int, gradit: int, tipotest: str,
                           asimmetriaascisse: float, asimmetriaordinate: float, distanzacomponenti: float):
        if distribuzione not in {"laplace", "normaleasimmetrica", "uniforme", "mistura", "lognormale", "t"}:
            raise Exception("Valori dei parametri sbagliati")
        if iterazioni <= 0 or gradit <= 0 or (tipotest != "henze-zirkler" and tipotest != "mardia"):
            raise Exception("Valori dei parametri sbagliati")
        if self.mardia is not None and (self.impostazioni["testhotelling"] or self.impostazioni["intervallohotelling"]
        or self.impostazioni["hotellingbayesiano"] or self.impostazioni["intervallobayesiano"]) and self.taglia > 2:
            if self.mardia and self.impostazioni["calcolaerrori"]:
                self.impostazioni["distribuzionebetamardia"] = distribuzione
                self.impostazioni["voltebetamardia"] = iterazioni
                self.impostazioni["graditbetamardia"] = gradit
                self.impostazioni["tipotestnormalità"] = tipotest
                self.impostazioni["asimmetriaascissebetamardia"] = asimmetriaascisse
                self.impostazioni["asimmetriaordinatebetamardia"] = asimmetriaordinate
                self.impostazioni["distanzacomponentibetamardia"] = distanzacomponenti
                if self.betamardia is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1-self.betamardia
                if self.tempobetamardia is not None:
                    self.tempototale -= self.tempobetamardia
                try:
                    tempo = time()
                    self.betamardia = betamardia(frecce=self.taglia,
                                                 distribuzione=distribuzione,
                                                 iterazioni=iterazioni,
                                                 gradit=gradit,
                                                 alfa=self.impostazioni["alfamardia"],
                                                 tipotest=tipotest,
                                                 asimmetriaascisse=asimmetriaascisse,
                                                 asimmetriaordinate=asimmetriaordinate,
                                                 distanzacomponenti=distanzacomponenti)
                    self.tempobetamardia = time() - tempo
                    self.tempototale += self.tempobetamardia
                    self.probabilitàtuttogiusto *= 1 - self.betamardia
                    self.alfaveromardia = None
                    self.tempoalfaveromardia = None
                except Exception as e:
                    self.betamardia = None
                    self.tempobetamardia = None
                    self.alfaveromardia = None
                    self.tempoalfaveromardia = None
                    self.probabilitàtuttogiusto = None
                    self.impostazioni["calcolaerrori"] = False
                    print(
                        f"\033[35mNon ho potuto calcolare il beta del test di multinormalità perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
        else:
            self.alfaveromardia = None
            self.tempoalfaveromardia = None
            self.betamardia = None
            self.tempobetamardia = None
            self.etichettefrecce = None

    def rieseguialfaveromardia(self, iterazioni: int, tipotest: str):
        if iterazioni <= 0 or (tipotest != "henze-zirkler" and tipotest != "mardia"):
            raise Exception("Valori dei parametri sbagliati")
        if self.mardia is not None and (self.impostazioni["testhotelling"] or self.impostazioni["intervallohotelling"]
        or self.impostazioni["hotellingbayesiano"] or self.impostazioni["intervallobayesiano"]) and self.taglia > 2:
            if not self.mardia and self.taglia < 50 and self.impostazioni["calcolaerrori"]:
                self.impostazioni["voltealfaveromardia"] = iterazioni
                self.impostazioni["tipotestnormalità"] = tipotest
                if self.alfaveromardia is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1-self.alfaveromardia
                if self.tempoalfaveromardia is not None:
                    self.tempototale -= self.tempoalfaveromardia
                try:
                    tempo = time()
                    self.alfaveromardia = alfaveromardia(frecce=self.taglia,
                                                         iterazioni=iterazioni,
                                                         alfa=self.impostazioni["alfamardia"],
                                                         tipotest=tipotest)
                    self.tempoalfaveromardia = time() - tempo
                    self.tempototale += self.tempoalfaveromardia
                    self.probabilitàtuttogiusto *= 1 - self.alfaveromardia
                    self.betamardia = None
                    self.tempobetamardia = None
                except Exception as e:
                    self.alfaveromardia = None
                    self.tempoalfaveromardia = None
                    self.betamardia = None
                    self.tempobetamardia = None
                    self.probabilitàtuttogiusto = None
                    self.impostazioni["calcolaerrori"] = False
                    print(
                        f"\033[35mNon ho potuto calcolare l'alfa effettivo del test di multinormalità perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
        else:
            self.alfaveromardia = None
            self.tempoalfaveromardia = None
            self.betamardia = None
            self.tempobetamardia = None

    def rieseguiclustering(self, inizializzazioneclustering: str, iterazionikmeans: int, componentimassime: int,
                           selezionemodello: str, numerobootstrapclustering: int, alfabootstrapclustering: float,
                           iterazioniinizializzazioneclustering: int, iterazioniEM: int, gaussianeclustering: bool,
                           algoritmoclustering: str, convergenzaclustering: float, criterioclustering: str,
                           bootstrapinterniclustering: int, foldclustering: int):
        if inizializzazioneclustering not in {"kmeans", "kmedoidi", "casuale"}:
            raise Exception("Valori dei parametri sbagliati")
        if iterazionikmeans <= 0 or componentimassime < 2:
            raise Exception("Valori dei parametri sbagliati")
        if selezionemodello not in {"mode", "criterio", "bootstrap", "doppiobootstrap", "crossvalidation"}:
            raise Exception("Valori dei parametri sbagliati")
        if (numerobootstrapclustering <= 1 or alfabootstrapclustering <= 0 or alfabootstrapclustering >= 1
            or iterazioniinizializzazioneclustering <= 0 or iterazioniEM <= 0):
            raise Exception("Valori dei parametri sbagliati")
        if (algoritmoclustering not in {"EM", "SEM", "CEM"} or convergenzaclustering <= 0
            or criterioclustering not in {"AIC", "BIC", "ICL"} or bootstrapinterniclustering <= 1 or foldclustering <= 1):
            raise Exception("Valori dei parametri sbagliati")
        self.impostazioni["clustering"] = True
        if self.taglia > 2:
            self.impostazioni["inizializzazioneclustering"] = inizializzazioneclustering
            self.impostazioni["iterazionikmeans"] = iterazionikmeans
            self.impostazioni["componentimassime"] = componentimassime
            self.impostazioni["selezionemodello"] = selezionemodello
            self.impostazioni["numerobootstrapclustering"] = numerobootstrapclustering
            self.impostazioni["alfabootstrapclustering"] = alfabootstrapclustering
            self.impostazioni["iterazioniinizializzazioneclustering"] = iterazioniinizializzazioneclustering
            self.impostazioni["iterazioniEM"] = iterazioniEM
            self.impostazioni["gaussianeclustering"] = gaussianeclustering
            self.impostazioni["algoritmoclustering"] = algoritmoclustering
            self.impostazioni["convergenzaclustering"] = convergenzaclustering
            self.impostazioni["criterioclustering"] = criterioclustering
            self.impostazioni["bootstrapinterniclustering"] = bootstrapinterniclustering
            self.impostazioni["foldclustering"] = foldclustering
            if self.tempoclustering is not None:
                self.tempototale -= self.tempoclustering
            try:
                tempo = time()
                self.etichettefrecce = miomodulo.clustering(inizializzazioneclustering,
                                                            self.taglia,
                                                            self.depura()[:, 0].tolist(),
                                                            self.depura()[:, 1].tolist(),
                                                            iterazionikmeans,
                                                            componentimassime,
                                                            selezionemodello,
                                                            numerobootstrapclustering,
                                                            alfabootstrapclustering,
                                                            iterazioniinizializzazioneclustering,
                                                            iterazioniEM,
                                                            gaussianeclustering,
                                                            algoritmoclustering,
                                                            convergenzaclustering,
                                                            criterioclustering,
                                                            bootstrapinterniclustering,
                                                            foldclustering)
                graficocluster(self.depura(), self.etichettefrecce, self.data)
                self.tempoclustering = time() - tempo
            except Exception as e:
                self.etichettefrecce = None
                self.tempoclustering = None
                print(f"\033[35mNon ho potuto completare il clustering o disegnarne il grafico perché {e}\033[0m")
        else:
            self.etichettefrecce = None
            self.tempoclustering = None

    def rieseguitesthotelling(self, alfa: float):
        if alfa <= 0 or alfa >= 1:
            raise Exception("Valori dei parametri sbagliati")
        self.impostazioni["testhotelling"] = True
        if self.taglia > 2:
            if self.ljungboxascisse is None:
                self.rieseguiljungboxascisse(self.impostazioni["alfaljungboxascisse"], self.impostazioni["hljungboxascisse"], True)
            if self.ljungboxordinate is None:
                self.rieseguiljungboxordinate(self.impostazioni["alfaljungboxordinate"], self.impostazioni["hljungboxordinate"], True)
            if self.mardia is None:
                self.rieseguimardia(self.impostazioni["alfamardia"], self.impostazioni["tipotestnormalità"])
            if self.betatesthotelling is not None and self.probabilitàtuttogiusto is not None:
                self.probabilitàtuttogiusto /= 1-self.betatesthotelling
                self.betatesthotelling = None
            elif not (self.testhotelling if self.testhotelling is not None else True) and self.probabilitàtuttogiusto is not None:
                self.probabilitàtuttogiusto /= 1-self.impostazioni["alfahotelling"]
            self.impostazioni["alfahotelling"] = alfa
            if self.tempotesthotelling is not None:
                self.tempototale -= self.tempotesthotelling
            if self.tempobetatesthotelling is not None:
                self.tempototale -= self.tempobetatesthotelling
                self.tempobetatesthotelling = None
            try:
                tempo = time()
                self.testhotelling = miomodulo.testhotelling(self.taglia,
                                                             self.depura()[:, 0].tolist(), self.depura()[:, 1].tolist(),
                                                             st.f.ppf(1 - alfa, 2, self.taglia - 2))
                self.tempotesthotelling = time() - tempo
                self.tempototale += self.tempotesthotelling
            except Exception as e:
                self.testhotelling = None
                self.tempotesthotelling = None
                print(f"\033[35mNon ho potuto eseguire il test di Hotelling perché {e}\033[0m")
            if self.testhotelling is not None:
                if self.testhotelling and self.impostazioni["calcolaerrori"]:
                    try:
                        tempo = time()
                        self.betatesthotelling = miomodulo.betatesthotelling(self.taglia,
                                                                             st.f.ppf(1 - alfa, 2, self.taglia - 2),
                                                                             self.impostazioni["voltebetahotelling"],
                                                                             self.impostazioni["distanzabetahotelling"],
                                                                             (self.varianzacoordinate[0]
                                                                                              if self.varianzacoordinate is not None
                                                                                              else np.var(
                                                                                 appiattisci(self.ascisse), ddof=1)),
                                                                             (self.varianzacoordinate[1]
                                                                                               if self.varianzacoordinate is not None
                                                                                               else np.var(
                                                                                 appiattisci(self.ordinate), ddof=1)))
                        self.tempobetatesthotelling = time() - tempo
                        self.tempototale += self.tempobetatesthotelling
                        self.probabilitàtuttogiusto *= 1 - self.betatesthotelling
                    except Exception as e:
                        self.betatesthotelling = None
                        self.tempobetatesthotelling = None
                        self.probabilitàtuttogiusto = None
                        self.impostazioni["calcolaerrori"] = False
                        print(f"\033[35mNon ho potuto calcolare il beta del test di Hotelling perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
                else:
                    self.betatesthotelling = None
                    self.tempobetatesthotelling = None
                    if self.impostazioni["calcolaerrori"]:
                        self.probabilitàtuttogiusto *= 1 - alfa
            else:
                self.testhotelling = None
                self.tempotesthotelling = None
                self.betatesthotelling = None
                self.tempobetatesthotelling = None
            if self.impostazioni["clustering"]:
                if self.probabilitàtestcluster is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= self.probabilitàtestcluster
                if self.tempotestcluster is not None:
                    self.tempototale -= self.tempotestcluster
                if self.tempoprobabilitàtestcluster is not None:
                    self.tempototale -= self.tempoprobabilitàtestcluster
                try:
                    tempo = time()
                    self.testcluster = testcluster(etichette=self.etichettefrecce,
                                                    allenamento=self.depura(),
                                                    alfa=alfa,
                                                    frecce=self.taglia)
                    self.tempotestcluster = time() - tempo
                    self.tempototale += self.tempotestcluster
                except Exception as e:
                    self.testcluster = None
                    self.tempotestcluster = None
                    print(f"\033[35mNon ho potuto eseguire il test di Hotelling sui cluster perché {e}\033[0m")
                if self.impostazioni["calcolaerrori"] and self.testcluster is not None:
                    try:
                        tempo = time()
                        self.probabilitàtestcluster = 1
                        for test in range(len(self.testcluster)):
                            numerositàcluster = sum((1 if numero == test else 0) for numero in self.etichettefrecce)
                            if self.testcluster[test]:
                                self.probabilitàtestcluster *= 1 - miomodulo.betatesthotelling(numerositàcluster,
                                                                                               st.f.ppf(1 - alfa, 2, self.taglia-2),
                                                                                               self.impostazioni["voltebetahotelling"],
                                                                                               self.impostazioni["distanzabetahotelling"],
                                                                                               self.varianzacoordinate[0] if self.varianzacoordinate is not None else np.var(appiattisci(self.ascisse), ddof=1),
                                                                                               self.varianzacoordinate[1] if self.varianzacoordinate is not None else np.var(appiattisci(self.ordinate), ddof=1))
                            else:
                                self.probabilitàtestcluster *= 1 - alfa
                        self.probabilitàtuttogiusto *= self.probabilitàtestcluster
                        self.tempoprobabilitàtestcluster = time() - tempo
                        self.tempototale += self.tempoprobabilitàtestcluster
                    except Exception as e:
                        self.probabilitàtestcluster = None
                        self.tempoprobabilitàtestcluster = None
                        self.impostazioni["calcolaerrori"] = False
                        print(f"\033[35mNon ho potuto calcolare le probabilità di errore dei test sui cluster perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
                else:
                    self.probabilitàtestcluster = None
                    self.tempoprobabilitàtestcluster = None
            else:
                self.testcluster = None
                self.tempotestcluster = None
                self.probabilitàtestcluster = None
                self.tempoprobabilitàtestcluster = None

    def rieseguibetatesthotelling(self, voltebetahotelling: int, distanzabetahotelling: float):
        if voltebetahotelling <= 0:
            raise Exception("Valori dei parametri sbagliati")
        if self.testhotelling is not None:
            if self.testhotelling and self.impostazioni["calcolaerrori"]:
                if self.betatesthotelling is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1-self.betatesthotelling
                if self.tempobetatesthotelling is not None:
                    self.tempototale -= self.tempobetatesthotelling
                self.impostazioni["voltebetahotelling"] = voltebetahotelling
                self.impostazioni["distanzabetahotelling"] = distanzabetahotelling
                try:
                    tempo = time()
                    self.betatesthotelling = miomodulo.betatesthotelling(self.taglia,
                                                                         st.f.ppf(1 - self.impostazioni["alfahotelling"], 2,
                                                                                  self.taglia - 2),
                                                                         voltebetahotelling,
                                                                         distanzabetahotelling,
                                                                         (self.varianzacoordinate[0]
                                                                          if self.varianzacoordinate is not None
                                                                          else np.var(
                                                                             appiattisci(self.ascisse), ddof=1)),
                                                                         (self.varianzacoordinate[1]
                                                                          if self.varianzacoordinate is not None
                                                                          else np.var(
                                                                             appiattisci(self.ordinate), ddof=1)))
                    self.tempobetatesthotelling = time() - tempo
                    self.probabilitàtuttogiusto *= 1 - self.betatesthotelling
                except Exception as e:
                    self.betatesthotelling = None
                    self.tempobetatesthotelling = None
                    self.probabilitàtuttogiusto = None
                    self.impostazioni["calcolaerrori"] = False
                    print(
                        f"\033[35mNon ho potuto calcolare il beta del test di Hotelling perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
            if self.impostazioni["calcolaerrori"] and self.testcluster is not None:
                if self.probabilitàtestcluster is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= self.probabilitàtestcluster
                if self.tempoprobabilitàtestcluster is not None:
                    self.tempototale -= self.tempoprobabilitàtestcluster
                self.impostazioni["voltebetahotelling"] = voltebetahotelling
                self.impostazioni["distanzabetahotelling"] = distanzabetahotelling
                try:
                    tempo = time()
                    self.probabilitàtestcluster = 1
                    for test in range(len(self.testcluster)):
                        numerositàcluster = sum((1 if numero == test else 0) for numero in self.etichettefrecce)
                        if self.testcluster[test]:
                            self.probabilitàtestcluster *= 1 - miomodulo.betatesthotelling(numerositàcluster,
                                                                                           st.f.ppf(1 - self.impostazioni["alfahotelling"], 2,
                                                                                                    self.taglia - 2),
                                                                                           voltebetahotelling,
                                                                                           distanzabetahotelling,
                                                                                           self.varianzacoordinate[
                                                                                               0] if self.varianzacoordinate is not None else np.var(
                                                                                               appiattisci(
                                                                                                   self.ascisse),
                                                                                               ddof=1),
                                                                                           self.varianzacoordinate[
                                                                                               1] if self.varianzacoordinate is not None else np.var(
                                                                                               appiattisci(
                                                                                                   self.ordinate),
                                                                                               ddof=1))
                        else:
                            self.probabilitàtestcluster *= 1 - self.impostazioni["alfahotelling"]
                    self.probabilitàtuttogiusto *= self.probabilitàtestcluster
                    self.tempoprobabilitàtestcluster = time() - tempo
                except Exception as e:
                    self.probabilitàtestcluster = None
                    self.tempoprobabilitàtestcluster = None
                    self.impostazioni["calcolaerrori"] = False
                    print(
                        f"\033[35mNon ho potuto calcolare le probabilità di errore dei test sui cluster perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
            else:
                self.probabilitàtestcluster = None
                self.tempoprobabilitàtestcluster = None

    def rieseguiintervallohotelling(self, confidenzahotelling: float, correggidipendenzahotelling: str,
                                    geometricastazionariahotelling: float, bootstrapstazionarihotelling: int,
                                    varianzaascissebootstrapstazionariohotelling: float,
                                    varianzaordinatebootstrapstazionariohotelling: float,
                                    iterazionistazionariehotelling: int, bootstrapmobilihotelling: int,
                                    varianzaascissebootstrapmobilehotelling: float,
                                    varianzaordinatebootstrapmobilehotelling: float,
                                    lunghezzabloccomobilehotelling: int,
                                    iterazionimobilihotelling: int):
        if confidenzahotelling <= 0 or confidenzahotelling >= 1:
            raise Exception("Valori dei parametri sbagliati")
        if correggidipendenzahotelling not in {"regolare", "stazionario", "mobile", "detrending"}:
            raise Exception("Valori dei parametri sbagliati")
        if geometricastazionariahotelling <= 0 or geometricastazionariahotelling >= 1:
            raise Exception("Valori dei parametri sbagliati")
        if bootstrapstazionarihotelling <= 0:
            raise Exception("Valori dei parametri sbagliati")
        if varianzaascissebootstrapstazionariohotelling <= 0 or varianzaordinatebootstrapstazionariohotelling <= 0 or iterazionistazionariehotelling <= 0:
            raise Exception("Valori dei parametri sbagliati")
        if lunghezzabloccomobilehotelling <= 0 or bootstrapmobilihotelling <= 0:
            raise Exception("Valori dei parametri sbagliati")
        if varianzaascissebootstrapmobilehotelling <= 0 or varianzaordinatebootstrapmobilehotelling <= 0:
            raise Exception("Valori dei parametri sbagliati")
        self.impostazioni["intervallohotelling"] = True
        if self.taglia > 2:
            if self.ljungboxascisse is None:
                self.rieseguiljungboxascisse(self.impostazioni["alfaljungboxascisse"], self.impostazioni["hljungboxordinate"], True)
            if self.ljungboxordinate is None:
                self.rieseguiljungboxordinate(self.impostazioni["alfaljungboxordinate"], self.impostazioni["hljungboxordinate"], True)
            if self.mardia is None:
                self.rieseguimardia(self.impostazioni["alfamardia"], self.impostazioni["tipotestnormalità"])
            if (esiste(f"./Grafici del {self.data}/graficointervallohotelling.png") or esiste(f"./Grafici del {self.data}/graficointervallohotellingdetrending.png")) and self.probabilitàtuttogiusto is not None:
                self.probabilitàtuttogiusto /= self.impostazioni["confidenzahotelling"]
            if self.confidenzaverabootstrapstazionariohotelling is not None and self.probabilitàtuttogiusto is not None:
                self.probabilitàtuttogiusto /= self.confidenzaverabootstrapstazionariohotelling
                self.confidenzaverabootstrapstazionariohotelling = None
            if self.confidenzaverabootstrapmobilehotelling is not None and self.probabilitàtuttogiusto is not None:
                self.probabilitàtuttogiusto /= self.confidenzaverabootstrapmobilehotelling
                self.confidenzaverabootstrapmobilehotelling = None
            self.impostazioni["confidenzahotelling"] = confidenzahotelling
            self.impostazioni["correggidipendenzahotelling"] = correggidipendenzahotelling
            self.impostazioni["geometricastazionariahotelling"] = geometricastazionariahotelling
            self.impostazioni["bootstrapstazionarihotelling"] = bootstrapstazionarihotelling
            self.impostazioni["varianzaascissebootstrapstazionariohotelling"] = varianzaascissebootstrapstazionariohotelling
            self.impostazioni["varianzaordinatebootstrapstazionariohotelling"] = varianzaordinatebootstrapstazionariohotelling
            self.impostazioni["iterazionistazionariehotelling"] = iterazionistazionariehotelling
            self.impostazioni["bootstrapmobilihotelling"] = bootstrapmobilihotelling
            self.impostazioni["varianzaascissebootstrapmobilehotelling"] = varianzaascissebootstrapmobilehotelling
            self.impostazioni["varianzaordinatebootstrapmobilehotelling"] = varianzaordinatebootstrapmobilehotelling
            self.impostazioni["lunghezzabloccomobilehotelling"] = lunghezzabloccomobilehotelling
            self.impostazioni["iterazionimobilihotelling"] = iterazionimobilihotelling
            if self.tempointervallohotelling is not None:
                self.tempototale -= self.tempointervallohotelling
            if self.tempoconfidenzaverabootstrapstazionariohotelling is not None:
                self.tempototale -= self.tempoconfidenzaverabootstrapstazionariohotelling
                self.tempoconfidenzaverabootstrapstazionariohotelling = None
            if self.tempoconfidenzaverabootstrapmobilehotelling is not None:
                self.tempototale -= self.tempoconfidenzaverabootstrapmobilehotelling
                self.tempoconfidenzaverabootstrapmobilehotelling = None
            try:
                tempo = time()
                if correggidipendenzahotelling == "regolare":
                    intervallo_hotelling = intervallohotelling(n=self.taglia,
                                                                   confidenza=confidenzahotelling,
                                                                   varianza=np.cov(self.depura(),
                                                                                   rowvar=False, bias=False),
                                                                   media=self.mediacoordinate
                                                                   if self.mediacoordinate is not None
                                                                   else (np.mean(appiattisci(self.ascisse)),
                                                                         np.mean(appiattisci(self.ordinate))))
                    graficointervallohotelling(intervallo_hotelling[0], intervallo_hotelling[1], self.data, "regolare", None)
                if correggidipendenzahotelling == "stazionario":
                    intervallostazionario = intervallohotelling(n=self.taglia,
                                                                confidenza=confidenzahotelling,
                                                                varianza=np.cov(self.depura(), rowvar=False,
                                                                                bias=False),
                                                                media=self.mediacoordinate if self.mediacoordinate is not None
                                                                else (np.mean(appiattisci(self.ascisse)),
                                                                      np.mean(appiattisci(self.ordinate))),
                                                                costante=miomodulo.bootstrapstazionariohotelling(
                                                                    self.taglia, appiattisci(self.ascisse), appiattisci(self.ordinate),
                                                                    geometricastazionariahotelling,
                                                                    bootstrapstazionarihotelling,
                                                                    np.mean(appiattisci(self.ascisse)),
                                                                    np.mean(appiattisci(self.ordinate)),
                                                                    1-confidenzahotelling)
                                                                )
                    graficointervallohotelling(intervallostazionario[0], intervallostazionario[1], self.data, "stazionario", None)
                else:
                    intervallostazionario = None
                if correggidipendenzahotelling == "mobile":
                    intervallomobile = intervallohotelling(n=self.taglia,
                                                                confidenza=confidenzahotelling,
                                                                varianza=np.cov(self.depura(), rowvar=False,
                                                                                bias=False),
                                                                media=self.mediacoordinate if self.mediacoordinate is not None
                                                                else (np.mean(appiattisci(self.ascisse)),
                                                                      np.mean(appiattisci(self.ordinate))),
                                                                costante=miomodulo.bootstrapmobilehotelling(
                                                                    self.taglia, appiattisci(self.ascisse), appiattisci(self.ordinate),
                                                                    lunghezzabloccomobilehotelling,
                                                                    bootstrapmobilihotelling,
                                                                    np.mean(appiattisci(self.ascisse)),
                                                                    np.mean(appiattisci(self.ordinate)),
                                                                    1-confidenzahotelling)
                                                                )
                    graficointervallohotelling(intervallomobile[0], intervallomobile[1], self.data, "mobile", None)
                else:
                    intervallomobile = None
                if correggidipendenzahotelling == "detrending":
                    if self.regressionecoordinate is None:
                        if not self.ordine:
                            rc = (np.polyfit([i for i in range(len(self.ascisse))],
                                                                ([media[0] for media in self.mediacoordinatevolée])
                                                                if self.mediacoordinatevolée is not None
                                                                else np.mean(self.ascisse, axis=0), 1),
                                                     np.polyfit([i for i in range(len(self.ordinate))],
                                                                ([media[1] for media in self.mediacoordinatevolée])
                                                                if self.mediacoordinatevolée is not None
                                                                else np.mean(self.ordinate, axis=0), 1))
                        else:
                            rc = (np.polyfit([i for i in range(len(self.ascisse))],
                                                                appiattisci(self.ascisse), 1),
                                                     np.polyfit([i for i in range(len(self.ordinate))],
                                                                appiattisci(self.ordinate), 1))
                    else:
                        rc = self.regressionecoordinate
                    va, vo, c, _, _ = miomodulo.detrenda(appiattisci(self.ascisse), appiattisci(self.ordinate),
                                                           rc[0][0], rc[0][1], rc[1][0], rc[1][1])
                    intervallodetrendato = intervallohotelling(n=self.taglia,
                                                                confidenza=confidenzahotelling,
                                                                varianza=np.array([[va, c], [c, vo]]),
                                                                media=(rc[0][1]+rc[0][0], rc[1][1]+rc[1][0])
                                                               )
                    graficointervallohotelling(intervallodetrendato[0], intervallodetrendato[1], self.data, "detrending", rc)
                else:
                    intervallodetrendato = None
                self.tempointervallohotelling = time() - tempo
                self.tempototale += self.tempointervallohotelling
            except Exception as e:
                intervallo_hotelling = None
                intervallostazionario = None
                intervallomobile = None
                intervallodetrendato = None
                self.tempointervallohotelling = None
                print(f"\033[35mNon ho potuto calcolare o disegnare l'intervallo di confidenza bivariato perché {e}\033[0m")
            if self.impostazioni["calcolaerrori"] and (intervallo_hotelling is not None or intervallodetrendato is not None):
                self.probabilitàtuttogiusto *= self.impostazioni["confidenzahotelling"]
            if self.impostazioni["calcolaerrori"] and intervallostazionario is not None:
                tempo = time()
                self.confidenzaverabootstrapstazionariohotelling = 1.0-miomodulo.alfaverobootstrapstazionariohotelling(self.taglia,
                                                                                                   varianzaascissebootstrapstazionariohotelling,
                                                                                                   varianzaordinatebootstrapstazionariohotelling,
                                                                                                   geometricastazionariahotelling,
                                                                                                   bootstrapstazionarihotelling,
                                                                                                   1.0-confidenzahotelling,
                                                                                                   iterazionistazionariehotelling)
                self.tempoconfidenzaverabootstrapstazionariohotelling = time()-tempo
                self.probabilitàtuttogiusto *= self.confidenzaverabootstrapstazionariohotelling
            else:
                self.confidenzaverabootstrapstazionariohotelling = None
            if self.impostazioni["calcolaerrori"] and intervallomobile is not None:
                tempo = time()
                self.confidenzaverabootstrapmobilehotelling = 1.0-miomodulo.alfaverobootstrapmobilehotelling(self.taglia,
                                                                                                             varianzaascissebootstrapmobilehotelling,
                                                                                                             varianzaordinatebootstrapmobilehotelling,
                                                                                                             lunghezzabloccomobilehotelling,
                                                                                                             bootstrapmobilihotelling,
                                                                                                             1.0-confidenzahotelling,
                                                                                                             iterazionimobilihotelling)
                self.tempoconfidenzaverabootstrapmobilehotelling = time()-tempo
                self.probabilitàtuttogiusto *= self.confidenzaverabootstrapmobilehotelling
            else:
                self.confidenzaverabootstrapmobilehotelling = None
            if self.impostazioni["clustering"]:
                if esiste(f"./Grafici del {self.data}/graficointervallicluster.png") and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= self.impostazioni["confidenzahotelling"]**(max(self.etichettefrecce)+1)
                if self.tempointervallicluster is not None:
                    self.tempototale -= self.tempointervallicluster
                try:
                    tempo = time()
                    intervalli_cluster = intervallicluster(etichette=self.etichettefrecce,
                                                               allenamento=self.depura(),
                                                               alfa=confidenzahotelling)
                    self.tempointervallicluster = time() - tempo
                    tempo = time()
                    graficointervallicluster(intervalli=intervalli_cluster, data=self.data)
                    self.tempograficointervallicluster = time() - tempo
                except Exception as e:
                    intervalli_cluster = None
                    self.tempointervallicluster = None
                    self.tempograficointervallicluster = None
                    print(f"\033[35mNon ho potuto calcolare o disegnare gli intervalli di confidenza bivariati per i cluster perché {e}\033[0m")
                if self.impostazioni["calcolaerrori"] and intervalli_cluster is not None:
                    self.probabilitàtuttogiusto *= confidenzahotelling**(max(self.etichettefrecce)+1)
            else:
                intervalli_cluster = None
                self.tempointervallicluster = None
                self.tempograficointervallicluster = None
        else:
            intervallo_hotelling = None
            self.tempointervallohotelling = None
            intervalli_cluster = None
            self.tempointervallicluster = None
            self.tempograficointervallicluster = None

    def rieseguiintervallobayesiano(self, catenehotelling: int, iterazionimcmchotelling: int, ROPEhotelling: float,
                                    confidenzahotellingbayesiano: float, noninformativitàhotellingbayesiano: bool,
                                    allenamentipriorihotellingbayesiano: int, apiacerehotelling: bool, aapiacerehotelling: float,
                                    bapiacerehotelling: float, etaapiacerehotelling: float, hotellinggerarchico: bool,
                                    intervallohotellingpreciso: bool, lambdaapiacerehotelling: list[list[float]],
                                    muapiacerehotelling: list[float], consoledistan: bool):
        if (catenehotelling <= 0 or iterazionimcmchotelling <= 0 or ROPEhotelling < 0 or confidenzahotellingbayesiano <= 0
            or confidenzahotellingbayesiano >= 1 or aapiacerehotelling <= 0 or bapiacerehotelling <= 0 or etaapiacerehotelling <= 0):
            raise Exception("Valori dei parametri sbagliati")
        if allenamentipriorihotellingbayesiano > len(leggifile("Date.txt")) or allenamentipriorihotellingbayesiano < 0:
            raise Exception("Valori dei parametri sbagliati")
        if allenamentipriorihotellingbayesiano == 0 and not noninformativitàhotellingbayesiano:
            raise Exception("Valori dei parametri sbagliati")
        if np.array(lambdaapiacerehotelling).shape != (2, 2):
            raise Exception("Valori dei parametri sbagliati")
        if lambdaapiacerehotelling[0][0] < 0 or lambdaapiacerehotelling[1][1] < 0:
            raise Exception("Valori dei parametri sbagliati")
        if lambdaapiacerehotelling[0][0]*lambdaapiacerehotelling[1][1]-lambdaapiacerehotelling[0][1]*lambdaapiacerehotelling[1][0] <= 0:
            raise Exception("Valori dei parametri sbagliati")
        self.impostazioni["hotellingbayesiano"] = True
        self.impostazioni["intervallobayesiano"] = True
        if self.taglia > 2:
            if self.hotellingbayesiano is not None and self.probabilitàtuttogiusto is not None:
                self.probabilitàtuttogiusto /= 1-min([self.hotellingbayesiano, 1-self.hotellingbayesiano])
            if esiste(f"./Grafici del {self.data}/graficohotellingbayesiano.png") and self.probabilitàtuttogiusto is not None:
                self.probabilitàtuttogiusto /= self.impostazioni["confidenzahotellingbayesiano"]
            self.impostazioni["catenehotelling"] = catenehotelling
            self.impostazioni["iterazionimcmchotelling"] = iterazionimcmchotelling
            self.impostazioni["ROPEhotelling"] = ROPEhotelling
            self.impostazioni["confidenzahotellingbayesiano"] = confidenzahotellingbayesiano
            self.impostazioni["noninformativitàhotellingbayesiano"] = noninformativitàhotellingbayesiano
            self.impostazioni["allenamentipriorihotellingbayesiano"] = allenamentipriorihotellingbayesiano
            self.impostazioni["apiacerehotelling"] = apiacerehotelling
            self.impostazioni["aapiacerehotelling"] = aapiacerehotelling
            self.impostazioni["bapiacerehotelling"] = bapiacerehotelling
            self.impostazioni["etaapiacerehotelling"] = etaapiacerehotelling
            self.impostazioni["hotellinggerarchico"] = hotellinggerarchico
            self.impostazioni["intervallohotellingpreciso"] = intervallohotellingpreciso
            self.impostazioni["lambdaapiacerehotelling"] = lambdaapiacerehotelling
            self.impostazioni["muapiacerehotelling"] = muapiacerehotelling
            self.impostazioni["consoledistan"] = consoledistan
            if self.tempohotellingbayesiano is not None:
                self.tempototale -= self.tempohotellingbayesiano
            try:
                tempo = time()
                self.hotellingbayesiano, intervallobayesiano = hotellingbayesiano(allenamento=self.depura(),
                                                                 numerocatene=catenehotelling,
                                                                 iterazioni=iterazionimcmchotelling,
                                                                 distanza=ROPEhotelling,
                                                                 confidenza=confidenzahotellingbayesiano,
                                                                 noninformativa=noninformativitàhotellingbayesiano,
                                                                 allenamentipriori=allenamentipriorihotellingbayesiano,
                                                                 test=True,
                                                                 apiacere=apiacerehotelling,
                                                                 aapiacere=aapiacerehotelling,
                                                                 bapiacere=bapiacerehotelling,
                                                                 etaapiacere=etaapiacerehotelling,
                                                                 gerarchica=hotellinggerarchico,
                                                                 intervallopreciso=intervallohotellingpreciso,
                                                                 intervallo=True,
                                                                 lambdaapiacere=lambdaapiacerehotelling,
                                                                 muapiacere=muapiacerehotelling,
                                                                 console=consoledistan)
                graficohotellingbayesiano(intervallobayesiano, self.data)
                self.tempohotellingbayesiano = time()-tempo
                self.tempototale += self.tempohotellingbayesiano
                if self.impostazioni["calcolaerrori"]:
                    self.probabilitàtuttogiusto *= confidenzahotellingbayesiano
                if self.impostazioni["calcolaerrori"]:
                    self.probabilitàtuttogiusto *= 1 - min([self.hotellingbayesiano, 1 - self.hotellingbayesiano])
            except Exception as e:
                self.hotellingbayesiano = None
                self.tempohotellingbayesiano = None
                intervallobayesiano = None
                print(f"\033[35mNon ho potuto calcolare l'intervallo di credibilità bayesiano perché {e}\033[0m")

    def rieseguimedianageometrica(self, iterazioni: int, gammamediana: list[float], distanzaverticale: float, coccamirino: float):
        if iterazioni <= 0:
            raise Exception("Valori dei parametri sbagliati")
        if not all(0 < gamma < 180 for gamma in gammamediana):
            raise Exception("Valori dei parametri sbagliati")
        if distanzaverticale <= 0 or coccamirino <= 0:
            raise Exception("Valori dei parametri sbagliati")
        self.impostazioni["medianageometrica"] = True
        try:
            if self.tempomedianageometrica is not None:
                self.tempototale -= self.tempomedianageometrica
            tempo = time()
            self.medianageometrica = medianageometrica(iterazioni=iterazioni,
                                                       media=(np.array(self.mediacoordinate)
                                                              if self.mediacoordinate is not None
                                                              else np.array([np.mean(appiattisci(self.ascisse)),
                                                                             np.mean(appiattisci(self.ordinate))])),
                                                       allenamento=self.depura(),
                                                       soglia=0.0001)
            self.mirinoideale = mirinoideale(gamma=gammamediana,
                                             distanza=[self.distanza, distanzaverticale],
                                             coccamirino=coccamirino,
                                             mediane=self.medianageometrica.tolist())
            graficomedianageometrica(self.medianageometrica, self.data)
            self.tempomedianageometrica = time() - tempo
            self.tempototale += self.tempomedianageometrica
        except Exception as e:
            self.medianageometrica = None
            self.tempomedianageometrica = None
            print(f"\033[35mNon ho potuto calcolare la mediana geometrica perché {e}\033[0m")

    def rieseguinormenormali(self, alfa: float):
        if alfa <= 0 or alfa >= 1:
            raise Exception("Valori dei parametri sbagliati")
        if (self.impostazioni["intervallonorme"] or self.impostazioni["testnorme"]) and self.taglia > 1:
            if self.betanormenormali is not None and self.probabilitàtuttogiusto is not None:
                self.probabilitàtuttogiusto /= 1-self.betanormenormali
                self.betanormenormali = None
            elif self.alfaveronormenormali is not None and self.probabilitàtuttogiusto is not None:
                self.probabilitàtuttogiusto /= 1-self.alfaveronormenormali
                self.alfaveronormenormali = None
            elif not (self.normenormali if self.normenormali is not None else True) and self.probabilitàtuttogiusto is not None:
                self.probabilitàtuttogiusto /= 1-self.impostazioni["alfashapiro"]
            self.impostazioni["alfashapiro"] = alfa
            if self.temponormenormali is not None:
                self.tempototale -= self.temponormenormali
            if self.tempobetanormenormali is not None:
                self.tempototale -= self.tempobetanormenormali
                self.tempobetanormenormali = None
            if self.tempoalfaveronormenormali is not None:
                self.tempototale -= self.tempoalfaveronormenormali
                self.tempoalfaveronormenormali = None
            tempo = time()
            self.normenormali = st.shapiro(appiattisci(self.norme)).pvalue > alfa
            self.temponormenormali = time() - tempo
            if self.normeindipendenti is None:
                self.affidabilitànorme = False
            elif self.normenormali and self.normeindipendenti:
                self.affidabilitànorme = True
            else:
                self.affidabilitànorme = False
            if self.normenormali and self.impostazioni["calcolaerrori"]:
                try:
                    tempo = time()
                    self.betanormenormali = betanormenormali(alternativa=self.impostazioni["alternativanormenormali"],
                                                             iterazioni=self.impostazioni["iterazionibetanormenormali"],
                                                             alfa=alfa,
                                                             frecce=self.taglia,
                                                             asimmetria=self.impostazioni["asimmetrianormenormali"],
                                                             distanzacomponenti=self.impostazioni["distanzacomponentinormenormali"],
                                                             gradit=self.impostazioni["graditnormenormali"])
                    self.tempobetanormenormali = time() - tempo
                    self.tempototale += self.tempobetanormenormali
                    self.probabilitàtuttogiusto *= 1 - self.betanormenormali
                    self.alfaveronormenormali = None
                    self.tempoalfaveronormenormali = None
                except Exception as e:
                    self.betanormenormali = None
                    self.tempobetanormenormali = None
                    self.alfaveronormenormali = None
                    self.tempoalfaveronormenormali = None
                    self.probabilitàtuttogiusto = None
                    self.impostazioni["calcolaerrori"] = False
                    print(f"\033[35mNon ho potuto calcolare il beta per il test di Shapiro perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
            elif self.taglia < 50 and self.impostazioni["calcolaerrori"]:
                try:
                    tempo = time()
                    self.alfaveronormenormali = alfaveronormenormali(iterazioni=self.impostazioni["iterazionialfaveronormenormali"],
                                                                     frecce=self.taglia,
                                                                     alfa=alfa)
                    self.tempoalfaveronormenormali = time() - tempo
                    self.tempototale += self.tempoalfaveronormenormali
                    self.probabilitàtuttogiusto *= 1 - self.alfaveronormenormali
                    self.betanormenormali = None
                    self.tempobetanormenormali = None
                except Exception as e:
                    self.alfaveronormenormali = None
                    self.tempoalfaveronormenormali = None
                    self.betanormenormali = None
                    self.tempobetanormenormali = None
                    self.probabilitàtuttogiusto = None
                    self.impostazioni["calcolaerrori"] = False
                    print(f"\033[35mNon ho potuto calcolare l'alfa effettivo per il test di Shapiro perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
            else:
                if self.impostazioni["calcolaerrori"]:
                    self.probabilitàtuttogiusto *= 1 - alfa
                self.betanormenormali = None
                self.tempobetanormenormali = None
                self.alfaveronormenormali = None
                self.tempoalfaveronormenormali = None

    def rieseguibetanormenormali(self, alternativanormenormali: str, iterazionibetanormenormali: int,
                                 asimmetrianormenormali: float, distanzacomponentinormenormali: float,
                                 graditnormenormali: int):
        if alternativanormenormali not in {"laplace", "normaleasimmetrica", "uniforme", "mistura", "lognormale", "t"}:
            raise Exception("Valori dei parametri sbagliati")
        if iterazionibetanormenormali <= 0 or graditnormenormali <= 0:
            raise Exception("Valori dei parametri sbagliati")
        if (self.impostazioni["intervallonorme"] or self.impostazioni["testnorme"]) and self.taglia > 1:
            if self.normenormali and self.impostazioni["calcolaerrori"]:
                if self.betanormenormali is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1-self.betanormenormali
                self.impostazioni["alternativanormenormali"] = alternativanormenormali
                self.impostazioni["iterazionibetanormenormali"] = iterazionibetanormenormali
                self.impostazioni["asimmetrianormenormali"] = asimmetrianormenormali
                self.impostazioni["distanzacomponentinormenormali"] = distanzacomponentinormenormali
                self.impostazioni["graditnormenormali"] = graditnormenormali
                if self.tempobetanormenormali is not None:
                    self.tempototale -= self.tempobetanormenormali
                try:
                    tempo = time()
                    self.betanormenormali = betanormenormali(alternativa=alternativanormenormali,
                                                             iterazioni=iterazionibetanormenormali,
                                                             alfa=self.impostazioni["alfashapiro"],
                                                             frecce=self.taglia,
                                                             asimmetria=asimmetrianormenormali,
                                                             distanzacomponenti=distanzacomponentinormenormali,
                                                             gradit=graditnormenormali)
                    self.tempobetanormenormali = time() - tempo
                    self.tempototale += self.tempobetanormenormali
                    self.probabilitàtuttogiusto *= 1 - self.betanormenormali
                    self.alfaveronormenormali = None
                    self.tempoalfaveronormenormali = None
                except Exception as e:
                    self.betanormenormali = None
                    self.tempobetanormenormali = None
                    self.alfaveronormenormali = None
                    self.tempoalfaveronormenormali = None
                    self.probabilitàtuttogiusto = None
                    self.impostazioni["calcolaerrori"] = False
                    print(f"\033[35mNon ho potuto calcolare il beta per il test di Shapiro perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")

    def rieseguialfaveronormenormali(self, iterazionialfaveronormenormali: int):
        if iterazionialfaveronormenormali <= 0:
            raise Exception("Valori dei parametri sbagliati")
        if (self.impostazioni["intervallonorme"] or self.impostazioni["testnorme"]) and self.taglia > 1:
            if not self.normenormali and self.taglia < 50 and self.impostazioni["calcolaerrori"]:
                if self.alfaveronormenormali is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1-self.alfaveronormenormali
                self.impostazioni["iterazionialfaveronormenormali"] = iterazionialfaveronormenormali
                if self.tempoalfaveronormenormali is not None:
                    self.tempototale -= self.tempoalfaveronormenormali
                try:
                    tempo = time()
                    self.alfaveronormenormali = alfaveronormenormali(iterazioni=iterazionialfaveronormenormali,
                                                                     frecce=self.taglia,
                                                                     alfa=self.impostazioni["alfashapiro"])
                    self.tempoalfaveronormenormali = time() - tempo
                    self.tempototale += self.tempoalfaveronormenormali
                    self.probabilitàtuttogiusto *= 1 - self.alfaveronormenormali
                    self.betanormenormali = None
                    self.tempobetanormenormali = None
                except Exception as e:
                    self.alfaveronormenormali = None
                    self.tempoalfaveronormenormali = None
                    self.betanormenormali = None
                    self.tempobetanormenormali = None
                    self.probabilitàtuttogiusto = None
                    self.impostazioni["calcolaerrori"] = False
                    print(f"\033[35mNon ho potuto calcolare l'alfa effettivo per il test di Shapiro perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")

    def rieseguinormeindipendenti(self, alfa: float, h: int):
        if alfa <= 0 or alfa >= 1 or h <= 0:
            raise Exception("Valori dei parametri sbagliati")
        if (self.impostazioni["intervallonorme"] or self.impostazioni["testnorme"]) and self.taglia > 1:
            cutoff3 = None
            soglie3 = None
            cutoffattuale3 = None
            try:
                if self.betanormeindipendenti is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1-self.betanormeindipendenti
                    self.betanormeindipendenti = None
                elif self.alfaveronormeindipendenti is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1-self.alfaveronormeindipendenti
                    self.alfaveronormeindipendenti = None
                elif not (self.normeindipendenti if self.normeindipendenti is not None else True) and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1-self.impostazioni["alfaljungboxnorme"]
                self.impostazioni["alfaljungboxnorme"] = alfa
                self.impostazioni["hljungboxnorme"] = h
                if self.temponormeindipendenti is not None:
                    self.tempototale -= self.temponormeindipendenti
                if self.tempobetanormeindipendenti is not None:
                    self.tempototale -= self.tempobetanormeindipendenti
                    self.tempobetanormeindipendenti = None
                if self.tempoalfaveronormeindipendenti is not None:
                    self.tempototale -= self.tempoalfaveronormeindipendenti
                    self.tempoalfaveronormeindipendenti = None
                tempo = time()
                aut = autocorrelazioni(
                    appiattisci(self.norme) if self.ordine else [np.mean(voléesingola) for voléesingola in self.norme],
                    np.var(appiattisci(self.norme)) if self.ordine else np.var(
                        [np.mean(voléesingola) for voléesingola in self.norme]))
                cutoff3 = list()
                soglie3 = list()
                if h < len(aut):
                    cutoffattuale3 = h
                    while cutoffattuale3 < len(aut):
                        cutoff3.append(cutoffattuale3)
                        soglie3.append(st.chi2.ppf(1 - alfa, df=cutoffattuale3))
                        cutoffattuale3 += h
                cutoff3.append(len(aut))
                soglie3.append(st.chi2.ppf(1 - alfa, df=len(aut)))
                self.normeindipendenti = miomodulo.ljungbox(aut.tolist(),
                                                            cutoff3, self.taglia if self.ordine else self.volée, soglie3,
                                                            len(cutoff3))
                self.temponormeindipendenti = time() - tempo
                self.tempototale += self.temponormeindipendenti
            except Exception as e:
                self.normeindipendenti = None
                self.temponormeindipendenti = None
                print(f"\033[35mNon ho potuto eseguire il test di Ljung-Box sulle norme perché {e}\033[0m")
            if self.normeindipendenti is None:
                self.affidabilitànorme = False
            elif self.normenormali and self.normeindipendenti:
                self.affidabilitànorme = True
            else:
                self.affidabilitànorme = False
            if (self.normeindipendenti if self.normeindipendenti is not None else False) and self.impostazioni[
                "calcolaerrori"]:
                try:
                    if cutoff3 is None or soglie3 is None:
                        raise TypeError("I cutoff o le soglie sono indeterminati")
                    tempo = time()
                    self.betanormeindipendenti = miomodulo.betaljungbox(self.taglia if self.ordine else self.volée,
                                                                        cutoff3, soglie3, len(cutoff3),
                                                                        self.impostazioni["voltebetaljungboxnorme"])
                    self.tempobetanormeindipendenti = time() - tempo
                    self.tempototale += self.tempobetanormeindipendenti
                    self.probabilitàtuttogiusto *= (1 - self.betanormeindipendenti)
                    self.alfaveronormeindipendenti = None
                    self.tempoalfaveronormeindipendenti = None
                except Exception as e:
                    self.betanormeindipendenti = None
                    self.tempobetanormeindipendenti = None
                    self.alfaveronormeindipendenti = None
                    self.tempoalfaveronormeindipendenti = None
                    self.probabilitàtuttogiusto = None
                    self.impostazioni["calcolaerrori"] = False
                    print(
                        f"\033[35mNon ho potuto calcolare il beta per il test di Ljung-Box sulle distanze dal centro perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
            elif (self.taglia if self.ordine else self.volée) < 50 and self.impostazioni[
                "calcolaerrori"] and self.normeindipendenti is not None:
                try:
                    if cutoff3 is None or soglie3 is None:
                        raise TypeError("I cutoff o le soglie sono indeterminati")
                    tempo = time()
                    self.alfaveronormeindipendenti = miomodulo.alfaveroljungbox(self.taglia if self.ordine else self.volée,
                                                                                cutoff3, soglie3, len(cutoff3),
                                                                                self.impostazioni["voltealfaljungboxnorme"])
                    self.tempoalfaveronormeindipendenti = time() - tempo
                    self.tempototale += self.tempoalfaveronormeindipendenti
                    self.probabilitàtuttogiusto *= (1 - self.alfaveronormeindipendenti)
                    self.betanormeindipendenti = None
                    self.tempobetanormeindipendenti = None
                except Exception as e:
                    self.betanormeindipendenti = None
                    self.tempobetanormeindipendenti = None
                    self.alfaveronormeindipendenti = None
                    self.tempoalfaveronormeindipendenti = None
                    self.impostazioni["calcolaerrori"] = False
                    print(
                        f"\033[35mNon ho potuto calcolare l'alfa effettivo per il test di Ljung-Box sulle distanze dal centro perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
            else:
                if self.impostazioni["calcolaerrori"] and self.normeindipendenti is not None:
                    self.probabilitàtuttogiusto *= (1 - alfa) ** len(cutoff3)
                self.betanormeindipendenti = None
                self.tempobetanormeindipendenti = None
                self.alfaveronormeindipendenti = None
                self.tempoalfaveronormeindipendenti = None

    def rieseguibetanormeindipendenti(self, voltebetaljungboxnorme: int):
        if voltebetaljungboxnorme <= 0:
            raise Exception("Valori dei parametri sbagliati")
        if (self.normeindipendenti if self.normeindipendenti is not None else False) and self.impostazioni["calcolaerrori"]:
            cutoff3 = None
            soglie3 = None
            cutoffattuale3 = None
            try:
                if self.betanormeindipendenti is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1-self.betanormeindipendenti
                self.impostazioni["voltebetaljungboxnorme"] = voltebetaljungboxnorme
                if self.tempobetanormeindipendenti is not None:
                    self.tempototale -= self.tempobetanormeindipendenti
                cutoff3 = list()
                soglie3 = list()
                if self.impostazioni["hljungboxnorme"] < self.taglia - 1:
                    cutoffattuale3 = self.impostazioni["hljungboxnorme"]
                    while cutoffattuale3 < self.taglia - 1:
                        cutoff3.append(cutoffattuale3)
                        soglie3.append(st.chi2.ppf(1 - self.impostazioni["alfaljungboxnorme"], df=cutoffattuale3))
                        cutoffattuale3 += self.impostazioni["hljungboxnorme"]
                cutoff3.append(self.taglia - 1)
                soglie3.append(st.chi2.ppf(1 - self.impostazioni["alfaljungboxnorme"], df=self.taglia - 1))
                if cutoff3 is None or soglie3 is None:
                    raise TypeError("I cutoff o le soglie sono indeterminati")
                tempo = time()
                self.betanormeindipendenti = miomodulo.betaljungbox(self.taglia if self.ordine else self.volée,
                                                                    cutoff3, soglie3, len(cutoff3),
                                                                    voltebetaljungboxnorme)
                self.tempobetanormeindipendenti = time() - tempo
                self.tempototale += self.tempobetanormeindipendenti
                self.probabilitàtuttogiusto *= (1 - self.betanormeindipendenti)
                self.alfaveronormeindipendenti = None
                self.tempoalfaveronormeindipendenti = None
            except Exception as e:
                self.betanormeindipendenti = None
                self.tempobetanormeindipendenti = None
                self.alfaveronormeindipendenti = None
                self.tempoalfaveronormeindipendenti = None
                self.probabilitàtuttogiusto = None
                self.impostazioni["calcolaerrori"] = False
                print(
                    f"\033[35mNon ho potuto calcolare il beta per il test di Ljung-Box sulle distanze dal centro perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")

    def rieseguialfaveronormeindipendenti(self, voltealfaljungboxnorme: int):
        if voltealfaljungboxnorme <= 0:
            raise Exception("Valori dei parametri sbagliati")
        if (self.taglia if self.ordine else self.volée) < 50 and self.impostazioni[
            "calcolaerrori"] and not (self.normeindipendenti if self.normeindipendenti is not None else True):
            cutoff3 = None
            soglie3 = None
            cutoffattuale3 = None
            try:
                if self.alfaveronormeindipendenti is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1-self.alfaveronormeindipendenti
                self.impostazioni["voltealfaljungboxnorme"] = voltealfaljungboxnorme
                if self.tempoalfaveronormeindipendenti is not None:
                    self.tempototale -= self.tempoalfaveronormeindipendenti
                cutoff3 = list()
                soglie3 = list()
                if self.impostazioni["hljungboxnorme"] < self.taglia - 1:
                    cutoffattuale3 = self.impostazioni["hljungboxnorme"]
                    while cutoffattuale3 < self.taglia - 1:
                        cutoff3.append(cutoffattuale3)
                        soglie3.append(st.chi2.ppf(1 - self.impostazioni["alfaljungboxnorme"], df=cutoffattuale3))
                        cutoffattuale3 += self.impostazioni["hljungboxnorme"]
                cutoff3.append(self.taglia - 1)
                soglie3.append(st.chi2.ppf(1 - self.impostazioni["alfaljungboxnorme"], df=self.taglia - 1))
                if cutoff3 is None or soglie3 is None:
                    raise TypeError("I cutoff o le soglie sono indeterminati")
                tempo = time()
                self.alfaveronormeindipendenti = miomodulo.alfaveroljungbox(self.taglia if self.ordine else self.volée,
                                                                            cutoff3, soglie3, len(cutoff3),
                                                                            voltealfaljungboxnorme)
                self.tempoalfaveronormeindipendenti = time() - tempo
                self.probabilitàtuttogiusto *= (1 - self.alfaveronormeindipendenti)
                self.betanormeindipendenti = None
                self.tempobetanormeindipendenti = None
            except Exception as e:
                self.betanormeindipendenti = None
                self.tempobetanormeindipendenti = None
                self.alfaveronormeindipendenti = None
                self.tempoalfaveronormeindipendenti = None
                self.impostazioni["calcolaerrori"] = False
                print(
                    f"\033[35mNon ho potuto calcolare l'alfa effettivo per il test di Ljung-Box sulle distanze dal centro perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")

    def rieseguiintervallonorme(self, confidenzanorme: float):
        self.impostazioni["intervallonorme"] = True
        if self.normenormali is None:
            self.rieseguinormenormali(self.impostazioni["alfashapiro"])
        if self.normeindipendenti is None:
            self.rieseguinormeindipendenti(self.impostazioni["alfaljungboxnorme"], self.impostazioni["hljungboxnorme"])
        if self.taglia > 1:
            try:
                if self.intervallonorme is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= self.impostazioni["confidenzanorme"]
                self.impostazioni["confidenzanorme"] = confidenzanorme
                if self.tempointervallonorme is not None:
                    self.tempototale -= self.tempointervallonorme
                if self.tempograficointervallonorme is not None:
                    self.tempototale -= self.tempograficointervallonorme
                tempo = time()
                self.intervallonorme = [np.mean(appiattisci(self.norme)) + (np.var(appiattisci(self.norme), ddof=1) / self.taglia) ** 0.5 * st.t.ppf((1 - confidenzanorme) / 2, df=self.taglia-1),
                                        np.mean(appiattisci(self.norme)) - (np.var(appiattisci(self.norme), ddof=1) / self.taglia) ** 0.5 * st.t.ppf((1 - confidenzanorme) / 2, df=self.taglia-1)]
                self.tempointervallonorme = time() - tempo
            except Exception as e:
                self.intervallonorme = None
                self.tempointervallonorme = None
                print(f"\033[35mNon ho potuto calcolare un intervallo di confidenza per le distanze dal centro perché {e}\033[0m")
            try:
                tempo = time()
                graficointervallonorme(self.intervallonorme, data)
                self.tempograficointervallonorme = time() - tempo
            except Exception as e:
                self.tempograficointervallonorme = None
                print(f"\033[35mNon ho potuto graficare l'intervallo di confidenza per le distanze dal centro perché {e}\033[0m")
            if self.intervallonorme is not None and self.impostazioni["calcolaerrori"]:
                self.probabilitàtuttogiusto *= confidenzanorme
        else:
            self.intervallonorme = None
            self.tempointervallonorme = None

    def rieseguitestnorme(self, alternativanorme: str, mediatestnorme: float, alfatestnorme: float):
        if alternativanorme not in {"two-sided", "less", "greater"}:
            raise Exception("Valori dei parametri sbagliati")
        if alfatestnorme <= 0 or alfatestnorme >= 1 or mediatestnorme < 0 or mediatestnorme > 10:
            raise Exception("Valori dei parametri sbagliati")
        self.impostazioni["testnorme"] = True
        if self.normenormali is None:
            self.rieseguinormenormali(self.impostazioni["alfashapiro"])
        if self.normeindipendenti is None:
            self.rieseguinormeindipendenti(self.impostazioni["alfaljungboxnorme"], self.impostazioni["hljungboxnorme"])
        if self.taglia > 1:
            try:
                if self.betatestnorme is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1-self.betatestnorme
                    self.betatestnorme = None
                elif not (self.testnorme if self.testnorme is not None else True) and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1-self.impostazioni["alfatestnorme"]
                self.impostazioni["alternativanorme"] = alternativanorme
                self.impostazioni["mediatestnorme"] = mediatestnorme
                self.impostazioni["alfatestnorme"] = alfatestnorme
                if self.tempotestnorme is not None:
                    self.tempototale -= self.tempotestnorme
                if self.tempobetatestnorme is not None:
                    self.tempototale -= self.tempobetatestnorme
                    self.tempobetatestnorme = None
                tempo = time()
                self.testnorme = st.ttest_1samp(appiattisci(self.norme),
                                                alternative=alternativanorme,
                                                popmean=mediatestnorme).pvalue > alfatestnorme
                self.tempotestnorme = time() - tempo
                self.tempototale += self.tempotestnorme
            except Exception as e:
                self.testnorme = None
                self.tempotestnorme = None
                print(f"\033[35mNon ho potuto effettuare il t-test sulle distanze dal centro perché {e}\033[0m")
            if (self.testnorme if self.testnorme is not None else False) and self.impostazioni["calcolaerrori"]:
                try:
                    tempo = time()
                    self.betatestnorme = betatestnorme(frecce=self.taglia,
                                                       alternativa=alternativanorme,
                                                       distanza=self.impostazioni["distanzabetanorme"],
                                                       iterazioni=self.impostazioni["iterazionibetanorme"],
                                                       media=mediatestnorme,
                                                       varianza=np.var(appiattisci(self.norme), ddof=1),
                                                       alfa=alfatestnorme)
                    self.tempobetatestnorme = time() - tempo
                    self.tempototale += self.tempobetatestnorme
                    self.probabilitàtuttogiusto *= 1 - self.betatestnorme
                except Exception as e:
                    self.betatestnorme = None
                    self.tempobetatestnorme = None
                    self.probabilitàtuttogiusto = None
                    self.impostazioni["calcolaerrori"] = False
                    print(f"\033[35mNon ho potuto calcolare il beta per il t-test sulle distanze dal centro perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
            else:
                self.betatestnorme = None
                self.tempobetatestnorme = None
                if self.impostazioni["calcolaerrori"] and self.testnorme is not None:
                    self.probabilitàtuttogiusto *= 1 - alfatestnorme
        else:
            self.testnorme = None
            self.tempotestnorme = None
            self.betatestnorme = None
            self.tempobetatestnorme = None

    def rieseguibetatestnorme(self, distanzabetanorme: float, iterazionibetanorme: int):
        if iterazionibetanorme <= 0:
            raise Exception("Valori dei parametri sbagliati")
        if (self.testnorme if self.testnorme is not None else False) and self.impostazioni["calcolaerrori"]:
            try:
                if self.betatestnorme is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1-self.betatestnorme
                self.impostazioni["distanzabetanorme"] = distanzabetanorme
                self.impostazioni["iterazionibetanorme"] = iterazionibetanorme
                if self.tempobetatestnorme is not None:
                    self.tempototale -= self.tempobetatestnorme
                tempo = time()
                self.betatestnorme = betatestnorme(frecce=self.taglia,
                                                   alternativa=self.impostazioni["alternativanorme"],
                                                   distanza=distanzabetanorme,
                                                   iterazioni=iterazionibetanorme,
                                                   media=self.impostazioni["mediatestnorme"],
                                                   varianza=np.var(appiattisci(self.norme), ddof=1),
                                                   alfa=self.impostazioni["alfatestnorme"])
                self.tempobetatestnorme = time() - tempo
                self.tempototale += self.tempobetatestnorme
                self.probabilitàtuttogiusto *= 1 - self.betatestnorme
            except Exception as e:
                self.betatestnorme = None
                self.tempobetatestnorme = None
                self.probabilitàtuttogiusto = None
                self.impostazioni["calcolaerrori"] = False
                print(
                    f"\033[35mNon ho potuto calcolare il beta per il t-test sulle distanze dal centro perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")

    def rieseguinormebayesiane(self, catenenorme: int, iterazionimcmcnorme: int, credibilitànormebayesiane: float,
                               ROPEnorme: float, noninformativitànormebayesiane: bool, allenamentipriorinormebayesiane: int,
                               normegerarchiche: bool, apiacerenorme: bool, alfaapiacerenorme: float,
                               betaapiacerenorme: float, alfa2apiacerenorme: float, beta2apiacerenorme: float):
        if catenenorme <= 0 or iterazionimcmcnorme <= 0 or credibilitànormebayesiane <= 0 or credibilitànormebayesiane >= 1:
            raise Exception("Valori dei parametri sbagliati")
        if allenamentipriorinormebayesiane > len(leggifile("Date.txt")) or allenamentipriorinormebayesiane < 0:
            raise Exception("Valori dei parametri sbagliati")
        if allenamentipriorinormebayesiane == 0 and not noninformativitànormebayesiane:
            raise Exception("Valori dei parametri sbagliati")
        if any(cosa <= 0 for cosa in {alfaapiacerenorme, betaapiacerenorme, alfa2apiacerenorme, beta2apiacerenorme}):
            raise Exception("Valori dei parametri sbagliati")
        self.impostazioni["normebayesiane"] = True
        self.impostazioni["intervallonormebayesiane"] = True
        if self.taglia > 1:
            try:
                if self.normebayesiane is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1 - min([self.normebayesiane, 1 - self.normebayesiane])
                if self.intervallonormebayesiano is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1-self.impostazioni["credibilitànormebayesiane"]
                self.impostazioni["catenenorme"] = catenenorme
                self.impostazioni["iterazionimcmcnorme"] = iterazionimcmcnorme
                self.impostazioni["credibilitànormebayesiane"] = credibilitànormebayesiane
                self.impostazioni["ROPEnorme"] = ROPEnorme
                self.impostazioni["noninformativitànormebayesiane"] = noninformativitànormebayesiane
                self.impostazioni["allenamentipriorinormebayesiane"] = allenamentipriorinormebayesiane
                self.impostazioni["normegerarchiche"] = normegerarchiche
                self.impostazioni["apiacerenorme"] = apiacerenorme
                self.impostazioni["alfaapiacerenorme"] = alfaapiacerenorme
                self.impostazioni["betaapiacerenorme"] = betaapiacerenorme
                self.impostazioni["alfa2apiacerenorme"] = alfa2apiacerenorme
                self.impostazioni["beta2apiacerenorme"] = beta2apiacerenorme
                if self.temponormebayesiane is not None:
                    self.tempototale -= self.temponormebayesiane
                tempo = time()
                self.normebayesiane, self.intervallonormebayesiane = normebayesiane(allenamento=appiattisci(self.norme),
                                                                                    catene=catenenorme,
                                                                                    iterazioni=iterazionimcmcnorme,
                                                                                    alternativa=self.impostazioni[
                                                                                        "alternativanorme"],
                                                                                    media=self.impostazioni[
                                                                                        "mediatestnorme"],
                                                                                    credibilità=credibilitànormebayesiane,
                                                                                    rope=ROPEnorme,
                                                                                    noninformativa=noninformativitànormebayesiane,
                                                                                    allenamentipriori=allenamentipriorinormebayesiane,
                                                                                    gerarchica=normegerarchiche,
                                                                                    apiacere=apiacerenorme,
                                                                                    alfaapiacere=alfaapiacerenorme,
                                                                                    betaapiacere=betaapiacerenorme,
                                                                                    alfa2apiacere=alfa2apiacerenorme,
                                                                                    beta2apiacere=beta2apiacerenorme,
                                                                                    console=self.impostazioni[
                                                                                        "consoledistan"])
                self.temponormebayesiane = time() - tempo
                self.tempototale += self.temponormebayesiane
                if self.impostazioni["calcolaerrori"]:
                    self.probabilitàtuttogiusto *= 1 - min([self.normebayesiane, 1 - self.normebayesiane])
                    self.probabilitàtuttogiusto *= credibilitànormebayesiane
            except Exception as e:
                self.normebayesiane = None
                self.intervallonormebayesiano = None
                self.temponormebayesiane = None
                print(
                    f"\033[35mNon ho potuto effettuare il t-test bayesiano o l'intervallo di credibilità sulle distanze dal centro perché {e}\033[0m")

    def rieseguiintervallivarianze(self, confidenzavarianzaascisse: float, confidenzavarianzaordinate: float):
        if confidenzavarianzaascisse <= 0 or confidenzavarianzaascisse >= 1 or confidenzavarianzaordinate <= 0 or confidenzavarianzaordinate >= 1:
            raise Exception("Valori dei parametri sbagliati")
        self.impostazioni["intervallivarianze"] = True
        if self.taglia > 2:
            try:
                if self.intervallivarianze is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= self.impostazioni["confidenzavarianzaascisse"]
                    self.probabilitàtuttogiusto /= self.impostazioni["confidenzavarianzaordinate"]
                self.impostazioni["confidenzavarianzaascisse"] = confidenzavarianzaascisse
                self.impostazioni["confidenzavarianzaordinate"] = confidenzavarianzaordinate
                if self.tempointervallivarianze is not None:
                    self.tempototale -= self.tempointervallivarianze
                if self.tempograficointervallivarianze is not None:
                    self.tempototale -= self.tempograficointervallivarianze
                tempo = time()
                varasc = (self.varianzacoordinate[0] if self.varianzacoordinate is not None
                          else np.var(appiattisci(self.ascisse), ddof=1))
                varord = (self.varianzacoordinate[1] if self.varianzacoordinate is not None
                          else np.var(appiattisci(self.ordinate), ddof=1))
                self.intervallivarianze = ([(self.taglia - 1) * varasc / st.chi2.ppf(
                    1 - (1-confidenzavarianzaascisse) / 2, df=self.taglia - 1),
                                            (self.taglia - 1) * varasc / st.chi2.ppf(
                                                (1-confidenzavarianzaascisse) / 2, df=self.taglia - 1)],
                                            [(self.taglia - 1) * varord / st.chi2.ppf(
                                                1 - (1-confidenzavarianzaordinate) / 2,
                                                df=self.taglia - 1),
                                            (self.taglia - 1) * varord / st.chi2.ppf(
                                                (1-confidenzavarianzaordinate) / 2, df=self.taglia - 1)])
                self.tempointervallivarianze = time() - tempo
                self.tempototale += self.tempointervallivarianze
                if self.impostazioni["calcolaerrori"]:
                    self.probabilitàtuttogiusto *= confidenzavarianzaascisse
                    self.probabilitàtuttogiusto *= confidenzavarianzaordinate
            except Exception as e:
                self.intervallivarianze = None
                self.tempointervallivarianze = None
                print(f"\033[35mNon ho potuto calcolare gli intervalli di confidenza delle varianze perché {e}\033[0m")
            if self.intervallivarianze is not None and self.mediacoordinate is not None:
                try:
                    tempo = time()
                    graficointervallivarianze(self.intervallivarianze, self.mediacoordinate, data)
                    self.tempograficointervallivarianze = time() - tempo
                    self.tempototale += self.tempograficointervallivarianze
                except Exception as e:
                    self.tempograficointervallivarianze = None
                    print(f"\033[35mNon ho potuto disegnare il grafico degli intervalli di confidenza delle varianze perché {e}\033[0m")
            else:
                self.tempograficointervallivarianze = None
        else:
            self.intervallivarianze = None
            self.tempointervallivarianze = None
            self.tempograficointervallivarianze = None

    def rieseguitestvarianze(self, alternativavarianzaascisse: str, ipotesinullavarianzaascisse: float,
                             alfavarianzaascisse: float, alternativavarianzaordinate: str,
                             ipotesinullavarianzaordinate: float, alfavarianzaordinate: float):
        if ipotesinullavarianzaascisse <= 0 or ipotesinullavarianzaordinate <= 0:
            raise Exception("Valori dei parametri sbagliati")
        if alfavarianzaascisse <= 0 or alfavarianzaascisse >= 1 or alfavarianzaordinate <= 0 or alfavarianzaordinate >= 1:
            raise Exception("Valori dei parametri sbagliati")
        if alternativavarianzaascisse not in {"maggiore", "disuguale", "minore"} or alternativavarianzaordinate not in {"maggiore", "disuguale", "minore"}:
            raise Exception("Valori dei parametri sbagliati")
        self.impostazioni["testvarianze"] = True
        if self.taglia > 2:
            try:
                if self.testvarianze is not None and self.probabilitàtuttogiusto is not None:
                    if self.betatestvarianzeascisse is not None and self.testvarianze[0]:
                        self.probabilitàtuttogiusto /= 1-self.betatestvarianzeascisse
                        self.betatestvarianzeascisse = None
                    elif not self.testvarianze[0]:
                        self.probabilitàtuttogiusto /= 1-self.impostazioni["alfavarianzaascisse"]
                    if self.betatestvarianzeordinate is not None and self.testvarianze[1]:
                        self.probabilitàtuttogiusto /= 1-self.betatestvarianzeordinate
                        self.betatestvarianzeordinate = None
                    elif not self.testvarianze[1]:
                        self.probabilitàtuttogiusto /= 1-self.impostazioni["alfavarianzaordinate"]
                self.impostazioni["alternativavarianzaascisse"] = alternativavarianzaascisse
                self.impostazioni["ipotesinullavarianzaascisse"] = ipotesinullavarianzaascisse
                self.impostazioni["alfavarianzaascisse"] = alfavarianzaascisse
                self.impostazioni["alternativavarianzaordinate"] = alternativavarianzaordinate
                self.impostazioni["ipotesinullavarianzaordinate"] = ipotesinullavarianzaordinate
                self.impostazioni["alfavarianzaordinate"] = alfavarianzaordinate
                if self.tempotestvarianze is not None:
                    self.tempototale -= self.tempotestvarianze
                if self.tempobetatestvarianzeascisse is not None:
                    self.tempototale -= self.tempobetatestvarianzeascisse
                    self.tempobetatestvarianzeascisse = None
                if self.tempobetatestvarianzeordinate is not None:
                    self.tempototale -= self.tempobetatestvarianzeordinate
                    self.tempobetatestvarianzeordinate = None
                tempo = time()
                varasc = (self.varianzacoordinate[0] if self.varianzacoordinate[0] is not None
                          else np.var(appiattisci(self.ascisse), ddof=1))
                varord = (self.varianzacoordinate[1] if self.varianzacoordinate[1] is not None
                          else np.var(appiattisci(self.ordinate), ddof=1))
                self.testvarianze = (testvarianze(alternativa=alternativavarianzaascisse,
                                                  varianza=varasc,
                                                  ipotesinulla=ipotesinullavarianzaascisse,
                                                  frecce=self.taglia) > alfavarianzaascisse,
                                     testvarianze(alternativa=alternativavarianzaordinate,
                                                  varianza=varord,
                                                  ipotesinulla=ipotesinullavarianzaordinate,
                                                  frecce=self.taglia) > alfavarianzaordinate)
                self.tempotestvarianze = time() - tempo
                self.tempototale += self.tempotestvarianze
            except Exception as e:
                self.testvarianze = None
                self.tempotestvarianze = None
                print(f"\033[35mNon ho potuto eseguire il test d'ipotesi sulle varianze perché {e}\033[0m")
            if (self.testvarianze[0] if self.testvarianze is not None else False) and self.impostazioni["calcolaerrori"]:
                try:
                    tempo = time()
                    self.betatestvarianzeascisse = betatestvarianze(
                        iterazioni=self.impostazioni["iterazionibetavarianzaascisse"],
                        alternativa=alternativavarianzaascisse,
                        ipotesinulla=ipotesinullavarianzaascisse,
                        distanza=self.impostazioni["distanzabetavarianzaascisse"],
                        frecce=self.taglia,
                        alfa=alfavarianzaascisse)
                    self.tempobetatestvarianzeascisse = time() - tempo
                    self.tempototale += self.tempobetatestvarianzeascisse
                    self.probabilitàtuttogiusto *= 1 - self.betatestvarianzeascisse
                except Exception as e:
                    self.betatestvarianzeascisse = None
                    self.tempobetatestvarianzeascisse = None
                    self.probabilitàtuttogiusto = None
                    self.impostazioni["calcolaerrori"] = False
                    print(f"\033[35mNon ho potuto calcolare il beta per il test d'ipotesi sulle varianze delle ascisse perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
            else:
                self.betatestvarianzeascisse = None
                self.tempobetatestvarianzeascisse = None
                if self.impostazioni["calcolaerrori"] and self.testvarianze is not None:
                    self.probabilitàtuttogiusto *= 1 - alfavarianzaascisse
            if (self.testvarianze[1] if self.testvarianze is not None else False) and self.impostazioni["calcolaerrori"]:
                try:
                    tempo = time()
                    self.betatestvarianzeordinate = betatestvarianze(iterazioni=self.impostazioni[
                                                                         "iterazionibetavarianzaordinate"],
                                                                     alternativa=alternativavarianzaordinate,
                                                                     ipotesinulla=ipotesinullavarianzaordinate,
                                                                     distanza=self.impostazioni["distanzabetavarianzaordinate"],
                                                                     frecce=self.taglia,
                                                                     alfa=alfavarianzaordinate)
                    self.tempobetatestvarianzeordinate = time() - tempo
                    self.tempototale += self.tempobetatestvarianzeordinate
                    self.probabilitàtuttogiusto *= 1 - self.betatestvarianzeordinate
                except Exception as e:
                    self.betatestvarianzeordinate = None
                    self.tempobetatestvarianzeordinate = None
                    self.probabilitàtuttogiusto = None
                    self.impostazioni["calcolaerrori"] = False
                    print(f"\033[35mNon ho potuto calcolare il beta per il test d'ipotesi sulle varianze delle ordinate perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
            else:
                self.betatestvarianzeordinate = None
                self.tempobetatestvarianzeordinate = None
                if self.impostazioni["calcolaerrori"] and self.testvarianze is not None:
                    self.probabilitàtuttogiusto *= 1 - alfavarianzaordinate
        else:
            self.testvarianze = None
            self.tempotestvarianze = None
            self.betatestvarianzeascisse = None
            self.tempobetatestvarianzeascisse = None
            self.betatestvarianzeordinate = None
            self.tempobetatestvarianzeordinate = None

    def rieseguibetatestvarianze(self, iterazionibetavarianzaascisse: int, distanzabetavarianzaascisse: float,
                                 iterazionibetavarianzaordinate: int, distanzabetavarianzaordinate: float):
        if iterazionibetavarianzaascisse <= 0 or iterazionibetavarianzaordinate <= 0:
            raise Exception("Valori dei parametri sbagliati")
        if (self.testvarianze[0] if self.testvarianze is not None else False) and self.impostazioni["calcolaerrori"]:
            try:
                if self.betatestvarianzeascisse is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1-self.betatestvarianzeascisse
                    self.betatestvarianzeascisse = None
                self.impostazioni["iterazionibetavarianzaascisse"] = iterazionibetavarianzaascisse
                self.impostazioni["distanzabetavarianzaascisse"] = distanzabetavarianzaascisse
                if self.tempobetatestvarianzeascisse is not None:
                    self.tempototale -= self.tempobetatestvarianzeascisse
                    self.tempobetatestvarianzeascisse = None
                tempo = time()
                self.betatestvarianzeascisse = betatestvarianze(
                    iterazioni=iterazionibetavarianzaascisse,
                    alternativa=self.impostazioni["alternativavarianzaascisse"],
                    ipotesinulla=self.impostazioni["ipotesinullavarianzaascisse"],
                    distanza=distanzabetavarianzaascisse,
                    frecce=self.taglia,
                    alfa=self.impostazioni["alfavarianzaascisse"])
                self.tempobetatestvarianzeascisse = time() - tempo
                self.tempototale += self.tempobetatestvarianzeascisse
                self.probabilitàtuttogiusto *= 1 - self.betatestvarianzeascisse
            except Exception as e:
                self.betatestvarianzeascisse = None
                self.tempobetatestvarianzeascisse = None
                self.probabilitàtuttogiusto = None
                self.impostazioni["calcolaerrori"] = False
                print(
                    f"\033[35mNon ho potuto calcolare il beta per il test d'ipotesi sulle varianze delle ascisse perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
        if (self.testvarianze[1] if self.testvarianze is not None else False) and self.impostazioni["calcolaerrori"]:
            try:
                if self.betatestvarianzeordinate is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1-self.betatestvarianzeordinate
                    self.betatestvarianzeordinate = None
                self.impostazioni["iterazionibetavarianzaordinate"] = iterazionibetavarianzaordinate
                self.impostazioni["distanzabetavarianzaordinate"] = distanzabetavarianzaordinate
                if self.tempobetatestvarianzeordinate is not None:
                    self.probabilitàtuttogiusto /= 1-self.tempobetatestvarianzeordinate
                    self.tempobetatestvarianzeordinate = None
                tempo = time()
                self.betatestvarianzeordinate = betatestvarianze(iterazioni=iterazionibetavarianzaordinate,
                                                                 alternativa=self.impostazioni[
                                                                     "alternativavarianzaordinate"],
                                                                 ipotesinulla=self.impostazioni[
                                                                     "ipotesinullavarianzaordinate"],
                                                                 distanza=distanzabetavarianzaordinate,
                                                                 frecce=self.taglia,
                                                                 alfa=self.impostazioni["alfavarianzaordinate"])
                self.tempobetatestvarianzeordinate = time() - tempo
                self.tempototale += self.tempobetatestvarianzeordinate
                self.probabilitàtuttogiusto *= 1 - self.betatestvarianzeordinate
            except Exception as e:
                self.betatestvarianzeordinate = None
                self.tempobetatestvarianzeordinate = None
                self.probabilitàtuttogiusto = None
                self.impostazioni["calcolaerrori"] = False
                print(
                    f"\033[35mNon ho potuto calcolare il beta per il test d'ipotesi sulle varianze delle ordinate perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")

    def rieseguivarianzebayesiane(self, catenevarianze: int, iterazionimcmcvarianze: int, ROPEvarianzaascisse: float,
                                  ROPEvarianzaordinate: float, noninformativitàvarianzebayesiane: bool,
                                  allenamentipriorivarianzebayesiane: int, credibilitàvarianzaascissebayesiane: float,
                                  credibilitàvarianzaordinatebayesiane: float, varianzebayesianegerarchiche: bool,
                                  apiacerevarianzebayesiane: bool, muapiacerevarianzebayesiane: list[float],
                                  sigmaapiacerevarianzebayesiane: list[list[float]], aapiacerevarianzebayesiane: float,
                                  bapiacerevarianzebayesiane: float, etaapiacerevarianzebayesiane: float):
        if catenevarianze <= 0 or iterazionimcmcvarianze <= 0:
            raise Exception("Valori dei parametri sbagliati")
        if allenamentipriorivarianzebayesiane > len(leggifile("Date.txt")) or allenamentipriorivarianzebayesiane < 0:
            raise Exception("Valori dei parametri sbagliati")
        if allenamentipriorivarianzebayesiane == 0 and not noninformativitàvarianzebayesiane:
            raise Exception("Valori dei parametri sbagliati")
        if credibilitàvarianzaascissebayesiane <= 0 or credibilitàvarianzaascissebayesiane >= 1:
            raise Exception("Valori dei parametri sbagliati")
        if credibilitàvarianzaordinatebayesiane <= 0 or credibilitàvarianzaordinatebayesiane >= 1:
            raise Exception("Valori dei parametri sbagliati")
        if sigmaapiacerevarianzebayesiane[0][0]*sigmaapiacerevarianzebayesiane[1][1]-sigmaapiacerevarianzebayesiane[0][1]*sigmaapiacerevarianzebayesiane[1][0] < 0:
            raise Exception("Valori dei parametri sbagliati")
        if sigmaapiacerevarianzebayesiane[0][1] != sigmaapiacerevarianzebayesiane[1][0]:
            raise Exception("Valori dei parametri sbagliati")
        self.impostazioni["varianzebayesiane"] = True
        self.impostazioni["intervallovarianzebayesiano"] = True
        if self.taglia > 2:
            try:
                if self.varianzebayesiane is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1 - min([self.varianzebayesiane[0], 1 - self.varianzebayesiane[0]])
                    self.probabilitàtuttogiusto /= 1 - min([self.varianzebayesiane[1], 1 - self.varianzebayesiane[1]])
                if self.intervallovarianzebayesiane is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= self.impostazioni["credibilitàvarianzaascissebayesiane"]
                    self.probabilitàtuttogiusto /= self.impostazioni["credibilitàvarianzaordinatebayesiane"]
                self.impostazioni["catenevarianze"] = catenevarianze
                self.impostazioni["iterazionimcmcvarianze"] = iterazionimcmcvarianze
                self.impostazioni["ROPEvarianzaascisse"] = ROPEvarianzaascisse
                self.impostazioni["ROPEvarianzaordinate"] = ROPEvarianzaordinate
                self.impostazioni["noninformativitàvarianzebayesiane"] = noninformativitàvarianzebayesiane
                self.impostazioni["allenamentipriorivarianzebayesiane"] = allenamentipriorivarianzebayesiane
                self.impostazioni["credibilitàvarianzaascissebayesiane"] = credibilitàvarianzaascissebayesiane
                self.impostazioni["credibilitàvarianzaordinatebayesiane"] = credibilitàvarianzaordinatebayesiane
                self.impostazioni["varianzebayesianegerarchiche"] = varianzebayesianegerarchiche
                self.impostazioni["apiacerevarianzebayesiane"] = apiacerevarianzebayesiane
                self.impostazioni["muapiacerevarianzebayesiane"] = muapiacerevarianzebayesiane
                self.impostazioni["sigmaapiacerevarianzebayesiane"] = sigmaapiacerevarianzebayesiane
                self.impostazioni["aapiacerevarianzebayesiane"] = aapiacerevarianzebayesiane
                self.impostazioni["bapiacerevarianzebayesiane"] = bapiacerevarianzebayesiane
                self.impostazioni["etaapiacerevarianzebayesiane"] = etaapiacerevarianzebayesiane
                if self.tempovarianzebayesiane is not None:
                    self.tempototale -= self.tempovarianzebayesiane
                if self.tempograficointervallovarianzebayesiano is not None:
                    self.tempototale -= self.tempograficointervallovarianzebayesiano
                tempo = time()
                self.varianzebayesiane, self.intervallovarianzebayesiane = varianzebayesiane(
                    allenamento=self.depura(),
                    catene=catenevarianze,
                    iterazioni=iterazionimcmcvarianze,
                    alternativaascisse=self.impostazioni["alternativavarianzaascisse"],
                    alternativaordinate=self.impostazioni["alternativavarianzaordinate"],
                    ipotesinullaascisse=self.impostazioni["ipotesinullavarianzaascisse"],
                    ipotesinullaordinate=self.impostazioni["ipotesinullavarianzaordinate"],
                    ropeascisse=ROPEvarianzaascisse,
                    ropeordinate=ROPEvarianzaordinate,
                    noninformativa=noninformativitàvarianzebayesiane,
                    allenamentipriori=allenamentipriorivarianzebayesiane,
                    alfaascisse=1-credibilitàvarianzaascissebayesiane,
                    alfaordinate=1-credibilitàvarianzaordinatebayesiane,
                    gerarchica=varianzebayesianegerarchiche,
                    apiacere=apiacerevarianzebayesiane,
                    muapiacere=muapiacerevarianzebayesiane,
                    sigmaapiacere=sigmaapiacerevarianzebayesiane,
                    aapiacere=aapiacerevarianzebayesiane,
                    bapiacere=bapiacerevarianzebayesiane,
                    etaapiacere=etaapiacerevarianzebayesiane,
                    console=self.impostazioni["consoledistan"])
                self.tempovarianzebayesiane = time() - tempo
                self.tempototale += self.tempovarianzebayesiane
                if self.impostazioni["calcolaerrori"]:
                    self.probabilitàtuttogiusto *= 1 - min([self.varianzebayesiane[0], 1 - self.varianzebayesiane[0]])
                    self.probabilitàtuttogiusto *= 1 - min([self.varianzebayesiane[1], 1 - self.varianzebayesiane[1]])
                    self.probabilitàtuttogiusto *= credibilitàvarianzaascissebayesiane
                    self.probabilitàtuttogiusto *= credibilitàvarianzaordinatebayesiane
            except Exception as e:
                self.varianzebayesiane = None
                self.intervallovarianzebayesiane = None
                self.tempovarianzebayesiane = None
                print(f"\033[35mNon ho potuto effettuare il test d'ipotesi bayesiano o l'intervallo di credibilità sulle varianze perché {e}\033[0m")
            if self.intervallovarianzebayesiane is not None:
                try:
                    tempo = time()
                    graficointervallovarianzebayesiano(self.intervallovarianzebayesiane, self.mediacoordinate, self.data)
                    self.tempograficointervallovarianzebayesiano = time() - tempo
                    self.tempototale += self.tempograficointervallovarianzebayesiano
                except Exception as e:
                    self.tempografricointervallovarianzebayesiano = None
                    print(f"\033[35mNon ho potuto graficare l'intervallo di credibilità sulle varianze perché {e}\033[0m")
            else:
                self.tempograficointervallovarianzebayesiano = None

    def rieseguifreccedifettose(self, alfahotellingduecampioni: float, iterazionibetahotellingduecampioni: int,
                                distanzabetahotellingduecampioni: float, alfavarianzeduecampioni: float,
                                iterazionibetavarianzeduecampioni: int, distanzabetavarianzeduecampioni: float):
        if alfahotellingduecampioni <= 0 and alfahotellingduecampioni >= 1:
            raise Exception("Valori dei parametri sbagliati")
        if alfavarianzeduecampioni <= 0 and alfavarianzeduecampioni >= 1:
            raise Exception("Valori dei parametri sbagliati")
        if iterazionibetahotellingduecampioni <= 0 or iterazionibetavarianzeduecampioni <= 0:
            raise Exception("Valori dei parametri sbagliati")
        self.impostazioni["freccedifettose"] = True
        if self.identificazione and self.frecce > 1:
            try:
                if self.difettosità is not None and self.probabilitàtuttogiusto is not None:
                    if all(self.difettosità):
                        self.probabilitàtuttogiusto /= self.impostazioni["alfahotellingduecampioni"]**self.frecce
                    elif self.betahotellingduecampioni is not None:
                        for difettosità in self.difettosità:
                            self.probabilitàtuttogiusto /= self.impostazioni["alfahotellingduecampioni"] if difettosità else self.betahotellingduecampioni
                        self.betahotellingduecampioni = None
                self.impostazioni["alfahotellingduecampioni"] = alfahotellingduecampioni
                self.impostazioni["iterazionibetahotellingduecampioni"] = iterazionibetahotellingduecampioni
                self.impostazioni["distanzabetahotellingduecampioni"] = distanzabetahotellingduecampioni
                if self.tempodifettosità is not None:
                    self.tempototale -= self.tempodifettosità
                if self.tempobetahotellingduecampioni is not None:
                    self.tempototale -= self.tempobetahotellingduecampioni
                    self.tempobetahotellingduecampioni = None
                tempo = time()
                ascisse = list()
                ordinate = list()
                frecce = list()
                for voléeascisse, voléeordinate in zip(self.ascisseidentificate, self.ordinateidentificate):
                    for ascissa, ordinata in zip(voléeascisse, voléeordinate):
                        ascisse.append(ascissa[0])
                        ordinate.append(ordinata[0])
                        if ascissa[1] != ordinata[1]:
                            raise Exception("Dataset mal formattati")
                        else:
                            frecce.append(ascissa[1])
                self.difettosità = miomodulo.freccedifettose(ascisse, ordinate, frecce, self.taglia,
                                                             st.f.ppf(1-alfahotellingduecampioni, 2, self.taglia-3),
                                                             self.frecce)
                self.tempodifettosità = time() - tempo
                self.tempototale += self.tempodifettosità
            except Exception as e:
                self.difettosità = None
                self.tempodifettosità = None
                print(f"\033[35mNon ho potuto valutare la difettosità delle frecce perché {e}\033[0m")
            try:
                if not (all(self.difettosità) if self.difettosità is not None else True) and self.impostazioni["calcolaerrori"]:
                    tempo = time()
                    matricecovarianze = np.cov(np.array([appiattisci(self.ascisse), appiattisci(self.ordinate)]), rowvar=True, bias=False)
                    self.betahotellingduecampioni = miomodulo.betahotellingduecampioni(
                        iterazionibetahotellingduecampioni,
                        matricecovarianze[0, 0],
                        matricecovarianze[1, 1],
                        matricecovarianze[0, 1],
                        self.taglia,
                        distanzabetahotellingduecampioni,
                        self.frecce,
                        alfahotellingduecampioni)
                    self.tempobetahotellingduecampioni = time() - tempo
                    self.tempototale += self.tempobetahotellingduecampioni
                else:
                    self.betahotellingduecampioni = None
                    self.tempobetahotellingduecampioni = None
                if self.impostazioni["calcolaerrori"] and self.difettosità is not None:
                    for difettosità in self.difettosità:
                        if difettosità:
                            self.probabilitàtuttogiusto *= 1 - alfahotellingduecampioni
                        else:
                            self.probabilitàtuttogiusto *= 1 - self.betahotellingduecampioni
            except Exception as e:
                self.betahotellingduecampioni = None
                self.tempobetahotellingduecampioni = None
                self.probabilitàtuttogiusto = None
                self.impostazioni["calcolaerrori"] = False
                print(f"\033[35mNon ho potuto calcolare il beta per il test di Hotelling a due campioni perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
            try:
                if self.difettositàvarianze is not None and self.probabilitàtuttogiusto is not None:
                    for difettositàascisse, difettositàordinate in self.difettositàvarianze:
                        if difettositàascisse:
                            self.probabilitàtuttogiusto /= 1-self.impostazioni["alfavarianzeduecampioni"]
                        elif self.betavarianzeduecampioni is not None:
                            self.probabilitàtuttogiusto /= 1-self.betavarianzeduecampioni
                        if difettositàordinate:
                            self.probabilitàtuttogiusto /= 1-self.impostazioni["alfavarianzeduecampioni"]
                        elif self.betavarianzeduecampioni is not None:
                            self.probabilitàtuttogiusto /= 1-self.betavarianzeduecampioni
                    self.betavarianzeduecampioni = None
                self.impostazioni["alfavarianzeduecampioni"] = alfavarianzeduecampioni
                self.impostazioni["iterazionibetavarianzeduecampioni"] = iterazionibetavarianzeduecampioni
                self.impostazioni["distanzabetavarianzeduecampioni"] = distanzabetavarianzeduecampioni
                if self.tempodifettositàvarianze:
                    self.tempototale -= self.tempodifettositàvarianze
                if self.tempobetavarianzeduecampioni:
                    self.tempototale -= self.tempobetavarianzeduecampioni
                    self.tempobetavarianzeduecampioni = None
                allenamento = list()
                for voléeascisse, voléeordinate in zip(self.ascisseidentificate, self.ordinateidentificate):
                    volée = list()
                    for ascissa, ordinata in zip(voléeascisse, voléeordinate):
                        if ascissa[1] != ordinata[1]:
                            raise Exception("Dataset mal formattati")
                        else:
                            volée.append([ascissa[0], ordinata[0], ascissa[1]])
                    allenamento.append(volée)
                tempo = time()
                self.difettositàvarianze = varianzedifettose(allenamento=allenamento,
                                                             alfa=alfavarianzeduecampioni,
                                                             numerofrecce=self.frecce)
                self.tempodifettositàvarianze = time() - tempo
                self.tempototale += self.tempodifettositàvarianze
            except Exception as e:
                self.difettositàvarianze = None
                self.tempodifettositàvarianze = None
                print(f"\033[35mNon ho potuto valutare la difettosità della dispersione delle frecce perché {e}\033[0m")
            if ((not all([valore for coppia in self.difettositàvarianze for valore in coppia]) if self.difettositàvarianze is not None else False)
                    and self.impostazioni["calcolaerrori"]):
                try:
                    tempo = time()
                    self.betavarianzeduecampioni = miomodulo.betavarianzeduecampioni(
                        iterazionibetavarianzeduecampioni,
                        distanzabetavarianzeduecampioni,
                        self.taglia,
                        self.frecce,
                        st.f.ppf(1 - alfavarianzeduecampioni,
                                        int(self.taglia / self.frecce),
                                        self.taglia - int(self.taglia / self.frecce)))
                    self.tempobetavarianzeduecampioni = time() - tempo
                    self.tempototale += self.tempobetavarianzeduecampioni
                except Exception as e:
                    self.betavarianzeduecampioni = None
                    self.tempobetavarianzeduecampioni = None
                    self.probabilitàtuttogiusto = None
                    self.impostazioni["calcolaerrori"] = False
                    print(f"\033[35mNon ho potuto calcolare il beta per il test d'ipotesi sulle varianze a due campioni perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
            else:
                self.betavarianzeduecampioni = None
                self.tempobetavarianzeduecampioni = None
            if self.impostazioni["calcolaerrori"] and self.difettositàvarianze is not None:
                for difettosità in [valore for coppia in self.difettositàvarianze for valore in coppia]:
                    if difettosità:
                        self.probabilitàtuttogiusto *= 1 - alfavarianzeduecampioni
                    else:
                        self.probabilitàtuttogiusto *= 1 - self.betavarianzeduecampioni
        else:
            self.difettosità = None
            self.tempodifettosità = None
            self.betahotellingduecampioni = None
            self.tempobetahotellingduecampioni = None
            self.difettositàvarianze = None
            self.tempodifettositàvarianze = None
            self.betavarianzeduecampioni = None
            self.tempobetavarianzeduecampioni = None

    def rieseguiuniformitàangoli(self, alfarayleigh: float):
        if alfarayleigh <= 0 or alfarayleigh >= 1:
            raise Exception("Valori dei parametri sbagliati")
        self.impostazioni["uniformitàangoli"] = True
        if self.taglia > 1:
            if self.betarayleigh is not None and self.probabilitàtuttogiusto is not None:
                self.probabilitàtuttogiusto /= 1-self.betarayleigh
            elif self.alfaverorayleigh is not None and self.probabilitàtuttogiusto is not None:
                self.probabilitàtuttogiusto /= 1-self.alfaverorayleigh
            elif not (self.uniformitàangoli if self.uniformitàangoli is not None else True) and self.probabilitàtuttogiusto is not None:
                self.probabilitàtuttogiusto /= 1-self.impostazioni["alfarayleigh"]
            self.impostazioni["alfarayleigh"] = alfarayleigh
            if self.tempouniformitàaangoli is not None:
                self.tempototale -= self.tempouniformitàangoli
            if self.tempobetarayleigh is not None:
                self.tempototale -= self.tempobetarayleigh
            if self.tempoalfaverorayleigh is not None:
                self.tempototale -= self.tempoalfaverorayleigh
            tempo = time()
            self.uniformitàangoli = self.varianzaangoli * self.taglia * 2 <= st.chi2.ppf(1 - alfarayleigh, df=2)
            self.tempouniformitàangoli = time() - tempo
            self.tempototale += self.tempouniformitàangoli
            if self.uniformitàangoli and self.impostazioni["calcolaerrori"]:
                try:
                    tempo = time()
                    self.betarayleigh = miomodulo.betarayleigh(self.impostazioni["iterazionibetarayleigh"],
                                                               self.impostazioni["kappabetarayleigh"],
                                                               self.taglia,
                                                               st.chi2.ppf(1 - alfarayleigh, df=2))
                    self.tempobetarayleigh = time() - tempo
                    self.tempototale += self.tempobetarayleigh
                    self.probabilitàtuttogiusto *= 1 - self.betarayleigh
                    self.alfaverorayleigh = None
                    self.tempoalfaverorayleigh = None
                except Exception as e:
                    self.betarayleigh = None
                    self.tempobetarayleigh = None
                    self.alfaverorayleigh = None
                    self.tempoalfaverorayleigh = None
                    self.probabilitàtuttogiusto = None
                    self.impostazioni["calcolaerrori"] = False
                    print(f"\033[35mNon ho potuto calcolare il beta per il test di Rayleigh perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
            elif self.taglia < 50 and self.impostazioni["calcolaerrori"]:
                try:
                    tempo = time()
                    self.alfaverorayleigh = miomodulo.alfaverorayleigh(
                        self.impostazioni["iterazionialfaverorayleigh"],
                        st.chi2.ppf(1 - alfarayleigh, df=2),
                        self.taglia)
                    self.tempoalfaverorayleigh = time() - tempo
                    self.betarayleigh = None
                    self.tempobetarayleigh = None
                except Exception as e:
                    self.alfaverorayleigh = None
                    self.tempoalfaverorayleigh = None
                    self.betarayleigh = None
                    self.tempobetarayleigh = None
                    print(f"\033[35mNon ho potuto calcolare l'alfa effettivo per il test di Rayleigh perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
            else:
                if self.impostazioni["calcolaerrori"]:
                    self.probabilitàtuttogiusto *= 1 - alfarayleigh
                self.alfaverorayleigh = None
                self.tempoalfaverorayleigh = None
                self.betarayleigh = None
                self.tempobetarayleigh = None

    def rieseguibetarayleigh(self, iterazionibetarayleigh: int, kappabetarayleigh: float):
        if iterazionibetarayleigh <= 0 or kappabetarayleigh <= 0:
            raise Exception("Valori dei parametri sbagliati")
        if self.uniformitàangoli and self.impostazioni["calcolaerrori"]:
            try:
                if self.betarayleigh is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1-self.betarayleigh
                self.impostazioni["iterazionibetarayleigh"] = iterazionibetarayleigh
                self.impostazioni["kappabetarayleigh"] = kappabetarayleigh
                if self.tempobetarayleigh is not None:
                    self.tempototale -= self.tempobetarayleigh
                tempo = time()
                self.betarayleigh = miomodulo.betarayleigh(iterazionibetarayleigh,
                                                           kappabetarayleigh,
                                                           self.taglia,
                                                           st.chi2.ppf(1 - self.impostazioni["alfarayleigh"], df=2))
                self.tempobetarayleigh = time() - tempo
                self.tempototale += self.tempobetarayleigh
                self.probabilitàtuttogiusto *= 1 - self.betarayleigh
                self.alfaverorayleigh = None
                self.tempoalfaverorayleigh = None
            except Exception as e:
                self.betarayleigh = None
                self.tempobetarayleigh = None
                self.alfaverorayleigh = None
                self.tempoalfaverorayleigh = None
                self.probabilitàtuttogiusto = None
                self.impostazioni["calcolaerrori"] = False
                print(
                    f"\033[35mNon ho potuto calcolare il beta per il test di Rayleigh perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")

    def rieseguialfaverorayleigh(self, iterazionialfaverorayleigh: int):
        if iterazionialfaverorayleigh <= 0:
            raise Exception("Valori dei parametri sbagliati")
        if self.taglia < 50 and self.impostazioni["calcolaerrori"]:
            try:
                if self.alfaverorayleigh is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1-self.alfaverorayleigh
                self.impostazioni["iterazionialfaverorayleigh"] = iterazionialfaverorayleigh
                if self.tempoalfaverorayleigh is not None:
                    self.tempototale -= self.tempoalfaverorayleigh
                tempo = time()
                self.alfaverorayleigh = miomodulo.alfaverorayleigh(
                    iterazionialfaverorayleigh,
                    st.chi2.ppf(1 - self.impostazioni["alfarayleigh"], df=2),
                    self.taglia)
                self.tempoalfaverorayleigh = time() - tempo
                self.tempototale += self.tempoalfaverorayleigh
                self.probabilitàtuttogiusto *= 1-self.alfaverorayleigh
                self.betarayleigh = None
                self.tempobetarayleigh = None
            except Exception as e:
                self.alfaverorayleigh = None
                self.tempoalfaverorayleigh = None
                self.betarayleigh = None
                self.tempobetarayleigh = None
                print(
                    f"\033[35mNon ho potuto calcolare l'alfa effettivo per il test di Rayleigh perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")

    def rieseguiaffidabilitàvonmises(self, alfatestvonmises: float):
        if alfatestvonmises <= 0 or alfatestvonmises >= 1:
            raise Exception("Valori dei parametri sbagliati")
        self.impostazioni["fittavonmises"] = True
        if self.taglia > 1:
            if self.kappa is not None:
                try:
                    if self.betaaffidabilitàvonmises is not None and self.probabilitàtuttogiusto is not None:
                        self.probabilitàtuttogiusto /= 1-self.betaaffidabilitàvonmises
                    elif self.alfaveroaffidabilitàvonmises is not None and self.probabilitàtuttogiusto is not None:
                        self.probabilitàtuttogiusto /= 1-self.alfaveroaffidabilitàvonmises
                    elif not (self.affidabilitàvonmises if self.affidabilitàvonmises is not None else True) and self.probabilitàtuttogiusto is not None:
                        self.probabilitàtuttogiusto /= 1-self.impostazioni["alfatestvonmises"]
                    self.impostazioni["alfatestvonmises"] = alfatestvonmises
                    if self.tempoaffidabilitàvonmises is not None:
                        self.tempototale -= self.tempoaffidabilitàvonmises
                    if self.tempobetaaffidabilitàvonmises is not None:
                        self.tempototale -= self.tempobetaaffidabilitàvonmises
                    if self.tempoalfaveroaffidabilitàvonmises is not None:
                        self.tempototale -= self.tempoalfaveroaffidabilitàvonmises
                    tempo = time()
                    self.affidabilitàvonmises = miomodulo.testvonmises(appiattisci(self.angoli) if isinstance(self.angoli, list) else appiattisci(self.angoli).tolist(),
                                                                       self.taglia,
                                                                       self.mediaangoli if self.mediaangoli is not None else st.circmean(appiattisci(self.angoli)),
                                                                       self.kappa,
                                                                       st.chi2.ppf(1 - alfatestvonmises, df=2))
                    self.tempoaffidabilitàvonmises = time() - tempo
                    self.tempototale += self.tempoaffidabilitàvonmises
                except Exception as e:
                    self.affidabilitàvonmises = None
                    self.tempoaffidabilitàvonmises = None
                    print(f"\033[35mNon ho potuto eseguire il test di goodness of fit della von Mises perché {e}\033[0m")
                if (self.affidabilitàvonmises if self.affidabilitàvonmises is not None else False) and self.impostazioni["calcolaerrori"]:
                    try:
                        tempo = time()
                        self.betaaffidabilitàvonmises = miomodulo.betaaffidabilitavonmises(
                            self.impostazioni["iterazionibetaaffidabilitàvonmises"],
                            self.impostazioni["uniformebetaaffidabilitàvonmises"],
                            self.taglia,
                            st.chi2.ppf(1 - alfatestvonmises, df=2),
                            self.impostazioni["distanzacomponentibetaaffidabilitàvonmises"],
                            self.impostazioni["kappabetaaffidabilitàvonmises"])
                        self.tempobetaaffidabilitàvonmises = time() - tempo
                        self.tempototale += self.tempobetaaffidabilitàvonmises
                        self.alfaveroaffidabilitàvonmises = None
                        self.tempoalfaveroaffidabilitàvonmises = None
                        self.probabilitàtuttogiusto *= 1 - self.betaaffidabilitàvonmises
                    except Exception as e:
                        self.betaaffidabilitàvonmises = None
                        self.tempobetaaffidabilitàvonmises = None
                        self.alfaveroaffidabilitàvonmises = None
                        self.tempoalfaveroaffidabilitàvonmises = None
                        self.probabilitàtuttogiusto = None
                        print(f"\033[35mNon ho potuto calcolare il beta per il test di goodness of fit della von Mises perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
                elif self.taglia < 50 and self.impostazioni["calcolaerrori"] and self.affidabilitàvonmises is not None:
                    try:
                        tempo = time()
                        self.alfaveroaffidabilitàvonmises = miomodulo.alfaveroaffidabilitavonmises(
                            self.impostazioni["iterazionialfaveroaffidabilitàvonmises"],
                            self.kappa,
                            self.taglia,
                            st.chi2.ppf(1 - alfatestvonmises, df=2))
                        self.tempoalfaveroaffidabilitàvonmises = time() - tempo
                        self.tempototale += self.tempoalfaveroaffidabilitàvonmises
                        self.betaaffidabilitàvonmises = None
                        self.tempobetaaffidabilitàvonmises = None
                        self.probabilitàtuttogiusto *= 1 - self.alfaveroaffidabilitàvonmises
                    except Exception as e:
                        self.alfaveroaffidabilitàvonmises = None
                        self.tempoalfaveroaffidabilitàvonmises = None
                        self.betaaffidabilitàvonmises = None
                        self.tempobetaaffidabilitàvonmises = None
                        self.probabilitàtuttogiusto = None
                        self.impostazioni["calcolaerrori"] = False
                        print(f"\033[35mNon ho potuto calcolare l'alfa effettivo per il test di goodness of fit della von Mises perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
                else:
                    if self.impostazioni["calcolaerrori"]:
                        self.probabilitàtuttogiusto *= 1 - alfatestvonmises
                    self.alfaveroaffidabilitàvonmises = None
                    self.tempoalfaveroaffidabilitàvonmises = None
                    self.betaaffidabilitàvonmises = None
                    self.tempobetaaffidabilitàvonmises = None

    def rieseguibetaaffidabilitàvonmises(self, iterazionibetaaffidabilitàvonmises: int, uniformeaffidabilitàvonmises: bool,
                                         distanzacomponentibetaaffidabilitàvonmises: float, kappabetaaffidabilitàvonmises: float):
        if iterazionibetaaffidabilitàvonmises <= 0 or kappabetaaffidabilitàvonmises <= 0:
            raise Exception("Valori dei parametri sbagliati")
        if (self.affidabilitàvonmises if self.affidabilitàvonmises is not None else False) and self.impostazioni[
            "calcolaerrori"]:
            try:
                if self.betaaffidabilitàvonmises is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1-self.betaaffidabilitàvonmises
                self.impostazioni["iterazionibetaaffidabilitàvonmises"] = iterazionibetaaffidabilitàvonmises
                self.impostazioni["uniformeaffidabilitàvonmises"] = uniformeaffidabilitàvonmises
                self.impostazioni["distanzacomponentibetaaffidabilitàvonmises"] = distanzacomponentibetaaffidabilitàvonmises
                self.impostazioni["kappabetaaffidabilitàvonmises"] = kappabetaaffidabilitàvonmises
                if self.tempobetaaffidabilitàvonmises is not None:
                    self.probabilitàtuttogiusto /= self.tempobetaaffidabilitàvonmises
                tempo = time()
                self.betaaffidabilitàvonmises = miomodulo.betaaffidabilitavonmises(
                    iterazionibetaaffidabilitàvonmises,
                    uniformeaffidabilitàvonmises,
                    self.taglia,
                    st.chi2.ppf(1 - self.impostazioni["alfatestvonmises"], df=2),
                    distanzacomponentibetaaffidabilitàvonmises,
                    kappabetaaffidabilitàvonmises)
                self.tempobetaaffidabilitàvonmises = time() - tempo
                self.tempototale += self.tempobetaaffidabilitàvonmises
                self.alfaveroaffidabilitàvonmises = None
                self.tempoalfaveroaffidabilitàvonmises = None
                self.probabilitàtuttogiusto *= 1 - self.betaaffidabilitàvonmises
            except Exception as e:
                self.betaaffidabilitàvonmises = None
                self.tempobetaaffidabilitàvonmises = None
                self.alfaveroaffidabilitàvonmises = None
                self.tempoalfaveroaffidabilitàvonmises = None
                self.probabilitàtuttogiusto = None
                print(
                    f"\033[35mNon ho potuto calcolare il beta per il test di goodness of fit della von Mises perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")

    def rieseguialfaveroaffidabilitàvonmises(self, iterazionialfaveroaffidabilitàvonmises: int):
        if iterazionialfaveroaffidabilitàvonmises <= 0:
            raise Exception("Valori dei parametri sbagliati")
        if self.taglia < 50 and self.impostazioni["calcolaerrori"] and not (self.affidabilitàvonmises if self.affidabilitàvonmises is not None else True):
            try:
                if self.alfaveroaffidabilitàvonmises is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1-self.alfaveroaffidabilitàvonmises
                self.impostazioni["iterazionialfaveroaffidabilitàvonmises"] = iterazionialfaveroaffidabilitàvonmises
                if self.tempoalfaveroaffidabilitàvonmises is not None:
                    self.tempototale -= self.tempoalfaveroaffidabilitàvonmises
                tempo = time()
                self.alfaveroaffidabilitàvonmises = miomodulo.alfaveroaffidabilitavonmises(
                    iterazionialfaveroaffidabilitàvonmises,
                    self.kappa,
                    self.taglia,
                    st.chi2.ppf(1 - self.impostazioni["alfatestvonmises"], df=2))
                self.tempoalfaveroaffidabilitàvonmises = time() - tempo
                self.tempototale += self.tempoalfaveroaffidabilitàvonmises
                self.betaaffidabilitàvonmises = None
                self.tempobetaaffidabilitàvonmises = None
                self.probabilitàtuttogiusto *= 1 - self.alfaveroaffidabilitàvonmises
            except Exception as e:
                self.alfaveroaffidabilitàvonmises = None
                self.tempoalfaveroaffidabilitàvonmises = None
                self.betaaffidabilitàvonmises = None
                self.tempobetaaffidabilitàvonmises = None
                self.probabilitàtuttogiusto = None
                self.impostazioni["calcolaerrori"] = False
                print(
                    f"\033[35mNon ho potuto calcolare l'alfa effettivo per il test di goodness of fit della von Mises perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")

    def rieseguiintervalloangolomedio(self, intervallomediavonmisesfacile: bool, confidenzaangolomedio: float):
        if confidenzaangolomedio <= 0 or confidenzaangolomedio >= 1:
            raise Exception("Valori dei parametri sbagliati")
        self.impostazioni["intervallomediavonmises"] = True
        self.impostazioni["fittavonmises"] = True
        if self.taglia > 1 and self.kappa is not None:
            try:
                if self.alfaverointervalloangolomedio is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1-self.alfaverointervalloangolomedio
                elif self.intervalloangolomedio is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= self.impostazioni["confidenzaangolomedio"]
                self.impostazioni["confidenzaangolomedio"] = confidenzaangolomedio
                if self.tempointervalloangolomedio is not None:
                    self.tempototale -= self.tempointervalloangolomedio
                tempo = time()
                self.intervalloangolomedio = intervalloangolomedio(
                    facile=intervallomediavonmisesfacile,
                    angolomedio=self.mediaangoli if self.mediaangoli is not None else st.circmean(self.angoli),
                    confidenza=confidenzaangolomedio,
                    kappa=self.kappa,
                    r=self.varianzaangoli if self.varianzaangoli is not None else varianzaangoli(self.angoli),
                    frecce=self.taglia)
                self.tempointervalloangolomedio = time() - tempo
                self.tempototale += self.tempointervalloangolomedio
            except Exception as e:
                self.intervalloangolomedio = None
                self.tempointervalloangolomedio = None
                print(f"\033[35mNon ho potuto calcolare l'intervallo di confidenza per l'angolo medio perché {e}")
            try:
                tempo = time()
                graficointervalloangolomedio(self.intervalloangolomedio, data)
                self.tempograficointervalloangolomedio = time() - tempo
                self.tempototale += self.tempograficointervalloangolomedio
            except Exception as e:
                self.tempograficointervalloangolomedio = None
                print(
                    f"\033[35mNon ho potuto disegnare il grafico dell'intervallo di confidenza per l'angolo medio perché {e}")
            if self.taglia < 50 and self.impostazioni["calcolaerrori"] and self.intervalloangolomedio is not None:
                try:
                    tempo = time()
                    self.alfaverointervalloangolomedio = miomodulo.alfaverointervalloangolomedio(
                        self.impostazioni["iterazionialfaveromediavonmises"],
                        self.kappa,
                        self.taglia,
                        intervallomediavonmisesfacile,
                        st.chi2.ppf(1 - confidenzaangolomedio, df=1))
                    self.tempoalfaverointervalloangolomedio = time() - tempo
                    self.tempototale += self.tempoalfaverointervalloangolomedio
                    self.probabilitàtuttogiusto *= 1 - self.alfaverointervalloangolomedio
                except Exception as e:
                    self.alfaverointervalloangolomedio = None
                    self.tempoalfaverointervalloangolomedio = None
                    self.probabilitàtuttogiusto = None
                    self.impostazioni["calcolaerrori"] = False
                    print(
                        f"\033[35mNon ho potuto calcolare l'alfa effettivo per l'intervallo di confidenza per l'angolo medio perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
            else:
                self.alfaverointervalloangolomedio = None
                self.tempoalfaverointervalloangolomedio = None
                if self.impostazioni["calcolaerrori"] and self.intervalloangolomedio is not None:
                    self.probabilitàtuttogiusto *= confidenzaangolomedio
        else:
            self.intervalloangolomedio = None
            self.tempointervalloangolomedio = None
            self.alfaverointervalloangolomedio = None
            self.tempoalfaverointervalloangolomedio = None
            self.tempograficointervalloangolomedio = None

    def rieseguialfaverointervalloangolomedio(self, iterazionialfaveromediavonmises: int):
        if iterazionialfaveromediavonmises <= 0:
            raise Exception("Valori dei parametri sbagliati")
        if self.taglia < 50 and self.impostazioni["calcolaerrori"] and self.intervalloangolomedio is not None:
            try:
                if self.alfaverointervalloangolomedio is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1-self.alfaverointervalloangolomedio
                self.impostazioni["iterazionialfaveromediavonmises"] = iterazionialfaveromediavonmises
                if self.tempoalfaverointervalloangolomedio is not None:
                    self.tempototale -= self.tempoalfaverointervalloangolomedio
                tempo = time()
                self.alfaverointervalloangolomedio = miomodulo.alfaverointervalloangolomedio(
                    iterazionialfaveromediavonmises,
                    self.kappa,
                    self.taglia,
                    self.impostazioni["intervallomediavonmisesfacile"],
                    st.chi2.ppf(1 - self.impostazioni["confidenzaangolomedio"], df=1))
                self.tempoalfaverointervalloangolomedio = time() - tempo
                self.tempototale += self.tempoalfaverointervalloangolomedio
                self.probabilitàtuttogiusto *= 1 - self.alfaverointervalloangolomedio
            except Exception as e:
                self.alfaverointervalloangolomedio = None
                self.tempoalfaverointervalloangolomedio = None
                self.probabilitàtuttogiusto = None
                self.impostazioni["calcolaerrori"] = False
                print(
                    f"\033[35mNon ho potuto calcolare l'alfa effettivo per l'intervallo di confidenza per l'angolo medio perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")

    def rieseguiintervallokappa(self, iterazioniintervallokappa: int, confidenzaintervallokappa: float):
        if iterazioniintervallokappa <= 0 or confidenzaintervallokappa <= 0 or confidenzaintervallokappa >= 1:
            raise Exception("Valori dei parametri sbagliati")
        self.impostazioni["intervallokappavonmises"] = True
        self.impostazioni["fittavonmises"] = True
        if self.taglia > 1 and self.kappa is not None:
            try:
                if self.alfaverointervallokappa is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1-self.alfaverointervallokappa
                elif self.intervallokappa is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= self.impostazioni["confidenzaintervallokappa"]
                self.impostazioni["iterazioniintervallokappa"] = iterazioniintervallokappa
                self.impostazioni["confidenzaintervallokappa"] = confidenzaintervallokappa
                if self.tempointervallokappa is not None:
                    self.tempototale -= self.tempointervallokappa
                tempo = time()
                self.intervallokappa = miomodulo.intervallokappa(appiattisci(self.angoli),
                                                                 iterazioniintervallokappa,
                                                                 confidenzaintervallokappa,
                                                                 self.taglia)
                self.tempointervallokappa = time() - tempo
                self.tempototale += self.tempointervallokappa
            except Exception as e:
                self.intervallokappa = None
                self.tempointervallokappa = None
                print(
                    f"\033[35mNon ho potuto calcolare l'intervallo di confidenza per il parametro di concentrazione perché {e}\033[0m")
            try:
                tempo = time()
                graficointervallokappa(intervallo=self.intervallokappa,
                                       media=self.mediaangoli,
                                       data=data)
                self.tempograficointervallokappa = time() - tempo
                self.tempototale += self.tempograficointervallokappa
            except Exception as e:
                self.tempograficointervallokappa = None
                print(
                    f"\033[35mNon ho potuto disegnare disegnare il grafico per l'intervallo di confidenza per il parametro di concentrazione perché {e}\033[0m")
            if self.taglia < 50 and self.impostazioni["calcolaerrori"] and self.intervallokappa is not None:
                try:
                    tempo = time()
                    self.alfaverointervallokappa = miomodulo.alfaverointervallokappa(
                        self.impostazioni["iterazionialfaverointervallokappa"],
                        confidenzaintervallokappa,
                        self.kappa,
                        iterazioniintervallokappa,
                        self.taglia)
                    self.tempoalfaverointervallokappa = time() - tempo
                    self.tempototale += self.tempoalfaverointervallokappa
                    self.probabilitàtuttogiusto *= 1 - self.alfaverointervallokappa
                except Exception as e:
                    self.alfaverointervallokappa = None
                    self.tempoalfaverointervallokappa = None
                    self.probabilitàtuttogiusto = None
                    self.impostazioni["calcolaerrori"] = False
                    print(
                        f"\033[35mNon ho potuto calcolare il livello di confidenza effettivo per l'intervallo di confidenza del parametro di concentrazione calcolato perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
            else:
                self.alfaverointervallokappa = None
                self.tempoalfaverointervallokappa = None
                if self.impostazioni["calcolaerrori"] and self.intervallokappa is not None:
                    self.probabilitàtuttogiusto *= 1 - confidenzaintervallokappa

    def rieseguialfaverointervallokappa(self, iterazionialfaverointervallokappa: int):
        if iterazionialfaverointervallokappa <= 0:
            raise Exception("Valori dei parametri sbagliati")
        if self.taglia < 50 and self.impostazioni["calcolaerrori"] and self.intervallokappa is not None:
            try:
                if self.alfaverointervallokappa is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1-self.alfaverointervallokappa
                self.impostazioni["iterazionialfaverointervallokappa"] = iterazionialfaverointervallokappa
                if self.tempoalfaverointervallokappa is not None:
                    self.tempototale -= self.tempoalfaverointervallokappa
                tempo = time()
                self.alfaverointervallokappa = miomodulo.alfaverointervallokappa(
                    iterazionialfaverointervallokappa,
                    self.impostazioni["confidenzaintervallokappa"],
                    self.kappa,
                    self.impostazioni["iterazioniintervallokappa"],
                    self.taglia)
                self.tempoalfaverointervallokappa = time() - tempo
                self.tempototale += self.tempoalfaverointervallokappa
                self.probabilitàtuttogiusto *= 1 - self.alfaverointervallokappa
            except Exception as e:
                self.alfaverointervallokappa = None
                self.tempoalfaverointervallokappa = None
                self.probabilitàtuttogiusto = None
                self.impostazioni["calcolaerrori"] = False
                print(
                    f"\033[35mNon ho potuto calcolare il livello di confidenza effettivo per l'intervallo di confidenza del parametro di concentrazione calcolato perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")

    def rieseguiangolomediobayesiano(self, cateneangolomediobayesiano: int,
                                     iterazionimcmcangolomediobayesiano: int, credibilitàangolomediobayesiano: float,
                                     credibilitàkappabayesiano: float, noninformativitàangolomediobayesiano: bool,
                                     allenamentiprioriangolomediobayesiano: int, variazionaleangolomediobayesiano: bool,
                                     angolomediobayesianogerarchico: bool, angolomediobayesianoapiacere: bool,
                                     muangolomediobayesiano: float, kappaangolomediobayesiano: float,
                                     alfaangolomediobayesiano: float, betaangolomediobayesiano: float):
        if cateneangolomediobayesiano <= 0 or iterazionimcmcangolomediobayesiano <= 0:
            raise Exception("Valori dei parametri sbagliati")
        if credibilitàangolomediobayesiano <= 0 or credibilitàangolomediobayesiano >= 1:
            raise Exception("Valori dei parametri sbagliati")
        if credibilitàkappabayesiano <= 0 or credibilitàkappabayesiano >= 1:
            raise Exception("Valori dei parametri sbagliati")
        if allenamentiprioriangolomediobayesiano > len(leggifile("Date.txt")) or allenamentiprioriangolomediobayesiano < 0:
            raise Exception("Valori dei parametri sbagliati")
        if allenamentiprioriangolomediobayesiano == 0 and not noninformativitàangolomediobayesiano:
            raise Exception("Valori dei parametri sbagliati")
        if muangolomediobayesiano <= 0 or muangolomediobayesiano >= 2*math.pi:
            raise Exception("Valori dei parametri sbagliati")
        if kappaangolomediobayesiano <= 0 or alfaangolomediobayesiano <= 0 or betaangolomediobayesiano <= 0:
            raise Exception("Valori dei parametri sbagliati")
        self.impostazioni["angolomediobayesiano"] = True
        self.impostazioni["kappabayesiano"] = True
        self.impostazioni["fittavonmises"] = True
        if self.taglia > 1:
            try:
                if self.intervalloangolomediobayesiano is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1-self.impostazioni["credibilitàangolomediobayesiano"]
                if self.intervallokappabayesiano is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= 1-self.impostazioni["credibilitàkappabayesiano"]
                self.impostazioni["cateneangolomediobayesiano"] = cateneangolomediobayesiano
                self.impostazioni["iterazionimcmcangolomediobayesiano"] = iterazionimcmcangolomediobayesiano
                self.impostazioni["credibilitàangolomediobayesiano"] = credibilitàangolomediobayesiano
                self.impostazioni["credibilitàkappabayesiano"] = credibilitàkappabayesiano
                self.impostazioni["noninformativitàangolomediobayesiano"] = noninformativitàangolomediobayesiano
                self.impostazioni["allenamentiprioriangolomediobayesiano"] = allenamentiprioriangolomediobayesiano
                self.impostazioni["variazionaleangolomediobayesiano"] = variazionaleangolomediobayesiano
                self.impostazioni["angolomediobayesianogerarchico"] = angolomediobayesianogerarchico
                self.impostazioni["angolomediobayesianoapiacere"] = angolomediobayesianoapiacere
                self.impostazioni["muangolomediobayesiano"] = muangolomediobayesiano
                self.impostazioni["kappaangolomediobayesiano"] = kappaangolomediobayesiano
                self.impostazioni["alfaangolomediobayesiano"] = alfaangolomediobayesiano
                self.impostazioni["betaangolomediobayesiano"] = betaangolomediobayesiano
                if self.tempointervalloangolomediobayesiano is not None:
                    self.tempototale -= self.tempointervalloangolomediobayesiano
                if self.tempograficoangolomediobayesiano is not None:
                    self.tempototale -= self.tempograficoangolomediobayesiano
                tempo = time()
                self.intervalloangolomediobayesiano, self.kappabayesiano, self.intervallokappabayesiano = angolomediobayesiano(
                    allenamento=appiattisci(self.angoli),
                    catene=cateneangolomediobayesiano,
                    iterazioni=iterazionimcmcangolomediobayesiano,
                    credibilitàmu=credibilitàangolomediobayesiano,
                    credibilitàkappa=credibilitàkappabayesiano,
                    noninformativa=noninformativitàangolomediobayesiano,
                    allenamentipriori=allenamentiprioriangolomediobayesiano,
                    variazionale=variazionaleangolomediobayesiano,
                    gerarchica=angolomediobayesianogerarchico,
                    apiacere=angolomediobayesianoapiacere,
                    mu=muangolomediobayesiano,
                    kappa=kappaangolomediobayesiano,
                    alfa=alfaangolomediobayesiano,
                    beta1=betaangolomediobayesiano,
                    console=self.impostazioni["consoledistan"])
                self.tempointervalloangolomediobayesiano = time() - tempo
                self.tempototale += self.tempointervalloangolomediobayesiano
                if self.impostazioni["calcolaerrori"]:
                    self.probabilitàtuttogiusto *= credibilitàangolomediobayesiano
                    self.probabilitàtuttogiusto *= credibilitàkappabayesiano
            except Exception as e:
                self.intervalloangolomediobayesiano, self.kappabayesiano, self.intervallokappabayesiano = None, None, None
                self.tempointervalloangolomediobayesiano = None
                print(f"\033[35mNon ho potuto calcolare le stime bayesiane per i parametri circolari perché {e}")
            try:
                tempo = time()
                graficoangolomediobayesiano(self.intervalloangolomediobayesiano, data)
                self.tempograficoangolomediobayesiano = time() - tempo
                self.tempototale += self.tempograficoangolomediobayesiano
            except Exception as e:
                self.tempograficoangolomediobayesiano = None
                print(f"\033[35mNon ho potuto disegnare il grafico per l'intervallo di credibilità dell'angolo medio perché {e}\033[0m")

    def rieseguimisturevonmises(self, componentifissatemisturevonmises: bool, noninformativitàmisturevonmises: bool,
                                allenamentipriorimisturevonmises: int, componentimisturevonmises: int,
                                credibilitàmumisturevonmises: float, credibilitàkappamisturevonmises: float):
        if allenamentipriorimisturevonmises > len(leggifile("Date.txt")) or allenamentipriorimisturevonmises < 0:
            raise Exception("Valori dei parametri sbagliati")
        if allenamentipriorimisturevonmises == 0 and not noninformativitàmisturevonmises:
            raise Exception("Valori dei parametri sbagliati")
        if componentimisturevonmises < 2 or componentimisturevonmises > self.taglia:
            raise Exception("Valori dei parametri sbagliati")
        if credibilitàmumisturevonmises <= 0 or credibilitàmumisturevonmises >= 1 or credibilitàkappamisturevonmises <= 0 or credibilitàkappamisturevonmises >= 1:
            raise Exception("Valori dei parametri sbagliati")
        self.impostazioni["angolomediobayesiano"] = True
        self.impostazioni["kappabayesiano"] = True
        self.impostazioni["misturevonmises"] = True
        self.impostazioni["fittavonmises"] = True
        if self.intervalloangolomediobayesiano is not None or self.kappabayesiano is not None or self.intervallokappabayesiano is not None:
            self.rieseguiangolomediobayesiano(self.impostazioni["cateneangolomediobayesiano"], self.impostazioni["iterazionimcmcangolomediobayesiano"],
                                              self.impostazioni["credibilitàangolomediobayesiano"], self.impostazioni["credibilitàkappabayesiano"],
                                              self.impostazioni["noninformativitàangolomediobayesiano"], self.impostazioni["allenamentiprioriangolomediobayesiano"],
                                              self.impostazioni["variazionaleangolomediobayesiano"], self.impostazioni["angolomediobayesianogerarchico"],
                                              self.impostazioni["angolomediobayesianoapiacere"], self.impostazioni["muangolomediobayesiano"],
                                              self.impostazioni["kappaangolomediobayesiano"], self.impostazioni["alfaangolomediobayesiano"],
                                              self.impostazioni["betaangolomediobayesiano"])
        if self.taglia > 1:
            try:
                if self.intervalliangolomedio is not None and self.componentimisturevonmises is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= self.impostazioni["credibilitàmumisturevonmises"]**len(self.componentimisturevonmises)
                if self.intervallikappa is not None and self.componentimisturevonmises is not None and self.probabilitàtuttogiusto is not None:
                    self.probabilitàtuttogiusto /= self.impostazioni["credibilitàkappamisturevonmises"]**len(self.componentimisturevonmises)
                self.impostazioni["componentifissatemisturevonmises"] = componentifissatemisturevonmises
                self.impostazioni["noninformativitàmisturevonmises"] = noninformativitàmisturevonmises
                self.impostazioni["allenamentipriorimisturevonmises"] = allenamentipriorimisturevonmises
                self.impostazioni["componentimisturevonmises"] = componentimisturevonmises
                self.impostazioni["credibilitàmumisturevonmises"] = credibilitàmumisturevonmises
                self.impostazioni["credibilitàkappamisturevonmises"] = credibilitàkappamisturevonmises
                if self.tempomisturevonmises is not None:
                    self.tempototale -= self.tempomisturevonmises
                if self.tempograficomisturevonmises is not None:
                    self.tempototale -= self.tempograficomisturevonmises
                tempo = time()
                self.intervalliangolomedio, self.intervallikappa, self.componentimisturevonmises, assegnazioni = misturevonmises(
                    componentifissate=componentifissatemisturevonmises,
                    allenamento=appiattisci(self.angoli),
                    noninformativa=noninformativitàmisturevonmises,
                    allenamentipriori=allenamentipriorimisturevonmises,
                    componentimassime=componentimisturevonmises,
                    credibilitàmu=credibilitàmumisturevonmises,
                    credibilitàkappa=credibilitàkappamisturevonmises)
                self.tempomisturevonmises = time() - tempo
                self.tempototale += self.tempomisturevonmises
                if self.impostazioni["calcolaerrori"]:
                    self.probabilitàtuttogiusto *= credibilitàmumisturevonmises ** len(self.componentimisturevonmises)
                    self.probabilitàtuttogiusto *= credibilitàkappamisturevonmises ** len(self.componentimisturevonmises)
            except Exception as e:
                self.intervalliangolomedio = None
                self.intervallikappa = None
                self.componentimisturevonmises = None
                self.tempomisturevonmises = None
                print(f"\033[35mNon ho potuto fittare il modello mistura per gli angoli perché {e}\033[0m")
            if self.intervalliangolomedio is not None and self.intervallikappa is not None:
                try:
                    tempo = time()
                    graficomisturevonmises(self.intervalliangolomedio, self.intervallikappa,
                                           self.componentimisturevonmises, appiattisci(self.angoli),
                                           assegnazioni, data)
                    self.tempograficomisturevonmises = time() - tempo
                    self.tempototale += self.tempograficomisturevonmises
                except Exception as e:
                    self.tempograficomisturevonmises = None
                    print(f"\033[35mNon ho potuto disegnare il grafico del modello mistura per gli angoli perché {e}")
            else:
                self.tempograficomisturevonmises = None

    def ricalcolatempototale(self):
        self.tempototale = sum([(tempo if tempo is not None else 0)
                                for tempo in {self.temporoutine, self.tempomediapunteggi, self.tempomediapunteggivolée,
                                              self.tempomediapunteggifrecce, self.tempomediacoordinate, self.tempomediacoordinatevolée,
                                              self.tempomediacoordinatefrecce, self.tempomediaangoli, self.tempomediaangolivolée,
                                              self.tempomediaangolifrecce, self.tempovarianzapunteggi, self.tempovarianzapunteggivolée,
                                              self.tempovarianzapunteggifrecce, self.tempovarianzacoordinate, self.tempovarianzacoordinatevolée,
                                              self.tempovarianzacoordinatefrecce, self.tempovarianzaangoli, self.tempovarianzaangolivolée,
                                              self.tempovarianzaangolifrecce, self.temporegressionepunteggi, self.temporegressionecoordinate,
                                              self.tempocorrelazione, self.tempocorrelazionevolée,
                                              self.tempocorrelazionefrecce, self.tempoautocorrelazionepunteggi, self.tempoautocorrelazioneascisse,
                                              self.tempoautocorrelazioneordinate, self.tempoautocorrelazioniangoli, self.tempograficodispersione,
                                              self.tempograficovolée, self.tempograficomisturevonmises, self.tempograficoangolomediobayesiano,
                                              self.tempograficomedie, self.tempograficoangoli, self.tempograficoautocorrelazioni,
                                              self.tempograficointervallicluster, self.tempograficointervallivarianze, self.tempograficointervalloangolomedio,
                                              self.tempograficointervallokappa, self.tempograficointervallonorme, self.tempograficointervallovarianzebayesiano,
                                              self.tempograficopunteggi, self.tempoljungboxascisse, self.tempoljungboxordinate,
                                              self.tempobetaljungboxascisse, self.tempobetaljungboxordinate, self.tempoalfaveroljungboxascisse,
                                              self.tempoalfaveroljungboxordinate, self.tempomardia, self.tempobetamardia,
                                              self.tempoalfaveromardia, self.tempoclustering, self.tempotesthotelling,
                                              self.tempobetatesthotelling, self.tempointervallohotelling, self.tempotestcluster,
                                              self.tempointervallicluster, self.tempohotellingbayesiano, self.tempoprobabilitàtestcluster,
                                              self.temponormenormali, self.tempobetanormenormali, self.tempoalfaveronormenormali,
                                              self.temponormeindipendenti, self.tempobetanormeindipendenti, self.tempoalfaveronormeindipendenti,
                                              self.temponormebayesiane, self.tempotestnorme, self.tempobetatestnorme,
                                              self.tempointervallonorme, self.tempodifettosità, self.tempodifettositàvarianze,
                                              self.tempobetahotellingduecampioni, self.tempobetavarianzeduecampioni, self.tempotestvarianze,
                                              self.tempointervallivarianze, self.tempobetatestvarianzeascisse, self.tempobetatestvarianzeordinate,
                                              self.tempouniformitàangoli,
                                              self.tempobetarayleigh, self.tempoalfaverorayleigh, self.tempoaffidabilitàvonmises,
                                              self.tempobetaaffidabilitàvonmises, self.tempoalfaveroaffidabilitàvonmises, self.tempointervalloangolomedio,
                                              self.tempokappa, self.tempointervalloangolomediobayesiano, self.tempokappabayesiano, self.tempointervallokappa,
                                              self.tempoalfaverointervalloangolomedio, self.tempoalfaverointervallokappa, self.tempomisturevonmises,
                                              self.tempoconfidenzaverabootstrapstazionariohotelling}])

    def calcolatuttierrori(self):
        self.impostazioni["calcolaerrori"] = True
        self.probabilitàtuttogiusto = 1
        self.betaljungboxascisse = None
        self.alfaveroljungboxascisse = None
        self.betaljungboxordinate = None
        self.alfaveroljungboxordinate = None
        self.betamardia = None
        self.alfaveromardia = None
        self.betahotelling = None
        self.probabilitàtestcluster = None
        self.betanormenormali = None
        self.alfaveronormenormali = None
        self.betanormeindipendenti = None
        self.alfaveronormeindipendenti = None
        self.betatestnorme = None
        self.betatestvarianzeascisse = None
        self.betatestvarianzeordinate = None
        self.betahotellingduecampioni = None
        self.betavarianzeduecampioni = None
        self.betarayleigh = None
        self.alfaverorayleigh = None
        self.betaaffidabilitàvonmises = None
        self.alfaveroaffidabilitàvonmises = None
        self.alfaverointervalloangolomedio = None
        self.alfaverointervallokappa = None
        self.confidenzaverabootstrapstazionariohotelling = None
        try:
            if self.ljungboxascisse is not None:
                if self.ljungboxascisse:
                    self.rieseguibetaljungboxascisse(self.impostazioni["voltebetaljungboxascisse"])
                elif self.taglia < 51:
                    self.rieseguialfaveroljungboxascisse(self.impostazioni["voltealfaljungboxascisse"])
                else:
                    h = self.impostazioni["hljungboxascisse"]
                    n = self.taglia
                    esponente = n//h if n%h == 0 else n//h+1
                    self.probabilitàtuttogiusto *= (1-self.impostazioni["alfaljungboxascisse"])**esponente
            if self.ljungboxordinate is not None:
                if self.ljungboxordinate:
                    self.rieseguibetaljungboxordinate(self.impostazioni["voltebetaljungboxordinate"])
                elif self.taglia < 51:
                    self.rieseguialfaveroljungboxascisse(self.impostazioni["voltealfaljungboxordinate"])
                else:
                    h = self.impostazioni["hljungboxordinate"]
                    n = self.taglia
                    esponente = n//h if n%h == 0 else n//h+1
                    self.probabilitàtuttogiusto *= (1-self.impostazioni["alfaljungboxordinate"])**esponente
            if self.mardia is not None:
                if self.mardia:
                    self.rieseguibetamardia(self.impostazioni["distribuzionebetamardia"], self.impostazioni["voltebetamardia"],
                                            self.impostazioni["graditbetamardia"], self.impostazioni["tipotestnormalità"],
                                            self.impostazioni["asimmetriaascissebetamardia"], self.impostazioni["asimmetriaordinatebetamardia"],
                                            self.impostazioni["distanzacomponentibetamardia"])
                elif self.taglia < 50:
                    self.rieseguialfaveromardia(self.impostazioni["voltealfamardia"], self.impostazioni["tipotestnormalità"])
                else:
                    self.probabilitàtuttogiusto *= (1-self.impostazioni["alfamardia"])
            if self.testhotelling is not None:
                if self.testhotelling:
                    self.rieseguibetatesthotelling(self.impostazioni["voltebetahotelling"], self.impostazioni["distanzabetahotelling"])
                else:
                    self.probabilitàtuttogiusto *= (1-self.impostazioni["alfahotelling"])
            if self.testcluster is not None:
                try:
                    tempo = time()
                    self.probabilitàtestcluster = 1
                    for test in range(len(self.testcluster)):
                        numerositàcluster = sum((1 if numero == test else 0) for numero in self.etichettefrecce)
                        if self.testcluster[test]:
                            self.probabilitàtestcluster *= 1 - miomodulo.betatesthotelling(numerositàcluster,
                                                                                           st.f.ppf(1 - self.impostazioni[
                                                                                               "alfahotelling"], 2,
                                                                                                    self.taglia - 2),
                                                                                           self.impostazioni[
                                                                                               "voltebetahotelling"],
                                                                                           self.impostazioni[
                                                                                               "distanzabetahotelling"],
                                                                                           self.varianzacoordinate[
                                                                                               0] if self.varianzacoordinate is not None else np.var(
                                                                                               appiattisci(
                                                                                                   self.ascisse),
                                                                                               ddof=1),
                                                                                           self.varianzacoordinate[
                                                                                               1] if self.varianzacoordinate is not None else np.var(
                                                                                               appiattisci(
                                                                                                   self.ordinate),
                                                                                               ddof=1))
                        else:
                            self.probabilitàtestcluster *= 1 - self.impostazioni["alfahotelling"]
                    self.probabilitàtuttogiusto *= self.probabilitàtestcluster
                    self.tempoprobabilitàtestcluster = time() - tempo
                except Exception:
                    self.probabilitàtestcluster = None
                    self.tempoprobabilitàtestcluster = None
                    raise Exception
            if esiste(f"./Grafici del {self.data}/graficointervallohotelling.png") or esiste(f"./Grafici del {self.data}/graficointervallohotellingdetrending.png"):
                self.probabilitàtuttogiusto *= self.impostazioni["confidenzahotelling"]
            if esiste(f"./Grafici del {self.data}/graficointervallohotellingstazionario.png"):
                self.confidenzaverabootstrapstazionariohotelling = 1.0-miomodulo.alfaverobootstrapstazionariohotelling(self.taglia,
                                                                                                                       self.impostazioni["varianzaascissebootstrapstazionariohotelling"],
                                                                                                                       self.impostazioni["varianzaordinatebootstrapstazionariohotelling"],
                                                                                                                       self.impostazioni["geometricastazionariahotelling"],
                                                                                                                       self.impostazioni["bootstrapstazionarihotelling"],
                                                                                                                       1-self.impostazioni["confidenzahotelling"],
                                                                                                                       self.impostazioni["iterazionistazionariehotelling"])
            if esiste(f"./Grafici del {self.data}/graficointervallohotellingmobile.png"):
                self.confidenzaverabootstrapmobilehotelling = 1.0-miomodulo.alfaverobootstrapmobilehotelling(self.taglia,
                                                                                                                       self.impostazioni["varianzaascissebootstrapmobilehotelling"],
                                                                                                                       self.impostazioni["varianzaordinatebootstrapmobilehotelling"],
                                                                                                                       self.impostazioni["lunghezzabloccomobilehotelling"],
                                                                                                                       self.impostazioni["bootstrapmobilihotelling"],
                                                                                                                       1-self.impostazioni["confidenzahotelling"],
                                                                                                                       self.impostazioni["iterazionimobilihotelling"])

            if esiste(f"./Grafici del {self.data}/graficointervallicluster.png"):
                self.probabilitàtuttogiusto *= self.impostazioni["confidenzahotelling"]**(max(self.etichettefrecce)+1)
            if self.normenormali is not None:
                if self.normenormali:
                    self.rieseguibetanormenormali(self.impostazioni["alternativanormenormali"], self.impostazioni["iterazionibetanormenormali"],
                                                  self.impostazioni["asimmetrianormenormali"], self.impostazioni["distanzacomponentinormenormali"],
                                                  self.impostazioni["graditnormenormali"])
                elif self.taglia < 50:
                    self.rieseguialfaveronormenormali(self.impostazioni["iterazionialfaveronormenormali"])
                else:
                    self.probabilitàtuttogiusto *= (1-self.impostazioni["alfashapiro"])
            if self.normeindipendenti is not None:
                if self.normeindipendenti:
                    self.rieseguibetanormeindipendenti(self.impostazioni["voltebetaljungboxnorme"])
                elif self.taglia < 50:
                    self.rieseguialfaveronormeindipendenti(self.impostazioni["voltealfaljungboxnorme"])
                else:
                    self.probabilitàtuttogiusto *= (1-self.impostazioni["alfaljungboxnorme"])
            if self.intervallonorme is not None:
                self.probabilitàtuttogiusto *= self.impostazioni["confidenzanorme"]
            if self.testnorme is not None:
                if self.testnorme:
                    self.rieseguibetatestnorme(self.impostazioni["distanzabetanorme"], self.impostazioni["iterazionibetanorme"])
                else:
                    self.probabilitàtuttogiusto *= 1-self.impostazioni["alfatestnorme"]
            if self.testvarianze is not None:
                self.rieseguibetatestvarianze(self.impostazioni["iterazionibetavarianzaascisse"], self.impostazioni["distanzabetavarianzaascisse"],
                                              self.impostazioni["iterazionibetavarianzaordinate"], self.impostazioni["distanzabetavarianzaordinate"])
            if self.difettosità is not None:
                self.rieseguifreccedifettose(self.impostazioni["alfahotellingduecampioni"], self.impostazioni["iterazionibetahotellingduecampioni"],
                                             self.impostazioni["distanzabetahotellingduecampioni"], self.impostazioni["alfavarianzeduecampioni"],
                                             self.impostazioni["iterazionibetavarianzeduecampioni"], self.impostazioni["distanzabetavarianzeduecampioni"])
            if self.uniformitàangoli is not None:
                if self.uniformitàangoli:
                    self.rieseguibetarayleigh(self.impostazioni["iterazionibetarayleigh"], self.impostazioni["kappabetarayleigh"])
                elif self.taglia < 50:
                    self.rieseguialfaverorayleigh(self.impostazioni["iterazionialfaverorayleigh"])
                else:
                    self.probabilitàtuttogiusto *= 1-self.impostazioni["alfarayleigh"]
            if self.affidabilitàvonmises is not None:
                if self.affidabilitàvonmises:
                    self.rieseguibetaaffidabilitàvonmises(self.impostazioni["iterazionibetaaffidabilitàvonmises"], self.impostazioni["uniformeaffidabilitàvonmises"],
                                                          self.impostazioni["distanzacomponentibetaaffidabilitàvonmises"], self.impostazioni["kappabetaaffidabilitàvonmises"])
                elif self.taglia < 50:
                    self.rieseguialfaveroaffidabilitàvonmises(self.impostazioni["iterazionialfaveroaffidabilitàvonmises"])
                else:
                    self.probabilitàtuttogiusto *= 1-self.impostazioni["alfatestvonmises"]
            if self.intervalloangolomedio is not None:
                self.rieseguialfaverointervalloangolomedio(self.impostazioni["iterazionialfaveromediavonmises"])
            if self.intervallokappa is not None:
                self.rieseguialfaverointervallokappa(self.impostazioni["iterazionialfaverointervallokappa"])
            self.probabilitàtuttogiusto *= max(self.hotellingbayesiano, 1-self.hotellingbayesiano)
            self.probabilitàtuttogiusto *= max(self.normebayesiane, 1-self.normebayesiane)
            self.probabilitàtuttogiusto *= max(self.varianzebayesiane, 1-self.varianzebayesiane)
        except Exception as e:
            self.impostazioni["calcolaerrori"] = False
            self.probabilitàtuttogiusto = None
            print(f"\033[35mRicalcolo degli errori fallito: {e}\033[0m")

    def applicacorrezionebonferroni(self, alfacomplessivo: float):
        if alfacomplessivo <= 0 or alfacomplessivo >= 1:
            raise Exception("Valori dei parametri sbagliati")
        self.impostazioni["calcolaerrori"] = True
        alfe = list()
        if self.impostazioni["testhotelling"]:
            alfe.append("alfahotelling")
            if self.impostazioni["clustering"]:
                for _ in range(max(self.etichettefrecce)+1):
                    alfe.append("alfahotelling")
        if self.impostazioni["difettosità"]:
            alfe.append("alfahotellingduecampioni")
            alfe.append("alfavarianzeduecampioni")
        if self.impostazioni["testhotelling"] or self.impostazioni["intervallohotelling"]:
            for _ in range((self.taglia//self.impostazioni["hljungboxascisse"])+(0 if self.taglia % self.impostazioni["hljungboxascisse"] == 0 else 1)):
                alfe.append("alfaljungboxascisse")
            for _ in range((self.taglia//self.impostazioni["hljungboxordinate"])+(0 if self.taglia % self.impostazioni["hljungboxordinate"] == 0 else 1)):
                alfe.append("alfaljungboxordinate")
            alfe.append("alfamardia")
        if self.impostazioni["testnorme"] or self.impostazioni["intervallonorme"]:
            alfe.append("alfashapiro")
            for _ in range((self.taglia//self.impostazioni["hljungboxnorme"])+(0 if self.taglia % self.impostazioni["hljungboxnorme"] else 1)):
                alfe.append("alfaljungboxnorme")
        if self.impostazioni["uniformitàangoli"]:
            alfe.append("alfarayleigh")
        if self.impostazioni["testnorme"]:
            alfe.append("alfatestnorme")
        if self.impostazioni["fittavonmises"]:
            alfe.append("alfatestvonmises")
        if self.impostazioni["testvarianze"]:
            alfe.append("alfavarianzaascisse")
            alfe.append("alfavarianzaordinate")
        crediconfidenze = list()
        if self.impostazioni["intervallomediavonmises"]:
            crediconfidenze.append("confidenzaangolomedio")
        if self.impostazioni["intervallohotelling"]:
            crediconfidenze.append("confidenzahotelling")
            if self.impostazioni["correggidipendenzahotelling"] == "stazionario":
                crediconfidenze.append("confidenzahotelling")
            if self.impostazioni["clustering"]:
                for _ in range(max(self.etichettefrecce)+1):
                    crediconfidenze.append("confidenzahotelling")
        if self.impostazioni["intervallobayesiano"]:
            crediconfidenze.append("confidenzahotellingbayesiano")
        if self.impostazioni["intervallokappavonmises"]:
            crediconfidenze.append("confidenzaintervallokappa")
        if self.impostazioni["intervallonorme"]:
            crediconfidenze.append("confidenzanorme")
        if self.impostazioni["intervallivarianze"]:
            crediconfidenze.append("confidenzavarianzaascisse")
            crediconfidenze.append("confidenzavarianzaordinate")
        if self.impostazioni["angolomediobayesiano"]:
            crediconfidenze.append("credibilitàangolomediobayesiano")
        if self.impostazioni["kappabayesiano"]:
            crediconfidenze.append("credibilitàkappabayesiano")
        if self.impostazioni["misturevonmises"]:
            for _ in range(len(self.componentimisturevonmises)):
                crediconfidenze.append("credibilitàkappamisturevonmises")
                crediconfidenze.append("credibilitàmumisturevonmises")
        if self.impostazioni["intervallonormebayesiano"]:
            crediconfidenze.append("credibilitànormebayesiane")
        if self.impostazioni["intervallovarianzebayesiano"]:
            crediconfidenze.append("credibilitàvarianzaascissebayesiane")
            crediconfidenze.append("credibilitàvarianzaordinatebayesiane")
        alfabonferroni = alfacomplessivo/(len(alfe)+len(crediconfidenze))
        for alfa in alfe:
            self.impostazioni[alfa] = alfabonferroni
        for crediconfidenza in crediconfidenze:
            self.impostazioni[crediconfidenza] = alfabonferroni
        if self.identificazione:
            allenamento = [[[a, o] for a, o in zip(va, vo)] for va, vo in zip(self.ascisse, self.ordinate)]
        else:
            allenamento = [[[a[0], o[0], a[1]] for a, o in zip(va, vo)] for va, vo in zip(self.ascisseidentificate, self.ordinateidentificate)]
        self.__init__(allenamento, self.ordine, self.identificazione, self.data, self.impostazioni, self.arco, self.distanza, self.tag)

    def __init__(self, allenamento, ordine: bool, identificazione: bool, data: str, impostazioni: dict, arco,
                 distanza: int, tag):
        for volée in range(len(allenamento)):
            for freccia in range(len(allenamento[volée])):
                if allenamento[volée][freccia][0] == 0.0 and allenamento[volée][freccia][1] == 0.0:
                    if identificazione:
                        allenamento[volée][freccia] = (0.0, 0.00001, allenamento[volée][freccia][2])
                    else:
                        allenamento[volée][freccia] = (0.0, 0.00001)
        print("Calcolo delle statistiche sull'allenamento in corso")
        tempoiniziale = time()
        makedirs(f"./Grafici del {data}", exist_ok=True)
        print("0,27%: cartella per i grafici creata")
        self.tag = tag
        self.ordine = ordine
        self.identificazione = identificazione
        self.volée = len(allenamento)
        if identificazione:
            self.frecce = max(freccia[2] for volée in allenamento for freccia in volée)+1
        else:
            self.frecce = None
        print("0,61%: volée e frecce contate")
        self.data = data
        self.arco = arco
        self.distanza = distanza
        print("0,88%: impostazioni dell'allenamento salvate")
        if identificazione:
            tempo = time()
            self.taglia = taglia(allenamento, True)
            if self.taglia < 1:
                raise IndexError("L'allenamento è vuoto")
            self.punteggi, self.punteggiidentificati = punteggi(allenamento, True, ordine)
            self.ascisse, self.ordinate, self.ascisseidentificate, self.ordinateidentificate = coordinate(allenamento,
                                                                                                          True, ordine)
            self.angoli, self.angoliidentificati = angoli(allenamento, True, ordine)
            self.norme, self.normeidentificate = norme(allenamento, True, ordine)
            self.temporoutine = time() - tempo
            print("1,49%: sistemazione dei dati eseguita")
            if impostazioni["mediapunteggifrecce"] and self.frecce > 1:
                try:
                    tempo = time()
                    self.mediapunteggifrecce = mediapunteggifrecce(self.punteggiidentificati)
                    self.tempomediapunteggifrecce = time() - tempo
                    print("1,83%: mediapunteggifrecce calcolata")
                except Exception as e:
                    self.mediapunteggifrecce = None
                    self.tempomediapunteggifrecce = None
                    print(f"\033[35mNon ho potuto calcolare mediapunteggifrecce perché {e}\033[0m")
            else:
                self.mediapunteggifrecce = None
                self.tempomediapunteggifrecce = None
                print("2,3%: mediapunteggifrecce non calcolata")
            if impostazioni["mediacoordinatefrecce"] and self.frecce > 1:
                try:
                    tempo = time()
                    self.mediacoordinatefrecce = mediacoordinatefrecce(self.ascisseidentificate,
                                                                       self.ordinateidentificate)
                    self.tempomediacoordinatefrecce = time() - tempo
                    print("2,64%: mediacoordinatefrecce calcolata")
                except Exception as e:
                    self.mediacoordinatefrecce = None
                    self.tempomediacoordinatefrecce = None
                    print(f"\033[35mNon ho potuto calcolare mediacoordinatefrecce perché {e}\033[0m")
            else:
                self.mediacoordinatefrecce = None
                self.tempomediacoordinatefrecce = None
                print("3,12%: mediacoordinatefrecce non calcolata")
            if impostazioni["mediaangolifrecce"] and self.frecce > 1:
                try:
                    tempo = time()
                    self.mediaangolifrecce = mediaangolifrecce(self.angoliidentificati)
                    self.tempomediaangolifrecce = time() - tempo
                    print("3,45%: mediaangolifrecce calcolata")
                except Exception as e:
                    self.mediaangolifrecce = None
                    self.tempomediacoordinatefrecce = None
                    print(f"\033[35mNon ho potuto calcolare mediaangolifrecce perché {e}\033[0m")
            else:
                self.mediaangolifrecce = None
                self.tempomediaangolifrecce = None
                print("3,93%: mediaangolifrecce non calcolata")
            if impostazioni["varianzapunteggifrecce"] and self.frecce > 1:
                try:
                    tempo = time()
                    self.varianzapunteggifrecce = varianzapunteggifrecce(self.punteggiidentificati)
                    self.tempovarianzapunteggifrecce = time() - tempo
                    print("4,27%: varianzapunteggifrecce calcolata")
                except Exception as e:
                    self.varianzapunteggifrecce = None
                    self.tempovarianzapunteggifrecce = None
                    print(f"\033[35mNon ho potuto calcolare varianzapunteggifrecce perché {e}\033[0m")
            else:
                self.varianzapunteggifrecce = None
                self.tempovarianzapunteggifrecce = None
                print("4,74%: varianzapunteggifrecce non calcolata")
            if impostazioni["varianzacoordinatefrecce"] and self.frecce > 1:
                try:
                    tempo = time()
                    self.varianzacoordinatefrecce = varianzacoordinatefrecce(self.ascisseidentificate,
                                                                             self.ordinateidentificate)
                    self.tempovarianzacoordinatefrecce = time() - tempo
                    print("5,08%: varianzacoordinatefrecce calcolata")
                except Exception as e:
                    self.varianzacoordinatefrecce = None
                    self.tempovarianzacoordinatefrecce = None
                    print(f"\033[35mNon ho potuto calcolare varianzacoordinatefrecce perché {e}\033[0m")
            else:
                self.varianzacoordinatefrecce = None
                self.tempovarianzacoordinatefrecce = None
                print("5,29%: varianzacoordinatefrecce non calcolata")
            if impostazioni["varianzaangolifrecce"] and self.frecce > 1:
                try:
                    tempo = time()
                    self.varianzaangolifrecce = varianzaangolifrecce(self.angoliidentificati)
                    self.tempovarianzaangolifrecce = time() - tempo
                    print("5,56%: varianzaangolifrecce calcolata")
                except Exception as e:
                    self.varianzaangolifrecce = None
                    self.tempovarianzaangolifrecce = None
                    print(f"\033[35mNon ho potuto calcolare varianzaangolifrecce perché {e}\033[0m")
            else:
                self.varianzaangolifrecce = None
                self.tempovarianzaangolifrecce = None
                print("5,76%: varianzaangolifrecce non calcolata")
            if impostazioni["correlazionefrecce"] and self.frecce > 1:
                try:
                    tempo = time()
                    self.correlazionefrecce = correlazionefrecce(self.ascisseidentificate,
                                                                 self.ordinateidentificate,
                                                                 self.frecce)
                    self.tempocorrelazionefrecce = time() - tempo
                    print("6,03%: correlazionefrecce calcolata")
                except Exception as e:
                    self.correlazionefrecce = None
                    self.tempocorrelazionefrecce = None
                    print(f"\033[35mNon ho potuto calcolare correlazionefrecce perché {e}\033[0m")
            else:
                self.correlazionefrecce = None
                self.tempocorrelazionefrecce = None
                print("6,24%: correlazionefrecce non calcolata")
        else:
            tempo = time()
            self.taglia = taglia(allenamento, False)
            if self.taglia < 1:
                raise IndexError("L'allenamento è vuoto")
            self.punteggi = punteggi(allenamento, False, ordine)
            self.ascisse, self.ordinate = coordinate(allenamento, False, ordine)
            self.angoli = angoli(allenamento, False, ordine)
            self.norme = norme(allenamento, False, ordine)
            self.temporoutine = time() - tempo
            print("6,78%: sistemazione dei dati eseguita")
            self.punteggiidentificati = None
            self.ascisseidentificate = None
            self.ordinateidentificate = None
            self.angoliidentificati = None
            self.normeidentificate = None
            self.mediapunteggifrecce = None
            self.tempomediapunteggifrecce = None
            self.mediacoordinatefrecce = None
            self.tempomediacoordinatefrecce = None
            self.mediaangolifrecce = None
            self.tempomediaangolifrecce = None
            self.varianzapunteggifrecce = None
            self.tempovarianzapunteggifrecce = None
            self.varianzacoordinatefrecce = None
            self.tempovarianzacoordinatefrecce = None
            self.varianzaangolifrecce = None
            self.tempovarianzaangolifrecce = None
            self.correlazionecoordinatefrecce = None
            self.tempocorrelazionecoordinatefrecce = None
            print("Statistiche per frecce non calcolate")
        try:
            tempo = time()
            graficodispersione(self.ascisse, self.ordinate, data)
            self.tempograficodispersione = time() - tempo
            print("8,27%: grafico di dispersione disegnato")
        except Exception as e:
            self.tempograficodispersione = None
            print(f"\033[35mNon ho potuto disegnare graficodispersione perché {e}\033[0m")
        if impostazioni["mediapunteggi"]:
            try:
                tempo = time()
                self.mediapunteggi = np.mean(appiattisci(self.punteggi))
                self.tempomediapunteggi = time() - tempo
                print("8,54%: mediapunteggi calcolata")
            except Exception as e:
                self.mediapunteggi = None
                self.tempomediapunteggi = None
                print(f"\033[35mNon ho potuto calcolare mediapunteggi perché {e}\033[0m")
        else:
            self.mediapunteggi = None
            self.tempomediapunteggi = None
            print("Mediapunteggi non calcolata")
        if impostazioni["mediapunteggivolée"] and self.volée > 1:
            try:
                tempo = time()
                self.mediapunteggivolée = np.array([np.mean(np.array(volée)) for volée in self.punteggi])
                self.tempomediapunteggivolée = time() - tempo
                print("9,02%: mediapunteggivolée calcolata")
            except Exception as e:
                self.mediapunteggivolée = None
                self.tempomediapunteggivolée = None
                print(f"\033[35mNon ho potuto calcolare mediapunteggivolée perché {e}\033[0m")
        else:
            self.mediapunteggivolée = None
            self.tempomediapunteggivolée = None
            print("Mediapunteggivolée non calcolata")
        if impostazioni["mediacoordinate"]:
            try:
                tempo = time()
                self.mediacoordinate = (np.mean(appiattisci(self.ascisse)),
                                        np.mean(appiattisci(self.ordinate)))
                self.tempomediacoordinate = time() - tempo
                print("9,49%: mediacoordinate calcolata")
            except Exception as e:
                self.mediacoordinate = None
                self.tempomediacoordinate = None
                print(f"\033[35mNon ho potuto calcolare mediacoordinate perché {e}\033[0m")
        else:
            self.mediacoordinate = None
            self.tempomediacoordinate = None
            print("Mediacoordinate non calcolata")
        if impostazioni["mediacoordinatevolée"] and self.volée > 1:
            try:
                tempo = time()
                self.mediacoordinatevolée = [(np.mean(np.array(self.ascisse[i])),
                                              np.mean(np.array(self.ordinate[i]))) for i in range(len(self.ascisse))]
                self.tempomediacoordinatevolée = time() - tempo
                print("10,1%: mediacoordinatevolée calcolata")
            except Exception as e:
                self.mediacoordinatevolée = None
                self.tempomediacoordinatevolée = None
                print(f"\033[35mNon ho potuto calcolare mediacoordinatevolée perché {e}\033[0m")
        else:
            self.mediacoordinatevolée = None
            self.tempomediacoordinatevolée = None
            print("Mediacoordinatevolée non calcolata")
        if impostazioni["mediaangoli"]:
            try:
                tempo = time()
                self.mediaangoli = st.circmean(appiattisci(self.angoli))
                self.tempomediaangoli = time() - tempo
                print("10,58%: mediaangoli calcolata")
            except Exception as e:
                self.mediaangoli = None
                self.tempomediaangoli = None
                print(f"\033[35mNon ho potuto calcolare mediaangoli perché {e}\033[0m")
        else:
            self.mediaangoli = None
            self.tempomediaangoli = None
            print("Mediaangoli non calcolata")
        if impostazioni["mediaangolivolée"] and self.volée > 1:
            try:
                tempo = time()
                self.mediaangolivolée = np.array([st.circmean(volée) for volée in self.angoli])
                self.tempomediaangolivolée = time() - tempo
                print("11,05%: mediaangolivolée calcolata")
            except Exception as e:
                self.mediaangolivolée = None
                self.tempomediaangolivolée = None
                print(f"\033[35mNon ho potuto calcolare mediaangolivolée perché {e}\033[0m")
        else:
            self.mediaangolivolée = None
            self.tempomediaangolivolée = None
            print("11,26%: mediaangolivolée non calcolata")
        if impostazioni["varianzapunteggi"] and self.taglia > 1:
            try:
                tempo = time()
                self.varianzapunteggi = np.var(appiattisci(self.punteggi), ddof=1)
                self.tempovarianzapunteggi = time() - tempo
                print("11,53%: varianzapunteggi calcolata")
            except Exception as e:
                self.varianzapunteggi = None
                self.tempovarianzapunteggi = None
                print(f"\033[35mNon ho potuto calcolare varianzapunteggi perché {e}\033[0m")
        else:
            self.varianzapunteggi = None
            self.tempovarianzapunteggi = None
            print("11,73%: varianzapunteggi non calcolata")
        if impostazioni["varianzapunteggivolée"] and self.volée > 1:
            try:
                tempo = time()
                self.varianzapunteggivolée = np.array([(np.var(np.array(volée), ddof=1) if len(volée) > 1 else 0)
                                                       for volée in self.punteggi])
                self.tempovarianzapunteggivolée = time() - tempo
                print("12,07%: varianzapunteggivolée calcolata")
            except Exception as e:
                self.varianzapunteggivolée = None
                self.tempovarianzapunteggivolée = None
                print(f"\033[35mNon ho potuto calcolare varianzapunteggivolée perché {e}\033[0m")
        else:
            self.varianzapunteggivolée = None
            self.tempovarianzapunteggivolée = None
            print("12,27%: varianzapunteggivolée non calcolata")
        if impostazioni["varianzacoordinate"] and self.taglia > 1:
            try:
                tempo = time()
                self.varianzacoordinate = (np.var(appiattisci(self.ascisse), ddof=1),
                                           np.var(appiattisci(self.ordinate), ddof=1))
                self.tempovarianzacoordinate = time() - tempo
                print("12,61%: varianzacoordinate calcolata")
            except Exception as e:
                self.varianzacoordinate = None
                self.tempovarianzacoordinate = None
                print(f"\033[35mNon ho potuto calcolare varianzacoordinate perché {e}\033[0m")
        else:
            self.varianzacoordinate = None
            self.tempovarianzacoordinate = None
            print("12,82%: varianzacoordinate non calcolata")
        if impostazioni["varianzacoordinatevolée"] and self.volée > 1:
            try:
                tempo = time()
                self.varianzacoordinatevolée = [((np.var(np.array(self.ascisse[i]), ddof=1),
                                                  np.var(np.array(self.ordinate[i]), ddof=1))
                                                 if len(self.ascisse[i]) > 1 else 0)
                                                for i in range(len(self.ascisse))]
                self.tempovarianzacoordinatevolée = time() - tempo
                print("13,29: varianzacoordinatevolée calcolata")
            except Exception as e:
                self.varianzacoordinatevolée = None
                self.tempovarianzacoordinatevolée = None
                print(f"\033[35mNon ho potuto calcolare varianzacoordinatevolée perché {e}\033[0m")
        else:
            self.varianzacoordinatevolée = None
            self.tempovarianzacoordiantevolée = None
            print("13,5%: varianzacoordinatevolée non calcolata")
        if impostazioni["varianzaangoli"] and self.taglia > 1:
            try:
                tempo = time()
                self.varianzaangoli = varianzaangoli(appiattisci(self.angoli))
                self.tempovarianzaangoli = time() - tempo
                print("13,77%: varianzaangoli calcolata")
            except Exception as e:
                self.varianzaangoli = None
                self.tempovarianzaangoli = None
                print(f"\033[35mNon ho potuto calcolare varianzaangoli perché {e}\033[0m")
        else:
            self.varianzaangoli = None
            self.tempovarianzaangoli = None
            print("13,97%: varianzaangoli non calcolata")
        if impostazioni["varianzaangolivolée"] and self.volée > 1:
            try:
                tempo = time()
                self.varianzaangolivolée = np.array([(varianzaangoli(volée) if len(volée) > 1 else 0)
                                                     for volée in self.angoli])
                self.tempovarianzaangolivolée = time() - tempo
                print("14,31%: varianzaangolivolée calcolata")
            except Exception as e:
                self.varianzaangolivolée = None
                self.tempovarianzaangolivolée = None
                print(f"\033[35mNon ho potuto calcolare varianzaangolivolée perché {e}\033[0m")
        else:
            self.varianzaangolivolée = None
            self.tempovarianzaangolivolée = None
            print("14,51%: varianzaangolivolée non calcolata")
        if impostazioni["correlazione"] and self.taglia > 1:
            try:
                tempo = time()
                self.correlazione = np.corrcoef(appiattisci(self.ascisse),
                                                appiattisci(self.ordinate))[0, 1]
                self.tempocorrelazione = time() - tempo
                print("14,85%: correlazione calcolata")
            except Exception as e:
                self.correlazione = None
                self.tempocorrelazione = None
                print(f"\033[35mNon ho potuto calcolare correlazione perché {e}\033[0m")
        else:
            self.correlazione = None
            self.tempocorrelazione = None
            print("15,06%: correlazione non calcolata")
        if impostazioni["correlazionevolée"] and self.volée > 1:
            try:
                tempo = time()
                self.correlazionevolée = np.array([(np.corrcoef(self.ascisse[i],
                                                                self.ordinate[i])[0, 1] if len(
                    self.ascisse[i]) > 1 else 0)
                                                   for i in range(len(self.ascisse))])
                self.tempocorrelazionevolée = time() - tempo
                print("15,46%: correlazionevolée calcolata")
            except Exception as e:
                self.correlazionevolée = None
                self.tempocorrelazionevolée = None
                print(f"\033[35mNon ho potuto calcolare correlazionevolée perché {e}\033[0m")
        else:
            self.correlazionevolée = None
            self.tempocorrelazionevolée = None
            print("15,67%: correlazionevolée non calcolata")
        if ordine and self.taglia > 1:
            if impostazioni["regressionepunteggi"]:
                try:
                    tempo = time()
                    self.regressionepunteggi = np.polyfit([i for i in range(self.taglia)],
                                                          appiattisci(self.punteggi),
                                                          1)
                    self.temporegressionepunteggi = time() - tempo
                    print("16,14%: regressionepunteggi calcolata")
                except Exception as e:
                    self.regressionepunteggi = None
                    self.temporegressionepunteggi = None
                    print(f"\033[35mNon ho potuto calcolare regressionepunteggi perché {e}\033[0m")
            else:
                self.regressionepunteggi = None
                self.temporegressionepunteggi = None
                print("16,35%: regressionepunteggi non calcolata")
            if impostazioni["regressionecoordinate"]:
                try:
                    tempo = time()
                    self.regressionecoordinate = (np.polyfit([i for i in range(self.taglia)],
                                                             appiattisci(self.ascisse),
                                                             1),
                                                  np.polyfit([i for i in range(self.taglia)],
                                                             appiattisci(self.ordinate),
                                                             1))
                    self.temporegressionecoordinate = time() - tempo
                    print("16,96%: regressionecoordinate calcolata")
                except Exception as e:
                    self.regressionecoordinate = None
                    self.temporegressionecoordinate = None
                    print(f"\033[35mNon ho potuto calcolare regressionecoordinate perché {e}\033[0m")
            else:
                self.regressionecoordinate = None
                self.temporegressionecoordinate = None
                print("17,16%: regressionecoordinate non calcolata")
            if impostazioni["autocorrelazionepunteggi"]:
                try:
                    tempo = time()
                    self.autocorrelazionepunteggi = autocorrelazioni(appiattisci(self.punteggi),
                                                                     self.varianzapunteggi
                                                                     if self.varianzapunteggi is not None
                                                                     else np.var(appiattisci(self.punteggi)))
                    self.tempoautocorrelazionepunteggi = time() - tempo
                    print("17,63%: autocorrelazionepunteggi calcolata")
                except Exception as e:
                    self.autocorrelazionepunteggi = None
                    self.tempoautocorrelazionepunteggi = None
                    print(f"\033[35mNon ho potuto calcolare autocorrelazionepunteggi perché {e}\033[0m")
            else:
                self.autocorrelazionepunteggi = None
                self.tempoautocorrelazionepunteggi = None
                print("17,84%: autocorrelazionepunteggi non calcolata")
            if impostazioni["autocorrelazioneascisse"]:
                try:
                    tempo = time()
                    self.autocorrelazioneascisse = autocorrelazioni(appiattisci(self.ascisse),
                                                                    self.varianzacoordinate[0]
                                                                    if self.varianzacoordinate is not None
                                                                    else np.var(appiattisci(self.ascisse)))
                    self.tempoautocorrelazioneascisse = time() - tempo
                    print("18,31%: autocorrelazioneascisse calcolata")
                except Exception as e:
                    self.autocorrelazioneascisse = None
                    self.tempoautocorrelazioneascisse = None
                    print(f"\033[35mNon ho potuto calcolare autocorrelazioneascisse perché {e}\033[0m")
            else:
                self.autocorrelazioneascisse = None
                self.tempoautocorrelazioneascisse = None
                print("18,52%: autocorrelazioneascisse non calcolata")
            if impostazioni["autocorrelazioneordinate"]:
                try:
                    tempo = time()
                    self.autocorrelazioneordinate = autocorrelazioni(appiattisci(self.ordinate),
                                                                     self.varianzacoordinate[1]
                                                                     if self.varianzacoordinate is not None
                                                                     else np.var(appiattisci(self.ordinate)))
                    self.tempoautocorrelazioneordinate = time() - tempo
                    print("18,99%: autocorrelazioneordinate calcolata")
                except Exception as e:
                    self.autocorrelazioneordinate = None
                    self.tempoautocorrelazioneordinate = None
                    print(f"\033[35mNon ho potuto calcolare autocorrelazioneordinate perché {e}\033[0m")
            else:
                self.autocorrelazioneordinate = None
                self.tempoautocorrelazioneordinate = None
                print("19,19%: autocorrelazione non calcolata")
            if impostazioni["autocorrelazioniangoli"]:
                try:
                    tempo = time()
                    self.autocorrelazioniangolipositive, self.autocorrelazioniangolinegative = miomodulo.autocorrelazioniangolari(
                        appiattisci(self.angoli), self.taglia)
                    self.tempoautocorrelazioniangoli = time() - tempo
                    print("19,47%: autocorrelazioniangoli calcolate")
                except Exception as e:
                    self.autocorrelazioniangolipositive = None
                    self.autocorrelazioniangolinegative = None
                    self.tempoautocorrelazioniangoli = None
                    print(f"\033[35mNon ho potuto calcolare autocorrelazioniangoli perché {e}\033[0m")
            else:
                self.autocorrelazioniangolipositive = None
                self.autocorrelazioniangolinegative = None
                self.tempoautocorrelazioniangoli = None
                print("19,74%: autocorrelazioniangoli non calcolate")
        elif self.volée > 1:
            if impostazioni["regressionepunteggi"]:
                try:
                    tempo = time()
                    self.regressionepunteggi = np.polyfit([i for i in range(len(self.punteggi))],
                                                          self.mediapunteggivolée if self.mediapunteggivolée is not None
                                                          else np.array([np.mean(volée) for volée in self.punteggi]),
                                                          1)
                    self.temporegressionepunteggi = time() - tempo
                    print("20,28%: regressionepunteggi calcolata")
                except Exception as e:
                    self.regressionepunteggi = None
                    self.temporegressionepunteggi = None
                    print(f"\033[35mNon ho potuto calcolare regressionepunteggi perché {e}\033[0m")
            else:
                self.regressionepunteggi = None
                self.temporegressionepunteggi = None
                print("20,48%: regressionepunteggi non calcolata")
            if impostazioni["regressionecoordinate"]:
                try:
                    tempo = time()
                    self.regressionecoordinate = (np.polyfit([i for i in range(len(self.ascisse))],
                                                             ([media[0] for media in self.mediacoordinatevolée])
                                                             if self.mediacoordinatevolée is not None
                                                             else np.mean(self.ascisse, axis=0), 1),
                                                  np.polyfit([i for i in range(len(self.ordinate))],
                                                             ([media[1] for media in self.mediacoordinatevolée])
                                                             if self.mediacoordinatevolée is not None
                                                             else np.mean(self.ordinate, axis=0), 1))
                    self.temporegressionecoordinate = time() - tempo
                    print("21,23%: regressionecoordinate calcolata")
                except Exception as e:
                    self.regressionecoordinate = None
                    self.temporegressionecoordinate = None
                    print(f"\033[35mNon ho potuto calcolare regressionecoordiante perché {e}\033[0m")
            else:
                self.regressionecoordinate = None
                self.temporegressionecoordinate = None
                print("21,43%: regressionecoordinate non calcolata")
            if impostazioni["autocorrelazionepunteggi"]:
                try:
                    tempo = time()
                    self.autocorrelazionepunteggi = autocorrelazioni(self.mediapunteggivolée
                                                                     if self.mediapunteggivolée is not None
                                                                     else np.array(
                                                                      [np.mean(volée) for volée in allenamento]),
                                                                     np.var(self.mediapunteggivolée
                                                                            if self.mediapunteggivolée is not None
                                                                            else np.array(
                                                                             [np.mean(volée) for volée in allenamento])))
                    self.tempoautocorrelazionepunteggi = time() - tempo
                    print("22,18%: autocorrelazionepunteggi calcolata")
                except Exception as e:
                    self.autocorrelazionepunteggi = None
                    self.tempoautocorrelazionepunteggi = None
                    print(f"\033[35mNon ho potuto calcolare autocorrelazionepunteggi perché {e}\033[0m")
            else:
                self.autocorrelazionepunteggi = None
                self.tempoautocorrelazionepunteggi = None
                print("22,38%: autocorrelazionepunteggi non calcolata")
            if impostazioni["autocorrelazioneascisse"]:
                try:
                    tempo = time()
                    medieascisse = (np.array([media[0] for media in self.mediacoordinatevolée])
                                    if self.mediacoordinatevolée is not None
                                    else np.mean(self.ascisse, axis=0))
                    self.autocorrelazioneascisse = autocorrelazioni(medieascisse, np.var(medieascisse))
                    self.tempoautocorrelazioneascisse = time() - tempo
                    print("22,86%: autocorrelazioneascisse calcolata")
                except Exception as e:
                    self.autocorrelazioneascisse = None
                    self.tempoautocorrelazioneascisse = None
                    print(f"\033[35mNon ho potuto calcolare autocorrelazioneascisse perché {e}\033[0m")
            else:
                self.autocorrelazioneascisse = None
                self.tempoautocorrelazioneascisse = None
                print("23,06%: autocorrelazioneascisse non calcolata")
            if impostazioni["autocorrelazioneordinate"]:
                try:
                    tempo = time()
                    medieordinate = (np.array([media[1] for media in self.mediacoordinatevolée])
                                     if self.mediacoordinatevolée is not None
                                     else np.mean(self.ordinate, axis=0))
                    self.autocorrelazioneordinate = autocorrelazioni(medieordinate, np.var(medieordinate))
                    self.tempoautocorrelazioneordinate = time() - tempo
                    print("23,54%: autocorrelazioneordinate calcolata")
                except Exception as e:
                    self.autocorrelazioneordinate = None
                    self.tempoautocorrelazioneordinate = None
                    print(f"\033[35mNon ho potuto calcolare autocorrelazioneordinate perché {e}\033[0m")
            else:
                self.autocorrelazioneordinate = None
                self.tempoautocorrelazioneordinate = None
                print("23,74%: autocorrelazioneordinate non calcolata")
            if impostazioni["autocorrelazioniangoli"]:
                try:
                    tempo = time()
                    self.autocorrelazioniangolipositive, self.autocorrelazioniangolinegative = miomodulo.autocorrelazioniangolari(
                        self.mediaangolivolée.tolist() if self.mediaangolivolée is not None else [st.circmean(volée) for volée in self.angoli],
                        self.volée)
                    self.tempoautocorrelazioniangoli = time() - tempo
                    print("24,08%: autocorrelazioniangoli calcolate")
                except Exception as e:
                    self.autocorrelazioniangolipositive = None
                    self.autocorrelazioniangolinegative = None
                    self.tempoautocorrelazioniangoli = None
                    print(f"\033[35mNon ho potuto calcolare autocorrelazioniangoli perché {e}\033[0m")
            else:
                self.autocorrelazioniangolipositive = None
                self.autocorrelazioniangolinegative = None
                self.tempoautocorrelazioniangoli = None
                print("24,35%: autocorrelazioniangoli non calcolate")
        else:
            self.regressionepunteggi = None
            self.regressionecoordinate = None
            self.autocorrelazionepunteggi = None
            self.autocorrelazioneascisse = None
            self.autocorrelazioneordinate = None
            self.autocorrelazioniangolipositive = None
            self.autocorrelazioniangolinegative = None
            self.temporegressionepunteggi = None
            self.temporegressionecoordinate = None
            self.tempoautocorrelazionepunteggi = None
            self.tempoautocorrelazioneascisse = None
            self.tempoautocorrelazioneordinate = None
            self.tempoautocorrelazioniangoli = None
            print("25,3%: non sono state calcolate regressioni né autocorrelazioni")
        if (self.mediacoordinate is not None or self.mediacoordinatevolée is not None
                or self.mediacoordinatefrecce is not None or self.regressionecoordinate is not None):
            try:
                tempo = time()
                graficomediesubersaglio(self.mediacoordinate, self.mediacoordinatevolée, self.mediacoordinatefrecce,
                                        self.regressionecoordinate, self.taglia if ordine else len(allenamento), data)
                self.tempograficomedie = time() - tempo
                print("25,71%: grafico delle medie disegnato")
            except Exception as e:
                self.tempograficomedie = None
                print(f"\033[35mNon ho potuto disegnare il grafico delle medie perché {e}\033[0m")
        else:
            self.tempograficomedie = None
            print("25,84%: grafico delle medie non disegnato")
        try:
            tempo = time()
            graficopunteggi(ordine, self.punteggi, self.mediapunteggi, self.regressionepunteggi, data)
            self.tempograficopunteggi = time() - tempo
            print("26,05%: grafico dei punteggi disegnato")
        except Exception as e:
            self.tempograficopunteggi = None
            print(f"\033[35mNon ho potuto disegnare il grafico dei punteggi perché {e}\033[0m")
        try:
            tempo = time()
            graficovolée(self.mediapunteggivolée, ordine, self.regressionepunteggi, self.mediapunteggifrecce, data)
            self.tempograficovolée = time() - tempo
            print("26,25%: grafico delle volée disegnato")
        except Exception as e:
            self.tempograficovolée = None
            print(f"\033[35mNon ho potuto disegnare il grafico delle volée perché {e}\033[0m")
        try:
            tempo = time()
            graficoangoli(self.angoli, self.mediaangoli, self.mediaangolivolée, self.mediaangolifrecce, data)
            self.tempograficoangoli = time() - tempo
            print("26,45%: grafico degli angoli disegnato")
        except Exception as e:
            self.tempograficoangoli = None
            print(f"\033[35mNon ho potuto disegnare il grafico degli angoli perché {e}\033[0m")
        try:
            tempo = time()
            graficiautocorrelazioni(self.autocorrelazionepunteggi, self.autocorrelazioneascisse,
                                    self.autocorrelazioneordinate, self.autocorrelazioniangolipositive,
                                    self.autocorrelazioniangolinegative, data)
            self.tempograficoautocorrelazioni = time() - tempo
            print("26,79%: grafici delle autocorrelazioni disegnati")
        except Exception as e:
            self.tempograficoautocorrelazioni = None
            print(f"\033[35mNon ho potuto disegnare i grafici delle autocorrelazioni perché {e}\033[0m")
        if impostazioni["calcolaerrori"]:
            self.probabilitàtuttogiusto = 1
        else:
            self.probabilitàtuttogiusto = None
        allenamentodepurato = np.array([[freccia[0], freccia[1]] for volée in allenamento for freccia in volée])
        if (impostazioni["testhotelling"] or impostazioni["intervallohotelling"] or impostazioni["hotellingbayesiano"]
            or impostazioni["intervallobayesiano"]) and self.taglia > 2:
            autasc = None
            autord = None
            cutoff1 = None
            cutoff2 = None
            soglie1 = None
            soglie2 = None
            try:
                tempo = time()
                if self.autocorrelazioneascisse is not None:
                    autasc = self.autocorrelazioneascisse
                else:
                    if ordine:
                        autasc = autocorrelazioni(appiattisci(self.ascisse), np.var(appiattisci(self.ascisse)))
                    else:
                        medieascisse = ([media[0] for media in self.mediacoordinatevolée]
                                        if self.mediacoordinatevolée is not None
                                        else [np.mean(np.array(self.ascisse[i]))
                                              for i in range(len(self.ascisse))])
                        autasc = autocorrelazioni(medieascisse, np.var(medieascisse))
                cutoff1 = list()
                soglie1 = list()
                if impostazioni["hljungboxascisse"] < self.taglia-1:
                    cutoffattuale1 = impostazioni["hljungboxascisse"]
                    while cutoffattuale1 < self.taglia - 1:
                        cutoff1.append(cutoffattuale1)
                        soglie1.append(st.chi2.ppf(1 - impostazioni["alfaljungboxascisse"], df=cutoffattuale1))
                        cutoffattuale1 += impostazioni["hljungboxascisse"]
                cutoff1.append(self.taglia - 1)
                soglie1.append(st.chi2.ppf(1 - impostazioni["alfaljungboxascisse"], df=self.taglia-1))
                self.ljungboxascisse = miomodulo.ljungbox(autasc.tolist(), cutoff1, len(autasc)+1,
                                                          soglie1, len(cutoff1))
                self.tempoljungboxascisse = time()-tempo
                tempo = time()
                if self.autocorrelazioneordinate is not None:
                    autord = self.autocorrelazioneordinate
                else:
                    if ordine:
                        autord = autocorrelazioni(appiattisci(self.ordinate), np.var(appiattisci(self.ordinate)))
                    else:
                        medieordinate = ([media[1] for media in self.mediacoordinatevolée]
                                         if self.mediacoordinatevolée is not None
                                         else [np.mean(np.array(self.ordinate[i]))
                                               for i in range(len(self.ordinate))])
                        autord = autocorrelazioni(medieordinate, np.var(medieordinate))
                if len(autasc) != len(autord) or not isinstance(autasc, np.ndarray) or not isinstance(autord,
                                                                                                      np.ndarray):
                    autasc = None
                    autord = None
                    raise IndexError("Le liste di autocorrelazioni delle ascisse e delle ordinate sono mal formattate")
                cutoff2 = list()
                soglie2 = list()
                if impostazioni["hljungboxordinate"] < self.taglia-1:
                    cutoffattuale2 = impostazioni["hljungboxordinate"]
                    while cutoffattuale2 < self.taglia - 1:
                        cutoff2.append(cutoffattuale2)
                        soglie2.append(st.chi2.ppf(1 - impostazioni["alfaljungboxordinate"], df=cutoffattuale2))
                        cutoffattuale2 += impostazioni["hljungboxordinate"]
                cutoff2.append(self.taglia - 1)
                soglie2.append(st.chi2.ppf(1 - impostazioni["alfaljungboxordinate"], df=self.taglia-1))
                self.ljungboxordinate = miomodulo.ljungbox(autord.tolist(), cutoff2, len(autord)+1,
                                                           soglie2, len(cutoff2))
                self.tempoljungboxordinate = time() - tempo
                print("30,32%: test di Ljung-Box eseguiti")
            except Exception as e:
                self.ljungboxascisse = None
                self.ljungboxordinate = None
                self.tempoljungboxascisse = None
                self.tempoljungboxordinate = None
                print(f"\033[35mNon ho potuto eseguire il test di Ljung-Box perché {e}\033[0m")
            try:
                tempo = time()
                # Si chiama self.mardia per comodità anche se non è per forza il test di Mardia
                self.mardia = testnormalità(allenamentodepurato, self.taglia, impostazioni["alfamardia"],
                                            impostazioni["tipotestnormalità"])
                self.tempomardia = time() - tempo
                print("30,66%: test di normalità multivariata eseguito")
            except Exception as e:
                self.mardia = None
                self.tempomardia = None
                print(f"\033[35mNon ho potuto eseguire il test di normalità multivariata perché {e}\033[0m")
            if self.ljungboxascisse is None or self.ljungboxordinate is None or self.mardia is None:
                self.affidabilitàhotelling = False
            elif self.ljungboxascisse and self.ljungboxordinate and self.mardia:
                self.affidabilitàhotelling = True
            else:
                self.affidabilitàhotelling = False
            if self.ljungboxascisse is not None and self.ljungboxordinate is not None:
                if autasc is not None and cutoff1 is not None and soglie1 is not None:
                    if self.ljungboxascisse and impostazioni["calcolaerrori"]:
                        try:
                            tempo = time()
                            self.betaljungboxascisse = miomodulo.betaljungbox(len(autasc)+1, cutoff1, soglie1,
                                                                              len(cutoff1),
                                                                              impostazioni["voltebetaljungboxascisse"])
                            self.tempobetaljungboxascisse = time() - tempo
                            self.probabilitàtuttogiusto *= (1 - self.betaljungboxascisse)
                            self.alfaveroljungboxascisse = None
                            self.tempoalfaveroljungboxascisse = None
                            print("31,47%: beta per il test di Ljung-Box calcolato")
                        except Exception as e:
                            self.betaljungboxascisse = None
                            self.tempobetaljungboxascisse = None
                            self.alfaveroljungboxascisse = None
                            self.tempoalfaveroljungboxascisse = None
                            self.probabilitàtuttogiusto = None
                            impostazioni["calcolaerrori"] = False
                            print(f"\033[35mNon ho potuto calcolare il beta del test di Ljung-Box perché {e}; smetteremo di calcolare le probabilità di errore\033[0m")
                    elif len(autasc) < 50 and impostazioni["calcolaerrori"]:
                        try:
                            tempo = time()
                            self.alfaveroljungboxascisse = miomodulo.alfaveroljungbox(len(autasc) + 1, cutoff1, soglie1, len(cutoff1),
                                                                                      impostazioni[
                                                                                          "voltealfaljungboxascisse"])
                            self.tempoalfaveroljungboxascisse = time() - tempo
                            self.probabilitàtuttogiusto *= (1 - self.alfaveroljungboxascisse)
                            self.betaljungboxascisse = None
                            self.tempobetaljungboxascisse = None
                            print("33,03%: alfa effettivo per il test di Ljung-Box calcolato")
                        except Exception as e:
                            self.alfaveroljungboxascisse = None
                            self.tempoalfaveroljungboxascisse = None
                            self.betaljungboxascisse = None
                            self.tempobetaljungboxascisse = None
                            self.probabilitàtuttogiusto = None
                            impostazioni["calcolaerrori"] = False
                            print(f"\033[35mNon ho potuto calcolare l'alfa effettivo per il test di Ljung-Box perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
                    else:
                        if impostazioni["calcolaerrori"]:
                            self.probabilitàtuttogiusto *= (1 - impostazioni["alfaljungboxascisse"])**len(cutoff1)
                        self.alfaveroljungboxascisse = None
                        self.betaljungboxascisse = None
                        self.tempoalfaveroljungboxascisse = None
                        self.tempobetaljungboxascisse = None
                        print("33,51%: la probabilità di errore del test di Ljung-Box si avvicina all'alfa nominale")
                else:
                    self.alfaveroljungboxascisse = None
                    self.betaljungboxascisse = None
                    self.tempoalfaveroljungboxascisse = None
                    self.tempobetaljungboxascisse = None
                    self.probabilitàtuttogiusto = None
                    impostazioni["calcolaerrori"] = False
                    print(f"\033[35mNon ho potuto calcolare la probabilità di errore per il test di Ljung-Box per un errore precedente; smetteremo di calcolare le probabilità degli errori\033[0m")
                if autord is not None and soglie2 is not None and cutoff2 is not None:
                    if self.ljungboxordinate and impostazioni["calcolaerrori"]:
                        try:
                            tempo = time()
                            self.betaljungboxordinate = miomodulo.betaljungbox(len(autord)+1, cutoff2, soglie2,
                                                                               len(cutoff2),
                                                                               impostazioni["voltebetaljungboxordinate"])
                            self.tempobetaljungboxordinate = time() - tempo
                            self.probabilitàtuttogiusto *= (1 - self.betaljungboxordinate)
                            self.alfaveroljungboxordinate = None
                            self.tempoalfaveroljungboxordinate = None
                            print("34,05%: beta del secondo test di Ljung-Box calcolato")
                        except Exception as e:
                            self.betaljungboxordinate = None
                            self.tempobetaljungboxordinate = None
                            self.alfaveroljungboxordinate = None
                            self.tempoalfaveroljungboxordinate = None
                            self.probabilitàtuttogiusto = None
                            impostazioni["calcolaerrori"] = False
                            print(f"\033[35mNon ho potuto calcolare il beta del test di Ljung-Box perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
                    elif len(autord) < 50 and impostazioni["calcolaerrori"]:
                        try:
                            tempo = time()
                            self.alfaveroljungboxordinate = miomodulo.alfaveroljungbox(len(autord) + 1, cutoff2, soglie2, len(cutoff2),
                                                                                       impostazioni["voltealfaljungboxordinate"])
                            self.tempoalfaveroljungboxordinate = time() - tempo
                            self.probabilitàtuttogiusto *= (1 - self.alfaveroljungboxordinate)
                            self.betaljungboxordinate = None
                            self.tempobetaljungboxordinate = None
                            print("35,54%: alfa per il secondo test di Ljung-Box calcolato")
                        except Exception as e:
                            self.alfaveroljungboxordinate = None
                            self.tempoalfaveroljungboxordinate = None
                            self.betaljungboxordinate = None
                            self.tempobetaljungboxordinate = None
                            self.probabilitàtuttogiusto = None
                            impostazioni["calcolaerrori"] = False
                            print(f"\033[35mNon ho potuto calcolare l'alfa effettivo del test di Ljung-Box perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
                    else:
                        if impostazioni["calcolaerrori"]:
                            self.probabilitàtuttogiusto *= (1 - impostazioni["alfaljungboxordinate"])**len(cutoff2)
                        self.alfaveroljungboxordinate = None
                        self.tempoalfaveroljungboxordinate = None
                        self.betaljungboxordinate = None
                        self.tempobetaljungboxordinate = None
                        print("36,02%: la probabilità di errore del secondo test di Ljung-Box si avvicina all'alfa nominale")
                else:
                    self.alfaveroljungboxordinate = None
                    self.tempoalfaveroljungboxordinate = None
                    self.betaljungboxordinate = None
                    self.tempobetaljungboxordinate = None
                    self.probabilitàtuttogiusto = None
                    impostazioni["calcolaerrori"] = False
                    print(f"\033[35mNon ho potuto calcolare la probabilità di errore del test di Ljung-Box per un errore precedente; smetteremo di calcolare le probabilità degli errori")
            else:
                self.alfaveroljungboxascisse = None
                self.alfaveroljungboxordinate = None
                self.tempoalfaveroljungboxascisse = None
                self.tempoalfaveroljungboxordinate = None
                self.betaljungboxascisse = None
                self.betaljungboxordinate = None
                self.tempobetaljungboxascisse = None
                self.tempobetaljungboxordinate = None
            if self.mardia is not None:
                if self.mardia and impostazioni["calcolaerrori"]:
                    try:
                        tempo = time()
                        self.betamardia = betamardia(frecce=self.taglia,
                                                     distribuzione=impostazioni["distribuzionebetamardia"],
                                                     iterazioni=impostazioni["voltebetamardia"],
                                                     gradit=impostazioni["graditbetamardia"],
                                                     alfa=impostazioni["alfamardia"],
                                                     tipotest=impostazioni["tipotestnormalità"],
                                                     asimmetriaascisse=impostazioni["asimmetriaascissebetamardia"],
                                                     asimmetriaordinate=impostazioni["asimmetriaordinatebetamardia"],
                                                     distanzacomponenti=impostazioni["distanzacomponentibetamardia"])
                        self.tempobetamardia = time() - tempo
                        self.probabilitàtuttogiusto *= 1 - self.betamardia
                        self.alfaveromardia = None
                        self.tempoalfaveromardia = None
                        print("36,77%: beta del test di multinormalità calcolato")
                    except Exception as e:
                        self.betamardia = None
                        self.tempobetamardia = None
                        self.alfaveromardia = None
                        self.tempoalfaveromardia = None
                        self.probabilitàtuttogiusto = None
                        impostazioni["calcolaerrori"] = False
                        print(f"\033[35mNon ho potuto calcolare il beta del test di multinormalità perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
                elif self.taglia < 50 and impostazioni["calcolaerrori"]:
                    try:
                        tempo = time()
                        self.alfaveromardia = alfaveromardia(frecce=self.taglia,
                                                             iterazioni=impostazioni["voltealfamardia"],
                                                             alfa=impostazioni["alfamardia"],
                                                             tipotest=impostazioni["tipotestnormalità"])
                        self.tempoalfaveromardia = time() - tempo
                        self.probabilitàtuttogiusto *= 1 - self.alfaveromardia
                        self.betamardia = None
                        self.tempobetamardia = None
                        print("37,38%: alfa effettivo del test di multinormalità calcolato")
                    except Exception as e:
                        self.alfaveromardia = None
                        self.tempoalfaveromardia = None
                        self.betamardia = None
                        self.tempobetamardia = None
                        self.probabilitàtuttogiusto = None
                        impostazioni["calcolaerrori"] = False
                        print(f"\033[35mNon ho potuto calcolare l'alfa effettivo del test di multinormalità perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
                else:
                    if impostazioni["calcolaerrori"]:
                        self.probabilitàtuttogiusto *= 1 - impostazioni["alfamardia"]
                    self.alfaveromardia = None
                    self.tempoalfaveromardia = None
                    self.betamardia = None
                    self.tempobetamardia = None
                    print("37,85%: la probabilità di errore del test di multinormalità è vicina all'alfa nominale")
            else:
                self.alfaveromardia = None
                self.tempoalfaveromardia = None
                self.betamardia = None
                self.tempobetamardia = None
                self.etichettefrecce = None
        else:
            self.ljungboxascisse = None
            self.tempoljungboxascisse = None
            self.ljungboxordinate = None
            self.tempoljungboxordinate = None
            self.mardia = None
            self.tempomardia = None
            self.affidabilitàhotelling = None
            self.betaljungboxascisse = None
            self.tempobetaljungboxascisse = None
            self.betaljungboxordinate = None
            self.tempobetaljungboxordinate = None
            self.alfaveroljungboxascisse = None
            self.tempoalfaveroljungboxascisse = None
            self.alfaveroljungboxordinate = None
            self.tempoalfaveroljungboxordinate = None
            self.betamardia = None
            self.tempobetamardia = None
            self.alfaveromardia = None
            self.tempoalfaveromardia = None
            print("39,28%: non sono stati eseguiti test di multinormalità o di Ljung-Box")
        if impostazioni["clustering"]:
            try:
                tempo = time()
                self.etichettefrecce = miomodulo.clustering(impostazioni["inizializzazioneclustering"],
                                                            self.taglia,
                                                            allenamentodepurato[:, 0].tolist(),
                                                            allenamentodepurato[:, 1].tolist(),
                                                            impostazioni["iterazionikmeans"],
                                                            impostazioni["componentimassime"],
                                                            impostazioni["selezionemodello"],
                                                            impostazioni["numerobootstrapclustering"],
                                                            impostazioni["alfabootstrapclustering"],
                                                            impostazioni["iterazioniinizializzazioneclustering"],
                                                            impostazioni["iterazioniEM"],
                                                            impostazioni["gaussianeclustering"],
                                                            impostazioni["algoritmoclustering"],
                                                            impostazioni["convergenzaclustering"],
                                                            impostazioni["criterioclustering"],
                                                            impostazioni["bootstrapinterniclustering"],
                                                            impostazioni["foldclustering"])
                graficocluster(allenamentodepurato, self.etichettefrecce, data)
                self.tempoclustering = time() - tempo
                print("39,41%: clustering completato")
            except Exception as e:
                self.etichettefrecce = None
                self.tempoclustering = None
                print(f"\033[35mNon ho potuto completare il clustering o disegnarne il grafico perché {e}\033[0m")
        else:
            self.etichettefrecce = None
            self.tempoclustering = None
            print("40,97%: clustering non effettuato")
        if impostazioni["testhotelling"] and self.taglia > 2:
            try:
                tempo = time()
                self.testhotelling = miomodulo.testhotelling(self.taglia,
                                                             allenamentodepurato[:, 0].tolist(), allenamentodepurato[:, 1].tolist(),
                                                             st.f.ppf(1 - impostazioni["alfahotelling"], 2, self.taglia - 2))
                self.tempotesthotelling = time() - tempo
                print("41,38%: test di Hotelling eseguito")
            except Exception as e:
                self.testhotelling = None
                self.tempotesthotelling = None
                print(f"\033[35mNon ho potuto eseguire il test di Hotelling perché {e}\033[0m")
            if self.testhotelling is not None:
                if self.testhotelling and impostazioni["calcolaerrori"]:
                    try:
                        tempo = time()
                        self.betatesthotelling = miomodulo.betatesthotelling(self.taglia,
                                                                             st.f.ppf(1 - impostazioni["alfahotelling"], 2, self.taglia - 2),
                                                                             impostazioni["voltebetahotelling"],
                                                                             impostazioni["distanzabetahotelling"],
                                                                             (self.varianzacoordinate[0]
                                                                                              if self.varianzacoordinate is not None
                                                                                              else np.var(
                                                                                 appiattisci(self.ascisse), ddof=1)),
                                                                             (self.varianzacoordinate[1]
                                                                                               if self.varianzacoordinate is not None
                                                                                               else np.var(
                                                                                 appiattisci(self.ordinate), ddof=1)))
                        self.tempobetatesthotelling = time() - tempo
                        self.probabilitàtuttogiusto *= 1 - self.betatesthotelling
                        print("42,6%: beta per il test di Hotelling calcolato")
                    except Exception as e:
                        self.betatesthotelling = None
                        self.tempobetatesthotelling = None
                        self.probabilitàtuttogiusto = None
                        impostazioni["calcolaerrori"] = False
                        print(f"\033[35mNon ho potuto calcolare il beta del test di Hotelling perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
                else:
                    self.betatesthotelling = None
                    self.tempobetatesthotelling = None
                    if impostazioni["calcolaerrori"]:
                        self.probabilitàtuttogiusto *= 1 - impostazioni["alfahotelling"]
                    print("42,94%: la probabilità di errore del test di Hotelling è l'alfa nominale")
            else:
                self.testhotelling = None
                self.tempotesthotelling = None
                self.betatesthotelling = None
                self.tempobetatesthotelling = None
            if impostazioni["clustering"]:
                try:
                    tempo = time()
                    self.testcluster = testcluster(etichette=self.etichettefrecce,
                                                    allenamento=allenamentodepurato,
                                                    alfa=impostazioni["alfahotelling"],
                                                    frecce=self.taglia)
                    self.tempotestcluster = time() - tempo
                    print("43,41%: test di Hotelling sui cluster eseguito")
                except Exception as e:
                    self.testcluster = None
                    self.tempotestcluster = None
                    print(f"\033[35mNon ho potuto eseguire il test di Hotelling sui cluster perché {e}\033[0m")
                if impostazioni["calcolaerrori"] and self.testcluster is not None:
                    try:
                        tempo = time()
                        self.probabilitàtestcluster = 1
                        for test in range(len(self.testcluster)):
                            numerositàcluster = sum((1 if numero == test else 0) for numero in self.etichettefrecce)
                            if self.testcluster[test]:
                                self.probabilitàtestcluster *= 1 - miomodulo.betatesthotelling(numerositàcluster,
                                                                                               st.f.ppf(1 - impostazioni["alfahotelling"], 2, self.taglia-2),
                                                                                               impostazioni["voltebetahotelling"],
                                                                                               impostazioni["distanzabetahotelling"],
                                                                                               self.varianzacoordinate[0] if self.varianzacoordinate is not None else np.var(appiattisci(self.ascisse), ddof=1),
                                                                                               self.varianzacoordinate[1] if self.varianzacoordinate is not None else np.var(appiattisci(self.ordinate), ddof=1))
                            else:
                                self.probabilitàtestcluster *= 1 - impostazioni["alfahotelling"]
                        self.probabilitàtuttogiusto *= self.probabilitàtestcluster
                        self.tempoprobabilitàtestcluster = time() - tempo
                        print("45,92%: probabilità di errore dei test di Hotelling sui cluster calcolate")
                    except Exception as e:
                        self.probabilitàtestcluster = None
                        self.tempoprobabilitàtestcluster = None
                        impostazioni["calcolaerrori"] = False
                        print(f"\033[35mNon ho potuto calcolare le probabilità di errore dei test sui cluster perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
                else:
                    self.probabilitàtestcluster = None
                    self.tempoprobabilitàtestcluster = None
                    print("46,06%: probabilità di errore dei test di Hotelling sui cluster non calcolate")
            else:
                self.testcluster = None
                self.tempotestcluster = None
                self.probabilitàtestcluster = None
                self.tempoprobabilitàtestcluster = None
                print("46,26%: test sui cluster non eseguito")
        if impostazioni["intervallohotelling"] and self.taglia > 2:
            try:
                tempo = time()
                intervallo_hotelling = intervallohotelling(n=self.taglia,
                                                               confidenza=impostazioni["confidenzahotelling"],
                                                               varianza=np.cov(allenamentodepurato,
                                                                               rowvar=False, bias=False),
                                                               media=self.mediacoordinate
                                                               if self.mediacoordinate is not None
                                                               else (np.mean(appiattisci(self.ascisse)),
                                                                     np.mean(appiattisci(self.ordinate))))
                graficointervallohotelling(intervallo_hotelling[0], intervallo_hotelling[1], self.data, "regolare")
                if impostazioni["correggidipendenzahotelling"] == "stazionario":
                    intervallostazionario = intervallohotelling(n=self.taglia,
                                                                confidenza=impostazioni["confidenzahotelling"],
                                                                varianza=np.cov(allenamentodepurato, rowvar=False, bias=False),
                                                                media=self.mediacoordinate if self.mediacoordinate is not None
                                                                else (np.mean(appiattisci(self.ascisse)), np.mean(appiattisci(self.ordinate))),
                                                                costante=miomodulo.bootstrapstazionariohotelling(self.taglia, appiattisci(self.ascisse), appiattisci(self.ordinate),
                                                                                                                 self.impostazioni["geometricastazionariahotelling"],
                                                                                                                 self.impostazioni["bootstrapstazionarihotelling"],
                                                                                                                 np.mean(appiattisci(self.ascisse)),
                                                                                                                 np.mean(appiattisci(self.ordinate)),
                                                                                                                 1-self.impostazioni["confidenzahotelling"])
                                                                )
                    graficointervallohotelling(intervallostazionario[0], intervallostazionario[1], self.data, "stazionario")
                else:
                    intervallostazionario = None
                if impostazioni["correggidipendenzahotelling"] == "mobile":
                    intervallomobile = intervallohotelling(n=self.taglia,
                                                                confidenza=impostazioni["confidenzahotelling"],
                                                                varianza=np.cov(allenamentodepurato, rowvar=False, bias=False),
                                                                media=self.mediacoordinate if self.mediacoordinate is not None
                                                                else (np.mean(appiattisci(self.ascisse)), np.mean(appiattisci(self.ordinate))),
                                                                costante=miomodulo.bootstrapmobilehotelling(self.taglia, appiattisci(self.ascisse), appiattisci(self.ordinate),
                                                                                                                 self.impostazioni["lunghezzabloccomobilehotelling"],
                                                                                                                 self.impostazioni["bootstrapmobilihotelling"],
                                                                                                                 np.mean(appiattisci(self.ascisse)),
                                                                                                                 np.mean(appiattisci(self.ordinate)),
                                                                                                                 1-self.impostazioni["confidenzahotelling"])
                                                                )
                    graficointervallohotelling(intervallomobile[0], intervallomobile[1], self.data, "mobile")
                else:
                    intervallomobile = None
                if correggidipendenzahotelling == "detrending":
                    if self.regressionecoordinate is None:
                        if not self.ordine:
                            rc = (np.polyfit([i for i in range(len(self.ascisse))],
                                                                ([media[0] for media in self.mediacoordinatevolée])
                                                                if self.mediacoordinatevolée is not None
                                                                else np.mean(self.ascisse, axis=0), 1),
                                                     np.polyfit([i for i in range(len(self.ordinate))],
                                                                ([media[1] for media in self.mediacoordinatevolée])
                                                                if self.mediacoordinatevolée is not None
                                                                else np.mean(self.ordinate, axis=0), 1))
                        else:
                            rc = (np.polyfit([i for i in range(len(self.ascisse))],
                                                                appiattisci(self.ascisse), 1),
                                                     np.polyfit([i for i in range(len(self.ordinate))],
                                                                appiattisci(self.ordinate), 1))
                    else:
                        rc = self.regressionecoordinate
                    va, vo, c, _, _ = miomodulo.detrenda(appiattisci(self.ascisse), appiattisci(self.ordinate),
                                                           rc[0][0], rc[0][1], rc[1][0], rc[1][1])
                    intervallodetrendato = intervallohotelling(n=self.taglia,
                                                                confidenza=confidenzahotelling,
                                                                varianza=np.array([[va, c], [c, vo]]),
                                                                media=(rc[0][1]+rc[0][0], rc[1][1]+rc[1][0])
                                                               )
                    graficointervallohotelling(intervallodetrendato[0], intervallodetrendato[1], self.data, "detrending", rc)
                else:
                    intervallodetrendato = None
                self.tempointervallohotelling = time() - tempo
                print("47,08%: intervallo di confidenza bivariato calcolato e disegnato")
            except Exception as e:
                intervallo_hotelling = None
                intervallostazionario = None
                intervallomobile = None
                intervallodetrendato = None
                self.tempointervallohotelling = None
                print(f"\033[35mNon ho potuto calcolare o disegnare l'intervallo di confidenza bivariato perché {e}\033[0m")
            if impostazioni["calcolaerrori"] and (intervallo_hotelling is not None or intervallodetrendato is not None):
                self.probabilitàtuttogiusto *= impostazioni["confidenzahotelling"]
            if impostazioni["calcolaerrori"] and intervallostazionario is not None:
                tempo = time()
                self.confidenzaverabootstrapstazionariohotelling = 1.0 - miomodulo.alfaverobootstrapstazionariohotelling(self.taglia,
                                                                                                     self.impostazioni[
                                                                                                         "varianzaascissebootstrapstazionariohotelling"],
                                                                                                     self.impostazioni[
                                                                                                         "varianzaordinatebootstrapstazionariohotelling"],
                                                                                                     self.impostazioni[
                                                                                                         "geometricastazionariahotelling"],
                                                                                                     self.impostazioni[
                                                                                                         "bootstrapstazionarihotelling"],
                                                                                                     1.0-self.impostazioni[
                                                                                                         "confidenzahotelling"],
                                                                                                     self.impostazioni[
                                                                                                         "iterazionistazionariehotelling"])
                self.tempoconfidenzaverabootstrapstazionariohotelling = time()-tempo
                self.probabilitàtuttogiusto *= self.confidenzaverabootstrapstazionariohotelling
            else:
                self.confidenzaverabootstrapstazionariohotelling = None
            if impostazioni["calcolaerrori"] and intervallomobile is not None:
                tempo = time()
                self.confidenzaverabootstrapmobilehotelling = 1.0 - miomodulo.alfaverobootstrapmobilehotelling(self.taglia,
                                                                                                     self.impostazioni[
                                                                                                         "varianzaascissebootstrapmobilehotelling"],
                                                                                                     self.impostazioni[
                                                                                                         "varianzaordinatebootstrapmobilehotelling"],
                                                                                                     self.impostazioni[
                                                                                                         "lunghezzabloccomobilehotelling"],
                                                                                                     self.impostazioni[
                                                                                                         "bootstrapmobilihotelling"],
                                                                                                     1.0-self.impostazioni[
                                                                                                         "confidenzahotelling"],
                                                                                                     self.impostazioni[
                                                                                                         "iterazionimobilihotelling"])
                self.tempoconfidenzaverabootstrapmobilehotelling = time()-tempo
                self.probabilitàtuttogiusto *= self.confidenzaverabootstrapmobilehotelling
            if impostazioni["clustering"]:
                try:
                    tempo = time()
                    intervalli_cluster = intervallicluster(etichette=self.etichettefrecce,
                                                               allenamento=allenamentodepurato,
                                                               alfa=impostazioni["confidenzahotelling"])
                    self.tempointervallicluster = time() - tempo
                    tempo = time()
                    graficointervallicluster(intervalli=intervalli_cluster, data=self.data)
                    self.tempograficointervallicluster = time() - tempo
                    print("47,69%: intervalli di confidenza bivariati per i cluster calcolati e disegnati")
                except Exception as e:
                    intervalli_cluster = None
                    self.tempointervallicluster = None
                    self.tempograficointervallicluster = None
                    print(f"\033[35mNon ho potuto calcolare o disegnare gli intervalli di confidenza bivariati per i cluster perché {e}\033[0m")
                if impostazioni["calcolaerrori"] and intervalli_cluster is not None:
                    self.probabilitàtuttogiusto *= impostazioni["confidenzahotelling"] ** (
                            max(self.etichettefrecce) + 1)
            else:
                intervalli_cluster = None
                self.tempointervallicluster = None
                self.tempograficointervallicluster = None
                print("48,1%: intervalli di confidenza bivariati per i cluster non calcolati")
        else:
            intervallo_hotelling = None
            self.tempointervallohotelling = None
            intervalli_cluster = None
            self.tempointervallicluster = None
            self.tempograficointervallicluster = None
            print("48,5%: intervallo di confidenza bivariato non calcolato")
        if impostazioni["hotellingbayesiano"] and impostazioni["intervallobayesiano"] and self.taglia > 2:
            try:
                tempo = time()
                self.hotellingbayesiano, intervallobayesiano = hotellingbayesiano(allenamento=allenamentodepurato,
                                                                                       numerocatene=impostazioni["catenehotelling"],
                                                                                       iterazioni=impostazioni["iterazionimcmchotelling"],
                                                                                       distanza=impostazioni["ROPEhotelling"],
                                                                                       confidenza=impostazioni["confidenzahotellingbayesiano"],
                                                                                       noninformativa=impostazioni["noninformativitàhotellingbayesiano"],
                                                                                       allenamentipriori=impostazioni["allenamentipriorihotellingbayesiano"],
                                                                                       apiacere=impostazioni["apiacerehotelling"],
                                                                                       aapiacere=impostazioni["aapiacerehotelling"],
                                                                                       bapiacere=impostazioni["bapiacerehotelling"],
                                                                                       etaapiacere=impostazioni["etaapiacerehotelling"],
                                                                                       gerarchica=impostazioni["hotellinggerarchico"],
                                                                                       intervallopreciso=impostazioni["intervallohotellingpreciso"],
                                                                                       test=True, intervallo=True,
                                                                                       lambdaapiacere=impostazioni["lambdaapiacerehotelling"],
                                                                                       muapiacere=impostazioni["muapiacerehotelling"],
                                                                                       console=impostazioni["consoledistan"])
                graficohotellingbayesiano(intervallobayesiano, data)
                self.tempohotellingbayesiano = time() - tempo
                print("49,59%: test di Hotelling bayesiano e intervallo di credibilità bivariato effettuati")
                if impostazioni["calcolaerrori"]:
                    self.probabilitàtuttogiusto *= 1 - min([self.hotellingbayesiano, 1 - self.hotellingbayesiano])
                    self.probabilitàtuttogiusto *= impostazioni["confidenzahotellingbayesiano"]
            except Exception as e:
                self.hotellingbayesiano = None
                self.tempohotellingbayesiano = None
                intervallobayesiano = None
                print(f"\033[35mNon ho potuto eseguire un test di Hotelling bayesiano o calcolare la regione di credibilità perché {e}\033[0m")
        elif impostazioni["hotellingbayesiano"] and self.taglia > 2:
            try:
                tempo = time()
                self.hotellingbayesiano, _ = hotellingbayesiano(allenamento=allenamentodepurato,
                                                                numerocatene=impostazioni["catenehotelling"],
                                                                iterazioni=impostazioni["iterazionimcmchotelling"],
                                                                distanza=impostazioni["ROPEhotelling"],
                                                                confidenza=impostazioni[
                                                                    "confidenzahotellingbayesiano"],
                                                                noninformativa=impostazioni[
                                                                    "noninformativitàhotellingbayesiano"],
                                                                allenamentipriori=impostazioni[
                                                                    "allenamentipriorihotellingbayesiano"],
                                                                intervallo=False,
                                                                apiacere=impostazioni["apiacerehotelling"],
                                                                aapiacere=impostazioni["aapiacerehotelling"],
                                                                bapiacere=impostazioni["bapiacerehotelling"],
                                                                etaapiacere=impostazioni["etaapiacerehotelling"],
                                                                gerarchica=impostazioni["hotellinggerarchico"],
                                                                intervallopreciso=impostazioni[
                                                                    "intervallohotellingpreciso"],
                                                                test=True,
                                                                lambdaapiacere=impostazioni["lambdaapiacerehotelling"],
                                                                muapiacere=impostazioni["muapiacerehotelling"],
                                                                console=impostazioni["consoledistan"])
                self.tempohotellingbayesiano = time() - tempo
                print("50,74%: test di Hotelling bayesiano effettuato")
                if impostazioni["calcolaerrori"]:
                    self.probabilitàtuttogiusto *= 1 - min([self.hotellingbayesiano, 1 - self.hotellingbayesiano])
                intervallobayesiano = None
            except Exception as e:
                self.hotellingbayesiano = None
                self.tempohotellingbayesiano = None
                intervallobayesiano = None
                print(f"\033[35mNon ho potuto effettuare il test di Hotelling bayesiano perché {e}\033[0m")
        elif impostazioni["intervallobayesiano"] and self.taglia > 2:
            try:
                tempo = time()
                _, intervallobayesiano = hotellingbayesiano(allenamento=allenamentodepurato,
                                                                 numerocatene=impostazioni["catenehotelling"],
                                                                 iterazioni=impostazioni["iterazionimcmchotelling"],
                                                                 distanza=impostazioni["ROPEhotelling"],
                                                                 confidenza=impostazioni[
                                                                     "confidenzahotellingbayesiano"],
                                                                 noninformativa=impostazioni[
                                                                     "noninformativitàhotellingbayesiano"],
                                                                 allenamentipriori=impostazioni[
                                                                     "allenamentipriorihotellingbayesiano"],
                                                                 test=False,
                                                                 apiacere=impostazioni["apiacerehotelling"],
                                                                 aapiacere=impostazioni["aapiacerehotelling"],
                                                                 bapiacere=impostazioni["bapiacerehotelling"],
                                                                 etaapiacere=impostazioni["etaapiacerehotelling"],
                                                                 gerarchica=impostazioni["hotellinggerarchico"],
                                                                 intervallopreciso=impostazioni[
                                                                     "intervallohotellingpreciso"],
                                                                 intervallo=True,
                                                                 lambdaapiacere=impostazioni["lambdaapiacerehotelling"],
                                                                 muapiacere=impostazioni["muapiacerehotelling"],
                                                                 console=impostazioni["consoledistan"])
                graficohotellingbayesiano(intervallobayesiano, data)
                self.tempohotellingbayesiano = time() - tempo
                print("52,03%: intervallo di credibilità bayesiano calcolato")
                if impostazioni["calcolaerrori"]:
                    self.probabilitàtuttogiusto *= impostazioni["confidenzahotellingbayesiano"]
                self.hotellingbayesiano = None
            except Exception as e:
                self.hotellingbayesiano = None
                self.tempohotellingbayesiano = None
                intervallobayesiano = None
                print(f"\033[35mNon ho potuto calcolare l'intervallo di credibilità bayesiano perché {e}\033[0m")
        else:
            self.hotellingbayesiano = None
            self.tempohotellingbayesiano = None
            intervallobayesiano = None
            print("52,44%: test di Hotelling bayesiano e intervallo di credibilità bivariato non effettuati")
        if impostazioni["medianageometrica"]:
            try:
                tempo = time()
                self.medianageometrica = medianageometrica(iterazioni=impostazioni["iterazionimedianageometrica"],
                                                           media=(np.array(self.mediacoordinate)
                                                                  if self.mediacoordinate is not None
                                                                  else np.array([np.mean(appiattisci(self.ascisse)),
                                                                                 np.mean(appiattisci(self.ordinate))])),
                                                           allenamento=allenamentodepurato,
                                                           soglia=0.0001)
                self.mirinoideale = mirinoideale(gamma=[self.impostazioni["gammamedianaorizzontale"], self.impostazioni["gammamedianaverticale"]],
                                                 distanza=[self.distanza, self.impostazioni["distanzaverticale"]],
                                                 coccamirino=self.impostazioni["coccamirino"],
                                                 mediane=self.medianageometrica.tolist())
                graficomedianageometrica(self.medianageometrica, data)
                self.tempomedianageometrica = time() - tempo
                print("53,05%: mediana geometrica calcolata")
            except Exception as e:
                self.medianageometrica = None
                self.mirinoideale = None
                self.tempomedianageometrica = None
                print(f"\033[35mNon ho potuto calcolare la mediana geometrica perché {e}\033[0m")
        else:
            self.medianageometrica = None
            self.mirinoideale = None
            self.tempomedianageometrica = None
            print("53,25%: mediana geometrica non calcolata")
        if (impostazioni["intervallonorme"] or impostazioni["testnorme"]) and self.taglia > 1:
            tempo = time()
            self.normenormali = st.shapiro(appiattisci(self.norme)).pvalue > impostazioni["alfashapiro"]
            self.temponormenormali = time() - tempo
            print("53,52%: test di Shapiro sulle distanze dal centro effettuato")
            if self.normenormali and impostazioni["calcolaerrori"]:
                try:
                    tempo = time()
                    self.betanormenormali = betanormenormali(alternativa=impostazioni["alternativanormenormali"],
                                                             iterazioni=impostazioni["iterazionibetanormenormali"],
                                                             alfa=impostazioni["alfashapiro"],
                                                             frecce=self.taglia,
                                                             asimmetria=impostazioni["asimmetrianormenormali"],
                                                             distanzacomponenti=impostazioni["distanzacomponentinormenormali"],
                                                             gradit=impostazioni["graditnormenormali"])
                    self.tempobetanormenormali = time() - tempo
                    self.probabilitàtuttogiusto *= 1 - self.betanormenormali
                    self.alfaveronormenormali = None
                    self.tempoalfaveronormenormali = None
                    print("54%: beta per il test di Shapiro calcolato")
                except Exception as e:
                    self.betanormenormali = None
                    self.tempobetanormenormali = None
                    self.alfaveronormenormali = None
                    self.tempoalfaveronormenormali = None
                    self.probabilitàtuttogiusto = None
                    impostazioni["calcolaerrori"] = False
                    print(f"\033[35mNon ho potuto calcolare il beta per il test di Shapiro perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
            elif self.taglia < 50 and impostazioni["calcolaerrori"]:
                try:
                    tempo = time()
                    self.alfaveronormenormali = alfaveronormenormali(iterazioni=impostazioni["iterazionialfaveronormenormali"],
                                                                     frecce=self.taglia,
                                                                     alfa=impostazioni["alfashapiro"])
                    self.tempoalfaveronormenormali = time() - tempo
                    self.probabilitàtuttogiusto *= 1 - self.alfaveronormenormali
                    self.betanormenormali = None
                    self.tempobetanormenormali = None
                    print("54,47%: alfa effettivo per il test di Shapiro calcolato")
                except Exception as e:
                    self.alfaveronormenormali = None
                    self.tempoalfaveronormenormali = None
                    self.betanormenormali = None
                    self.tempobetanormenormali = None
                    self.probabilitàtuttogiusto = None
                    impostazioni["calcolaerrori"] = False
                    print(f"\033[35mNon ho potuto calcolare l'alfa effettivo per il test di Shapiro perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
            else:
                if impostazioni["calcolaerrori"]:
                    self.probabilitàtuttogiusto *= 1 - impostazioni["alfashapiro"]
                self.betanormenormali = None
                self.tempobetanormenormali = None
                self.alfaveronormenormali = None
                self.tempoalfaveronormenormali = None
                print("54,95%: la probabilità di errore del test di Shapiro è vicina all'alfa nominale")
            cutoff3 = None
            soglie3 = None
            cutoffattuale3 = None
            try:
                tempo = time()
                cutoff3 = list()
                soglie3 = list()
                if impostazioni["hljungboxnorme"] < self.taglia-1:
                    cutoffattuale3 = impostazioni["hljungboxnorme"]
                    while cutoffattuale3 < self.taglia - 1:
                        cutoff3.append(cutoffattuale3)
                        soglie3.append(st.chi2.ppf(1 - impostazioni["alfaljungboxnorme"], df=cutoffattuale3))
                        cutoffattuale3 += impostazioni["hljungboxnorme"]
                cutoff3.append(self.taglia - 1)
                soglie3.append(st.chi2.ppf(1 - impostazioni["alfaljungboxnorme"], df=self.taglia - 1))
                self.normeindipendenti = miomodulo.ljungbox(autocorrelazioni(appiattisci(self.norme) if ordine else [np.mean(voléesingola) for voléesingola in self.norme],
                                                                             np.var(appiattisci(self.norme)) if ordine else np.var([np.mean(voléesingola) for voléesingola in self.norme])).tolist(),
                                                            cutoff3, self.taglia if ordine else self.volée, soglie3, len(cutoff3))
                self.temponormeindipendenti = time() - tempo
                print("55,42%: test di Ljung-Box sulle distanze dal centro effettuato")
            except Exception as e:
                self.normeindipendenti = None
                self.temponormeindipendenti = None
                print(f"\033[35mNon ho potuto eseguire il test di Ljung-Box sulle norme perché {e}\033[0m")
            if self.normeindipendenti is None:
                self.affidabilitànorme = False
            elif self.normenormali and self.normeindipendenti:
                self.affidabilitànorme = True
            else:
                self.affidabilitànorme = False
            if (self.normeindipendenti if self.normeindipendenti is not None else False) and impostazioni["calcolaerrori"]:
                try:
                    if cutoff3 is None or soglie3 is None:
                        raise TypeError("I cutoff o le soglie sono indeterminati")
                    tempo = time()
                    self.betanormeindipendenti = miomodulo.betaljungbox(self.taglia if ordine else self.volée,
                                                                        cutoff3, soglie3, len(cutoff3),
                                                                        impostazioni["voltebetaljungboxnorme"])
                    self.tempobetanormeindipendenti = time() - tempo
                    self.probabilitàtuttogiusto *= (1 - self.betanormeindipendenti)
                    self.alfaveronormeindipendenti = None
                    self.tempoalfaveronormeindipendenti = None
                    print("56,78%: beta per il test di Ljung-Box sulle distanze dal centro calcolato")
                except Exception as e:
                    self.betanormeindipendenti = None
                    self.tempobetanormeindipendenti = None
                    self.alfaveronormeindipendenti = None
                    self.tempoalfaveronormeindipendenti = None
                    self.probabilitàtuttogiusto = None
                    impostazioni["calcolaerrori"] = False
                    print(f"\033[35mNon ho potuto calcolare il beta per il test di Ljung-Box sulle distanze dal centro perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
            elif (self.taglia if ordine else self.volée) < 50 and impostazioni["calcolaerrori"] and self.normeindipendenti is not None:
                try:
                    if cutoff3 is None or soglie3 is None:
                        raise TypeError("I cutoff o le soglie sono indeterminati")
                    tempo = time()
                    self.alfaveronormeindipendenti = miomodulo.alfaveroljungbox(self.taglia if ordine else self.volée,
                                                                                cutoff3, soglie3, len(cutoff3),
                                                                                impostazioni["voltealfaljungboxnorme"])
                    self.tempoalfaveronormeindipendenti = time() - tempo
                    self.probabilitàtuttogiusto *= (1 - self.alfaveronormeindipendenti)
                    self.betanormeindipendenti = None
                    self.tempobetanormeindipendenti = None
                    print("58,41%: alfa effettivo del test di Ljung-Box sulle distanze dal centro calcolato")
                except Exception as e:
                    self.betanormeindipendenti = None
                    self.tempobetanormeindipendenti = None
                    self.alfaveronormeindipendenti = None
                    self.tempoalfaveronormeindipendenti = None
                    impostazioni["calcolaerrori"] = False
                    print(f"\033[35mNon ho potuto calcolare l'alfa effettivo per il test di Ljung-Box sulle distanze dal centro perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
            else:
                if impostazioni["calcolaerrori"] and self.normeindipendenti is not None:
                    self.probabilitàtuttogiusto *= (1 - impostazioni["alfaljungboxnorme"])**len(cutoff3)
                self.betanormeindipendenti = None
                self.tempobetanormeindipendenti = None
                self.alfaveronormeindipendenti = None
                self.tempoalfaveronormeindipendenti = None
                print("58,88%: la probabilità di errore del test di Ljung-Box sulle distanze dal centro è vicina all'alfa nominale")
        else:
            self.normenormali = None
            self.normeindipendenti = None
            self.temponormenormali = None
            self.temponormeindipendenti = None
        if impostazioni["intervallonorme"] and self.taglia > 1:
            try:
                tempo = time()
                self.intervallonorme = [np.mean(appiattisci(self.norme)) + (np.var(appiattisci(self.norme), ddof=1) / self.taglia) ** 0.5 * st.t.ppf((1 - impostazioni["confidenzanorme"]) / 2, df=self.taglia-1),
                                        np.mean(appiattisci(self.norme)) - (np.var(appiattisci(self.norme), ddof=1) / self.taglia) ** 0.5 * st.t.ppf((1 - impostazioni["confidenzanorme"]) / 2, df=self.taglia-1)]
                self.tempointervallonorme = time() - tempo
                print("59,9%: intervallo di confidenza sulle distanze dal centro calcolato")
            except Exception as e:
                self.intervallonorme = None
                self.tempointervallonorme = None
                print(f"\033[35mNon ho potuto calcolare un intervallo di confidenza per le distanze dal centro perché {e}\033[0m")
            try:
                tempo = time()
                graficointervallonorme(self.intervallonorme, data)
                self.tempograficointervallonorme = time() - tempo
                print("60,1%: intervallo di confidenza sulle distanze dal centro graficato")
            except Exception as e:
                self.tempograficointervallonorme = None
                print(f"\033[35mNon ho potuto graficare l'intervallo di confidenza per le distanze dal centro perché {e}\033[0m")
            if self.intervallonorme is not None and impostazioni["calcolaerrori"]:
                self.probabilitàtuttogiusto *= impostazioni["confidenzanorme"]
        else:
            self.intervallonorme = None
            self.tempointervallonorme = None
            print("60,44%: intervallo di confidenza sulle distanze dal centro non calcolato")
        if impostazioni["testnorme"] and self.taglia > 1:
            try:
                tempo = time()
                self.testnorme = st.ttest_1samp(appiattisci(self.norme),
                                                alternative=impostazioni["alternativanorme"],
                                                popmean=impostazioni["mediatestnorme"]).pvalue > impostazioni["alfatestnorme"]
                self.tempotestnorme = time() - tempo
                print("60,85%: t-test sulle distanze dal centro effettuato")
            except Exception as e:
                self.testnorme = None
                self.tempotestnorme = None
                print(f"\033[35mNon ho potuto effettuare il t-test sulle distanze dal centro perché {e}\033[0m")
            if (self.testnorme if self.testnorme is not None else False) and impostazioni["calcolaerrori"]:
                try:
                    tempo = time()
                    self.betatestnorme = betatestnorme(frecce=self.taglia,
                                                       alternativa=impostazioni["alternativanorme"],
                                                       distanza=impostazioni["distanzabetanorme"],
                                                       iterazioni=impostazioni["iterazionibetanorme"],
                                                       media=impostazioni["mediatestnorme"],
                                                       varianza=np.var(appiattisci(self.norme), ddof=1),
                                                       alfa=impostazioni["alfatestnorme"])
                    self.tempobetatestnorme = time() - tempo
                    self.probabilitàtuttogiusto *= 1 - self.betatestnorme
                    print("61,6%: beta per il t-test sulle distanze dal centro calcolato")
                except Exception as e:
                    self.betatestnorme = None
                    self.tempobetatestnorme = None
                    self.probabilitàtuttogiusto = None
                    impostazioni["calcolaerrori"] = False
                    print(f"\033[35mNon ho potuto calcolare il beta per il t-test sulle distanze dal centro perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
            else:
                self.betatestnorme = None
                self.tempobetatestnorme = None
                if impostazioni["calcolaerrori"] and self.testnorme is not None:
                    self.probabilitàtuttogiusto *= 1 - impostazioni["alfatestnorme"]
                print("61,94%: probabilità di errore del t-test sulle distanze dal centro non calcolate")
        else:
            self.testnorme = None
            self.tempotestnorme = None
            self.betatestnorme = None
            self.tempobetatestnorme = None
            print("62,27%: t-test sulle distanze dal centro non eseguito")
        if impostazioni["normebayesiane"] and impostazioni["intervallonormebayesiano"] and self.taglia > 1:
            try:
                tempo = time()
                self.normebayesiane, self.intervallonormebayesiano = normebayesiane(allenamento=appiattisci(self.norme),
                                                                                    catene=impostazioni["catenenorme"],
                                                                                    iterazioni=impostazioni["iterazionimcmcnorme"],
                                                                                    alternativa=impostazioni["alternativanorme"],
                                                                                    media=impostazioni["mediatestnorme"],
                                                                                    credibilità=impostazioni["credibilitànormebayesiane"],
                                                                                    rope=impostazioni["ROPEnorme"],
                                                                                    noninformativa=impostazioni["noninformativitànormebayesiane"],
                                                                                    allenamentipriori=impostazioni["allenamentipriorinormebayesiane"],
                                                                                    gerarchica=impostazioni["normegerarchiche"],
                                                                                    apiacere=impostazioni["apiacerenorme"],
                                                                                    alfaapiacere=impostazioni["alfaapiacerenorme"],
                                                                                    betaapiacere=impostazioni["betaapiacerenorme"],
                                                                                    alfa2apiacere=impostazioni["alfa2apiacerenorme"],
                                                                                    beta2apiacere=impostazioni["beta2apiacerenorme"],
                                                                                    console=impostazioni["consoledistan"])
                self.temponormebayesiane = time() - tempo
                if impostazioni["calcolaerrori"]:
                    self.probabilitàtuttogiusto *= 1 - min([self.normebayesiane, 1 - self.normebayesiane])
                    self.probabilitàtuttogiusto *= impostazioni["credibilitànormebayesiane"]
                print("63,7%: t-test bayesiano e intervallo di credibilità sulle distanze dal centro effettuati")
            except Exception as e:
                self.normebayesiane = None
                self.intervallonormebayesiano = None
                self.temponormebayesiane = None
                print(f"\033[35mNon ho potuto effettuare il t-test bayesiano o l'intervallo di credibilità sulle distanze dal centro perché {e}\033[0m")
        elif impostazioni["normebayesiane"] and self.taglia > 1:
            try:
                tempo = time()
                self.normebayesiane, _ = normebayesiane(allenamento=appiattisci(self.norme),
                                                        catene=impostazioni["catenenorme"],
                                                        iterazioni=impostazioni["iterazionimcmcnorme"],
                                                        alternativa=impostazioni["alternativanorme"],
                                                        media=impostazioni["mediatestnorme"],
                                                        credibilità=impostazioni["credibilitànormebayesiane"],
                                                        rope=impostazioni["ROPEnorme"],
                                                        noninformativa=impostazioni["noninformativitànormebayesiane"],
                                                        allenamentipriori=impostazioni["allenamentipriorinormebayesiane"],
                                                        gerarchica=impostazioni["normegerarchiche"],
                                                        apiacere=impostazioni["apiacerenorme"],
                                                        alfaapiacere=impostazioni["alfaapiacerenorme"],
                                                        betaapiacere=impostazioni["betaapiacerenorme"],
                                                        alfa2apiacere=impostazioni["alfa2apiacerenorme"],
                                                        beta2apiacere=impostazioni["beta2apiacerenorme"],
                                                        console=impostazioni["consoledistan"])
                self.temponormebayesiane = time() - tempo
                self.intervallonormebayesiano = None
                if impostazioni["calcolaerrori"]:
                    self.probabilitàtuttogiusto *= 1 - min([self.normebayesiane, 1 - self.normebayesiane])
                print("64,78%: t-test bayesiano sulle distanze dal centro effettuato")
            except Exception as e:
                self.normebayesiane = None
                self.intervallonormebayesiano = None
                self.temponormebayesiane = None
                print(f"\03[35mNon ho potuto effettuare il t-test bayesiano sulle distanze dal centro perché {e}\033[0m")
        elif impostazioni["intervallonormebayesiano"] and self.taglia > 1:
            try:
                tempo = time()
                _, self.intervallonormebayesiano = normebayesiane(allenamento=appiattisci(self.norme),
                                                                  catene=impostazioni["catenenorme"],
                                                                  iterazioni=impostazioni["iterazionimcmcnorme"],
                                                                  alternativa=impostazioni["alternativanorme"],
                                                                  media=impostazioni["mediatestnorme"],
                                                                  credibilità=impostazioni["credibilitànormebayesiane"],
                                                                  rope=impostazioni["ROPEnorme"],
                                                                  noninformativa=impostazioni["noninformativitànormebayesiane"],
                                                                  allenamentipriori=impostazioni["allenamentipriorinormebayesiane"],
                                                                  gerarchica=impostazioni["normegerarchiche"],
                                                                  apiacere=impostazioni["apiacerenorme"],
                                                                  alfaapiacere=impostazioni["alfaapiacerenorme"],
                                                                  betaapiacere=impostazioni["betaapiacerenorme"],
                                                                  alfa2apiacere=impostazioni["alfa2apiacerenorme"],
                                                                  beta2apiacere=impostazioni["beta2apiacerenorme"],
                                                                  console=impostazioni["consoledistan"])
                self.temponormebayesiane = time() - tempo
                self.normebayesiane = None
                if impostazioni["calcolaerrori"]:
                    self.probabilitàtuttogiusto *= impostazioni["credibilitànormebayesiane"]
                print("66,01%: intervallo di credibilità sulle distanze dal centro calcolato")
            except Exception as e:
                self.normebayesiane = None
                self.intervallonormebayesiane = None
                self.temponormebayesiane = None
                print(f"\033[35mNon ho potuto calcolare l'intervallo di credibilità sulle distanze dal centro perché {e}\033[0m")
        else:
            self.normebayesiane = None
            self.temponormebayesiane = None
            self.intervallonormebayesiano = None
            print("66,28%: t-test bayesiano e intervallo di credibilità sulle distanze dal centro non effettuati")
        if impostazioni["intervallivarianze"] and self.taglia > 2:
            try:
                tempo = time()
                varasc = (self.varianzacoordinate[0] if self.varianzacoordinate is not None
                          else np.var(appiattisci(self.ascisse), ddof=1))
                varord = (self.varianzacoordinate[1] if self.varianzacoordinate is not None
                          else np.var(appiattisci(self.ordinate), ddof=1))
                self.intervallivarianze = ([(self.taglia - 1) * varasc / st.chi2.ppf(
                    1 - (1-impostazioni["confidenzavarianzaascisse"]) / 2, df=self.taglia - 1),
                                            (self.taglia - 1) * varasc / st.chi2.ppf(
                                                (1-impostazioni["confidenzavarianzaascisse"]) / 2, df=self.taglia - 1)],
                                            [(self.taglia - 1) * varord / st.chi2.ppf(
                                                1 - (1-impostazioni["confidenzavarianzaordinate"]) / 2,
                                                df=self.taglia - 1),
                                            (self.taglia - 1) * varord / st.chi2.ppf(
                                                (1-impostazioni["confidenzavarianzaordinate"]) / 2, df=self.taglia - 1)])
                self.tempointervallivarianze = time() - tempo
                if impostazioni["calcolaerrori"]:
                    self.probabilitàtuttogiusto *= impostazioni["confidenzavarianzaascisse"]
                    self.probabilitàtuttogiusto *= impostazioni["confidenzavarianzaordinate"]
                print("67,57%: intervalli di confidenza delle varianze calcolati")
            except Exception as e:
                self.intervallivarianze = None
                self.tempointervallivarianze = None
                print(f"\033[35mNon ho potuto calcolare gli intervalli di confidenza delle varianze perché {e}\033[0m")
            if self.intervallivarianze is not None and self.mediacoordinate is not None:
                try:
                    tempo = time()
                    graficointervallivarianze(self.intervallivarianze, self.mediacoordinate, data)
                    self.tempograficointervallivarianze = time() - tempo
                    print("67,77%: grafico degli intervalli di confidenza delle varianze disegnato")
                except Exception as e:
                    self.tempograficointervallivarianze = None
                    print(f"\033[35mNon ho potuto disegnare il grafico degli intervalli di confidenza delle varianze perché {e}\033[0m")
            else:
                self.tempograficointervallivarianze = None
        else:
            self.intervallivarianze = None
            self.tempointervallivarianze = None
            self.tempograficointervallivarianze = None
            print("68,04%: intervalli di confidenza delle varianze non calcolati")
        if impostazioni["testvarianze"] and self.taglia > 2:
            try:
                tempo = time()
                varasc = (self.varianzacoordinate[0] if self.varianzacoordinate[0] is not None
                          else np.var(appiattisci(self.ascisse), ddof=1))
                varord = (self.varianzacoordinate[1] if self.varianzacoordinate[1] is not None
                          else np.var(appiattisci(self.ordinate), ddof=1))
                self.testvarianze = (testvarianze(alternativa=impostazioni["alternativavarianzaascisse"],
                                                  varianza=varasc,
                                                  ipotesinulla=impostazioni["ipotesinullavarianzaascisse"],
                                                  frecce=self.taglia) > impostazioni["alfavarianzaascisse"],
                                     testvarianze(alternativa=impostazioni["alternativavarianzaordinate"],
                                                  varianza=varord,
                                                  ipotesinulla=impostazioni["ipotesinullavarianzaordinate"],
                                                  frecce=self.taglia) > impostazioni["alfavarianzaordinate"])
                self.tempotestvarianze = time() - tempo
                print("69,06%: test d'ipotesi sulle varianze eseguito")
            except Exception as e:
                self.testvarianze = None
                self.tempotestvarianze = None
                print(f"\033[35mNon ho potuto eseguire il test d'ipotesi sulle varianze perché {e}\033[0m")
            if (self.testvarianze[0] if self.testvarianze is not None else False) and impostazioni["calcolaerrori"]:
                try:
                    tempo = time()
                    self.betatestvarianzeascisse = betatestvarianze(
                        iterazioni=impostazioni["iterazionibetavarianzaascisse"],
                        alternativa=impostazioni["alternativavarianzaascisse"],
                        ipotesinulla=impostazioni["ipotesinullavarianzaascisse"],
                        distanza=impostazioni["distanzabetavarianzaascisse"],
                        frecce=self.taglia,
                        alfa=impostazioni["alfavarianzaascisse"])
                    self.tempobetatestvarianzeascisse = time() - tempo
                    self.probabilitàtuttogiusto *= 1 - self.betatestvarianzeascisse
                    print("69,81%: beta per il test d'ipotesi sulle varianze delle ascisse calcolato")
                except Exception as e:
                    self.betatestvarianzeascisse = None
                    self.tempobetatestvarianzeascisse = None
                    self.probabilitàtuttogiusto = None
                    impostazioni["calcolaerrori"] = False
                    print(f"\033[35mNon ho potuto calcolare il beta per il test d'ipotesi sulle varianze delle ascisse perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
            else:
                self.betatestvarianzeascisse = None
                self.tempobetatestvarianzeascisse = None
                if impostazioni["calcolaerrori"] and self.testvarianze is not None:
                    self.probabilitàtuttogiusto *= 1 - impostazioni["alfavarianzaascisse"]
                    print("70,14%: la probabilità di errore del test delle varianze delle ascisse è pari all'alfa nominale")
            if (self.testvarianze[1] if self.testvarianze is not None else False) and impostazioni["calcolaerrori"]:
                try:
                    tempo = time()
                    self.betatestvarianzeordinate = betatestvarianze(iterazioni=impostazioni[
                                                                         "iterazionibetavarianzaordinate"],
                                                                     alternativa=impostazioni[
                                                                         "alternativavarianzaordinate"],
                                                                     ipotesinulla=impostazioni[
                                                                         "ipotesinullavarianzaordinate"],
                                                                     distanza=impostazioni["distanzabetavarianzaordinate"],
                                                                     frecce=self.taglia,
                                                                     alfa=impostazioni["alfavarianzaordinate"])
                    self.tempobetatestvarianzeordinate = time() - tempo
                    self.probabilitàtuttogiusto *= 1 - self.betatestvarianzeordinate
                    print("71,03%: beta per il test d'ipotesi sulle varianze delle ordinate calcolato")
                except Exception as e:
                    self.betatestvarianzeordinate = None
                    self.tempobetatestvarianzeordinate = None
                    self.probabilitàtuttogiusto = None
                    impostazioni["calcolaerrori"] = False
                    print(f"\033[35mNon ho potuto calcolare il beta per il test d'ipotesi sulle varianze delle ordinate perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
            else:
                self.betatestvarianzeordinate = None
                self.tempobetatestvarianzeordinate = None
                if impostazioni["calcolaerrori"] and self.testvarianze is not None:
                    self.probabilitàtuttogiusto *= 1 - impostazioni["alfavarianzaordinate"]
                    print("71,37%: la probabilità di errore del test d'ipotesi sulle varianze delle ordinate è pari all'alfa nominale")
        else:
            self.testvarianze = None
            self.tempotestvarianze = None
            self.betatestvarianzeascisse = None
            self.tempobetatestvarianzeascisse = None
            self.betatestvarianzeordinate = None
            self.tempobetatestvarianzeordinate = None
            print("71,84%: test d'ipotesi sulle varianze non eseguiti")
        if impostazioni["varianzebayesiane"] and impostazioni["intervallovarianzebayesiano"] and self.taglia > 2:
            try:
                tempo = time()
                self.varianzebayesiane, self.intervallovarianzebayesiane = varianzebayesiane(
                    allenamento=allenamentodepurato,
                    catene=impostazioni["catenevarianze"],
                    iterazioni=impostazioni["iterazionimcmcvarianze"],
                    alternativaascisse=impostazioni["alternativavarianzaascisse"],
                    alternativaordinate=impostazioni["alternativavarianzaordinate"],
                    ipotesinullaascisse=impostazioni["ipotesinullavarianzaascisse"],
                    ipotesinullaordinate=impostazioni["ipotesinullavarianzaordinate"],
                    ropeascisse=impostazioni["ROPEvarianzaascisse"],
                    ropeordinate=impostazioni["ROPEvarianzaordinate"],
                    noninformativa=impostazioni["noninformativitàvarianzebayesiane"],
                    allenamentipriori=impostazioni["allenamentipriorivarianzebayesiane"],
                    alfaascisse=1-impostazioni["credibilitàvarianzaascissebayesiane"],
                    alfaordinate=1-impostazioni["credibilitàvarianzaordinatebayesiane"],
                    gerarchica=impostazioni["varianzebayesianegerarchiche"],
                    apiacere=impostazioni["apiacerevarianzebayesiane"],
                    muapiacere=impostazioni["muapiacerevarianzebayesiane"],
                    sigmaapiacere=impostazioni["sigmaapiacerevarianzebayesiane"],
                    aapiacere=impostazioni["aapiacerevarianzebayesiane"],
                    bapiacere=impostazioni["bapiacerevarianzebayesiane"],
                    etaapiacere=impostazioni["etaapiacerevarianzebayesiane"],
                    console=impostazioni["consoledistan"])
                self.tempovarianzebayesiane = time() - tempo
                if impostazioni["calcolaerrori"]:
                    self.probabilitàtuttogiusto *= 1 - min([self.varianzebayesiane[0], 1 - self.varianzebayesiane[0]])
                    self.probabilitàtuttogiusto *= 1 - min([self.varianzebayesiane[1], 1 - self.varianzebayesiane[1]])
                    self.probabilitàtuttogiusto *= impostazioni["credibilitàvarianzaascissebayesiane"]
                    self.probabilitàtuttogiusto *= impostazioni["credibilitàvarianzaordinatebayesiane"]
                print("73,27%: test d'ipotesi bayesiano e intervallo di credibilità sulle varianze effettuati")
            except Exception as e:
                self.varianzebayesiane = None
                self.intervallovarianzebayesiane = None
                self.tempovarianzebayesiane = None
                print(f"\033[35mNon ho potuto effettuare il test d'ipotesi bayesiano o l'intervallo di credibilità sulle varianze perché {e}\033[0m")
            if self.intervallovarianzebayesiane is not None:
                try:
                    tempo = time()
                    graficointervallovarianzebayesiano(self.intervallovarianzebayesiane, self.mediacoordinate, data)
                    self.tempograficointervallovarianzebayesiano = time() - tempo
                    print("73,47%: intervallo di credibilità sulle varianze graficato")
                except Exception as e:
                    self.tempografricointervallovarianzebayesiano = None
                    print(f"\033[35mNon ho potuto graficare l'intervallo di credibilità sulle varianze perché {e}\033[0m")
            else:
                self.tempograficointervallovarianzebayesiano = None
        elif impostazioni["varianzebayesiane"] and self.taglia > 2:
            try:
                tempo = time()
                self.varianzebayesiane, _ = varianzebayesiane(
                    allenamento=allenamentodepurato,
                    catene=impostazioni["catenevarianze"],
                    iterazioni=impostazioni["iterazionimcmcvarianze"],
                    alternativaascisse=impostazioni["alternativavarianzascisse"],
                    alternativaordinate=impostazioni["alternativavarianzaordinate"],
                    ipotesinullaascisse=impostazioni["ipotesinullavarianzeascisse"],
                    ipotesinullaordinate=impostazioni["ipotesinullavarianzaordinate"],
                    ropeascisse=impostazioni["ROPEvarianzascisse"],
                    ropeordinate=impostazioni["ROPEvarianzaordinate"],
                    noninformativa=impostazioni["noninformativitàvarianzebayesiane"],
                    allenamentipriori=impostazioni["allenamentipriorivarianzebayesiane"],
                    alfaascisse=1-impostazioni["credibilitàvarianzaascissebayesiane"],
                    alfaordinate=1-impostazioni["credibilitàvarianzaordinatebayesiane"],
                    gerarchica=impostazioni["varianzebayesianegerarchiche"],
                    apiacere=impostazioni["apiacerevarianzebayesiane"],
                    muapiacere=impostazioni["muapiacerevarianzebayesiane"],
                    sigmaapiacere=impostazioni["sigmaapiacerevarianzebayesiane"],
                    aapiacere=impostazioni["aapiacerevarianzebayesiane"],
                    bapiacere=impostazioni["bapiacerevarianzebayesiane"],
                    etaapiacere=impostazioni["etaapiacerevarianzebayesiane"],
                    console=impostazioni["consoledistan"])
                self.tempovarianzebayesiane = time() - tempo
                self.intervallovarianzebayesiano = None
                self.tempograficointervallovarianzebayesiano = None
                if impostazioni["calcolaerrori"]:
                    self.probabilitàtuttogiusto *= 1 - min([self.varianzebayesiane[0], 1 - self.varianzebayesiane[0]])
                    self.probabilitàtuttogiusto *= 1 - min([self.varianzebayesiane[1], 1 - self.varianzebayesiane[1]])
                print("74,96%: test d'ipotesi sulle varianze bayesiano effettuato")
            except Exception as e:
                self.varianzebayesiane = None
                self.intervallovarianzebayesiano = None
                self.tempovarianzebayesiane = None
                self.tempograficointervallovarianzebayesiano = None
                print(f"\033[35mNon ho potuto effettuare il test d'ipotesi sulle varianze bayesiano perché {e}\033[0m")
        elif impostazioni["intervallovarianzebayesiano"] and self.taglia > 2:
            try:
                tempo = time()
                _, self.intervallovarianzebayesiane = varianzebayesiane(
                    allenamento=allenamentodepurato,
                    catene=impostazioni["catenevarianze"],
                    iterazioni=impostazioni["iterazionimcmcvarianze"],
                    alternativaascisse=impostazioni["alternativavarianzascisse"],
                    alternativaordinate=impostazioni["alternativavarianzaordinate"],
                    ipotesinullaascisse=impostazioni["ipotesinullavarianzeascisse"],
                    ipotesinullaordinate=impostazioni["ipotesinullavarianzaordinate"],
                    ropeascisse=impostazioni["ROPEvarianzascisse"],
                    ropeordinate=impostazioni["ROPEvarianzaordinate"],
                    noninformativa=impostazioni["noninformativitàvarianzebayesiane"],
                    allenamentipriori=impostazioni["allenamentipriorivarianzebayesiane"],
                    alfaascisse=1-impostazioni["credibilitàvarianzaascissebayesiane"],
                    alfaordinate=1-impostazioni["credibilitàvarianzaordinatebayesiane"],
                    gerarchica=impostazioni["varianzebayesianegerarchiche"],
                    apiacere=impostazioni["apiacerevarianzebayesiane"],
                    muapiacere=impostazioni["muapiacerevarianzebayesiane"],
                    sigmaapiacere=impostazioni["sigmaapiacerevarianzebayesiane"],
                    aapiacere=impostazioni["aapiacerevarianzebayesiane"],
                    bapiacere=impostazioni["bapiacerevarianzebayesiane"],
                    etaapiacere=impostazioni["etaapiacerevarianzebayesiane"],
                    console=impostazioni["consoledistan"])
                self.tempovarianzebayesiane = time() - tempo
                self.varianzebayesiane = None
                if impostazioni["calcolaerrori"]:
                    self.probabilitàtuttogiusto *= impostazioni["credibilitàvarianzaascissebayesiane"]
                    self.probabilitàtuttogiusto *= impostazioni["credibilitàvarianzaordinatebayesiane"]
                print("76,18%: intervallo di credibilità sulle varianze calcolato")
            except Exception as e:
                self.varianzebayesiane = None
                self.intervallovarianzebayesiane = None
                self.tempovarianzebayesiane = None
                print(f"\033[35mNon ho potuto calcolare l'intervallo di credibilità sulle varianze perché {e}\033[0m")
            try:
                tempo = time()
                graficointervallovarianzebayesiano(self.intervallovarianzebayesiane, self.mediacoordinate, data)
                self.tempograficointervallovarianzebayesiano = time() - tempo
                print("76,52%: intervallo di credibilità sulle varianze graficato")
            except Exception as e:
                self.tempograficointervallovarianzebayesiano = None
                print(f"\033[35mNon ho potuto graficare l'intervallo di credibilità sulle varianze perché {e}\033[0m")
        else:
            self.varianzebayesiane = None
            self.tempovarianzebayesiane = None
            self.intervallovarianzebayesiano = None
            self.tempograficointervallovarianzebayesiano = None
            print("76,86%: test d'ipotesi bayesiano e intervallo di credibilità sulle varianze non effettuati")
        if identificazione and impostazioni["freccedifettose"] and self.frecce > 1:
            try:
                tempo = time()
                ascisse = list()
                ordinate = list()
                frecce = list()
                for volée in allenamento:
                    for freccia in volée:
                        ascisse.append(freccia[0])
                        ordinate.append(freccia[1])
                        frecce.append(freccia[2])
                self.difettosità = miomodulo.freccedifettose(ascisse, ordinate, frecce, self.taglia,
                                                             st.chi2.ppf(1 - impostazioni["alfahotellingduecampioni"],
                                                                 df=2),
                                                             self.frecce)
                self.tempodifettosità = time() - tempo
                print("77,4%: difettosità delle frecce valutata")
            except Exception as e:
                self.difettosità = None
                self.tempodifettosità = None
                print(f"\033[35mNon ho potuto valutare la difettosità delle frecce perché {e}\033[0m")
            try:
                if not (all(self.difettosità) if self.difettosità is not None else True) and impostazioni["calcolaerrori"]:
                    tempo = time()
                    matricecovarianze = np.cov(np.array([appiattisci(self.ascisse), appiattisci(self.ordinate)]), rowvar=False, bias=False)
                    self.betahotellingduecampioni = miomodulo.betahotellingduecampioni(
                        impostazioni["iterazionibetahotellingduecampioni"],
                        matricecovarianze[0, 0],
                        matricecovarianze[1, 1],
                        matricecovarianze[0, 1],
                        self.taglia,
                        impostazioni["distanzabetahotellingduecampioni"],
                        self.frecce,
                        impostazioni["alfahotellingduecampioni"])
                    self.tempobetahotellingduecampioni = time() - tempo
                    print("78,29%: beta per il test di Hotelling a due campioni calcolato")
                else:
                    self.betahotellingduecampioni = None
                    self.tempobetahotellingduecampioni = None
                if impostazioni["calcolaerrori"] and self.difettosità is not None:
                    for difettosità in self.difettosità:
                        if difettosità:
                            self.probabilitàtuttogiusto *= 1 - self.betahotellingduecampioni
                        else:
                            self.probabilitàtuttogiusto *= 1 - impostazioni["alfahotellingduecampioni"]
            except Exception as e:
                self.betahotellingduecampioni = None
                self.tempobetahotellingduecampioni = None
                self.probabilitàtuttogiusto = None
                impostazioni["calcolaerrori"] = False
                print(f"\033[35mNon ho potuto calcolare il beta per il test di Hotelling a due campioni perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
            try:
                tempo = time()
                self.difettositàvarianze = varianzedifettose(allenamento=allenamento,
                                                             alfa=impostazioni["alfavarianzeduecampioni"],
                                                             numerofrecce=self.frecce)
                self.tempodifettositàvarianze = time() - tempo
                print("79,17%: difettosità della dispersione delle frecce valutata")
            except Exception as e:
                self.difettositàvarianze = None
                self.tempodifettositàvarianze = None
                print(f"\033[35mNon ho potuto valutare la difettosità della dispersione delle frecce perché {e}\033[0m")
            if ((not all([valore for coppia in self.difettositàvarianze for valore in coppia]) if self.difettositàvarianze is not None else False)
                    and impostazioni["calcolaerrori"]):
                try:
                    tempo = time()
                    self.betavarianzeduecampioni = miomodulo.betavarianzeduecampioni(
                        impostazioni["iterazionibetavarianzeduecampioni"],
                        impostazioni["distanzabetavarianzeduecampioni"],
                        self.taglia,
                        self.frecce,
                        st.f.ppf(1 - impostazioni["alfavarianzeduecampioni"],
                                        int(self.taglia / self.frecce),
                                        self.taglia - int(self.taglia / self.frecce)))
                    self.tempobetavarianzeduecampioni = time() - tempo
                    print("79,98%: beta per il test d'ipotesi sulle varianze a due campioni calcolato")
                except Exception as e:
                    self.betavarianzeduecampioni = None
                    self.tempobetavarianzeduecampioni = None
                    self.probabilitàtuttogiusto = None
                    impostazioni["calcolaerrori"] = False
                    print(f"\033[35mNon ho potuto calcolare il beta per il test d'ipotesi sulle varianze a due campioni perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
            else:
                self.betavarianzeduecampioni = None
                self.tempobetavarianzeduecampioni = None
            if impostazioni["calcolaerrori"] and self.difettositàvarianze is not None:
                for difettosità in [valore for coppia in self.difettositàvarianze for valore in coppia]:
                    if difettosità:
                        self.probabilitàtuttogiusto *= 1 - self.betavarianzeduecampioni
                    else:
                        self.probabilitàtuttogiusto *= 1 - impostazioni["alfavarianzeduecampioni"]
        else:
            self.difettosità = None
            self.tempodifettosità = None
            self.betahotellingduecampioni = None
            self.tempobetahotellingduecampioni = None
            self.difettositàvarianze = None
            self.tempodifettositàvarianze = None
            self.betavarianzeduecampioni = None
            self.tempobetavarianzeduecampioni = None
            print("81,2%: difettosità delle frecce non valutata")
        if (impostazioni["uniformitàangoli"] or impostazioni["fittavonmises"]) and self.taglia > 1:
            tempo = time()
            self.uniformitàangoli = self.varianzaangoli * self.taglia * 2 <= st.chi2.ppf(1 - impostazioni["alfarayleigh"], df=2)
            self.tempouniformitàangoli = time() - tempo
            print("81,61%: test di Rayleigh eseguito")
            if self.uniformitàangoli and impostazioni["calcolaerrori"]:
                try:
                    tempo = time()
                    self.betarayleigh = miomodulo.betarayleigh(impostazioni["iterazionibetarayleigh"],
                                                               impostazioni["kappabetarayleigh"],
                                                               self.taglia,
                                                               st.chi2.ppf(1 - impostazioni["alfarayleigh"], df=2))
                    self.tempobetarayleigh = time() - tempo
                    self.probabilitàtuttogiusto *= 1 - self.betarayleigh
                    self.alfaverorayleigh = None
                    self.tempoalfaverorayleigh = None
                    print("82,29%: beta per il test di Rayleigh calcolato")
                except Exception as e:
                    self.betarayleigh = None
                    self.tempobetarayleigh = None
                    self.alfaverorayleigh = None
                    self.tempoalfaverorayleigh = None
                    self.probabilitàtuttogiusto = None
                    impostazioni["calcolaerrori"] = False
                    print(f"\033[35mNon ho potuto calcolare il beta per il test di Rayleigh perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
            elif self.taglia < 50 and impostazioni["calcolaerrori"]:
                try:
                    tempo = time()
                    self.alfaverorayleigh = miomodulo.alfaverorayleigh(
                        impostazioni["iterazionialfaverorayleigh"],
                        st.chi2.ppf(1 - impostazioni["alfarayleigh"], df=2),
                        self.taglia)
                    self.tempoalfaverorayleigh = time() - tempo
                    self.probabilitàtuttogiusto *= 1-self.alfaverorayleigh
                    self.betarayleigh = None
                    self.tempobetarayleigh = None
                    print("82,76%: alfa effettivo per il test di Rayleigh calcolato")
                except Exception as e:
                    self.alfaverorayleigh = None
                    self.tempoalfaverorayleigh = None
                    self.betarayleigh = None
                    self.tempobetarayleigh = None
                    print(f"\033[35mNon ho potuto calcolare l'alfa effettivo per il test di Rayleigh perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
            else:
                if impostazioni["calcolaerrori"]:
                    self.probabilitàtuttogiusto *= 1 - impostazioni["alfarayleigh"]
                    print("83,24%: la probabilità di errore del test di Rayleigh è vicina all'alfa nominale")
                self.alfaverorayleigh = None
                self.tempoalfaverorayleigh = None
                self.betarayleigh = None
                self.tempobetarayleigh = None
            if impostazioni["fittavonmises"]:
                try:
                    tempo = time()
                    self.kappa = pycircstat.kappa(np.array(appiattisci(self.angoli)))[0]
                    self.tempokappa = time() - tempo
                    print("83,51%: stima del parametro di concentrazione calcolata")
                    if self.kappa < 1:
                        print(f"\033[37mStima del parametro di concentrazione instabile\033[0m")
                except Exception as e:
                    self.kappa = None
                    self.tempokappa = None
                    print(f"\033[35mNon ho potuto stimare il parametro di concentrazione perché {e}; interromperemo l'analisi sui dati angolari\033[0m")
            if impostazioni["fittavonmises"] and self.kappa is not None:
                try:
                    tempo = time()
                    self.affidabilitàvonmises = miomodulo.testvonmises(appiattisci(self.angoli) if isinstance(self.angoli, list) else appiattisci(self.angoli).tolist(),
                                                                       self.taglia,
                                                                       self.mediaangoli if self.mediaangoli is not None else st.circmean(appiattisci(self.angoli)),
                                                                       self.kappa,
                                                                       st.chi2.ppf(1 - impostazioni["alfatestvonmises"], df=2))
                    self.tempoaffidabilitàvonmises = time() - tempo
                    print("84,32%: test di goodness of fit della von Mises eseguito")
                except Exception as e:
                    self.affidabilitàvonmises = None
                    self.tempoaffidabilitàvonmises = None
                    print(f"\033[35mNon ho potuto eseguire il test di goodness of fit della von Mises perché {e}\033[0m")
                if (self.affidabilitàvonmises if self.affidabilitàvonmises is not None else False) and impostazioni["calcolaerrori"]:
                    try:
                        tempo = time()
                        self.betaaffidabilitàvonmises = miomodulo.betaaffidabilitavonmises(
                            impostazioni["iterazionibetaaffidabilitàvonmises"],
                            impostazioni["uniformebetaaffidabilitàvonmises"],
                            self.taglia,
                            st.chi2.ppf(1 - impostazioni["alfatestvonmises"], df=2),
                            impostazioni["distanzacomponentibetaaffidabilitàvonmises"],
                            impostazioni["kappabetaaffidabilitàvonmises"])
                        self.tempobetaaffidabilitàvonmises = time() - tempo
                        self.alfaveroaffidabilitàvonmises = None
                        self.tempoalfaveroaffidabilitàvonmises = None
                        self.probabilitàtuttogiusto *= 1 - self.betaaffidabilitàvonmises
                        print("85,21%: beta per il test di goodness of fit della von Mises calcolato")
                    except Exception as e:
                        self.betaaffidabilitàvonmises = None
                        self.tempobetaaffidabilitàvonmises = None
                        self.alfaveroaffidabilitàvonmises = None
                        self.tempoalfaveroaffidabilitàvonmises = None
                        self.probabilitàtuttogiusto = None
                        print(f"\033[35mNon ho potuto calcolare il beta per il test di goodness of fit della von Mises perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
                elif self.taglia < 50 and impostazioni["calcolaerrori"] and self.affidabilitàvonmises is not None:
                    try:
                        tempo = time()
                        self.alfaveroaffidabilitàvonmises = miomodulo.alfaveroaffidabilitavonmises(
                            impostazioni["iterazionialfaveroaffidabilitàvonmises"],
                            self.kappa,
                            self.taglia,
                            st.chi2.ppf(1 - impostazioni["alfatestvonmises"], df=2))
                        self.tempoalfaveroaffidabilitàvonmises = time() - tempo
                        self.betaaffidabilitàvonmises = None
                        self.tempobetaaffidabilitàvonmises = None
                        self.probabilitàtuttogiusto *= 1 - self.alfaveroaffidabilitàvonmises
                        print("85,95%: alfa effettivo per il test di goodness of fit della von Mises calcolato")
                    except Exception as e:
                        self.alfaveroaffidabilitàvonmises = None
                        self.tempoalfaveroaffidabilitàvonmises = None
                        self.betaaffidabilitàvonmises = None
                        self.tempobetaaffidabilitàvonmises = None
                        self.probabilitàtuttogiusto = None
                        impostazioni["calcolaerrori"] = False
                        print(f"\033[35mNon ho potuto calcolare l'alfa effettivo per il test di goodness of fit della von Mises perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
                else:
                    if impostazioni["calcolaerrori"]:
                        self.probabilitàtuttogiusto *= 1 - impostazioni["alfatestvonmises"]
                        print("86,43%: la probabilità di errore del test di goodness of fit della von Mises è vicino all'alfa nominale")
                    self.alfaveroaffidabilitàvonmises = None
                    self.tempoalfaveroaffidabilitàvonmises = None
                    self.betaaffidabilitàvonmises = None
                    self.tempobetaaffidabilitàvonmises = None
                    # Fino a qui sono 1274 righe su 1474
                if impostazioni["intervallomediavonmises"]:
                    try:
                        tempo = time()
                        self.intervalloangolomedio = intervalloangolomedio(
                            facile=impostazioni["intervallomediavonmisesfacile"],
                            angolomedio=self.mediaangoli if self.mediaangoli is not None else st.circmean(self.angoli),
                            confidenza=impostazioni["confidenzaangolomedio"],
                            kappa=self.kappa,
                            r=self.varianzaangoli if self.varianzaangoli is not None else varianzaangoli(self.angoli),
                            frecce=self.taglia)
                        self.tempointervalloangolomedio = time() - tempo
                        print("87,1%: intervallo di confidenza per l'angolo medio calcolato")
                    except Exception as e:
                        self.intervalloangolomedio = None
                        self.tempointervalloangolomedio = None
                        print(f"\033[35mNon ho potuto calcolare l'intervallo di confidenza per l'angolo medio perché {e}")
                    try:
                        tempo = time()
                        graficointervalloangolomedio(self.intervalloangolomedio, data)
                        self.tempograficointervalloangolomedio = time() - tempo
                        print("87,31%: grafico per l'intervallo di confidenza per l'angolo medio disegnato")
                    except Exception as e:
                        self.tempograficointervalloangolomedio = None
                        print(f"\033[35mNon ho potuto disegnare il grafico dell'intervallo di confidenza per l'angolo medio perché {e}")
                    if self.taglia < 50 and impostazioni["calcolaerrori"] and self.intervalloangolomedio is not None:
                        try:
                            tempo = time()
                            self.alfaverointervalloangolomedio = miomodulo.alfaverointervalloangolomedio(
                                impostazioni["iterazionialfaveromediavonmises"],
                                self.kappa,
                                self.taglia,
                                impostazioni["intervallomediavonmisesfacile"],
                                st.chi2.ppf(1 - impostazioni["confidenzaangolomedio"], df=1))
                            self.tempoalfaverointervalloangolomedio = time() - tempo
                            self.probabilitàtuttogiusto *= 1 - self.alfaverointervalloangolomedio
                            print("87,99%: alfa effettivo per l'intervallo di confidenza per l'angolo medio calcolato")
                        except Exception as e:
                            self.alfaverointervalloangolomedio = None
                            self.tempoalfaverointervalloangolomedio = None
                            self.probabilitàtuttogiusto = None
                            impostazioni["calcolaerrori"] = False
                            print(f"\033[35mNon ho potuto calcolare l'alfa effettivo per l'intervallo di confidenza per l'angolo medio perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
                    else:
                        self.alfaverointervalloangolomedio = None
                        self.tempoalfaverointervalloangolomedio = None
                        if impostazioni["calcolaerrori"] and self.intervalloangolomedio is not None:
                            self.probabilitàtuttogiusto *= impostazioni["confidenzaangolomedio"]
                            print("88,33%: il livello di confidenza effettivo dell'intervallo di confidenza per l'angolo medio è vicino a quello nominale")
                else:
                    self.intervalloangolomedio = None
                    self.tempointervalloangolomedio = None
                    self.alfaverointervalloangolomedio = None
                    self.tempoalfaverointervalloangolomedio = None
                    self.tempograficointervalloangolomedio = None
                    print("88,73%: intervallo di confidenza per l'angolo medio non calcolato")
                if impostazioni["intervallokappavonmises"]:
                    try:
                        tempo = time()
                        self.intervallokappa = miomodulo.intervallokappa(appiattisci(self.angoli),
                                                                         impostazioni["iterazioniintervallokappa"],
                                                                         impostazioni["confidenzaintervallokappa"],
                                                                         self.taglia)
                        self.tempointervallokappa = time() - tempo
                        print("89,28: intervallo di confidenza per il parametro di concentrazione calcolato")
                    except Exception as e:
                        self.intervallokappa = None
                        self.tempointervallokappa = None
                        print(f"\033[35mNon ho potuto calcolare l'intervallo di confidenza per il parametro di concentrazione perché {e}\033[0m")
                    try:
                        tempo = time()
                        graficointervallokappa(intervallo=self.intervallokappa,
                                                media=self.mediaangoli,
                                                data=data)
                        self.tempograficointervallokappa = time() - tempo
                        print("89,48%: grafico per l'intervallo di confidenza per il parametro di concentrazione disegnato")
                    except Exception as e:
                        self.tempograficointervallokappa = None
                        print(f"\033[35mNon ho potuto disegnare disegnare il grafico per l'intervallo di confidenza per il parametro di concentrazione perché {e}\033[0m")
                    if self.taglia < 50 and impostazioni["calcolaerrori"] and self.intervallokappa is not None:
                        try:
                            tempo = time()
                            self.alfaverointervallokappa = miomodulo.alfaverointervallokappa(
                                impostazioni["iterazionialfaverointervallokappa"],
                                impostazioni["confidenzaintervallokappa"],
                                self.kappa,
                                impostazioni["iterazioniintervallokappa"],
                                self.taglia)
                            self.tempoalfaverointervallokappa = time() - tempo
                            self.probabilitàtuttogiusto *= 1 - self.alfaverointervallokappa
                            print("90,16%: alfa effettivo per l'intervallo di confidenza del parametro di concentrazione calcolato")
                        except Exception as e:
                            self.alfaverointervallokappa = None
                            self.tempoalfaverointervallokappa = None
                            self.probabilitàtuttogiusto = None
                            impostazioni["calcolaerrori"] = False
                            print(f"\033[35mNon ho potuto calcolare il livello di confidenza effettivo per l'intervallo di confidenza del parametro di concentrazione calcolato perché {e}; smetteremo di calcolare le probabilità degli errori\033[0m")
                    else:
                        self.alfaverointervallokappa = None
                        self.tempoalfaverointervallokappa = None
                        if impostazioni["calcolaerrori"] and self.intervallokappa is not None:
                            self.probabilitàtuttogiusto *= 1 - impostazioni["confidenzaintervallokappa"]
                            print("90,5%: il livello di confidenza effettivo dell'intervallo di confidenza per il parametro di concentrazione è vicino a quello nominale")
                else:
                    self.intervallokappa = None
                    self.tempointervallokappa = None
                    self.alfaverointervallokappa = None
                    self.tempoalfaverointervallokappa = None
                    self.tempograficointervallokappa = None
                    print("90,9%: intervallo di confidenza per il parametro di concentrazione non calcolato")
            if (impostazioni["angolomediobayesiano"] and impostazioni["kappabayesiano"]
                    and self.taglia > 1 and impostazioni["fittavonmises"]):
                try:
                    tempo = time()
                    self.intervalloangolomediobayesiano, self.kappabayesiano, self.intervallokappabayesiano = angolomediobayesiano(
                        allenamento=appiattisci(self.angoli),
                        catene=impostazioni["cateneangolomediobayesiano"],
                        iterazioni=impostazioni["iterazionimcmcangolomediobayesiano"],
                        credibilitàmu=impostazioni["credibilitàangolomediobayesiano"],
                        credibilitàkappa=impostazioni["credibilitàkappabayesiano"],
                        noninformativa=impostazioni["noninformativitàangolomediobayesiano"],
                        allenamentipriori=impostazioni["allenamentiprioriangolomediobayesiano"],
                        variazionale=impostazioni["variazionaleangolomediobayesiano"],
                        gerarchica=impostazioni["angolomediobayesianogerarchico"],
                        apiacere=impostazioni["angolomediobayesianoapiacere"],
                        mu=impostazioni["muangolomediobayesiano"],
                        kappa=impostazioni["kappaangolomediobayesiano"],
                        alfa=impostazioni["alfaangolomediobayesiano"],
                        beta1=impostazioni["betaangolomediobayesiano"],
                        console=impostazioni["consoledistan"])
                    self.tempointervalloangolomediobayesiano = time() - tempo
                    if impostazioni["calcolaerrori"]:
                        self.probabilitàtuttogiusto *= impostazioni["credibilitàangolomediobayesiano"]
                        self.probabilitàtuttogiusto *= impostazioni["credibilitàkappabayesiano"]
                    print("91,92%: stime bayesiane per i parametri circolari calcolate")
                except Exception as e:
                    self.intervalloangolomediobayesiano, self.kappabayesiano, self.intervallokappabayesiano = None, None, None
                    self.tempointervalloangolomediobayesiano = None
                    print(f"\033[35mNon ho potuto calcolare le stime bayesiane per i parametri circolari perché {e}")
                try:
                    tempo = time()
                    graficoangolomediobayesiano(self.intervalloangolomediobayesiano, data)
                    self.tempograficoangolomediobayesiano = time() - tempo
                    print("92,13%: grafico per l'intervallo di credibilità dell'angolo medio disegnato")
                except Exception as e:
                    self.tempograficoangolomediobayesiano = None
                    print(f"\033[35mNon ho potuto disegnare il grafico per l'intervallo di credibilità dell'angolo medio perché {e}\033[0m")
            elif impostazioni["angolomediobayesiano"] and impostazioni["fittavonmises"] and self.taglia > 1:
                try:
                    tempo = time()
                    self.intervalloangolomediobayesiano, _, _ = angolomediobayesiano(
                        allenamento=appiattisci(self.angoli),
                        catene=impostazioni["cateneangolomediobayesiano"],
                        iterazioni=impostazioni["iterazionimcmcangolomediobayesiano"],
                        credibilitàmu=impostazioni["credibilitàangolomediobayesiano"],
                        credibilitàkappa=impostazioni["credibilitàkappabayesiano"],
                        noninformativa=impostazioni["noninformativitàangolomediobayesiano"],
                        allenamentipriori=impostazioni["allenamentiprioriangolomediobayesiano"],
                        variazionale=impostazioni["variazionaleangolomediobayesiano"],
                        gerarchica=impostazioni["angolomediobayesianogerarchico"],
                        apiacere=impostazioni["angolomediobayesianoapiacere"],
                        mu=impostazioni["muangolomediobayesiano"],
                        kappa=impostazioni["kappaangolomediobayesiano"],
                        alfa=impostazioni["alfaangolomediobayesiano"],
                        beta1=impostazioni["betaangolomediobayesiano"],
                        console=impostazioni["consoledistan"])
                    self.tempointervalloangolomediobayesiano = time() - tempo
                    self.kappabayesiano = None
                    self.intervallokappabayesiano = None
                    if impostazioni["calcolaerrori"]:
                        self.probabilitàtuttogiusto *= impostazioni["credibilitàangolomediobayesiano"]
                    print("93,48%: intervallo di credibilità per l'angolo medio calcolato")
                except Exception as e:
                    self.intervalloangolomediobayesiano = None
                    self.tempointervalloangolomediobayesiano = None
                    self.kappabayesiano = None
                    self.intervallokappabayesiano = None
                    print(f"\033[35mNon ho potuto calcolare l'intervallo di credibilità per l'angolo medio perché {e}\033[0m")
                try:
                    tempo = time()
                    graficoangolomediobayesiano(self.intervalloangolomediobayesiano, data)
                    self.tempograficoangolomediobayesiano = time() - tempo
                    print("93,69%: grafico per l'intervallo di credibilità dell'angolo medio disegnato")
                except Exception as e:
                    self.tempograficoangolomediobayesiano = None
                    print(f"\033[35mNon ho potuto disegnare il grafico per l'intervallo di credibilità dell'angolo medio perché {e}\033[0m")
            elif impostazioni["kappabayesiano"] and impostazioni["fittavonmises"] and self.taglia > 1:
                try:
                    tempo = time()
                    _, self.kappabayesiano, self.intervallokappabayesiano = angolomediobayesiano(
                        allenamento=appiattisci(self.angoli),
                        catene=impostazioni["cateneangolomediobayesiano"],
                        iterazioni=impostazioni["iterazionimcmcangolomediobayesiano"],
                        credibilitàmu=impostazioni["credibilitàangolomediobayesiano"],
                        credibilitàkappa=impostazioni["credibilitàkappabayesiano"],
                        noninformativa=impostazioni["noninformativitàangolomediobayesiano"],
                        allenamentipriori=impostazioni["allenamentiprioriangolomediobayesiano"],
                        variazionale=impostazioni["variazionaleangolomediobayesiano"],
                        gerarchica=impostazioni["angolomediobayesianogerarchico"],
                        apiacere=impostazioni["angolomediobayesianoapiacere"],
                        mu=impostazioni["muangolomediobayesiano"],
                        kappa=impostazioni["kappaangolomediobayesiano"],
                        alfa=impostazioni["alfaangolomediobayesiano"],
                        beta1=impostazioni["betaangolomediobayesiano"],
                        console=impostazioni["consoledistan"]
                    )
                    self.tempoangolomediobayesiano = time() - tempo
                    self.intervalloangolomediobayesiano = None
                    self.tempograficoangolomediobayesiano = None
                    if impostazioni["calcolaerrori"]:
                        self.probabilitàtuttogiusto *= impostazioni["credibilitàkappabayesiano"]
                    print("95,38%: stime bayesiane per il parametro di concentrazione calcolate")
                except Exception as e:
                    self.intervalloangolomediobayesiano = None
                    self.kappabayesiano = None
                    self.intervallokappabayesiano = None
                    self.tempograficoangolomediobayesiano = None
                    self.tempograficoangolomediobayesiano = None
                    print(f"\033[35mNon ho potuto calcolare le stime bayesiane per il parametro di concentrazione perché {e}\033[0m")
            else:
                self.intervalloangolomediobayesiano = None
                self.tempograficoangolomediobayesiano = None
                self.kappabayesiano = None
                self.intervallokappabayesiano = None
                self.tempokappabayesiano = None
                print("95,79%: stime bayesiane per l'angolo medio e il parametro di concentrazione non calcolate")
            if (impostazioni["angolomediobayesiano"] or impostazioni[
                "kappabayesiano"]) and impostazioni["misturevonmises"] and impostazioni["fittavonmises"] and self.taglia > 1:
                try:
                    tempo = time()
                    self.intervalliangolomedio, self.intervallikappa, self.componentimisturevonmises, assegnazioni = misturevonmises(
                        componentifissate=impostazioni["componentifissatemisturevonmises"],
                        allenamento=appiattisci(self.angoli),
                        noninformativa=impostazioni["noninformativitàmisturevonmises"],
                        allenamentipriori=impostazioni["allenamentipriorimisturevonmises"],
                        componentimassime=impostazioni["componentimisturevonmises"],
                        credibilitàmu=impostazioni["credibilitàmumisturevonmises"],
                        credibilitàkappa=impostazioni["credibilitàkappamisturevonmises"])
                    self.tempomisturevonmises = time() - tempo
                    if impostazioni["calcolaerrori"]:
                        self.probabilitàtuttogiusto *= (impostazioni["credibilitàmumisturevonmises"]) ** len(self.componentimisturevonmises)
                        self.probabilitàtuttogiusto *= (impostazioni["credibilitàkappamisturevonmises"]) ** len(self.componentimisturevonmises)
                    print("96,87%: modello mistura per gli angoli fittato")
                except Exception as e:
                    self.intervalliangolomedio = None
                    self.intervallikappa = None
                    self.componentimisturevonmises = None
                    self.tempomisturevonmises = None
                    print(f"\033[35mNon ho potuto fittare il modello mistura per gli angoli perché {e}\033[0m")
                if self.intervalliangolomedio is not None and self.intervallikappa is not None:
                    try:
                        tempo = time()
                        graficomisturevonmises(self.intervalliangolomedio, self.intervallikappa,
                                               self.componentimisturevonmises, appiattisci(self.angoli),
                                               assegnazioni, data)
                        self.tempograficomisturevonmises = time() - tempo
                        print("97,08%: grafico del modello mistura per gli angoli disegnato")
                    except Exception as e:
                        self.tempograficomisturevonmises = None
                        print(f"\033[35mNon ho potuto disegnare il grafico del modello mistura per gli angoli perché {e}")
                else:
                    self.tempograficomisturevonmises = None
            else:
                self.intervalliangolomedio = None
                self.intervallikappa = None
                self.componentimisturevonmises = None
                self.tempomisturevonmises = None
                self.tempograficomisturevonmises = None
                print("97,48%: modello mistura per gli angoli non fittato")
        else:
            self.uniformitàangoli = None
            self.tempouniformitàangoli = None
            self.betarayleigh = None
            self.tempobetarayleigh = None
            self.kappa = None
            self.tempokappa = None
            self.affidabilitàvonmises = None
            self.tempoaffidabilitàvonmises = None
            self.intervalloangolomedio = None
            self.tempointervalloangolomedio = None
            self.alfaverointervalloangolomedio = None
            self.tempoalfaverointervalloangolomedio = None
            self.intervallokappa = None
            self.tempointervallokappa = None
            self.alfaverointervallokappa = None
            self.tempoalfaverointervallokappa = None
            self.intervalliangolomedio = None
            self.intervallikappa = None
            self.componentimisturevonmises = None
            self.tempomisturevonmises = None
            print("98,91%: inferenza circolare non eseguita")
        self.impostazioni = impostazioni
        self.tempototale = time() - tempoiniziale
        print("\033[36mHo calcolato tutto!\033[0m")


# Da questo punto
def graficoholt(date: list[str], periodi: int, smorzamento: float):
    medie = pd.DataFrame([leggifile(f"Sessione del {data}").mediapunteggi for data in date], columns=["medie"])
    modello = Holt(medie["medie"], damped_trend=smorzamento).fit()
    previsione = modello.forecast(periodi)
    plt.plot(medie["medie"], label="Dati")
    plt.plot(previsione, label="Previsione")
    plt.savefig("graficoholt.png")


def graficoarima(date: list[str], periodi: int):
    medie = [leggifile(f"Sessione del {data}").mediapunteggi for data in date]
    serie = pd.Series(medie)
    modello = auto_arima(serie)
    previsione = modello.forecast(n_periods=periodi)
    plt.plot(serie, label="Dati")
    plt.plot(previsione, label="Previsione")
    plt.savefig("graficoarima.png")


def regressionegenerale(date: list[str], iterazioni: int):
    dataset = pd.DataFrame({"Punteggio": [leggifile(f"Sessione del {data}").mediapunteggi for data in date],
                            "Sessione": indicatoridate(date),
                            "Distanza": [leggifile(f"Sessione del {data}").distanza for data in date],
                            "Arco": [leggifile(f"Sessione del {data}").arco for data in date]})
    dataset["Arco"] = pd.Categorical(dataset["Arco"])
    matricedisegno = pd.get_dummies(dataset[["Sessione", "Distanza", "Arco"]], drop_first=True)
    y = dataset["Punteggio"]
    modello = GLSAR(y, matricedisegno, rho=1).iterative_fit(maxiter=iterazioni)
    return modello.params.filter(like="Arco"), modello.params["Distanza"]


# Agli allenamenti non è associata una distanza dal bersaglio
class StatisticheGenerali:
    def __init__(self, date, impostazioni):
        self.impostazioni = impostazioni
        # Dovresti farlo solo se hai almeno due osservazioni
        if impostazioni["metodoprevisione"] == "Holt":
            graficoholt(date, impostazioni["periodiprevisione"], impostazioni["smorzamentoholt"])
        else:
            graficoarima(date, impostazioni["periodiprevisione"])
        self.effettodistanza, self.effettoarchi = regressionegenerale(date, impostazioni["iterazioniglsar"])


def disegnabersaglio(tela, posizione: tuple | None, x: float | int, y: float | int, larghezza: float | int):
    # Disegna un bersaglio dato un oggetto canvas di una certa larghezza.
    if posizione is None:
        posizione = (x, y)
    tela.add(Color(0, 3, 0, 1))
    tela.add(Rectangle(pos=posizione, size=(larghezza, larghezza)))
    tela.add(Color(3, 3, 3, 1))
    tela.add(Ellipse(pos=posizione, size=(larghezza, larghezza)))
    tela.add(Color(0, 0, 0, 1))
    tela.add(Line(ellipse=(x + larghezza / 20, y + larghezza / 20, larghezza * 9 / 10, larghezza * 9 / 10)))
    tela.add(Ellipse(pos=(x + larghezza / 10, y + larghezza / 10), size=(larghezza * 8 / 10, larghezza * 8 / 10)))
    tela.add(Color(3, 3, 3, 1))
    tela.add(Line(ellipse=(x + larghezza * 3 / 20, y + larghezza * 3 / 20, larghezza * 7 / 10, larghezza * 7 / 10)))
    tela.add(Color(0, 0, 3, 1))
    tela.add(Ellipse(pos=(x + larghezza * 4 / 20, y + larghezza * 4 / 20),
                     size=(larghezza * 6 / 10, larghezza * 6 / 10)))
    tela.add(Color(0, 0, 0, 1))
    tela.add(Line(ellipse=(x + larghezza * 5 / 20, y + larghezza * 5 / 20, larghezza * 5 / 10, larghezza * 5 / 10)))
    tela.add(Color(3, 0, 0, 1))
    tela.add(Ellipse(pos=(x + larghezza * 6 / 20, y + larghezza * 6 / 20),
                     size=(larghezza * 4 / 10, larghezza * 4 / 10)))
    tela.add(Color(0, 0, 0, 1))
    tela.add(Line(ellipse=(x + larghezza * 7 / 20, y + larghezza * 7 / 20, larghezza * 3 / 10, larghezza * 3 / 10)))
    tela.add(Color(3, 3, 0, 1))
    tela.add(Ellipse(pos=(x + larghezza * 8 / 20, y + larghezza * 8 / 20),
                     size=(larghezza * 2 / 10, larghezza * 2 / 10)))
    tela.add(Color(0, 0, 0, 1))
    tela.add(Line(ellipse=(x + larghezza * 9 / 20, y + larghezza * 9 / 20, larghezza / 10, larghezza / 10)))


class Bersaglio(Widget):
    def __init__(self, pos_hint, size_hint, **kwargs):
        super(Bersaglio, self).__init__(**kwargs)
        self.pos = ((pos_hint["right"] - size_hint[0]) * Window.width,
                    (pos_hint["top"] - size_hint[1]) * Window.height)
        disegnabersaglio(self.canvas, None, (pos_hint["right"] - size_hint[0]) * Window.width,
                         (pos_hint["top"] - size_hint[1]) * Window.height, size_hint[0] * Window.width)


class CerchioMobile(Widget):
    def __init__(self, listacerchi, **kwargs):
        super().__init__(**kwargs)
        self.listacerchi = listacerchi
        self.size = (15, 15)  # Circle size
        with self.canvas:
            Color(1, 0, 1, 1)  # Blue circle
            self.circle = Ellipse(pos=self.pos, size=self.size)
        self.bind(pos=self.update_circle)  # Keep visual in sync

    def update_circle(self, *args):
        self.circle.pos = self.pos  # Update drawn position

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.touch_offset = (self.x - touch.x, self.y - touch.y)
            return True
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if hasattr(self, "touch_offset"):  # If we're dragging
            self.pos = (touch.x + self.touch_offset[0], touch.y + self.touch_offset[1])
            if self.x < .4 * Window.width:
                print("Distrutto")
                self.parent.remove_widget(self)
                self.listacerchi.remove(self)
            return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if hasattr(self, "touch_offset"):
            del self.touch_offset  # Stop dragging
        return super().on_touch_up(touch)


class BottoneMobile(Button):
    def __init__(self, dropdown, listacerchi, **kwargs):
        super(BottoneMobile, self).__init__(**kwargs)
        self.dropdown = dropdown
        self.listacerchi = listacerchi

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.touch_offset = (self.x - touch.x, self.y - touch.y)
            return True
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if hasattr(self, "touch_offset"):  # If we're dragging
            self.pos = (touch.x + self.touch_offset[0], touch.y + self.touch_offset[1])
            if self.x < .4 * Window.width:
                self.parent.remove_widget(self)
                self.listacerchi.remove(self)
            return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.collide_point(*touch.pos) and abs(touch.dx) < 5 and abs(touch.dy) < 5:
            self.dropdown.open(self)
            return True
        return super().on_touch_up(touch)


class DoppioDropDown(Widget):
    def __init__(self, bottoniprimodropdown, bottonisecondodropdown, **kwargs):
        super(DoppioDropDown, self).__init__(**kwargs)
        self.size = (30, 15)
        self.primodropdown = DropDown()
        self.secondodropdown = DropDown()
        for bottone in bottoniprimodropdown:
            bottone.bind(on_release=lambda btn=bottone, dd=self.primodropdown: dd.select(btn.text))
            self.primodropdown.add_widget(bottone)
        for bottone in bottonisecondodropdown:
            bottone.bind(on_release=lambda btn2=bottone, dd2=self.secondodropdown: dd2.select(btn2.text))
            self.secondodropdown.add_widget(bottone)
        self.bottoneprimodropdown = Button(text="Ord.")
        self.bottonesecondodropdown = Button(text="Frec.")
        self.bottoneprimodropdown.size = (15, 15)
        self.bottoneprimodropdown.pos = (self.pos[0] - 8, self.pos[1])
        self.bottonesecondodropdown.pos = (self.pos[0] + 8, self.pos[1])
        if self.parent:
            self.parent.add_widget(self.bottoneprimodropdown)
            self.parent.add_widget(self.bottonesecondodropdown)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.touch_offset = (self.x - touch.x, self.y - touch.y)
            return True
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if hasattr(self, "touch_offset"):  # If we're dragging
            self.pos = (touch.x + self.touch_offset[0], touch.y + self.touch_offset[1])
            if self.x < .4 * Window.width:
                self.bottoniprimodropdown.remove(self.bottoniprimodropdown[-1])
                self.bottonisecondodropdown.remove(self.bottonisecondodropdown[-1])
                self.parent.remove_widget(self)
                self.listacerchi.remove(self)
            return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.collide_point(*touch.pos) and abs(touch.dx) < 5 and abs(touch.dy) < 5:
            self.dropdown.open(self)
            return True
        return super().on_touch_up(touch)


# Devi fare in modo che le etichette delle frecce siano numeri
class BersaglioToccabile(Widget):
    # Classe di un oggetto che funge da bersaglio su cui si possono annotare le posizioni delle frecce,
    # per scrivere le pagine del diario degli allenamenti.
    def annullafreccia(self, instance):
        # Premendo il pulsante bianco, se è stata già contrassegnata almeno una freccia alla nuova volée,
        # l'ultima viene annullata. Se la volée attuale così diventa vuota, viene anche cambiata la scritta
        # sul pulsante bianco.
        if len(self.posizionicliccate) > 0:
            self.rimuovendo = self.listaistruzioni.pop(-1)
            self.canvas.remove(self.rimuovendo)
            self.posizionicliccate.remove(self.posizionicliccate[-1])
            if len(self.posizionicliccate) == 0:
                self.annullatore.text = "Annulla \n l'ultima \n volée"
        # Se invece la volée attuale è vuota, tutta la volée precedente viene cancellata per essere rifatta da capo.
        # Viene anche segnalato che la volée è stata annullata.
        else:
            if len(self.allenamentocorrente) > 0:
                self.allenamentocorrente.remove(self.allenamentocorrente[-1])
                self.etichettaerrore.text = "Volée annullata"

    def cambiavolée(self, instance):
        # Se c'è almeno una freccia nella volée, questa viene salvata nella lista self.allenamentocorrente.
        # Altrimenti viene mostrato l'errore "Volée vuota".
        if len(self.posizionicliccate) > 0:
            self.etichettaerrore.text = ""
            self.annullatore.text = "Annulla \n l'ultima \n volée"
            for istruzione in self.listaistruzioni:
                self.canvas.remove(istruzione)
            self.listaistruzioni = list()
            self.allenamentocorrente.append(self.posizionicliccate)
            self.posizionicliccate = list()
        else:
            self.etichettaerrore.text = "Volée vuota"

    def salvaallenamento(self, instance):
        # Il processo di salvataggio viene iniziato solo se è stata contrassegnata almeno una freccia.
        # Se così non è, è mostrato un errore.
        if len(self.allenamentocorrente) > 0 or len(self.posizionicliccate) > 0:
            # La data scelta per l'allenamento viene assemblata.
            self.datascelta = ' '.join([self.presinadata[0].text, self.presinadata[1].text, self.presinadata[2].text])
            # Se esiste già un file con questa data, viene mostrato un errore.
            if esiste(f"Sessione del {self.datascelta}"):
                self.etichettaerrore.text = "Questa data ha già un allenamento"
            # Altrimenti, la volée corrente viene salvata come ultima volée, e tutte le coordinate grezze
            # vengono standardizzate e convertite in oggetti Freccia. Tale lista di frecce, assieme alla data ed
            # eventuali tag, viene salvata in un file chiamato "Sessione del {data}".
            else:
                self.cambiavolée(instance=None)
                self.allenamentonormalizzato = list()
                # Devi rendere possibile salvare come insiemi anziché liste
                for volée in self.allenamentocorrente:
                    self.volée = list()
                    for freccia in volée:
                        self.volée.append((-(self.centro[0] - freccia[0]) / self.larghezza * 2,
                                           -(self.centro[1] - freccia[1]) / self.altezza * 2))
                    self.allenamentonormalizzato.append(self.volée)
                try:
                    dati = np.array(self.allenamentonormalizzato)
                except Exception:
                    dati = self.allenamentonormalizzato
                else:
                    dati = np.array(self.allenamentonormalizzato)
                allenamento = Allenamento(dati, self.ordinefrecce,
                                          self.identificazionefrecce, self.datascelta, leggifile("Impostazioni.txt"),
                                          self.arco, 25, self.tag) # Andrà resa modificabile
                creafile(f"Sessione del {self.datascelta}", allenamento)
                self.listadate = leggifile("Date.txt")
                self.listadate.append(self.datascelta)
                # Inoltre, la data scelta viene aggiunta alla lista nel file delle date.
                # NOTA BENE: VA IMPLEMENTATO UN ORDINAMENTO DELLE DATE.
                scrivifile(self.listadate, "Date.txt")
                statistichegenerali = StatisticheGenerali(self.listadate, leggifile("Impostazioni.txt"))
                scrivifile(statistichegenerali, "Statistichegenerali.txt")
                # A questo punto viene chiusa la schermata di scrittura degli allenamenti.
                self.funzioneritorno(instance=None)
        else:
            self.etichettaerrore.text = "Volée vuota"

    def __init__(self, annullatore, altezza, larghezza, cambiatore, salvatore, presinadata, funzioneritorno, presinatag,
                 centro, etichettaerrore, ordinefrecce, identificazionefrecce, arco, **kwargs):
        super().__init__(**kwargs)
        self.ordinefrecce = ordinefrecce
        self.identificazionefrecce = identificazionefrecce
        # Grandezze del bersaglio.
        self.larghezza = larghezza
        self.altezza = altezza
        # Pulsante bianco di annullamento delle frecce o delle volée.
        self.annullatore = annullatore
        # Pulsante per salvare.
        self.salvatore = salvatore
        # Oggetti che contengano la data dell'allenamento.
        self.presinadata = presinadata
        # Funzione per smettere di scrivere l'allenamento.
        self.funzioneritorno = funzioneritorno
        self.centro = centro
        # Campo in cui scrivere i tag.
        self.tag = presinatag.text
        self.arco = arco
        # Etichetta su cui vengono scritti gli errori, se ne vengono commessi durante la scrittura.
        self.etichettaerrore = etichettaerrore
        # Lista in cui sono salvate le frecce contrassegnate.
        self.posizionicliccate = []
        # Lista tecnica in cui sono conservate le istruzioni per bollare le frecce sul bersaglio a ogni clic.
        # Vengono usate durante l'annullamento per cancellare i bollini delle frecce sul bersaglio.
        self.listaistruzioni = list()
        # Lista in cui vengono passate le posizioni cliccate ogni volta che si inizia una nuova volée.
        self.allenamentocorrente = list()
        # Ai tre pulsanti passati come argomento vengono assegnate le funzioni definite sopra.
        annullatore.bind(on_press=self.annullafreccia)
        cambiatore.bind(on_press=self.cambiavolée)
        salvatore.bind(on_press=self.salvaallenamento)

    def on_touch_down(self, touch):
        # Se viene cliccato un punto all'interno del bersaglio, vengono attivate queste istruzioni:
        if self.collide_point(*touch.pos) and 0 < touch.pos[0] < self.larghezza and Window.height / 10 < touch.pos[
            1] < Window.height / 10 + self.altezza:
            # La scritta sul pulsante bianco viene cambiata e vengono tolti eventuali errori.
            self.annullatore.text = "Annulla \n l'ultima \n freccia"
            self.etichettaerrore.text = ""
            # Alle posizioni cliccate si aggiungono le coordinate del punto cliccato.
            self.posizionicliccate.append(touch.pos)
            # Viene aggiunta alla lista delle istruzioni un'istruzione per disegnare un pallino rosa sul punto
            # che è stato cliccato.
            if self.ordinefrecce and self.identificazionefrecce:
                self.istruzioni = list()
                self.istruzioni.append(DoppioDropDown())
            # Da continuare
            self.istruzioni = InstructionGroup()
            self.istruzioni.add(Color(3, 0, 3, 1))
            self.istruzioni.add(Ellipse(pos=touch.pos, size=(9, 9)))
            self.listaistruzioni.append(self.istruzioni)
            self.canvas.add(self.istruzioni)
            return True
        return super().on_touch_down(touch)


zerosenone = lambda numero: 0 if numero is None else numero


class SagittariumIuvo6App(App):
    def chiudi(self, instance):
        self.stop()

    def chiudiallenamento(self, instance):
        mettibottoneindietro(lambda: self.chiudi(instance=None))
        rimuovioggetti(self.cornice, self.bottoniallenamento)
        aggiungioggetti(self.cornice, self.bottoniprincipali)

    def tornaaallenamentodafotografia(self, instance):
        mettibottoneindietro(lambda: self.chiudiallenamento(instance=None))
        rimuovioggetti(self.cornice, self.bottonifotografia)
        aggiungioggetti(self.cornice, self.bottoniallenamento)

    def analizzafoto(self, filepath):
        pass

    def fotocamera(self, instance):
        pass

    def tornaafotografiadagalleria(self, instance):
        mettibottoneindietro(lambda: self.tornaaallenamentodafotografia(instance=None))
        rimuovioggetti(self.cornice, self.oggettigalleria)
        aggiungioggetti(self.cornice, self.bottonifotografia)

    def confermaordinefrecce(self, instance):
        self.confermatoreordinefrecce.background_color = (0, 0, 3, 1)
        self.negatoreordinefrecce.background_color = (.7, .7, .7, 1)
        self.negatoreordinefrecce.color = (1, 1, 1, 1)
        self.ordinefrecce = True

    def negaordinefrecce(self, instance):
        self.confermatoreordinefrecce.background_color = (.7, .7, .7, 1)
        self.negatoreordinefrecce.background_color = (2, 2, 0, 1)
        self.negatoreordinefrecce.color = (0, 0, 0, 1)
        self.ordinefrecce = False

    def confermaidentificazionefrecce(self, instance):
        self.confermatoreidentificazionefrecce.background_color = (0, 0, 3, 1)
        self.negatoreidentificazionefrecce.background_color = (.7, .7, .7, 1)
        self.negatoreidentificazionefrecce.color = (1, 1, 1, 1)
        self.identificazionefrecce = True

    def negaidentificazionefrecce(self, instance):
        self.confermatoreidentificazionefrecce.background_color = (.7, .7, .7, 1)
        self.negatoreidentificazionefrecce.background_color = (2, 2, 0, 1)
        self.negatoreidentificazionefrecce.color = (0, 0, 0, 1)
        self.identificazionefrecce = False

    def tornaafotografiadaimmaginescelta(self, instance):
        mettibottoneindietro(lambda: self.tornaaallenamentodafotografia(instance=None))
        rimuovioggetti(self.cornice, self.oggettiimmaginescelta)
        aggiungioggetti(self.cornice, self.bottonifotografia)

    def aggiungifrecciascelta(self, instance):
        if self.ordinefrecce and self.identificazionefrecce:
            for dropdown in self.cerchimobili:
                for bottone in self.bottoniprimodropdown:
                    dropdown.remove_widget(bottone)
                for bottone in self.bottonisecondodropdown:
                    dropdown.remove_widget(bottone)
            self.bottoniprimodropdown.append(Button(text=str(len(self.cerchimobili)), size_hint_y=None, size=(20, 10)))
            self.bottonisecondodropdown.append(
                Button(text=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"][len(self.cerchimobili)],
                       size_hint_y=None, size=(20, 15)))
            for dropdown in self.cerchimobili:
                for bottone in self.bottoniprimodropdown:
                    dropdown.add_widget(bottone)
                for bottone in self.bottonisecondodropdown:
                    dropdown.add_widget(bottone)
            self.nuovodropdown = DoppioDropDown(self.bottoniprimodropdown, self.bottonisecondodropdown)
            self.nuovodropdown.pos_hint = {"top": .8, "right": .8}
            self.cerchimobili.append(self.nuovodropdown)
            self.cornice.add_widget(self.nuovodropdown)
        elif self.ordinefrecce:
            for dropdown in self.cerchimobili:
                for bottone in self.bottonidropdown:
                    dropdown.remove_widget(bottone)
            self.bottonidropdown.append(Button(text=str(len(self.cerchimobili)), size_hint_y=None, size=(20, 10)))
            for dropdown in self.cerchimobili:
                for bottone in self.bottonidropdown:
                    dropdown.add_widget(bottone)
            self.nuovodropdown = DropDown()
            self.bottonedropdown = BottoneMobile(text="Scegli", size_hint=(None, None), size=(10, 10),
                                                 dropdown=self.nuovodropdown, listacerchi=self.cerchimobili)
            self.bottonedropdown.pos_hint = {"top": .8, "right": .8}
            self.nuovodropdown.bind(on_select=lambda instance, x, btn=self.bottonedropdown: setattr(btn, 'text', x))
            self.cerchimobili.append(self.bottonedropdown)
        elif self.identificazionefrecce:
            for dropdown in self.cerchimobili:
                for bottone in self.bottonidropdown:
                    dropdown.remove_widget(bottone)
            self.bottonidropdown.append(
                Button(text=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"][len(self.bottonidropdown)],
                       size_hint_y=None, size=(20, 10)))
            for dropdown in self.cerchimobili:
                for bottone in self.bottonidropdown:
                    dropdown.add_widget(bottone)
            self.nuovodropdown = DropDown()
            self.bottonedropdown = BottoneMobile(text="Scegli", size_hint=(None, None), size=(10, 10),
                                                 dropdown=self.nuovodropdown, listacerchi=self.cerchimobili)
            self.bottonedropdown.pos_hint = {"top": .8, "right": .8}
            self.nuovodropdown.bind(on_select=lambda instance, x, btn=self.bottonedropdown: setattr(btn, 'text', x))
            self.cerchimobili.append(self.bottonedropdown)
        else:
            self.nuovocerchio = CerchioMobile(pos=(.8 * Window.width, .8 * Window.height),
                                              listacerchi=self.cerchimobili)
            self.cerchimobili.append(self.nuovocerchio)
            self.cornice.add_widget(self.nuovocerchio)

    def nuovavoléescelta(self, instance):
        if self.ordinefrecce and self.identificazionefrecce:
            self.nuovavolée = [None for _ in range(len(self.cerchimobili))]
            for oggetto in self.cerchimobili:
                self.nuovavolée[int(oggetto.bottoneprimodropdown.text)] = ((
                    (oggetto.pos[0] - .4 * Window.width) / (.6 * Window.width),
                    (oggetto.pos[1] - .1 * Window.height) / (.6 * Window.width), oggetto.bottonesecondodropdown.text))
        elif self.ordinefrecce:
            self.nuovavolée = [None for _ in range(len(self.cerchimobili))]
            for oggetto in self.cerchimobili:
                self.nuovavolée[int(oggetto.text)] = (((oggetto.pos[0] - .4 * Window.width) / (.6 * Window.width),
                                                       (oggetto.pos[1] - .1 * Window.height) / (.6 * Window.width)))
        elif self.identificazionefrecce:
            self.nuovavolée = list()
            for oggetto in self.cerchimobili:
                self.nuovavolée.append(((oggetto.pos[0] - .4 * Window.width) / (.6 * Window.width),
                                        (oggetto.pos[1] - .1 * Window.height) / (.6 * Window.width), oggetto.text))
        else:
            self.nuovavolée = list()
            for oggetto in self.cerchimobili:
                self.nuovavolée.append(((oggetto.pos[0] - .4 * Window.width) / (.6 * Window.width),
                                        (oggetto.pos[1] - .1 * Window.height) / (.6 * Window.width)))
        self.allenamentogalleria.append(self.nuovavolée)
        filechooser.open_file(on_selection=self.immaginescelta)

    def tornaaprincipaledasalvato(self, instance):
        pickle.dump(Allenamento(np.array(self.allenamentogalleria), self.ordinefrecce, self.identificazionefrecce,
                                self.confermatoredata.text, self.attesa),
                    open(f"Sessione del {self.confermatoredata.text}", "xb"))
        self.vecchiedate = leggifile("Date.txt")
        self.vecchiedate.append(self.confermatoredata.text)
        scrivifile(self.vecchiedate, "Date.txt")
        scrivifile(StatisticheGenerali(), "Statistichegenerali.txt")
        self.build()

    def salvadavveroallenamentoscelto(self, instance):
        self.cornice.remove_widget(self.chiedidata)
        self.cornice.remove_widget(self.prendidata)
        self.cornice.remove_widget(self.confermatoredata)
        self.attesa = Label(text="Caricamento...")
        self.cornice.add_widget(self.attesa)
        Clock.schedule_once(self.tornaaprincipaledasalvato, 1)

    def salvaallenamentoscelto(self, instance):
        if self.ordinefrecce and self.identificazionefrecce:
            self.nuovavolée = [None for _ in range(len(self.cerchimobili))]
            for oggetto in self.cerchimobili:
                self.nuovavolée[int(oggetto.bottoneprimodropdown.text)] = ((
                    (oggetto.pos[0] - .4 * Window.width) / (.6 * Window.width),
                    (oggetto.pos[1] - .1 * Window.height) / (.6 * Window.width), oggetto.bottonesecondodropdown.text))
        elif self.ordinefrecce:
            self.nuovavolée = [None for _ in range(len(self.cerchimobili))]
            for oggetto in self.cerchimobili:
                self.nuovavolée[int(oggetto.text)] = (((oggetto.pos[0] - .4 * Window.width) / (.6 * Window.width),
                                                       (oggetto.pos[1] - .1 * Window.height) / (.6 * Window.width)))
        elif self.identificazionefrecce:
            self.nuovavolée = set()
            for oggetto in self.cerchimobili:
                self.nuovavolée.add(((oggetto.pos[0] - .4 * Window.width) / (.6 * Window.width),
                                     (oggetto.pos[1] - .1 * Window.height) / (.6 * Window.width), oggetto.text))
        else:
            self.nuovavolée = set()
            for oggetto in self.cerchimobili:
                self.nuovavolée.add(((oggetto.pos[0] - .4 * Window.width) / (.6 * Window.width),
                                     (oggetto.pos[1] - .1 * Window.height) / (.6 * Window.width)))
        self.allenamentogalleria.append(self.nuovavolée)
        rimuovioggetti(self.cornice, self.oggettiimmaginescelta)
        self.chiedidata = Label(text="Data dell'allenamento:")
        self.chiedidata.pos_hint = {"top": .6, "right": .5}
        self.chiedidata.size_hint = (.3, .2)
        self.prendidata = TextInput()
        self.prendidata.pos_hint = {"top": .6, "right": .9}
        self.prendidata.size_hint = (.4, .2)
        self.confermatoredata = Button(text="Conferma")
        self.confermatoredata.pos_hint = {"top": .4, "right": .7}
        self.confermatoredata.size_hint = (.4, .2)
        self.confermatoredata.bind(on_press=self.salvadavveroallenamentoscelto)
        self.cornice.add_widget(self.chiedidata)
        self.cornice.add_widget(self.prendidata)
        self.cornice.add_widget(self.confermatoredata)

    def immaginescelta(self, selection):
        self.percorsoimmagine = selection[0]
        self.chiusoreimmaginescelta = Button(text="Torna indietro")
        self.chiusoreimmaginescelta.size_hint = (1, .1)
        self.chiusoreimmaginescelta.pos_hint = {"top": 1, "right": 1}
        self.chiusoreimmaginescelta.background_color = (3, 0, 0, 1)
        self.bersaglioimmaginescelta = Bersaglio({"top": .7, "right": 1}, (.6, .6))
        if len(self.allenamentogalleria) == 0:
            self.chiusoreimmaginescelta.bind(on_press=self.tornaafotografiadaimmaginescelta)
        self.immagine = cv2.cvtColor(cv2.imread(self.percorsoimmagine), cv2.COLOR_BGR2RGB)
        self.cerchimobili = list()
        self.previsioni = self.modello.predict(self.percorsoimmagine).predictions
        self.bersagliato = False
        for previsione in self.previsioni:
            if previsione["class"] == "target":
                for previsione2 in self.previsioni:
                    if previsione2["class"] != "target" and previsione["x"] - previsione["width"] / 2 < previsione2[
                        "x"] < previsione["x"] + previsione["width"] / 2 and previsione["y"] - previsione[
                        "height"] / 2 < previsione2["y"] < previsione["y"] + previsione["height"] / 2:
                        cv2.rectangle(self.immagine, (int(previsione["x"] - previsione["width"] / 2),
                                                      int(previsione["y"] - previsione["height"] / 2)),
                                      (int(previsione["x"] + previsione["width"] / 2),
                                       int(previsione["y"] + previsione["height"] / 2)),
                                      (0, 0, 10, 1), thickness=4)
                        self.posizionebersaglio = (
                            previsione["x"] - previsione["width"] / 2, previsione["y"] - previsione["height"] / 2,
                            previsione["width"], previsione["height"])
                        self.bersagliato = True
                        break
            if self.bersagliato:
                break
            else:
                if self.bersagliato:
                    cv2.circle(self.immagine, (int(previsione["x"]), int(previsione["y"])), 8, (0, 10, 0, 1),
                               thickness=4)
                    self.posizionecerchiomobile = (self.bersaglioimmaginescelta.pos[0] + .6 * Window.width * (
                            previsione["x"] - self.posizionebersaglio[0]) / (self.posizionebersaglio[2]),
                                                   self.bersaglioimmaginescelta.pos[1] + .6 * Window.width * (
                                                           previsione["y"] - self.posizionebersaglio[1]) / (
                                                       self.posizionebersaglio[3]))
                    if self.ordinefrecce and self.identificazionefrecce:
                        self.bottoniprimodropdown = list()
                        self.bottonisecondodropdown = list()
                        for testo in [str(i + 1) for i in range(len(self.previsioni))]:
                            bottone = Button(text=testo, size_hint_y=None, width=10, height=20)
                            self.bottoniprimodropdown.append(bottone)
                        for testo in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"][0:len(self.previsioni) - 1]:
                            bottone = Button(text=testo, size_hint_y=None, width=10, height=20)
                            self.bottonisecondodropdown.append(bottone)
                        dropdown = DoppioDropDown(self.bottoniprimodropdown, self.bottonisecondodropdown)
                        self.cerchimobili.append(dropdown)
                    elif self.ordinefrecce:
                        dropdown = DropDown()
                        self.bottonidropdown = list()
                        for testo in [str(i + 1) for i in range(len(self.previsioni))]:
                            bottone = Button(text=testo, size_hint_y=None, width=10, height=20)
                            bottone.bind(on_release=lambda btn=bottone, dd=dropdown: dd.select(btn.text))
                            dropdown.add_widget(bottone)
                            self.bottonidropdown.append(bottone)
                        self.bottonedropdown = BottoneMobile(text="Scegli", size_hint=(None, None), size=(10, 10),
                                                             dropdown=dropdown)
                        self.bottonedropdown.pos = self.posizionecerchiomobile
                        dropdown.bind(on_select=lambda instance, x, btn=self.bottonedropdown: setattr(btn, 'text', x))
                        self.cerchimobili.append(self.bottonedropdown)
                    elif self.identificazionefrecce:
                        dropdown = DropDown()
                        self.bottonidropdown = list()
                        for lettera in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"][
                                       0:len(self.previsioni) - 1]:
                            bottone = Button(text=lettera, size_hint_y=None, height=20, width=10)
                            bottone.bind(on_release=lambda btn=bottone, dd=dropdown: dd.select(btn.text))
                            dropdown.add_widget(bottone)
                            self.bottonidropdown.append(bottone)
                        self.bottonedropdown = BottoneMobile(text="Scegli", size_hint=(None, None), size=(10, 10),
                                                             dropdown=dropdown)
                        self.bottonedropdown.pos = self.posizionecerchiomobile
                        dropdown.bind(on_select=lambda instance, x, btn=self.bottonedropdown: setattr(btn, 'text', x))
                        self.cerchimobili.append(self.bottonedropdown)
                    else:
                        self.cerchiocorrente = CerchioMobile(pos=self.posizionecerchiomobile,
                                                             listacerchi=self.cerchimobili)
                        self.cerchimobili.append(self.cerchiocorrente)
        self.immagine = cv2.flip(self.immagine, 0)
        self.texture = Texture.create(size=(self.immagine.shape[1], self.immagine.shape[0]), colorfmt="rgb")
        self.texture.blit_buffer(self.immagine.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
        self.immagineesposta = Image()
        self.immagineesposta.texture = self.texture
        self.immagineesposta.size_hint = (.4, .8)
        self.immagineesposta.pos_hint = {"top": .9, "right": .4}
        self.cornice.remove_widget(self.attesa)
        self.aggiuntorefreccia = Button(text="Aggiungi freccia")
        self.aggiuntorefreccia.background_color = (0, 3, 0, 1)
        self.aggiuntorefreccia.color = (1, 1, 1, 1)
        self.aggiuntorefreccia.size_hint = (1 / 3, .1)
        self.aggiuntorefreccia.pos_hint = {"top": .1, "right": 1 / 3}
        self.aggiuntorefreccia.bind(on_press=self.aggiungifrecciascelta)
        self.innovatorevolée = Button(text="Nuova volée")
        self.innovatorevolée.background_color = (1, 1, 1, 1)
        self.innovatorevolée.color = (0, 0, 0, 1)
        self.innovatorevolée.size_hint = (1 / 3, .1)
        self.innovatorevolée.pos_hint = {"top": .1, "right": 2 / 3}
        self.innovatorevolée.bind(on_press=self.nuovavoléescelta)
        self.salvatoreallenamento = Button(text="Salva allenamento")
        self.salvatoreallenamento.background_color = (0, 0, 1, 1)
        self.salvatoreallenamento.color = (1, 1, 1, 1)
        self.salvatoreallenamento.size_hint = (1 / 3, .1)
        self.salvatoreallenamento.pos_hint = {"top": .1, "right": 1}
        self.salvatoreallenamento.bind(on_press=self.salvaallenamentoscelto)
        self.oggettiimmaginescelta = (self.chiusoreimmaginescelta, self.immagineesposta, self.bersaglioimmaginescelta,
                                      *self.cerchimobili, self.aggiuntorefreccia, self.innovatorevolée,
                                      self.salvatoreallenamento)
        aggiungioggetti(self.cornice, self.oggettiimmaginescelta)

    def scegliimmagine(self):
        mettibottoneindietro(lambda: self.tornaafotografiadaimmaginescelta(instance=None))
        rimuovioggetti(self.cornice, self.oggettigalleria)
        self.attesa = Label(text="Caricamento...")
        self.cornice.add_widget(self.attesa)
        filechooser.open_file(on_selection=self.immaginescelta)

    def aprigalleria(self, instance):
        self.allenamentogalleria = list()
        try:
            self.rf = Roboflow(api_key="bsVEDcXpIrAmmUdnvBJv")
        except ConnectionError:
            self.erroreconnessione = Label(text="Manca la connessione")
            with self.erroreconnessione.canvas.before:
                Color(3, 0, 0, 1)
                Rectangle(pos=(Window.width / 2 - 80, Window.height / 2 - 40), size=(160, 80))
            self.cornice.add_widget(self.erroreconnessione)
            Clock.schedule_once(lambda dt: self.cornice.remove_widget(self.erroreconnessione), 4)
        else:
            self.progetto = self.rf.workspace("archery-zrbei").project("target-and-arrow-detection")
            self.versione = self.progetto.version(6)
            self.modello = self.versione.model
            self.scegliimmagine()

    def galleria(self, instance):
        mettibottoneindietro(lambda: self.tornaafotografiadagalleria(instance=None))
        rimuovioggetti(self.cornice, self.bottonifotografia)
        self.ordinefrecce = False
        self.identificazionefrecce = False
        self.chiusoregalleria = Button(text="Torna indietro")
        self.chiusoregalleria.size_hint = (1, .125)
        self.chiusoregalleria.pos_hint = {"top": 1, "right": 1}
        self.chiusoregalleria.bind(on_press=self.tornaafotografiadagalleria)
        self.chiusoregalleria.background_color = (3, 0, 0, 1)
        self.chiusoregalleria.color = (1, 1, 1, 1)
        self.chiediordinefrecce = Label(text="Vuoi indicare l'ordine in cui\nsono state tirate le frecce?")
        self.chiediordinefrecce.size_hint = (.5, .25)
        self.chiediordinefrecce.pos_hint = {"top": .875, "right": .5}
        self.chiediidentificazionefrecce = Label(text="Vuoi identificare con una lettera\nogni freccia specifica?")
        self.chiediidentificazionefrecce.size_hint = (.5, .25)
        self.chiediidentificazionefrecce.pos_hint = {"top": .625, "right": .5}
        self.confermatoreordinefrecce = Button(text="Sì")
        self.confermatoreordinefrecce.size_hint = (.25, .25)
        self.confermatoreordinefrecce.pos_hint = {"top": .875, "right": .75}
        self.confermatoreordinefrecce.bind(on_press=self.confermaordinefrecce)
        self.negatoreordinefrecce = Button(text="No")
        self.negatoreordinefrecce.size_hint = (.25, .25)
        self.negatoreordinefrecce.pos_hint = {"top": .875, "right": 1}
        self.negatoreordinefrecce.background_color = (2, 2, 0, 1)
        self.negatoreordinefrecce.color = (0, 0, 0, 1)
        self.negatoreordinefrecce.bind(on_press=self.negaordinefrecce)
        self.confermatoreidentificazionefrecce = Button(text="Sì")
        self.confermatoreidentificazionefrecce.size_hint = (.25, .25)
        self.confermatoreidentificazionefrecce.pos_hint = {"top": .625, "right": .75}
        self.confermatoreidentificazionefrecce.bind(on_press=self.confermaidentificazionefrecce)
        self.negatoreidentificazionefrecce = Button(text="No")
        self.negatoreidentificazionefrecce.size_hint = (.25, .25)
        self.negatoreidentificazionefrecce.pos_hint = {"top": .625, "right": 1}
        self.negatoreidentificazionefrecce.background_color = (2, 2, 0, 1)
        self.negatoreidentificazionefrecce.color = (0, 0, 0, 1)
        self.negatoreidentificazionefrecce.bind(on_press=self.negaidentificazionefrecce)
        self.apertoregalleria = Button(text="Continua (scegli una sola immagine per volta)")
        self.apertoregalleria.size_hint = (1, .125)
        self.apertoregalleria.pos_hint = {"top": .125, "right": 1}
        self.apertoregalleria.background_color = (0, 0, 3, 1)
        self.apertoregalleria.bind(on_press=self.aprigalleria)
        self.oggettigalleria = (self.chiusoregalleria, self.chiediordinefrecce, self.chiediidentificazionefrecce,
                                self.confermatoreordinefrecce, self.negatoreordinefrecce,
                                self.confermatoreidentificazionefrecce, self.negatoreidentificazionefrecce,
                                self.apertoregalleria)
        aggiungioggetti(self.cornice, self.oggettigalleria)

    def fotografaallenamento(self, instance):
        mettibottoneindietro(lambda: self.tornaaallenamentodafotografia(instance=None))
        rimuovioggetti(self.cornice, self.bottoniallenamento)
        self.chiusorefotografia = Button(text="Torna indietro")
        self.fotocameratore = Button(text="Usa la fotocamera")
        self.galleriatore = Button(text="Scegli dalla galleria")
        self.bottonifotografia = (self.chiusorefotografia, self.fotocameratore, self.galleriatore)
        self.sfondifotografia = ((3, 0, 0, 1), (2, 1, 0, 1), (1, 2, 0, 1))
        self.funzionifotografia = (self.tornaaallenamentodafotografia, self.fotocamera, self.galleria)
        for bottone in range(3):
            self.bottonifotografia[bottone].size_hint = (1, 1 / 3)
            self.bottonifotografia[bottone].pos_hint = {"top": 1 - .3 * bottone, "right": 1}
            self.bottonifotografia[bottone].background_color = self.sfondifotografia[bottone]
            self.bottonifotografia[bottone].color = (1, 1, 1, 1)
            self.bottonifotografia[bottone].bind(on_press=self.funzionifotografia[bottone])
        aggiungioggetti(self.cornice, self.bottonifotografia)

    def tornaaallenamentodadisegno(self, instance):
        rimuovioggetti(self.cornice, self.oggettidisegno)
        aggiungioggetti(self.cornice, self.bottoniallenamento)

    def select_button(self, nome):
        self.bottoneselezionatorearco.text = nome
        self.selezionatorearco.dismiss()

    def disegnaallenamento(self, instance):
        rimuovioggetti(self.cornice, self.bottoniallenamento)
        self.ordinefrecce = False
        self.identificazionefrecce = False
        self.chiusoregalleria = Button(text="Torna indietro")
        self.chiusoregalleria.size_hint = (1, .125)
        self.chiusoregalleria.pos_hint = {"top": 1, "right": 1}
        self.chiusoregalleria.bind(on_press=self.tornaafotografiadagalleria)
        self.chiusoregalleria.background_color = (3, 0, 0, 1)
        self.chiusoregalleria.color = (1, 1, 1, 1)
        self.chiediordinefrecce = Label(text="Vuoi indicare l'ordine in cui\nsono state tirate le frecce?")
        self.chiediordinefrecce.size_hint = (.5, .25)
        self.chiediordinefrecce.pos_hint = {"top": .875, "right": .5}
        self.chiediidentificazionefrecce = Label(text="Vuoi identificare con una lettera\nogni freccia specifica?")
        self.chiediidentificazionefrecce.size_hint = (.5, .25)
        self.chiediidentificazionefrecce.pos_hint = {"top": .625, "right": .5}
        self.confermatoreordinefrecce = Button(text="Sì")
        self.confermatoreordinefrecce.size_hint = (.25, .25)
        self.confermatoreordinefrecce.pos_hint = {"top": .875, "right": .75}
        self.confermatoreordinefrecce.bind(on_press=self.confermaordinefrecce)
        self.negatoreordinefrecce = Button(text="No")
        self.negatoreordinefrecce.size_hint = (.25, .25)
        self.negatoreordinefrecce.pos_hint = {"top": .875, "right": 1}
        self.negatoreordinefrecce.background_color = (2, 2, 0, 1)
        self.negatoreordinefrecce.color = (0, 0, 0, 1)
        self.negatoreordinefrecce.bind(on_press=self.negaordinefrecce)
        self.confermatoreidentificazionefrecce = Button(text="Sì")
        self.confermatoreidentificazionefrecce.size_hint = (.25, .25)
        self.confermatoreidentificazionefrecce.pos_hint = {"top": .625, "right": .75}
        self.confermatoreidentificazionefrecce.bind(on_press=self.confermaidentificazionefrecce)
        self.negatoreidentificazionefrecce = Button(text="No")
        self.negatoreidentificazionefrecce.size_hint = (.25, .25)
        self.negatoreidentificazionefrecce.pos_hint = {"top": .625, "right": 1}
        self.negatoreidentificazionefrecce.background_color = (2, 2, 0, 1)
        self.negatoreidentificazionefrecce.color = (0, 0, 0, 1)
        self.negatoreidentificazionefrecce.bind(on_press=self.negaidentificazionefrecce)
        self.chiediselezionearco = Label(text="Scegli un arco:")
        self.chiediselezionearco.size_hint = (.8, .1)
        self.chiediselezionearco.pos_hint = {"top": .4, "right": .9}
        self.selezionatorearco = DropDown()
        self.bottoneselezionatorearco = Button(text="Scegli")
        for setup in leggifile("ImpostazioniArco.txt"):
            nome = setup.nomesetup
            bottone = Button(text=str(nome), size_hint_y=None, height=44)
            bottone.bind(on_release=lambda btn, nome=nome: self.select_button(nome))
            self.selezionatorearco.add_widget(bottone)
        self.bottoneselezionatorearco.size_hint = (.6, .1)
        self.bottoneselezionatorearco.pos_hint = {"top": .3, "right": .7}
        self.bottoneselezionatorearco.bind(on_release=lambda btn: self.selezionatorearco.open(btn))
        self.apertoregalleria = Button(text="Continua (scegli una sola immagine per volta)")
        self.apertoregalleria.size_hint = (1, .125)
        self.apertoregalleria.pos_hint = {"top": .125, "right": 1}
        self.apertoregalleria.background_color = (0, 0, 3, 1)
        self.apertoregalleria.bind(on_press=self.disegnadavveroallenamento)
        self.oggettigalleria = (self.chiusoregalleria, self.chiediordinefrecce, self.chiediidentificazionefrecce,
                                self.confermatoreordinefrecce, self.negatoreordinefrecce,
                                self.confermatoreidentificazionefrecce, self.negatoreidentificazionefrecce,
                                self.chiediselezionearco, self.bottoneselezionatorearco, self.apertoregalleria)
        aggiungioggetti(self.cornice, self.oggettigalleria)

    def disegnadavveroallenamento(self, instance):
        # Crea la schermata per disegnare l'allenamento.
        # La schermata di selezione di modalità di scrittura viene rimossa.
        for bottone in self.oggettigalleria:
            self.cornice.remove_widget(bottone)
        # Crea un bottone per tornare indietro direttamente, senza salvare.
        self.chiusoredisegno = Button(text="Torna indietro senza salvare")
        self.chiusoredisegno.background_color = (3, 0, 0, 1)
        self.chiusoredisegno.size_hint, self.chiusoredisegno.pos_hint = (.8, .1), {"top": 1, "right": .9}
        self.chiusoredisegno.bind(on_press=self.tornaaallenamentodadisegno)
        self.cornice.add_widget(self.chiusoredisegno)
        # Crea un bottone bianco che, passato attraverso l'oggetto BersaglioToccabile, permette di annullare
        # le frecce contrassegnate.
        self.annullatorefreccia = Button(text="Annulla \nl'ultima\n freccia")
        self.annullatorefreccia.background_color = (3, 3, 3, 1)
        self.annullatorefreccia.color = (0, 0, 0, 1)
        self.annullatorefreccia.size_hint, self.annullatorefreccia.pos_hint = (.2, .1), {"top": .1, "right": .6}
        self.cornice.add_widget(self.annullatorefreccia)
        # Menù a tendina per selezionare l'anno dell'allenamento. Parte da dieci anni prima di adesso fino all'anno
        # successivo.
        self.selezioneanno = DropDown()
        for anno in range(int(datetime.datetime.now().year) - 10, int(datetime.datetime.now().year) + 1):
            bottone = Button(text=str(anno), size_hint=(None, None), height=20)
            bottone.font_size = 8
            bottone.bind(on_release=lambda bottone: self.selezioneanno.select(bottone.text))
            self.selezioneanno.add_widget(bottone)
        self.annoselezionato = Button(text=str(datetime.datetime.now().year))
        self.cornice.add_widget(self.annoselezionato)
        self.annoselezionato.bind(on_press=self.selezioneanno.open)
        self.annoselezionato.size_hint, self.annoselezionato.pos_hint = (.15, .1), {"top": .9, "right": .15}
        self.selezioneanno.bind(on_select=lambda instance, x: setattr(self.annoselezionato, "text", x))
        # Menù a tendina per selezionare il mese.
        self.selezionemese = DropDown()
        self.mesi = [None, "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto",
                     "settembre",
                     "ottobre", "novembre", "dicembre"]
        for mese in self.mesi:
            if mese is not None:
                bottone = Button(text=mese, size_hint=(None, None), height=20)
                bottone.font_size = 8
                bottone.bind(on_release=lambda bottone: self.selezionemese.select(bottone.text))
                self.selezionemese.add_widget(bottone)
        self.meseselezionato = Button(text=self.mesi[int(datetime.datetime.now().month)])
        self.cornice.add_widget(self.meseselezionato)
        self.meseselezionato.bind(on_press=self.selezionemese.open)
        self.meseselezionato.size_hint, self.meseselezionato.pos_hint = (.2, .1), {"top": .9, "right": .35}
        self.selezionemese.bind(on_select=lambda instance, x: setattr(self.meseselezionato, "text", x))
        # Menù a tendina per selezionare il giorno.
        self.selezionegiorno = DropDown()
        self.giorni = {"gennaio": 31, "marzo": 31, "aprile": 30, "maggio": 31, "giugno": 30, "luglio": 31,
                       "agosto": 31, "settembre": 30, "ottobre": 31, "novembre": 30, "dicembre": 31}
        if int(self.annoselezionato.text) % 4 == 0:
            self.giorni["febbraio"] = 29
        else:
            self.giorni["febbraio"] = 28
        for giorno in range(1, self.giorni[self.meseselezionato.text] + 1):
            bottone = Button(text=str(giorno), size_hint=(None, None), height=15)
            bottone.font_size = 8
            bottone.bind(on_release=lambda bottone: self.selezionegiorno.select(bottone.text))
            self.selezionegiorno.add_widget(bottone)
        self.giornoselezionato = Button(text=str(datetime.datetime.now().day))
        self.cornice.add_widget(self.giornoselezionato)
        self.giornoselezionato.bind(on_release=self.selezionegiorno.open)
        self.giornoselezionato.size_hint, self.giornoselezionato.pos_hint = (.15, .1), {"top": .9, "right": .5}
        self.selezionegiorno.bind(on_select=lambda instance, x: setattr(self.giornoselezionato, "text", x))
        # Bottone verde che passato attraverso l'oggetto BersaglioToccabile permette di passare
        # alla volée successiva da contrassegnare.
        self.cambiatorevolée = Button(text="Nuova volée")
        self.cambiatorevolée.background_color = (0, 3, 0, 1)
        self.cambiatorevolée.size_hint, self.cambiatorevolée.pos_hint = (.4, .1), {"top": .1, "right": 1}
        self.cornice.add_widget(self.cambiatorevolée)
        # Bottone blu che passato attraverso l'oggetto BersaglioToccabile permette di
        # salvare l'allenamento.
        self.salvatoreallenamento = Button(text="Salva allenamento")
        self.salvatoreallenamento.background_color = (0, 0, 3, 1)
        self.salvatoreallenamento.size_hint, self.salvatoreallenamento.pos_hint = (.4, .1), {"top": .1, "right": .4}
        self.cornice.add_widget(self.salvatoreallenamento)
        # Campo in cui scrivere i tag, con annessa un'etichetta che dà le istruzioni per metterli.
        self.richiestatag = Label(text="Scrivi i tag separati da \n una virgola, senza spazio dopo:")
        self.richiestatag.size_hint, self.richiestatag.pos_hint = (.4, .1), {"top": .8, "right": .5}
        self.cornice.add_widget(self.richiestatag)
        self.presatag = TextInput(multiline=True)
        self.presatag.size_hint, self.presatag.pos_hint = (.4, .1), {"top": .8, "right": .9}
        self.cornice.add_widget(self.presatag)
        # Etichetta che passata attraverso l'oggetto BersaglioToccabile diventa in grado di mostrare
        # eventuali errori commessi durante la scrittura dell'allenamento.
        self.etichettaerrore = Label(text="")
        self.etichettaerrore.size_hint, self.etichettaerrore.pos_hint = (.5, .1), {"top": .9, "right": 1}
        self.cornice.add_widget(self.etichettaerrore)
        # Se lo schermo è molto largo, il bersaglio è mostrato in verticale a sinistra;
        # altrimenti, viene messo in cima alla pagina e fatto più stretto.
        # Le istruzioni seguenti aggiustano la sua larghezza.
        if Window.width > Window.height * 6 / 10:
            self.larghezza = Window.height * 6 / 10
        else:
            self.larghezza = Window.width
        # noinspection PyTypeChecker
        self.bersaglio = BersaglioToccabile(self.annullatorefreccia, self.larghezza, self.larghezza,
                                            self.cambiatorevolée,
                                            self.salvatoreallenamento, [self.giornoselezionato,
                                                                        self.meseselezionato,
                                                                        self.annoselezionato],
                                            self.tornaaallenamentodadisegno, self.presatag,
                                            (self.larghezza / 2, Window.height / 10 + self.larghezza / 2),
                                            self.etichettaerrore, self.ordinefrecce, self.identificazionefrecce,
                                            self.bottoneselezionatorearco.text)
        disegnabersaglio(self.bersaglio.canvas, posizione=None, x=0, y=Window.height / 10, larghezza=self.larghezza)
        self.cornice.add_widget(self.bersaglio)
        # Tutti gli oggetti creati qui vengono messi in questa lista per essere rimossi al bisogno.
        self.oggettidisegno = [self.chiusoredisegno, self.annullatorefreccia, self.cambiatorevolée,
                               self.salvatoreallenamento, self.bersaglio, self.richiestatag, self.presatag,
                               self.annoselezionato, self.meseselezionato, self.giornoselezionato, self.etichettaerrore]

    def allena(self, instance):
        mettibottoneindietro(lambda: self.chiudiallenamento(instance=None))
        rimuovioggetti(self.cornice, self.bottoniprincipali)
        self.chiusoreallenamento = Button(text="Torna al menù principale")
        self.fotografatoreallenamento = Button(text="Utilizza una foto")
        self.disegnatoreallenamento = Button(text="Disegna a mano")
        self.bottoniallenamento = (self.chiusoreallenamento, self.fotografatoreallenamento, self.disegnatoreallenamento)
        self.sfondiallenamento = ((3, 0, 0, 1), (0, 2, 1, 1), (0, 1, 2, 1))
        self.funzioniallenamento = (self.chiudiallenamento, self.fotografaallenamento, self.disegnaallenamento)
        for bottone in range(3):
            self.bottoniallenamento[bottone].size_hint = (1, 1 / 3)
            self.bottoniallenamento[bottone].pos_hint = {"top": 1 - .3 * bottone, "right": 1}
            self.bottoniallenamento[bottone].background_color = self.sfondiallenamento[bottone]
            self.bottoniallenamento[bottone].color = (1, 1, 1, 1)
            self.bottoniallenamento[bottone].bind(on_press=self.funzioniallenamento[bottone])
        aggiungioggetti(self.cornice, self.bottoniallenamento)

    def tornaaprincipaledadiario(self, instance):
        self.cornice.remove_widget(self.chiusorediario)
        self.cornice.remove_widget(self.grigliadate)
        aggiungioggetti(self.cornice, self.bottoniprincipali)

    def paginadiario(self, instance, data):
        self.allenamentopagina = leggifile(f"Sessione del {data}")
        self.cornice.remove_widget(self.chiusorediario)
        self.cornice.remove_widget(self.scorsoredate)
        self.chiusorepagina = Button(text="Torna al diario")
        self.chiusorepagina.size_hint = (1, .1)
        self.chiusorepagina.pos_hint = {"top": 1, "right": 1}
        self.chiusorepagina.background_color = (3, 0, 0, 1)
        self.graficodispersione = Image(source=f"./Grafici del {data}/graficodispersione.png")
        self.graficodispersione.size_hint = (.5, .5)
        self.graficodispersione.pos_hint = {"top": .7, "right": .5}
        self.scorsorestatistiche = ScrollView()
        self.scorsorestatistiche.size_hint = (.5, .9)
        self.scorsorestatistiche.pos_hint = {"top": .9, "right": 1}
        self.righegrigliastatistiche = 6
        self.altezzagrigliastatistiche = 2 * Window.width + 20
        if self.allenamentopagina.mediapunteggi is not None:
            self.righegrigliastatistiche += 1
            self.altezzagrigliastatistiche += 40
        if self.allenamentopagina.mediapunteggivolée is not None:
            self.righegrigliastatistiche += 2
            self.altezzagrigliastatistiche += 40 * (len(self.allenamentopagina.mediapunteggivolée) + 1)
        if self.allenamentopagina.mediapunteggifrecce is not None:
            self.righegrigliastatistiche += 2
            self.altezzagrigliastatistiche += 40 * (len(self.allenamentopagina.mediapunteggifrecce) + 1)
        if (
                self.allenamentopagina.mediacoordinate is not None or self.allenamentopagina.mediacoordinatevolée is not None
                or self.allenamentopagina.mediacoordinatefrecce is not None or self.allenamentopagina.regressionecoordinate is not None):
            self.righegrigliastatistiche += 2
            self.altezzagrigliastatistiche += Window.width / 2 + 40
        if self.allenamentopagina.mediaangoli is not None:
            self.righegrigliastatistiche += 1
            self.altezzagrigliastatistiche += 40
        if self.allenamentopagina.mediaangolivolée is not None:
            self.righegrigliastatistiche += 2
            self.altezzagrigliastatistiche += 40 * (1 + len(self.allenamentopagina.mediaangolivolée))
        if self.allenamentopagina.varianzapunteggi is not None:
            self.righegrigliastatistiche += 1
            self.altezzagrigliastatistiche += 40
        if self.allenamentopagina.varianzapunteggivolée is not None:
            self.righegrigliastatistiche += 2
            self.altezzagrigliastatistiche += 40 * (len(self.allenamentopagina.varianzapunteggivolée) + 1)
        if self.allenamentopagina.varianzapunteggifrecce is not None:
            self.righegrigliastatistiche += 2
            self.altezzagrigliastatistiche += 40 * (len(self.allenamentopagina.varianzapunteggifrecce) + 1)
        if self.allenamentopagina.varianzacoordinate is not None:
            self.righegrigliastatistiche += 1
            self.altezzagrigliastatistiche += 40
        if self.allenamentopagina.varianzacoordinatevolée is not None:
            self.righegrigliastatistiche += 2
            self.altezzagrigliastatistiche += 40 * (len(self.allenamentopagina.varianzacoordinatevolée) + 1)
        if self.allenamentopagina.varianzacoordinatefrecce is not None:
            self.righegrigliastatistiche += 2
            self.altezzagrigliastatistiche += 40 * (len(self.allenamentopagina.varianzacoordinatefrecce) + 1)
        if self.allenamentopagina.varianzaangoli is not None:
            self.righegrigliastatistiche += 1
            self.altezzagrigliastatistiche += 40
        if self.allenamentopagina.varianzaangolivolée is not None:
            self.righegrigliastatistiche += 2
            self.altezzagrigliastatistiche += 40 * (len(self.allenamentopagina.varianzaangolivolée) + 1)
        if self.allenamentopagina.varianzaangolifrecce is not None:
            self.righegrigliastatistiche += 2
            self.altezzagrigliastatistiche += 40 * (len(self.allenamentopagina.varianzaangolifrecce) + 1)
        if self.allenamentopagina.correlazione is not None:
            self.righegrigliastatistiche += 1
            self.altezzagrigliastatistiche += 40
        if self.allenamentopagina.correlazionevolée is not None:
            self.righegrigliastatistiche += 2
            self.altezzagrigliastatistiche += 40 * (len(self.allenamentopagina.correlazionevolée) + 1)
        if self.allenamentopagina.autocorrelazionepunteggi is not None:
            self.righegrigliastatistiche += 1
            self.altezzagrigliastatistiche += Window.width / 2
        if self.allenamentopagina.autocorrelazioneascisse is not None:
            self.righegrigliastatistiche += 1
            self.altezzagrigliastatistiche += Window.width / 2
        if self.allenamentopagina.autocorrelazioneordinate is not None:
            self.righegrigliastatistiche += 1
            self.altezzagrigliastatistiche += Window.width / 2
        if self.allenamentopagina.autocorrelazioniangolipositive is not None:
            self.righegrigliastatistiche += 1
            self.altezzagrigliastatistiche += Window.width / 2
        if self.allenamentopagina.affidabilitàhotelling is not None:
            if not self.allenamentopagina.affidabilitàhotelling:
                self.righegrigliastatistiche += 1
                self.altezzagrigliastatistiche += 40
        if self.allenamentopagina.ljungboxascisse is not None:
            self.righegrigliastatistiche += 1
            self.altezzagrigliastatistiche += 40
        if self.allenamentopagina.ljungboxordinate is not None:
            self.righegrigliastatistiche += 1
            self.altezzagrigliastatistiche += 40
        if self.allenamentopagina.ljungboxascisse is not None or self.allenamentopagina.ljungboxordinate is not None:
            self.righegrigliastatistiche += 1
            self.altezzagrigliastatistiche += 40
        if self.allenamentopagina.mardia is not None:
            self.righegrigliastatistiche += 2
            self.altezzagrigliastatistiche += 80
            if not self.allenamentopagina.mardia:
                self.righegrigliastatistiche += 1
                self.altezzagrigliastatistiche += 40 + Window.width / 2
        if self.allenamentopagina.testhotelling is not None:
            self.righegrigliastatistiche += 2
            self.altezzagrigliastatistiche += 80
            if not self.allenamentopagina.mardia:
                self.righegrigliastatistiche += 2 + len(self.allenamentopagina.testcluster)
                self.altezzagrigliastatistiche += 40 * (2 + len(self.allenamentopagina.testcluster))
        if self.allenamentopagina.intervallohotelling is not None:
            self.righegrigliastatistiche += 2
            self.altezzagrigliastatistiche += Window.width / 2 + 40
        if self.allenamentopagina.hotellingbayesiano is not None:
            self.righegrigliastatistiche += 1
            self.altezzagrigliastatistiche += 40
        if self.allenamentopagina.intervallobayesiano is not None:
            self.righegrigliastatistiche += 1
            self.altezzagrigliastatistiche += 40
        if self.allenamentopagina.medianageometrica is not None:
            self.righegrigliastatistiche += 1
            self.altezzagrigliastatistiche += Window.width / 2
        if self.allenamentopagina.normenormali is not None:
            self.righegrigliastatistiche += 1
            self.altezzagrigliastatistiche += 40
        if self.allenamentopagina.normeindipendenti is not None:
            self.righegrigliastatistiche += 2
            self.altezzagrigliastatistiche += 80
        if self.allenamentopagina.intervallonorme is not None:
            self.righegrigliastatistiche += 1
            self.altezzagrigliastatistiche += 40
        if self.allenamentopagina.testnorme is not None:
            self.righegrigliastatistiche += 2
            self.altezzagrigliastatistiche += 80
        if self.allenamentopagina.normebayesiane is not None:
            self.righegrigliastatistiche += 1
            self.altezzagrigliastatistiche += 40
        if self.allenamentopagina.intervallonormebayesiano is not None:
            self.righegrigliastatistiche += 1
            self.altezzagrigliastatistiche += 40
        if self.allenamentopagina.intervallivarianze is not None:
            self.righegrigliastatistiche += 2
            self.altezzagrigliastatistiche += 80
        if self.allenamentopagina.testvarianze is not None:
            self.righegrigliastatistiche += 3
            self.altezzagrigliastatistiche += 120
        if self.allenamentopagina.varianzebayesiane is not None:
            self.righegrigliastatistiche += 2
            self.altezzagrigliastatistiche += 80
        if self.allenamentopagina.intervallovarianzebayesiano is not None:
            self.righegrigliastatistiche += 2
            self.altezzagrigliastatistiche += 80
        if self.allenamentopagina.difettosità is not None:
            self.righegrigliastatistiche += 2
            self.altezzagrigliastatistiche += 40 * (1 + len(self.allenamentopagina.difettosità))
        if self.allenamentopagina.difettositàvarianze is not None:
            self.righegrigliastatistiche += 2
            self.altezzagrigliastatistiche += 40 * (1 + len(self.allenamentopagina.difettositàvarianze))
        if self.allenamentopagina.uniformitàangoli is not None:
            self.righegrigliastatistiche += 2
            self.altezzagrigliastatistiche += 80
        if self.allenamentopagina.kappa is not None:
            self.righegrigliastatistiche += 1
            self.altezzagrigliastatistiche += 40
        if self.allenamentopagina.affidabilitàvonmises is not None:
            self.righegrigliastatistiche += 1
            self.altezzagrigliastatistiche += 40
        if self.allenamentopagina.intervalloangolomedio is not None:
            self.righegrigliastatistiche += 1
            self.altezzagrigliastatistiche += 40
        if self.allenamentopagina.intervallokappa is not None:
            self.righegrigliastatistiche += 1
            self.altezzagrigliastatistiche += 40
        # Mancano un paio di cose
        self.grigliastatistiche = GridLayout(rows=self.righegrigliastatistiche, cols=1)
        self.grigliastatistiche.size_hint = (1, None)
        self.grigliastatistiche.size = (Window.width / 2, self.altezzagrigliastatistiche)
        self.scorsorestatistiche.add_widget(self.grigliastatistiche)
        self.oggettipagina = (self.chiusorepagina, self.graficodispersione, self.scorsorestatistiche)
        if self.allenamentopagina.mediapunteggi is not None:
            self.grigliamediapunteggi = GridLayout(rows=1, cols=3)
            self.grigliamediapunteggi.size = (Window.width / 2, 20)
            self.grigliamediapunteggi.add_widget(Label(text="Punteggio medio:"))
            self.grigliamediapunteggi.add_widget(Label(text=str(round(self.allenamentopagina.mediapunteggi, 3))))
            self.grigliamediapunteggi.add_widget(
                Label(text="Tempo: " + str(round(self.allenamentopagina.tempomediapunteggi, 3))))
            self.grigliastatistiche.add_widget(self.grigliamediapunteggi)
        if self.allenamentopagina.mediapunteggivolée is not None:
            self.grigliastatistiche.add_widget(Label(text="Punteggio medio di ogni volée", size=(Window.width / 2, 20)))
            self.grigliamediapunteggivoléegrande = GridLayout(rows=1, cols=2)
            self.grigliamediapunteggivolée = GridLayout(rows=len(self.allenamentopagina.mediapunteggivolée), cols=2)
            self.grigliamediapunteggivolée.size = (
                Window.width / 2, 20 * len(self.allenamentopagina.mediapunteggivolée))
            for indice, media in enumerate(self.allenamentopagina.mediapunteggivolée):
                self.grigliamediapunteggivolée.add_widget(Label(text=str(indice + 1) + ":"))
                self.grigliamediapunteggivolée.add_widget(Label(text=str(round(media, 3))))
            self.grigliamediapunteggivoléegrande.add_widget(self.grigliamediapunteggivolée)
            self.grigliamediapunteggivoléegrande.add_widget(
                Label(text="Tempo: " + str(round(self.allenamentopagina.tempomediapunteggivolée, 3))))
            self.grigliastatistiche.add_widget(self.grigliamediapunteggivoléegrande)
        if self.allenamentopagina.mediapunteggifrecce is not None:
            self.grigliastatistiche.add_widget(
                Label(text="Punteggio medio di ogni freccia", size=(Window.width / 2, 20)))
            self.grigliamediapunteggifreccegrande = GridLayout(rows=1, cols=2)
            self.grigliamediapunteggifrecce = GridLayout(rows=len(self.allenamentopagina.mediapunteggifrecce), cols=2)
            self.grigliamediapunteggifrecce.size = (
                Window.width / 2, 20 * len(self.allenamentopagina.mediapunteggifrecce))
            for etichetta in self.allenamentopagina.mediapunteggifrecce:
                self.grigliamediapunteggifrecce.add_widget(Label(text=str(etichetta + 1) + ":"))
                self.grigliamediapunteggifrecce.add_widget(
                    Label(text=str(round(self.allenamentopagina.mediapunteggifrecce[etichetta], 3))))
            self.grigliamediapunteggifreccegrande.add_widget(self.grigliamediapunteggifrecce)
            self.grigliamediapunteggifreccegrande.add_widget(
                Label(text="Tempo: " + str(round(self.allenamentopagina.tempomediapunteggifrecce, 3))))
        if (
                self.allenamentopagina.mediacoordinate is not None or self.allenamentopagina.mediacoordinatevolée is not None
                or self.allenamentopagina.mediacoordinatefrecce is not None or self.allenamentopagina.regressionecoordinate is not None):
            tempograficomedie = str(round(zerosenone(self.allenamentopagina.tempomediacoordinate) +
                                          zerosenone(self.allenamentopagina.tempomediacoordinatevolée) +
                                          zerosenone(self.allenamentopagina.tempomediacoordinatefrecce) +
                                          zerosenone(self.allenamentopagina.temporegressionecoordinate) +
                                          zerosenone(self.allenamentopagina.tempograficomedie), 3))
            self.grigliastatistiche.add_widget(Label(text=f"Grafico delle medie (tempo: {tempograficomedie}):"))
            self.grigliastatistiche.add_widget(
                Image(source=f"./Grafici del {data}/graficomedie.png", size=(Window.width / 2, Window.width / 2),
                      size_hint=(None, None)))
        if self.allenamentopagina.mediaangoli is not None:
            self.grigliastatistiche.add_widget(Label(
                text=f"Angolo medio: {str(round(180 + math.degrees(self.allenamentopagina.mediaangoli), 3))}, tempo di calcolo: {str(round(self.allenamentopagina.tempomediaangoli, 3))}"))
        if self.allenamentopagina.mediaangolivolée is not None:
            self.grigliastatistiche.add_widget(Label(
                text=f"Angoli medi per ogni volée (tempo di calcolo: {str(round(self.allenamentopagina.tempomediaangolivolée, 3))}):"))
            self.grigliamediaangolivolée = GridLayout(rows=len(self.allenamentopagina.mediaangolivolée), cols=2)
            for etichetta, media in enumerate(self.allenamentopagina.mediaangolivolée):
                self.grigliamediaangolivolée.add_widget(Label(text=str(etichetta + 1) + ":"))
                self.grigliamediaangolivolée.add_widget(Label(text=str(round(180 + math.degrees(media), 3))))
            self.grigliastatistiche.add_widget(self.grigliamediaangolivolée)
        self.grigliastatistiche.add_widget(
            Image(source=f"./Grafici del {data}/graficoangoli.png", size=(Window.width / 2, Window.width / 2),
                  size_hint=(None, None)))
        if self.allenamentopagina.mediaangolifrecce is not None:
            self.grigliastatistiche.add_widget(Label(
                text=f"Angoli medi per ogni freccia (tempo di calcolo: {str(round(self.allenamentopagina.tempomediaangolifrecce, 3))}):"))
            self.grigliamediaangolifrecce = GridLayout(rows=len(self.allenamentopagina.mediaangolifrecce), cols=2)
            for etichetta in self.allenamentopagina.mediaangolivolée:
                self.grigliamediaangolifrecce.add_widget(Label(text=str(etichetta + 1) + ":"))
                self.grigliamediaangolifrecce.add_widget(
                    Label(text=str(round(180 + math.degrees(self.allenamentopagina.mediaangolivolée[etichetta]), 3))))
            self.grigliastatistiche.add_widget(self.grigliamediaangolifrecce)
        if self.allenamentopagina.varianzapunteggi is not None:
            self.grigliastatistiche.add_widget(Label(
                text=f"Varianza dei punteggi (tempo di calcolo: {str(round(self.allenamentopagina.tempovarianzapunteggi, 3))}): {str(round(self.allenamentopagina.varianzapunteggi, 3))}"))
        if self.allenamentopagina.varianzapunteggivolée is not None:
            self.grigliastatistiche.add_widget(Label(
                text=f"Varianza dei punteggi per ogni volée (tempo di calcolo: {str(round(self.allenamentopagina.tempovarianzapunteggivolée, 3))}):"))
            self.grigliavarianzapunteggivolée = GridLayout(rows=len(self.allenamentopagina.varianzapunteggivolée),
                                                           cols=2)
            for etichetta, varianza in enumerate(self.allenamentopagina.varianzapunteggivolée):
                self.grigliavarianzapunteggivolée.add_widget(Label(text=str(etichetta + 1) + ":"))
                self.grigliavarianzapunteggivolée.add_widget(Label(text=str(round(varianza, 3))))
            self.grigliastatistiche.add_widget(self.grigliavarianzapunteggivolée)
        if self.allenamentopagina.varianzapunteggifrecce is not None:
            self.grigliastatistiche.add_widget(Label(
                text=f"Varianza dei punteggi per ogni freccia (tempo di calcolo: {str(round(self.allenamentopagina.tempovarianzapunteggifrecce, 3))}):"))
            self.grigliavarianzapunteggifrecce = GridLayout(rows=len(self.allenamentopagina.varianzapunteggifrecce),
                                                            cols=2)
            for etichetta in self.allenamentopagina.varianzapunteggifrecce:
                self.grigliavarianzapunteggifrecce.add_widget(Label(text=str(etichetta + 1) + ":"))
                self.grigliavarianzapunteggifrecce.add_widget(
                    Label(text=str(round(self.allenamentopagina.varianzapunteggifrecce[etichetta], 3))))
            self.grigliastatistiche.add_widget(self.grigliavarianzapunteggifrecce)
        if self.allenamentopagina.varianzacoordinate is not None:
            self.grigliastatistiche.add_widget(Label(
                text=f"Varianza delle coordinate (tempo di calcolo: {str(round(self.allenamentopagina.tempovarianzacoordinate, 3))}): {(round(self.allenamentopagina.varianzacoordinate[0], 3), round(self.allenamentopagina.varianzacoordinate[1], 3))}"))
        if self.allenamentopagina.varianzacoordinatevolée is not None:
            self.grigliastatistiche.add_widget(Label(
                text=f"Varianza delle coordinate per ogni volée (tempo di calcolo: {str(round(self.allenamentopagina.tempovarianzacoordinatevolée, 3))}):"))
            self.grigliavarianzacoordinatevolée = GridLayout(rows=len(self.allenamentopagina.varianzacoordinatevolée),
                                                             cols=2)
            for etichetta, varianza in enumerate(self.allenamentopagina.varianzacoordinatevolée):
                self.grigliavarianzacoordinatevolée.add_widget(Label(text=str(etichetta + 1) + ":"))
                self.grigliavarianzacoordinatevolée.add_widget(
                    Label(text=str((round(varianza[0], 3), round(varianza[1], 3)))))
            self.grigliastatistiche.add_widget(self.grigliavarianzacoordinatevolée)
        if self.allenamentopagina.varianzacoordinatefrecce is not None:
            self.grigliastatistiche.add_widget(Label(
                text=f"Varianza delle coordinate per ogni freccia (tempo di calcolo: {str(round(self.allenamentopagina.tempovarianzacoordinatefrecce, 3))}):"))
            self.grigliavarianzacoordinatefrecce = GridLayout(rows=len(self.allenamentopagina.varianzacoordinatefrecce),
                                                              cols=2)
            for etichetta, varianza in enumerate(self.allenamentopagina.varianzacoordinatevolée):
                self.grigliavarianzacoordinatefrecce.add_widget(Label(text=str(etichetta + 1) + ":"))
                self.grigliavarianzacoordinatefrecce.add_widget(
                    Label(text=str((round(varianza[0], 3), round(varianza[1], 3)))))
            self.grigliastatistiche.add_widget(self.grigliavarianzacoordinatefrecce)
        if self.allenamentopagina.varianzaangoli is not None:
            self.grigliastatistiche.add_widget(Label(
                text=f"Lunghezza risultante media (tempo di calcolo: {str(round(self.allenamentopagina.tempovarianzaangoli, 3))}): {round(self.allenamentopagina.varianzaangoli, 3)}"))
        if self.allenamentopagina.varianzaangolivolée is not None:
            self.grigliastatistiche.add_widget(Label(
                text=f"Lunghezza risultante media per ogni volée (tempo di calcolo: {str(round(self.allenamentopagina.tempovarianzaangolivolée, 3))}):"))
            self.grigliavarianzaangolivolée = GridLayout(rows=len(self.allenamentopagina.varianzaangolivolée), cols=2)
            for etichetta, varianza in enumerate(self.allenamentopagina.varianzaangolivolée):
                self.grigliavarianzaangolivolée.add_widget(Label(text=str(etichetta + 1) + ":"))
                self.grigliavarianzaangolivolée.add_widget(Label(text=str(round(varianza, 3))))
            self.grigliastatistiche.add_widget(self.grigliavarianzaangolivolée)
        if self.allenamentopagina.varianzaangolifrecce is not None:
            self.grigliastatistiche.add_widget(Label(
                text=f"Lunghezza risultante media per ogni freccia (tempo di calcolo: {str(round(self.allenamentopagina.tempovarianzaangolifrecce, 3))}):"))
            self.grigliavarianzaangolifrecce = GridLayout(rows=len(self.allenamentopagina.varianzaangolifrecce), cols=2)
            for etichetta, varianza in enumerate(self.allenamentopagina.varianzaangolifrecce):
                self.grigliavarianzaangolifrecce.add_widget(Label(text=str(etichetta + 1) + ":"))
                self.grigliavarianzaangolifrecce.add_widget(Label(text=str(round(varianza, 3))))
            self.grigliastatistiche.add_widget(self.grigliavarianzaangolifrecce)
        if self.allenamentopagina.correlazione is not None:
            self.grigliastatistiche.add_widget(Label(
                text=f"Correlazione di ascisse e ordinate (tempo di calcolo: {round(self.allenamentopagina.tempocorrelazione, 3)}): {str(round(self.allenamentopagina.correlazione, 3))}"))
        if self.allenamentopagina.correlazionevolée is not None:
            self.grigliastatistiche.add_widget(Label(
                text=f"Correlazione di ascisse e ordinate per ogni volée (tempo di calcolo: {round(self.allenamentopagina.tempocorrelazionevolée, 3)}):"))
            self.grigliacorrelazionevolée = GridLayout(rows=len(self.allenamentopagina.correlazionevolée), cols=2)
            for etichetta, correlazione in enumerate(self.allenamentopagina.correlazionevolée):
                self.grigliacorrelazionevolée.add_widget(Label(text=str(etichetta + 1) + ":"))
                self.grigliacorrelazionevolée.add_widget(Label(text=str(round(correlazione, 3))))
            self.grigliastatistiche.add_widget(self.grigliacorrelazionevolée)
        self.grigliastatistiche.add_widget(
            Image(source=f"./Grafici del {data}/graficopunteggi.png", size=(Window.width / 2, Window.width / 2),
                  size_hint=(None, None)))
        self.grigliastatistiche.add_widget(
            Image(source=f"./Grafici del {data}/graficovolée.png", size=(Window.width / 2, Window.width / 2),
                  size_hint=(None, None)))
        if self.allenamentopagina.autocorrelazionepunteggi is not None:
            self.grigliastatistiche.add_widget(Image(source=f"./Grafici del {data}/graficoautocorrelazionipunteggi.png",
                                                     size=(Window.width / 2, Window.width / 2), size_hint=(None, None)))
        if self.allenamentopagina.autocorrelazioneascisse is not None:
            self.grigliastatistiche.add_widget(Image(source=f"./Grafici del {data}/graficoautocorrelazioniascisse.png",
                                                     size=(Window.width / 2, Window.width / 2), size_hint=(None, None)))
        if self.allenamentopagina.autocorrelazioneordinate is not None:
            self.grigliastatistiche.add_widget(Image(source=f"./Grafici del {data}/graficoautocorrelazioniordinate.png",
                                                     size=(Window.width / 2, Window.width / 2), size_hint=(None, None)))
        if self.allenamentopagina.autocorrelazioniangolipositive is not None:
            self.grigliastatistiche.add_widget(
                Image(source=f"./Grafici del {data}/graficoautocorrelazioniangolipositive.png",
                      size=(Window.width / 2, Window.width / 2), size_hint=(None, None)))
        if self.allenamentopagina.autocorrelazioniangolinegative is not None:
            self.grigliastatistiche.add_widget(
                Image(source=f"./Grafici del {data}/graficoautocorrelazioniangolinegative.png",
                      size=(Window.width / 2, Window.width / 2), size_hint=(None, None)))
        self.grigliastatistiche.add_widget(Label(
            text=f"Probabilità che tutte le affermazioni di seguito siano corrette: {100 * round(self.allenamentopagina.probabilitàtuttogiusto, 3)}%"))
        if self.allenamentopagina.probabilitàtuttogiusto == 1:
            self.grigliastatistiche.add_widget(Label(text="Non è stata fatta nessuna affermazione significativa."))
        else:
            if self.allenamentopagina.affidabilitàhotelling is not None:
                if not self.allenamentopagina.affidabilitàhotelling:
                    self.grigliastatistiche.add_widget(Label(text="ATTENZIONE: il test di Hotelling non è affidabile"))
            if self.allenamentopagina.ljungboxascisse is not None:
                if self.allenamentopagina.ljungboxascisse:
                    self.grigliastatistiche.add_widget(Label(
                        text=f"Non c'è dipendenza nelle ascisse (probabilità di errore: {str(round(self.allenamentopagina.betaljungboxascisse, 3))}, tempo di calcolo: {str(round(self.allenamentopagina.tempobetaljungboxascisse, 3))})"))
                else:
                    self.grigliastatistiche.add_widget(Label(
                        text=f"C'È DIPENDENZA NELLE ASCISSE (probabilità di errore: {self.allenamentopagina.impostazioni['alfaljungbox'] if self.allenamentopagina.alfaveroljungboxascisse is None else round(self.allenamentopagina.alfaveroljungboxascisse, 3)})"))
            if self.allenamentopagina.ljungboxordinate is not None:
                if self.allenamentopagina.ljungboxordinate:
                    self.grigliastatistiche.add_widget(Label(
                        text=f"Non c'è dipendenza nelle ordinate (probabilità di errore: {str(round(self.allenamentopagina.betaljungboxordinate, 3))}, tempo di calcolo: {str(round(self.allenamentopagina.tempobetaljungboxordinate, 3))})"))
                else:
                    self.grigliastatistiche.add_widget(Label(
                        text=f"C'È DIPENDENZA NELLE ORDINATE (probabilità di errore: {self.allenamentopagina.impostazioni['alfaljungbox'] if self.allenamentopagina.alfaveroljungboxordinate is None else round(self.allenamentopagina.alfaveroljungboxordinate, 3)})"))
            if self.allenamentopagina.ljungboxordinate is not None or self.allenamentopagina.ljungboxascisse is not None:
                self.grigliastatistiche.add_widget(Label(
                    text=f"Tempo per eseguire il test di Ljung-Box: {str(round(self.allenamentopagina.tempoljungbox, 3))}"))
            if self.allenamentopagina.mardia is not None:
                if self.allenamentopagina.mardia:
                    self.grigliastatistiche.add_widget(Label(
                        text=f"Viene accettata l'ipotesi di normalità (probabilità di errore: {str(round(self.allenamentopagina.betamardia, 3))}, tempo di calcolo: {str(round(self.allenamentopagina.tempobetamardia, 3))})"))
                else:
                    self.grigliastatistiche.add_widget(Label(
                        text=f"Il test di normalità fallisce (probabilità di errore: {self.allenamentopagina.impostazioni['alfamardia'] if self.allenamentopagina.alfaveromardia is None else round(self.allenamentopagina.alfaveromardia, 3)})"))
                self.grigliastatistiche.add_widget(Label(
                    text=f"Tempo per eseguire il test di normalità: {str(round(self.allenamentopagina.tempomardia, 3))}"))
                if not self.allenamentopagina.mardia:
                    self.grigliastatistiche.add_widget(Image(source=f"./Grafici del {data}/graficocluster.png"))
            if self.allenamentopagina.testhotelling is not None:
                if self.allenamentopagina.testhotelling:
                    self.grigliastatistiche.add_widget(Label(
                        text=f"Il test di Hotelling accetta l'ipotesi nulla (probabilità di errore: {str(round(self.allenamentopagina.betatesthotelling, 3))}, tempo di calcolo: {str(round(self.allenamentopagina.tempobetatesthotelling, 3))})"))
                else:
                    self.grigliastatistiche.add_widget(Label(
                        text=f"Il test di Hotelling rifiuta l'ipotesi nulla (probabilità di errore: {str(self.allenamentopagina.impostazioni['alfahotelling'])})"))
                self.grigliastatistiche.add_widget(Label(
                    text=f"Tempo per eseguire il test di Hotelling: {str(round(self.allenamentopagina.tempotesthotelling, 3))}"))
                if not self.allenamentopagina.mardia:
                    self.grigliastatistiche.add_widget(Label(
                        text=f"Test di Hotelling per i cluster (probabilità di errore: {str(round(1 - self.allenamentopagina.probabilitàtestcluster, 3))}, tempo di calcolo: {str(round(self.allenamentopagina.tempoprobabiliitàtestcluster, 3))})"))
                    self.grigliatestcluster = GridLayout(rows=len(self.allenamentopagina.testcluster), cols=2)
                    for cluster, test in enumerate(self.allenamentopagina.testcluster):
                        self.grigliatestcluster.add_widget(Label(text=str(cluster + 1)))
                        self.grigliatestcluster.add_widget(Label(text="Accettata" if test else "Rifiutata"))
                    self.grigliastatistiche.add_widget(self.grigliatestcluster)
                    self.grigliastatistiche.add_widget(Label(
                        text=f"Tempo per eseguire i test sui cluster: {str(round(self.allenamentopagina.tempotestcluster, 3))}"))
            if self.allenamentopagina.intervallohotelling is not None:
                self.grigliastatistiche.add_widget(Label(
                    text=f"Intervallo di confidenza bivariato (probabilità di errore: {str(round(1 - self.allenamentopagina.impostazioni['confidenzahotelling'], 3))},\n tempo di calcolo: {str(round(self.allenamentopagina.tempointervallohotelling, 3))}):"))
                self.grigliastatistiche.add_widget(Image(source=f"./Grafici del {data}/graficointervallohotelling.png",
                                                         size=(Window.width / 2, Window.width / 2),
                                                         size_hint=(None, None)))
            # Aggiungi il grafico per gli intervalli dei cluster
            if self.allenamentopagina.hotellingbayesiano is not None:
                self.grigliastatistiche.add_widget(Label(
                    text=f"Probabilità dell'ipotesi nulla: {str(round(self.allenamentopagina.hotellingbayesiano, 3))} (tempo di calcolo: {str(round(self.allenamentopagina.tempohotellingbayesiano, 3))})"))
            if self.allenamentopagina.intervallobayesiano is not None:
                self.grigliastatistiche.add_widget(Image(source=f"./Grafici del {data}/graficohotellingbayesiano.png",
                                                         size=(Window.width / 2, Window.width / 2),
                                                         size_hint=(None, None)))
            if self.allenamentopagina.medianageometrica is not None:
                self.grigliastatistiche.add_widget(Image(source=f"./Grafici del {data}/graficomedianageometrica.png",
                                                         size=(Window.width / 2, Window.width / 2),
                                                         size_hint=(None, None)))
            if self.allenamentopagina.normenormali is not None:
                if self.allenamentopagina.normenormali:
                    self.grigliastatistiche.add_widget(Label(
                        text=f"I punteggi reali si distribuiscono come una normale (tempo di calcolo: {str(round(self.allenamentopagina.temponormenormali, 3))})"))
                else:
                    # Non hai la probabilità di II tipo del test di Shapiro
                    self.grigliastatistiche.add_widget(Label(
                        text=f"I punteggi reali NON si distribuiscono come una normale (probabilità di errore: {self.allenamentopagina.impostazioni['alfashapiro']}, tempo di calcolo: {str(round(self.allenamentopagina.temponormenormali, 3))})"))
            if self.allenamentopagina.normeindipendenti is not None:
                if self.allenamentopagina.normeindipendenti:
                    self.grigliastatistiche.add_widget(Label(
                        text=f"I punteggi reali sono indipendenti (probabilità di errore: {str(round(self.allenamentopagina.betanormenormali, 3))}, tempo di calcolo: {str(round(self.allenamentopagina.tempobetanormenormali, 3))})"))
                else:
                    # Ricorda che il test di Ljung-Box è asintotico
                    self.grigliastatistiche.add_widget(Label(
                        text=f"I punteggi reali NON sono indipendenti (probabilità di errore: {self.allenamentopagina.impostazioni['alfaljungbox']}"))
                self.grigliastatistiche.add_widget(Label(
                    text=f"Tempo per l'analisi delle autocorrelazioni: {str(round(self.allenamentopagina.temponormeindipendenti, 3))}"))
            if self.allenamentopagina.intervallonorme is not None:
                self.grigliastatistiche.add_widget(Label(
                    text=f"Intervallo di confidenza al {str(self.allenamentopagina.impostazioni['confidenzanorme'])}% dei punteggi reali: [{str(round(self.allenamentopagina.intervallonorme[0], 3))}, {str(round(self.allenamentopagina.intervallonorme[1], 3))}]"))
            if self.allenamentopagina.testnorme is not None:
                if self.allenamentopagina.testnorme:
                    self.grigliastatistiche.add_widget(Label(
                        text=f"Il test sulle norme accetta l'ipotesi nulla (probabilità di errore: {str(round(self.allenamentopagina.betatestnorme, 3))}, tempo di calcolo: {str(round(self.allenamentopagina.tempobetatestnorme, 3))})"))
                else:
                    self.grigliastatistiche.add_widget(Label(
                        text=f"Il test sulle norme rifiuta l'ipotesi nulla (probabilità di errore: {self.allenamentopagina.impostazioni['alfatestnorme']})"))
                self.grigliastatistiche.add_widget(Label(
                    text=f"Tempo per eseguire il test sulle norme: {str(round(self.allenamentopagina.tempotestnorme, 3))}"))
            if self.allenamentopagina.normebayesiane is not None:
                self.grigliastatistiche.add_widget(Label(
                    text=f"La probabilità che l'ipotesi nulla sia vera è {str(round(self.allenamentopagina.normebayesiane, 3))}%"))
            if self.allenamentopagina.intervallonormebayesiano is not None:
                self.grigliastatistiche.add_widget(Label(
                    text=f"Intervallo di credibilità al {str(self.allenamentopagina.impostazioni['credibilitànormebayesiane'])}% dei punteggi reali bayesiano: [{str(round(self.allenamentopagina.intervallonormebayesiano[0], 3))}, {str(round(self.allenamentopagina.intervallonormebayesiano[1], 3))}]"))
            if self.allenamentopagina.intervallivarianze is not None:
                self.grigliastatistiche.add_widget(Label(
                    text=f"Intervallo di confidenza al {str(self.allenamentopagina.impostazioni['alfavarianze'])}% della varianza delle ascisse: [{str(round(self.allenamentopagina.intervallivarianze[0], 3))}, {str(round(self.allenamentopagina.intervallivarianze[1], 3))}]"))
                self.grigliastatistiche.add_widget(Label(
                    text=f"Intervallo di confidenza al {str(self.allenamentopagina.impostazioni['alfavarianze'])}% della varianza delle ordinate: [{str(round(self.allenamentopagina.intervallivarianze[2], 3))}, {str(round(self.allenamentopagina.intervallivarianze[3], 3))}]"))
            if self.allenamentopagina.testvarianze is not None:
                if self.allenamentopagina.testvarianze[0]:
                    self.grigliastatistiche.add_widget(Label(
                        text=f"Il test per la varianza delle ascisse accetta l'ipotesi nulla \n(probabilità di errore: {str(round(self.allenamentopagina.betatestvarianze, 3))}, tempo di calcolo: {str(round(self.allenamentopagina.tempobetatestvarianze, 3))})"))
                else:
                    self.grigliastatistiche.add_widget(Label(
                        text=f"Il test per la varianza delle ascisse rifiuta l'ipotesi nulla \n(probabilità di errore: {self.allenamentopagina.impostazioni['alfavarianze']})"))
                if self.allenamentopagina.testvarianze[1]:
                    self.grigliastatistiche.add_widget(Label(
                        text=f"Il test per la varianza delle ordinate accetta l'ipotesi nulla \n(probabilità di errore: {str(round(self.allenamentopagina.betatestvarianze, 3))}, tempo di calcolo: {str(round(self.allenamentopagina.tempobetatestvarianze, 3))})"))
                else:
                    self.grigliastatistiche.add_widget(Label(
                        text=f"Il test per la varianza delle ordinate rifiuta l'ipotesi nulla \n(probabilità di errore: {self.allenamentopagina.impostazioni['alfavarianze']})"))
                self.grigliastatistiche.add_widget(Label(
                    text=f"Tempo per il test sulle varianze: {str(round(self.allenamentopagina.tempotestvarianze, 3))}"))
            if self.allenamentopagina.varianzebayesiane is not None:
                self.grigliastatistiche.add_widget(Label(
                    text=f"La probabilità che l'ipotesi nulla sia vera per la varianza delle ascisse è {str(round(self.allenamentopagina.varianzebayesiane[0], 3))}"))
                self.grigliastatistiche.add_widget(Label(
                    text=f"La probabilità che l'ipotesi nulla sia vera per la varianza delle ordinate è {str(round(self.allenamentopagina.varianzebayesiane[1], 3))}"))
            if self.allenamentopagina.intervallovarianzebayesiano is not None:
                self.grigliastatistiche.add_widget(Label(
                    text=f"L'intervallo di credibilità al {str(self.allenamentopagina.impostazioni['credibilitàvarianzebayesiane'])}% per la varianza delle ascisse è [{str(round(self.allenamentopagina.intervallovarianzebayesiano[0][0], 3))}, {str(round(self.allenamentopagina.intervallovarianzebayesiano[0][1], 3))}]"))
                self.grigliastatistiche.add_widget(Label(
                    text=f"L'intervallo di credibilità al {str(self.allenamentopagina.impostazioni['credibilitàvarianzebayesiane'])}% per la varianza delle ordinate è [{str(round(self.allenamentopagina.intervallovarianzebayesiano[1][0], 3))}, {str(round(self.allenamentopagina.intervallovarianzebayesiano[1][1], 3))}]"))
            if self.allenamentopagina.difettosità is not None:
                self.grigliastatistiche.add_widget(Label(
                    text=f"Difettosità delle frecce per posizione (tempo di calcolo: {str(round(self.allenamentopagina.tempodifettosità, 3))}):"))
                self.grigliadifettosità = GridLayout(rows=len(self.allenamentopagina.difettosità), cols=2)
                for etichetta, difettosità in enumerate(self.allenamentopagina.difettosità):
                    self.grigliadifettosità.add_widget(Label(text=str(etichetta + 1) + ":"))
                    self.grigliadifettosità.add_widget(Label(
                        text=f"Sì (probabilità di errore: {str(round(self.allenamentopagina.betadifettosità, 3))}, tempo di calcolo: {str(round(self.allenamentopagina.tempobetadifettosità, 3))})" if difettosità else f"No (probabilità di errore: {str(self.allenamentopagina.impostazioni['alfahotellingduecampioni'])})"))
                self.grigliastatistiche.add_widget(self.grigliadifettosità)
            if self.allenamentopagina.difettositàvarianze is not None:
                self.grigliastatistiche.add_widget(Label(
                    text=f"Difettosità delle frecce per varianza (tempo di calcolo: {str(round(self.allenamentopagina.tempodifettositàvarianze, 3))}:"))
                self.grigliadifettositàvarianze = GridLayout(rows=len(self.allenamentopagina.difettositàvarianze),
                                                             cols=2)
                for etichetta, difettosità in enumerate(self.allenamentopagina.difettositàvarianze):
                    self.grigliadifettositàvarianze.add_widget(Label(text=str(etichetta + 1) + ":"))
                    self.grigliadifettositàvarianze.add_widget(Label(
                        text=f"Sì (probabilità di errore: {str(round(self.allenamentopagina.betadifettositàvarianze, 3))}, tempo di calcolo: {str(round(self.allenamentopagina.tempobetadifettositàvarianze, 3))})" if difettosità else f"No (probabilità di errore: {str(self.allenamentopagina.impostazioni['alfavarianzeduecampioni'])})"))
                self.grigliastatistiche.add_widget(self.grigliadifettositàvarianze)
            if self.allenamentopagina.uniformitàangoli is not None:
                self.grigliastatistiche.add_widget(Label(
                    text=f"Uniformità degli angoli (tempo di calcolo: {str(round(self.allenamentopagina.tempouniformitàangoli, 3))}):"))
                if self.allenamentopagina.uniformitàangoli:
                    # Non c'è una probabilità di errore di II tipo
                    self.grigliastatistiche.add_widget(Label(
                        text=f"Gli angoli sono uniformi (probabilità di errore: {str(round(self.allenamentopagina.betarayleigh, 3))}, tempo di calcolo: {str(round(self.allenamentopagina.tempobetarayleigh, 3))})"))
                else:
                    self.grigliastatistiche.add_widget(Label(
                        text=f"Gli angoli NON sono uniformi (probabilità di errore: {str(self.allenamentopagina.impostazioni['alfarayleigh'])})"))
            if self.allenamentopagina.kappa is not None:
                self.grigliastatistiche.add_widget(Label(
                    text=f"Stima puntuale del parametro di concentrazione degli angoli: {str(round(self.allenamentopagina.kappa[0], 3))} (tempo di calcolo: {str(round(self.allenamentopagina.tempokappa, 3))})"))
            if self.allenamentopagina.affidabilitàvonmises is not None:
                if self.allenamentopagina.affidabilitàvonmises:
                    # Non hai calcolato una probabilità di errore di II tipo
                    self.grigliastatistiche.add_widget(Label(
                        text=f"Gli angoli sono ben descritti da una distribuzione di von Mises (tempo di calcolo: {str(round(self.allenamentopagina.tempoaffidabilitàvonmises, 3))})"))
                else:
                    self.grigliastatistiche.add_widget(Label(
                        text=f"Gli angoli NON sono ben descritti da una distribuzione di von Mises (tempo di calcolo: {str(round(self.allenamentopagina.tempoaffidabilitàvonmises, 3))}, probabilità di errore: {self.allenamentopagina.impostazioni['alfavonmises']})"))
            if self.allenamentopagina.intervalloangolomedio is not None:
                self.grigliastatistiche.add_widget(Label(
                    text=f"Intervallo di confidenza al {self.allenamentopagina.impostazioni['confidenzaangolomedio'] if self.allenamentopagina.alfaverointervalloangolomedio is None else round(self.allenamentopagina.alfaverointervalloangolomedio, 3)} dell'angolo medio: [{round(180 + math.degrees(self.allenamentopagina.intervalloangolomedio[0]), 3)}, {round(180 + math.degrees(self.allenamentopagina.intervalloangolomedio[1]), 3)}] (tempo di calcolo: {round(self.allenamentopagina.tempointervalloangolomedio, 3)}"))
            if self.allenamentopagina.intervallokappa is not None:
                # Non è un intervallo asintotico?
                self.grigliastatistiche.add_widget(Label(
                    text=f"Intervallo di confidenza al {self.allenamentopagina.impostazioni['alfakappa'] if self.allenamentopagina.alfaverointervallokappa is None else round(self.allenamentopagina.alfaverointervallokappa, 3)}% per il parametro di concentrazione degli angoli:\n [{round(self.allenamentopagina.intervallokappa[0], 3)}, {round(self.allenamentopagina.intervallokappa[1], 3)}] (tempo di calcolo: {round(self.allenamentopagina.tempointervallokappa, 3)})"))

        aggiungioggetti(self.cornice, self.oggettipagina)

    def tornaadiariodacancellazione(self, instance):
        rimuovioggetti(self.cornice, [self.richiestacancellazioneallenamento,
                                      self.confermatorecancellazioneallenamento,
                                      self.annullatorecancellazioneallenamento])
        aggiungioggetti(self.cornice, [self.chiusorediario, self.scorsoredate])

    def cancelladavveroallenamento(self, instance, data):
        self.date = leggifile("Date.txt")
        self.date.remove(data)
        scrivifile(self.date, "Date.txt")
        remove(f"Sessione del {data}")
        rimuovioggetti(self.cornice, [self.richiestacancellazioneallenamento,
                                      self.confermatorecancellazioneallenamento,
                                      self.annullatorecancellazioneallenamento])
        self.diario(instance=None)

    def cancellaallenamento(self, instance, data):
        rimuovioggetti(self.cornice, [self.chiusorediario, self.scorsoredate])
        self.richiestacancellazioneallenamento = Label(text="Sei sicuro?")
        self.richiestacancellazioneallenamento.size_hint = (.8, .3)
        self.richiestacancellazioneallenamento.pos_hint = {"top": .8, "right": .9}
        self.confermatorecancellazioneallenamento = Button(text="Sì")
        self.confermatorecancellazioneallenamento.size_hint = (.4, .3)
        self.confermatorecancellazioneallenamento.pos_hint = {"top": .5, "right": .5}
        self.confermatorecancellazioneallenamento.background_color = (3, 0, 0, 1)
        self.confermatorecancellazioneallenamento.bind(
            on_press=lambda instance: self.cancelladavveroallenamento(instance, data))
        self.annullatorecancellazioneallenamento = Button(text="No")
        self.annullatorecancellazioneallenamento.size_hint = (.4, .3)
        self.annullatorecancellazioneallenamento.pos_hint = {"top": .5, "right": .9}
        self.annullatorecancellazioneallenamento.background_color = (0, 0, 3, 1)
        self.annullatorecancellazioneallenamento.bind(
            on_press=lambda instance: self.tornaadiariodacancellazione(instance))
        aggiungioggetti(self.cornice, [self.richiestacancellazioneallenamento,
                                       self.confermatorecancellazioneallenamento,
                                       self.annullatorecancellazioneallenamento])

    def diario(self, instance):
        rimuovioggetti(self.cornice, self.bottoniprincipali)
        self.chiusorediario = Button(text="Torna indietro")
        self.chiusorediario.size_hint = (1, .1)
        self.chiusorediario.pos_hint = {"top": 1, "right": 1}
        self.chiusorediario.background_color = (3, 0, 0, 1)
        self.chiusorediario.bind(on_press=self.tornaaprincipaledadiario)
        self.date = leggifile("Date.txt")
        self.scorsoredate = ScrollView()
        self.scorsoredate.size_hint = (1, .9)
        self.scorsoredate.pos_hint = {"top": .9, "right": 1}
        self.grigliadate = GridLayout(rows=len(self.date), cols=2)
        self.grigliadate.size_hint = (1, None)
        self.grigliadate.size = (Window.width * 9 / 10, 20 * len(self.date))
        self.scorsoredate.add_widget(self.grigliadate)
        for data in self.date:
            bottone = Button(text=data)
            bottone.bind(on_press=lambda instance, data=data: self.paginadiario(instance, data))
            bottone2 = Button(text="Cancella", background_color=(3, 0, 0, 1))
            bottone2.bind(on_press=lambda instance, data=data: self.cancellaallenamento(instance, data))
            self.grigliadate.add_widget(bottone)
            self.grigliadate.add_widget(bottone2)
        self.cornice.add_widget(self.chiusorediario)
        self.cornice.add_widget(self.scorsoredate)

    def generiche(self, instance):
        pass

    def chiudiimpostazioni(self, instance):
        mettibottoneindietro(self.chiudi)
        rimuovioggetti(self.cornice, self.bottoniimpostazioni)
        aggiungioggetti(self.cornice, self.bottoniprincipali)

    def tornaaimpostazionidaarco(self, instance):
        mettibottoneindietro(lambda: self.chiudiimpostazioni(instance=None))
        rimuovioggetti(self.cornice, self.oggettiarco)
        aggiungioggetti(self.cornice, self.bottoniimpostazioni)

    def tornaaarcodaaggiuntaarco(self, instance):
        mettibottoneindietro(lambda: self.tornaaimpostazionidaarco(instance=None))
        rimuovioggetti(self.cornice, (self.chiusoreaggiuntaarco, *self.bottoniaggiuntaarco))
        aggiungioggetti(self.cornice, self.oggettiarco)

    def tornaadaggiuntaarcodaeffettivo(self, instance):
        mettibottoneindietro(lambda: self.tornaaimpostazionidaarco(instance=None))
        rimuovioggetti(self.cornice, self.oggettieffettivo)
        aggiungioggetti(self.cornice, (self.chiusoreaggiuntaarco, *self.bottoniaggiuntaarco))

    def tornaaeffettivodasalvataggioeffettivo(self, instance):
        mettibottoneindietro(lambda: self.tornaadaggiuntaarcodaeffettivo(instance=None))
        rimuovioggetti(self.cornice, self.oggettisalvataggioeffettivo)
        aggiungioggetti(self.cornice, self.oggettigrigliaeffettivo)

    def salvaeffettivo(self, instance, oggettigriglia, tipo):
        self.dizionarioeffettivo = dict()
        self.i = 1
        if tipo == "olimpico":
            self.chiavi = ("Nomesetup", "Nomemirino", "Modelloarco", "Libbraggio", "Pivotcorda", "Allungo", "Tipocorda",
                           "Estensione", "Puntoincocco", "Tiller", "Occhiopin", "Occhiocorda", "Molla",
                           "Centershotbottone",
                           "Settaggiobottone", "Note")
        elif tipo == "nudo":
            self.chiavi = ("Nomesetup", "Modelloarco", "Libbraggio", "Pivotcorda", "Allungo", "Tipocorda",
                           "Puntoincocco", "Tiller", "Occhiopin", "Occhiocorda", "Molla", "Centershotbottone",
                           "Settaggiobottone", "Note")
        else:
            self.chiavi = ("Nomesetup", "Nomemirino", "Modelloarco", "Libbraggio", "Pivotcorda", "Allungo", "Tipocorda",
                           "Estensione", "Puntoincocco", "Tiller", "Visettediottra", "Caricovalle", "Release",
                           "Sincronizzazione", "Orizzontalerest", "Verticalerest", "Rigidezzarest", "Tiporest",
                           "Ingrandimentodiottra", "Visettecocca", "Note")
        for chiave in self.chiavi:
            self.dizionarioeffettivo[chiave] = oggettigriglia[self.i].text
            self.i += 2
        if tipo == "olimpico":
            self.nuovoeffettivo = ArcoOlimpico(self.dizionarioeffettivo)
        elif tipo == "nudo":
            self.nuovoeffettivo = ArcoNudo(self.dizionarioeffettivo)
        else:
            self.nuovoeffettivo = ArcoCompound(self.dizionarioeffettivo)
        if esiste("ImpostazioniArco.txt"):
            self.archiesistenti = leggifile("ImpostazioniArco.txt")
            self.archiesistenti.append(self.nuovoeffettivo)
            scrivifile(self.archiesistenti, "ImpostazioniArco.txt")
        else:
            creafile("ImpostazioniArco.txt", [self.nuovoeffettivo])
        rimuovioggetti(self.cornice, self.oggettisalvataggioeffettivo)
        self.impostaarco(instance=None)

    def rimettisalvatoreeffettivo(self, dt):
        self.salvatoreeffettivo.text = "Salva"

    def chiedisalvataggioeffettivo2(self, oggettigriglia, tipo):
        mettibottoneindietro(lambda: self.tornaaeffettivodasalvataggioeffettivo(instance=None))
        rimuovioggetti(self.cornice, self.oggettieffettivo)
        self.richiestasalvataggioeffettivo = Label(text="Vuoi davvero salvare?")
        self.richiestasalvataggioeffettivo.size_hint = (.8, .1)
        self.richiestasalvataggioeffettivo.pos_hint = {"top": .7, "right": .9}
        self.confermatoreeffettivo = Button(text="Sì")
        self.confermatoreeffettivo.size_hint = (.4, .3)
        self.confermatoreeffettivo.pos_hint = {"top": .6, "right": .5}
        self.confermatoreeffettivo.background_color = (0, 0, 3, 1)
        self.confermatoreeffettivo.color = (1, 1, 1, 1)
        self.confermatoreeffettivo.bind(on_press=lambda instance: self.salvaeffettivo(instance, oggettigriglia, tipo))
        self.negatoreeffettivo = Button(text="No")
        self.negatoreeffettivo.size_hint = (.4, .3)
        self.negatoreeffettivo.pos_hint = {"top": .6, "right": .9}
        self.negatoreeffettivo.background_color = (3, 0, 0, 1)
        self.negatoreeffettivo.color = (1, 1, 1, 1)
        self.negatoreeffettivo.bind(on_press=self.tornaaeffettivodasalvataggioeffettivo)
        self.oggettisalvataggioeffettivo = (self.richiestasalvataggioeffettivo, self.confermatoreeffettivo,
                                            self.negatoreeffettivo)
        aggiungioggetti(self.cornice, self.oggettisalvataggioeffettivo)

    def chiedisalvataggioeffettivo(self, instance, oggettigriglia, tipo):
        if oggettigriglia[1].text == "":
            self.salvatoreeffettivo.text = "Il nome non può essere vuoto"
            Clock.schedule_once(self.rimettisalvatoreeffettivo, 3)
        else:
            try:
                if tipo == "olimpico":
                    for indice in (7, 9, 11, 15, 17, 19, 21, 23, 27):
                        if oggettigriglia[indice].text != "" and "." not in list(oggettigriglia[indice].text):
                            int(oggettigriglia[indice].text)
                        elif "." in list(oggettigriglia[indice].text):
                            float(oggettigriglia[indice].text)
                elif tipo == "nudo":
                    for indice in (5, 7, 9, 13, 15, 17, 19, 23):
                        if oggettigriglia[indice].text != "" and "." not in list(oggettigriglia[indice].text):
                            int(oggettigriglia[indice].text)
                        elif "." in list(oggettigriglia[indice].text):
                            float(oggettigriglia[indice].text)
                else:
                    for indice in (7, 9, 11, 15, 17, 19, 21, 23, 37):
                        if oggettigriglia[indice].text != "" and "." not in list(oggettigriglia[indice].text):
                            int(oggettigriglia[indice].text)
                        elif "." in list(oggettigriglia[indice].text):
                            float(oggettigriglia[indice].text)
                archiesistenti = leggifile("ImpostazioniArco.txt")
                nomiarchiesistenti = [arco.nomesetup for arco in archiesistenti]
                if oggettigriglia[1].text in nomiarchiesistenti:
                    raise ValueError
            except ValueError:
                self.salvatoreeffettivo.text = ("Uno dei valori è invalido. Ricorda che devi "
                                                "usare il punto\n per le cifre decimali.")
                Clock.schedule_once(self.rimettisalvatoreeffettivo, 3)
            else:
                self.chiedisalvataggioeffettivo2(oggettigriglia, tipo)

    def arcoeffettivo(self, instance, tipo):
        mettibottoneindietro(lambda: self.tornaadaggiuntaarcodaeffettivo(instance=None))
        rimuovioggetti(self.cornice, (self.chiusoreaggiuntaarco, *self.bottoniaggiuntaarco))
        self.chiusoreeffettivo = Button(text="Torna alla selezione dell'arco")
        self.chiusoreeffettivo.background_color = (3, 0, 0, 1)
        self.chiusoreeffettivo.color = (1, 1, 1, 1)
        self.chiusoreeffettivo.size_hint = (1, .1)
        self.chiusoreeffettivo.pos_hint = {"top": 1, "right": 1}
        self.chiusoreeffettivo.bind(on_press=self.tornaadaggiuntaarcodaeffettivo)
        self.scorsoreeffettivo = ScrollView()
        self.scorsoreeffettivo.size_hint = (.8, .8)
        self.scorsoreeffettivo.pos_hint = {"top": .9, "right": .9}
        if tipo == "olimpico":
            self.grigliaeffettivo = GridLayout(rows=16, cols=2)
        elif tipo == "nudo":
            self.grigliaeffettivo = GridLayout(rows=14, cols=2)
        else:
            self.grigliaeffettivo = GridLayout(rows=22, cols=2)
        self.grigliaeffettivo.size_hint = (1, None)
        self.grigliaeffettivo.size = (Window.width * 8 / 10, 640)
        self.scorsoreeffettivo.add_widget(self.grigliaeffettivo)
        if tipo == "olimpico":
            self.oggettigrigliaeffettivo = (Label(text="Nome del setup:"), TextInput(multiline=False),
                                            Label(text="Nome mirino:"), TextInput(multiline=False),
                                            Label(text="Modello di arco:"), TextInput(multiline=False),
                                            Label(text="Libbraggio in libbre:"), TextInput(multiline=False),
                                            Label(text="Distanza pivot-corda:"), TextInput(multiline=False),
                                            Label(text="Allungo in centimetri:"), TextInput(multiline=False),
                                            Label(text="Tipo di corda:"), TextInput(multiline=False),
                                            Label(text="Estensione del mirino:"), TextInput(multiline=False),
                                            Label(text="Punto d'incocco in millimetri:"), TextInput(multiline=False),
                                            Label(text="Tiller in millimetri:"), TextInput(multiline=False),
                                            Label(text="Distanza occhio-pin in millimetri:"),
                                            TextInput(multiline=False),
                                            Label(text="Distanza occhio-corda in millimetri:"),
                                            TextInput(multiline=False),
                                            Label(text="Molla del bottone:"), Button(text="Seleziona"),
                                            Label(text="Centershot del bottone in millimetri:"),
                                            TextInput(multiline=False),
                                            Label(text="Settaggio del bottone:"), TextInput(multiline=False),
                                            Label(text="Note a piacere:"), TextInput())
        elif tipo == "nudo":
            self.oggettigrigliaeffettivo = (Label(text="Nome del setup:"), TextInput(multiline=False),
                                            Label(text="Modello di arco:"), TextInput(multiline=False),
                                            Label(text="Libbraggio in libbre:"), TextInput(multiline=False),
                                            Label(text="Distanza pivot-corda:"), TextInput(multiline=False),
                                            Label(text="Allungo in centimetri:"), TextInput(multiline=False),
                                            Label(text="Tipo di corda:"), TextInput(multiline=False),
                                            Label(text="Punto d'incocco in millimetri:"), TextInput(multiline=False),
                                            Label(text="Tiller in millimetri:"), TextInput(multiline=False),
                                            Label(text="Distanza occhio-pin in millimetri:"),
                                            TextInput(multiline=False),
                                            Label(text="Distanza occhio-corda in millimetri:"),
                                            TextInput(multiline=False),
                                            Label(text="Molla del bottone:"), Button(text="Seleziona"),
                                            Label(text="Centershot del bottone in millimetri:"),
                                            TextInput(multiline=False),
                                            Label(text="Settaggio del bottone:"), TextInput(multiline=False),
                                            Label(text="Note a piacere:"), TextInput())
        else:
            self.oggettigrigliaeffettivo = (Label(text="Nome del setup:"), TextInput(multiline=False),
                                            Label(text="Nome mirino:"), TextInput(multiline=False),
                                            Label(text="Modello di arco:"), TextInput(multiline=False),
                                            Label(text="Libbraggio in libbre:"), TextInput(multiline=False),
                                            Label(text="Distanza pivot-corda:"), TextInput(multiline=False),
                                            Label(text="Allungo in centimetri:"), TextInput(multiline=False),
                                            Label(text="Tipo di corda:"), TextInput(multiline=False),
                                            Label(text="Estensione del mirino:"), TextInput(multiline=False),
                                            Label(text="Punto d'incocco in millimetri:"), TextInput(multiline=False),
                                            Label(text="Tiller in millimetri:"), TextInput(multiline=False),
                                            Label(text="Distanza visette-diottra in millimetri:"),
                                            TextInput(multiline=False),
                                            Label(text="Carico in valle in libbre:"),
                                            TextInput(multiline=False),
                                            Label(text="Release:"),
                                            TextInput(multiline=False),
                                            Label(text="Sincronizzazione Cam superiore:"), TextInput(multiline=False),
                                            Label(text="Posizione orizzontale del rest:"), TextInput(multiline=False),
                                            Label(text="Posizione verticale del rest:"), TextInput(multiline=False),
                                            Label(text="Rigidezza del rest:"), TextInput(multiline=False),
                                            Label(text="Tipo di rest:"), TextInput(multiline=False),
                                            Label(text="Ingrandimento diottra:"), TextInput(multiline=False),
                                            Label(text="Distanza visette-cocca in millimetri:"),
                                            TextInput(multiline=False),
                                            Label(text="Note a piacere:"), TextInput())
        for oggetto in self.oggettigrigliaeffettivo:
            self.grigliaeffettivo.add_widget(oggetto)
        if tipo == "olimpico" or tipo == "nudo":
            self.mollabottone = DropDown()
            self.possibilimolle = (Button(text="Seleziona", size_hint=(None, None), size=(150, 20)),
                                   Button(text="Soffice", size_hint=(None, None), size=(150, 20)),
                                   Button(text="Medio", size_hint=(None, None), size=(150, 20)),
                                   Button(text="Fissato", size_hint=(None, None), size=(150, 20)),
                                   Button(text="Forte", size_hint=(None, None), size=(150, 20)))
            for molla in self.possibilimolle:
                molla.bind(on_release=lambda molla: self.mollabottone.select(molla.text))
                self.mollabottone.add_widget(molla)
            if tipo == "olimpico":
                self.mollabottone.bind(
                    on_select=lambda instance, x: setattr(self.oggettigrigliaeffettivo[25], "text", x))
                self.oggettigrigliaeffettivo[25].bind(on_press=self.mollabottone.open)
            else:
                self.mollabottone.bind(
                    on_select=lambda instance, x: setattr(self.oggettigrigliaeffettivo[21], "text", x))
                self.oggettigrigliaeffettivo[21].bind(on_press=self.mollabottone.open)
        self.salvatoreeffettivo = Button(text="Salva")
        self.salvatoreeffettivo.background_color = (0, 0, 3, 1)
        self.salvatoreeffettivo.color = (1, 1, 1, 1)
        self.salvatoreeffettivo.bind(on_press=lambda instance: self.chiedisalvataggioeffettivo(instance,
                                                                                               self.oggettigrigliaeffettivo,
                                                                                               tipo))
        self.salvatoreeffettivo.size_hint = (1, .1)
        self.salvatoreeffettivo.pos_hint = {"top": .1, "right": 1}
        self.oggettieffettivo = (self.chiusoreeffettivo, self.scorsoreeffettivo, self.salvatoreeffettivo)
        aggiungioggetti(self.cornice, self.oggettieffettivo)

    def aggiungiarco(self, instance):
        mettibottoneindietro(lambda: self.tornaaarcodaaggiuntaarco(instance=None))
        rimuovioggetti(self.cornice, self.oggettiarco)
        self.chiusoreaggiuntaarco = Button(text="Torna alla lista di archi")
        self.chiusoreaggiuntaarco.background_color = (3, 0, 0, 1)
        self.chiusoreaggiuntaarco.color = (1, 1, 1, 1)
        self.chiusoreaggiuntaarco.size_hint = (1, .1)
        self.chiusoreaggiuntaarco.pos_hint = {"top": 1, "right": 1}
        self.chiusoreaggiuntaarco.bind(on_press=self.tornaaarcodaaggiuntaarco)
        self.cornice.add_widget(self.chiusoreaggiuntaarco)
        self.olimpicatore = Button(text="Arco olimpico")
        self.nudatore = Button(text="Arco nudo")
        self.compoundatore = Button(text="Arco compound")
        self.sfondiaggiuntaarco = ((2, 2, 0, 1), (0, 0, 3, 1), (0, 3, 0, 1))
        self.coloriaggiuntaarco = ((0, 0, 0, 1), (1, 1, 1, 1), (1, 1, 1, 1))
        self.etichettebottoniaggiuntaarco = ("olimpico", "nudo", "compound")
        self.bottoniaggiuntaarco = (self.olimpicatore, self.nudatore, self.compoundatore)
        for bottone in range(3):
            bottoneattuale = self.bottoniaggiuntaarco[bottone]
            bottoneattuale.background_color = self.sfondiaggiuntaarco[bottone]
            bottoneattuale.color = self.coloriaggiuntaarco[bottone]
            bottoneattuale.bind(on_press=lambda instance, bottone=bottone: self.arcoeffettivo(instance,
                                                                                              self.etichettebottoniaggiuntaarco[
                                                                                                  bottone]))
            bottoneattuale.size_hint = (.8, .2)
            bottoneattuale.pos_hint = {"top": .75 - bottone * .2, "right": .9}
            self.cornice.add_widget(bottoneattuale)

    def tornaaarcodamodificaarco(self, instance):
        mettibottoneindietro(lambda: self.tornaaimpostazionidaarco(instance=None))
        rimuovioggetti(self.cornice, self.oggettimodifica)
        aggiungioggetti(self.cornice, self.oggettiarco)

    def tornaamodificaarcodachiedisalvataggiomodifica(self, instance):
        mettibottoneindietro(lambda: self.tornaaarcodamodificaarco(instance=None))
        rimuovioggetti(self.cornice, self.oggettisalvataggiomodifica)
        aggiungioggetti(self.cornice, self.oggettimodifica)

    def salvamodifica(self, instance, oggettigriglia: tuple, arco: ArcoOlimpico | ArcoNudo | ArcoCompound):
        mettibottoneindietro(lambda: self.tornaaimpostazionidaarco(instance=None))
        rimuovioggetti(self.cornice, self.oggettisalvataggiomodifica)
        self.dizionariomodifica = dict()
        self.i = 1
        if type(arco) is ArcoOlimpico:
            self.chiavi = ("Nomesetup", "Nomemirino", "Modelloarco", "Libbraggio", "Pivotcorda", "Allungo", "Tipocorda",
                           "Estensione", "Puntoincocco", "Tiller", "Occhiopin", "Occhiocorda", "Molla",
                           "Centershotbottone",
                           "Settaggiobottone", "Note")
        elif type(arco) is ArcoNudo:
            self.chiavi = ("Nomesetup", "Modelloarco", "Libbraggio", "Pivotcorda", "Allungo", "Tipocorda",
                           "Puntoincocco", "Tiller", "Occhiopin", "Occhiocorda", "Molla", "Centershotbottone",
                           "Settaggiobottone", "Note")
        else:
            self.chiavi = ("Nomesetup", "Nomemirino", "Modelloarco", "Libbraggio", "Pivotcorda", "Allungo", "Tipocorda",
                           "Estensione", "Puntoincocco", "Tiller", "Visettediottra", "Caricovalle", "Release",
                           "Sincronizzazione", "Orizzontalerest", "Verticalerest", "Rigidezzarest", "Tiporest",
                           "Ingrandimentodiottra", "Visettecocca", "Note")
        for chiave in self.chiavi:
            self.dizionariomodifica[chiave] = oggettigriglia[self.i].text
            self.i += 2
        if type(arco) is ArcoOlimpico:
            self.nuovamodifica = ArcoOlimpico(self.dizionariomodifica)
        elif type(arco) is ArcoNudo:
            self.nuovamodifica = ArcoNudo(self.dizionariomodifica)
        else:
            self.nuovamodifica = ArcoCompound(self.dizionariomodifica)
        self.archiesistenti[self.archiesistenti.index(arco)] = self.nuovamodifica
        scrivifile(self.archiesistenti, "ImpostazioniArco.txt")
        self.impostaarco(instance=None)

    def chiedisalvataggiomodifica(self, instance, oggettigriglia, arco):
        mettibottoneindietro(lambda: self.tornaamodificaarcodachiedisalvataggiomodifica(instance=None))
        rimuovioggetti(self.cornice, self.oggettimodifica)
        self.richiestasalvataggiomodifica = Label(text="Sei sicuro di voler salvare?")
        self.richiestasalvataggiomodifica.size_hint = (1, .1)
        self.richiestasalvataggiomodifica.pos_hint = {"top": .7, "right": 1}
        self.confermatoremodifica = Button(text="Sì")
        self.confermatoremodifica.size_hint = (.4, .3)
        self.confermatoremodifica.pos_hint = {"top": .6, "right": .5}
        self.confermatoremodifica.background_color = (0, 0, 3, 1)
        self.confermatoremodifica.color = (1, 1, 1, 1)
        self.confermatoremodifica.bind(on_press=lambda instance: self.salvamodifica(instance, oggettigriglia, arco))
        self.negatoremodifica = Button(text="No")
        self.negatoremodifica.size_hint = (.4, .3)
        self.negatoremodifica.pos_hint = {"top": .6, "right": .9}
        self.negatoremodifica.background_color = (3, 0, 0, 1)
        self.negatoremodifica.color = (1, 1, 1, 1)
        self.negatoremodifica.bind(on_press=self.tornaamodificaarcodachiedisalvataggiomodifica)
        self.oggettisalvataggiomodifica = (self.richiestasalvataggiomodifica, self.confermatoremodifica,
                                           self.negatoremodifica)
        aggiungioggetti(self.cornice, self.oggettisalvataggiomodifica)

    def modificaarco(self, instance, arco: ArcoOlimpico | ArcoNudo | ArcoCompound):
        mettibottoneindietro(lambda: self.tornaaarcodamodificaarco(instance=None))
        rimuovioggetti(self.cornice, self.oggettiarco)
        self.chiusoremodifica = Button(text="Torna alla lista di archi")
        self.chiusoremodifica.background_color = (3, 0, 0, 1)
        self.chiusoremodifica.color = (1, 1, 1, 1)
        self.chiusoremodifica.size_hint = (1, .1)
        self.chiusoremodifica.pos_hint = {"top": 1, "right": 1}
        self.chiusoremodifica.bind(on_press=self.tornaaarcodamodificaarco)
        self.scorsoremodifica = ScrollView()
        self.scorsoremodifica.size_hint = (.8, .8)
        self.scorsoremodifica.pos_hint = {"top": .9, "right": .9}
        if type(arco) is ArcoOlimpico:
            self.grigliamodifica = GridLayout(rows=16, cols=2)
        elif type(arco) is ArcoNudo:
            self.grigliamodifica = GridLayout(rows=14, cols=2)
        else:
            self.grigliamodifica = GridLayout(rows=22, cols=2)
        self.grigliamodifica.size_hint = (1, None)
        self.grigliamodifica.size = (Window.width * 8 / 10, 640)
        self.scorsoremodifica.add_widget(self.grigliamodifica)
        if type(arco) is ArcoOlimpico:
            self.oggettigrigliamodifica = (Label(text="Nome del setup:"),
                                           TextInput(text=arco.nomesetup, multiline=False),
                                           Label(text="Nome mirino:"),
                                           TextInput(text=arco.nomemirino, multiline=False),
                                           Label(text="Modello di arco:"),
                                           TextInput(text=arco.modelloarco, multiline=False),
                                           Label(text="Libbraggio in libbre:"),
                                           TextInput(text=arco.libbraggio, multiline=False),
                                           Label(text="Distanza pivot-corda:"),
                                           TextInput(text=arco.pivotcorda, multiline=False),
                                           Label(text="Allungo in centimetri:"),
                                           TextInput(text=arco.allungo, multiline=False),
                                           Label(text="Tipo di corda:"),
                                           TextInput(text=arco.tipocorda, multiline=False),
                                           Label(text="Estensione del mirino:"),
                                           TextInput(text=arco.estensione, multiline=False),
                                           Label(text="Punto d'incocco in millimetri:"),
                                           TextInput(text=arco.puntoincocco, multiline=False),
                                           Label(text="Tiller in millimetri:"),
                                           TextInput(text=arco.tiller, multiline=False),
                                           Label(text="Distanza occhio-pin in millimetri:"),
                                           TextInput(text=arco.occhiopin, multiline=False),
                                           Label(text="Distanza occhio-corda in millimetri:"),
                                           TextInput(text=arco.occhiocorda, multiline=False),
                                           Label(text="Molla del bottone:"), Button(text=arco.molla),
                                           Label(text="Centershot del bottone in millimetri:"),
                                           TextInput(text=arco.centershotbottone, multiline=False),
                                           Label(text="Settaggio del bottone:"),
                                           TextInput(text=arco.settaggiobottone, multiline=False),
                                           Label(text="Note a piacere:"), TextInput(text=arco.note))
        elif type(arco) is ArcoNudo:
            self.oggettigrigliamodifica = (Label(text="Nome del setup:"),
                                           TextInput(text=arco.nomesetup, multiline=False),
                                           Label(text="Modello di arco:"),
                                           TextInput(text=arco.modelloarco, multiline=False),
                                           Label(text="Libbraggio in libbre:"),
                                           TextInput(text=arco.libbraggio, multiline=False),
                                           Label(text="Distanza pivot-corda:"),
                                           TextInput(text=arco.pivotcorda, multiline=False),
                                           Label(text="Allungo in centimetri:"),
                                           TextInput(text=arco.allungo, multiline=False),
                                           Label(text="Tipo di corda:"),
                                           TextInput(text=arco.tipocorda, multiline=False),
                                           Label(text="Punto d'incocco in millimetri:"),
                                           TextInput(text=arco.puntoincocco, multiline=False),
                                           Label(text="Tiller in millimetri:"),
                                           TextInput(text=arco.tiller, multiline=False),
                                           Label(text="Distanza occhio-pin in millimetri:"),
                                           TextInput(text=arco.occhiopin, multiline=False),
                                           Label(text="Distanza occhio-corda in millimetri:"),
                                           TextInput(text=arco.occhiocorda, multiline=False),
                                           Label(text="Molla del bottone:"), Button(text=arco.molla),
                                           Label(text="Centershot del bottone in millimetri:"),
                                           TextInput(text=arco.centershotbottone, multiline=False),
                                           Label(text="Settaggio del bottone:"),
                                           TextInput(text=arco.settaggiobottone, multiline=False),
                                           Label(text="Note a piacere:"), TextInput(text=arco.note))
        else:
            self.oggettigrigliamodifica = (Label(text="Nome del setup:"),
                                           TextInput(text=arco.nomesetup, multiline=False),
                                           Label(text="Nome mirino:"),
                                           TextInput(text=arco.nomemirino, multiline=False),
                                           Label(text="Modello di arco:"),
                                           TextInput(text=arco.modelloarco, multiline=False),
                                           Label(text="Libraggio in libbre:"),
                                           TextInput(text=arco.libbraggio, multiline=False),
                                           Label(text="Distanza pivot-corda:"),
                                           TextInput(text=arco.pivotcorda, multiline=False),
                                           Label(text="Allungo in centimetri:"),
                                           TextInput(text=arco.allungo, multiline=False),
                                           Label(text="Tipo di corda:"),
                                           TextInput(text=arco.tipocorda, multiline=False),
                                           Label(text="Estensione del mirino:"),
                                           TextInput(text=arco.estensione, multiline=False),
                                           Label(text="Punto d'incocco in millimetri:"),
                                           TextInput(text=arco.puntoincocco, multiline=False),
                                           Label(text="Tiller in millimetri:"),
                                           TextInput(text=arco.tiller, multiline=False),
                                           Label(text="Distanza visette-diottra in millimetri:"),
                                           TextInput(text=arco.visettediottra, multiline=False),
                                           Label(text="Carico in valle in libbre:"),
                                           TextInput(text=arco.caricovalle, multiline=False),
                                           Label(text="Release:"),
                                           TextInput(text=arco.release, multiline=False),
                                           Label(text="Sincronizzazione Cam superiore:"),
                                           TextInput(text=arco.sincronizzazione, multiline=False),
                                           Label(text="Posizione orizzontale del rest:"),
                                           TextInput(text=arco.orizzontalerest, multiline=False),
                                           Label(text="Posizione verticale del rest:"),
                                           TextInput(text=arco.verticalerest, multiline=False),
                                           Label(text="Rigidezza del rest:"),
                                           TextInput(text=arco.rigidezzarest, multiline=False),
                                           Label(text="Tipo di rest:"),
                                           TextInput(text=arco.tiporest, multiline=False),
                                           Label(text="Ingrandimento diottra:"),
                                           TextInput(text=arco.ingrandimentodiottra, multiline=False),
                                           Label(text="Distanza visette-cocca in millimetri:"),
                                           TextInput(text=arco.visettecocca, multiline=False),
                                           Label(text="Note a piacere:"), TextInput(text=arco.note))
        for oggetto in self.oggettigrigliamodifica:
            self.grigliamodifica.add_widget(oggetto)
        if type(arco) is ArcoOlimpico or type(arco) is ArcoNudo:
            self.mollabottone = DropDown()
            self.possibilimolle = (Button(text="Seleziona", size_hint=(None, None), size=(150, 20)),
                                   Button(text="Soffice", size_hint=(None, None), size=(150, 20)),
                                   Button(text="Medio", size_hint=(None, None), size=(150, 20)),
                                   Button(text="Fissato", size_hint=(None, None), size=(150, 20)),
                                   Button(text="Forte", size_hint=(None, None), size=(150, 20)))
            for molla in self.possibilimolle:
                molla.bind(on_release=lambda molla: self.mollabottone.select(molla.text))
                self.mollabottone.add_widget(molla)
            if type(arco) is ArcoOlimpico:
                self.mollabottone.bind(
                    on_select=lambda instance, x: setattr(self.oggettigrigliamodifica[25], "text", x))
                self.oggettigrigliamodifica[25].bind(on_press=self.mollabottone.open)
            else:
                self.mollabottone.bind(
                    on_select=lambda instance, x: setattr(self.oggettigrigliamodifica[21], "text", x))
                self.oggettigrigliamodifica[21].bind(on_press=self.mollabottone.open)
        self.salvatoremodifica = Button(text="Salva")
        self.salvatoremodifica.background_color = (0, 0, 3, 1)
        self.salvatoremodifica.color = (1, 1, 1, 1)
        self.salvatoremodifica.bind(on_press=lambda instance: self.chiedisalvataggiomodifica(instance,
                                                                                             self.oggettigrigliamodifica,
                                                                                             arco))
        self.salvatoremodifica.size_hint = (1, .1)
        self.salvatoremodifica.pos_hint = {"top": .1, "right": 1}
        self.oggettimodifica = (self.chiusoremodifica, self.scorsoremodifica, self.salvatoremodifica)
        aggiungioggetti(self.cornice, self.oggettimodifica)

    def tornaaarcodacancellazione(self, instance):
        mettibottoneindietro(self.tornaaimpostazionidaarco)
        rimuovioggetti(self.cornice, self.oggetticancellazionearco)
        aggiungioggetti(self.cornice, self.oggettiarco)

    def cancellaarco2(self, instance, arco):
        rimuovioggetti(self.cornice, self.oggetticancellazionearco)
        self.indicerimuovendo = self.archiesistenti.index(arco)
        self.archiesistenti = leggifile("ImpostazioniArco.txt")
        self.archiesistenti.remove(self.archiesistenti[self.indicerimuovendo])
        scrivifile(self.archiesistenti, "ImpostazioniArco.txt")
        self.impostaarco(instance=None)

    def cancellaarco(self, instance, arco):
        mettibottoneindietro(lambda: self.tornaaarcodacancellazione(instance=None))
        rimuovioggetti(self.cornice, self.oggettiarco)
        self.richiestacancellazionearco = Label(text="Sei davvero sicuro di voler cancellare l'arco?")
        self.richiestacancellazionearco.size_hint = (1, .1)
        self.richiestacancellazionearco.pos_hint = {"top": .7, "right": 1}
        self.confermatorecancellazionearco = Button(text="Sì")
        self.confermatorecancellazionearco.background_color = (3, 0, 0, 1)
        self.confermatorecancellazionearco.color = (1, 1, 1, 1)
        self.confermatorecancellazionearco.size_hint = (.4, .3)
        self.confermatorecancellazionearco.pos_hint = {"top": .6, "right": .5}
        self.confermatorecancellazionearco.bind(on_press=lambda instance: self.cancellaarco2(instance, arco))
        self.negatorecancellazionearco = Button(text="No")
        self.negatorecancellazionearco.background_color = (0, 0, 3, 1)
        self.negatorecancellazionearco.color = (1, 1, 1, 1)
        self.negatorecancellazionearco.size_hint = (.4, .3)
        self.negatorecancellazionearco.pos_hint = {"top": .6, "right": .9}
        self.negatorecancellazionearco.bind(on_press=self.tornaaarcodacancellazione)
        self.oggetticancellazionearco = (self.richiestacancellazionearco, self.confermatorecancellazionearco,
                                         self.negatorecancellazionearco)
        aggiungioggetti(self.cornice, self.oggetticancellazionearco)

    def impostaarco(self, instance):
        mettibottoneindietro(lambda: self.tornaaimpostazionidaarco(instance=None))
        rimuovioggetti(self.cornice, self.bottoniimpostazioni)
        self.sfondiarco = ((3, 0, 0, 1), (0, 0, 0, 1), (0, 0, 3, 1))
        self.chiusorearco = Button(text="Torna al menù principale")
        self.chiusorearco.bind(on_press=self.tornaaimpostazionidaarco)
        self.chiusorearco.background_color = (3, 0, 0, 1)
        self.chiusorearco.color = (1, 1, 1, 1)
        self.chiusorearco.size_hint = (1, .1)
        self.chiusorearco.pos_hint = {"top": 1, "right": 1}
        self.oggettiarco = [self.chiusorearco]
        if esiste("Impostazioniarco.txt"):
            self.archiesistenti = leggifile("ImpostazioniArco.txt")
            if len(self.archiesistenti) > 0:
                self.scorsorearco = ScrollView()
                self.scorsorearco.size_hint = (1, .8)
                self.scorsorearco.pos_hint = {"top": .9, "right": 1}
                self.oggettiarco.append(self.scorsorearco)
                self.grigliaarco = GridLayout(rows=len(self.archiesistenti), cols=3)
                self.grigliaarco.size_hint = (1, None)
                self.grigliaarco.size = (Window.width, len(self.archiesistenti) * 50)
                self.scorsorearco.add_widget(self.grigliaarco)
                for arco in self.archiesistenti:
                    if type(arco) is ArcoOlimpico:
                        self.nominatorearco = Label(text="".join([arco.nomesetup, ", ", "arco olimpico"]))
                    elif type(arco) is ArcoNudo:
                        self.nominatorearco = Label(text="".join([arco.nomesetup, ", ", "arco nudo"]))
                    else:
                        self.nominatorearco = Label(text="".join([arco.nomesetup, ", ", "arco compound"]))
                    self.grigliaarco.add_widget(self.nominatorearco)
                    self.modificatorearco = Button(text="Modifica")
                    self.modificatorearco.bind(on_press=lambda instance, arco=arco: self.modificaarco(instance, arco))
                    self.grigliaarco.add_widget(self.modificatorearco)
                    self.cancellatorearco = Button(text="Cancella")
                    self.cancellatorearco.bind(on_press=lambda instance, arco=arco: self.cancellaarco(instance, arco))
                    self.grigliaarco.add_widget(self.cancellatorearco)
            else:
                self.indicazionenessunarco = Label(text="Non è presente alcun arco.\n Usa il bottone in basso "
                                                        "\n per aggiungerne uno.")
                self.oggettiarco.append(self.indicazionenessunarco)
        else:
            self.indicazionenessunarco = Label(text="Non è presente alcun arco.\n Usa il bottone in basso "
                                                    "\n per aggiungerne uno.")
            self.oggettiarco.append(self.indicazionenessunarco)
        self.nuovoarco = Button(text="Nuovo arco")
        self.nuovoarco.bind(on_press=self.aggiungiarco)
        self.nuovoarco.background_color = (0, 0, 3, 1)
        self.nuovoarco.color = (1, 1, 1, 1)
        self.nuovoarco.size_hint = (1, .1)
        self.nuovoarco.pos_hint = {"top": .1, "right": 1}
        self.oggettiarco.append(self.nuovoarco)
        aggiungioggetti(self.cornice, self.oggettiarco)

    def tornaaimpostazionidastatistiche(self, instance):
        pass

    def impostastatistiche(self, instance):
        mettibottoneindietro(self.tornaaimpostazionidastatistiche(instance=None))
        rimuovioggetti(self.cornice, self.bottoniimpostazioni)

    def imposta(self, instance):
        mettibottoneindietro(lambda: self.chiudiimpostazioni(instance=None))
        rimuovioggetti(self.cornice, self.bottoniprincipali)
        self.sfondiimpostazioni = ((3, 0, 0, 1), (3, 3, 3, 1), (0, 0, 0, 1))
        self.coloriimpostazioni = ((1, 1, 1, 1), (0, 0, 0, 1), (1, 1, 1, 1))
        self.funzioniimpostazioni = (self.chiudiimpostazioni, self.impostaarco, self.impostastatistiche)
        self.chiusoreimpostazioni = Button(text="Torna al menù principale")
        self.arciatore = Button(text="Impostazioni dell'arco")
        self.statisticatore = Button(text="Impostazioni delle statistiche e dell'applicazione")
        self.bottoniimpostazioni = (self.chiusoreimpostazioni, self.arciatore, self.statisticatore)
        for bottone in range(3):
            bottoneattuale = self.bottoniimpostazioni[bottone]
            bottoneattuale.background_color = self.sfondiimpostazioni[bottone]
            bottoneattuale.color = self.coloriimpostazioni[bottone]
            bottoneattuale.size_hint = (1, 1 / 3)
            bottoneattuale.pos_hint = {"top": 1 - bottone / 3, "right": 1}
            bottoneattuale.bind(on_press=self.funzioniimpostazioni[bottone])
        aggiungioggetti(self.cornice, self.bottoniimpostazioni)

    def build(self):
        Window.bind(on_back_button=lambda: self.chiudi(instance=None))
        self.cornice = RelativeLayout()
        self.sfondiprincipali = ((3, 0, 0, 1), (0, 0, 3, 1), (2, 2, 0, 1), (0, 3, 0, 1), (3, 3, 3, 1))
        self.coloriprincipali = ((1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 1), (1, 1, 1, 1), (0, 0, 0, 1))
        self.funzioniprincipali = (self.chiudi, self.allena, self.diario, self.generiche, self.imposta)
        self.chiusore = Button(text="Chiudi tutto")
        self.allenatore = Button(text="Aggiungi un allenamento")
        self.diariatore = Button(text="Apri il diario")
        self.genericatore = Button(text="Vedi le statistiche generiche")
        self.impostatore = Button(text="Impostazioni di statistiche e archi")
        self.bottoniprincipali = (self.chiusore, self.allenatore, self.diariatore, self.genericatore, self.impostatore)
        for bottone in range(5):
            bottoneattuale = self.bottoniprincipali[bottone]
            bottoneattuale.background_color = self.sfondiprincipali[bottone]
            bottoneattuale.color = self.coloriprincipali[bottone]
            bottoneattuale.size_hint = (1, .2)
            bottoneattuale.pos_hint = {"top": 1 - 0.2 * bottone, "right": 1}
            bottoneattuale.bind(on_press=self.funzioniprincipali[bottone])
        aggiungioggetti(self.cornice, self.bottoniprincipali)
        return self.cornice


if __name__ == "__main__":
    # VA AGGIUSTATO IL CALCOLO DEL BETA AL TEST DI HENZE-ZIRKLER;
    # in particolare, devi applicargli le stesse correzioni che hai applicato al test di Henze-Zirkler singolo
    # Idem per il test di Mardia
    SagittariumIuvo6App().run()

#Non hai implementato le regressioni quadratiche con test di significatività
#Correggi il probabilitàtuttogiusto al test di Ljung-Box
#Controlla che i metodi bayesiani gerarchici non sollevino errori a causa di parametri sbagliati
