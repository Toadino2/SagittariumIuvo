from dataclasses import dataclass
import numpy as np
from typing import Any, Callable
from classi.tipi import (MetadatiSessione, punteggia, irpa, angola, controlladati,
                         controllaindici, controllametadati, controllaimpostazioni)
from letturascrittura.memoria import elencasessioni, leggifile
from time import time
from datetime import date
from os import makedirs


class Dati:
    def __init__(self, xy: np.ndarray, ordine: bool, idv: np.ndarray, idf: np.ndarray | None = None):
        xy = controlladati(xy)
        controllaindici(idv)
        controllaindici(idf)
        self.xy = xy
        self.punteggi = punteggia(xy)
        self.irp = irpa(xy)
        self.angoli = angola(xy)
        self.taglia = xy.shape[0]
        self.idvolée = idv
        self.volée = np.max(idv)+1
        self.idfrecce = idf
        self.frecce = None if idf is None else np.max(idf)+1
        self.ordine = ordine


class SessioneGrezza:
    def __init__(self, id_: int, dati: Dati, metadati: MetadatiSessione):
        controllametadati(metadati)
        self.id_ = id_
        self.dati = dati
        self.metadati = metadati


@dataclass
class Risultato:
    nome: str
    valore: Any
    errore: Exception | None
    tempo: float | None


@dataclass
class Libro:
    id_: int | None
    contenuto: dict[str, Risultato]
    tempototale: float


def monitora(nome: str):
    def decoratore(funzione):
        def confezione(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> Risultato:
            avvio = time()
            try:
                valore = funzione(libro, dati, impostazioni)
                return Risultato(nome=nome, valore=valore, errore=None, tempo=time() - avvio if valore is not None else None)
            except Exception as e:
                print(f"\033[35mNon ho potuto calcolare {REGISTRO[nome].prodotto} perché {e}\033[0m")
                return Risultato(nome=nome, valore=None, errore=e, tempo=None)
        return confezione
    return decoratore


@dataclass
class Compito:
    nome: str
    requisiti: set[str]
    prodotto: str
    funzione: Callable[[Libro, SessioneGrezza | list, dict], Risultato]


REGISTRO: dict[str, Compito] = dict()
FALDONE: dict[str, Compito] = dict()


def registra(nome: str, requisiti: set[str], prodotto: str, registro: dict):
    def decoratore(funzione):
        compito = Compito(nome=nome, requisiti=requisiti, prodotto=prodotto, funzione=funzione)
        if nome in registro:
            print(f"Compito {nome} già registrato")
        else:
            registro[nome] = compito
        return funzione
    return decoratore


def esecutore(registro: dict[str, Compito], nomecompito: str, libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> Libro:
    if nomecompito in libro.contenuto:
        return libro
    if nomecompito not in registro:
        print(f"Compito {nomecompito} non registrato")
    compito = registro[nomecompito]
    for requisito in compito.requisiti:
        libro = esecutore(registro, requisito, libro, dati, impostazioni)
    risultato = compito.funzione(libro, dati, impostazioni)
    libro.contenuto[nomecompito] = risultato
    return libro


def eseguitutto(registro: dict[str, Compito], compiti: list[str], libro: Libro, dati, impostazioni: dict, iniziale: bool) -> Libro:
    numerocompiti = len(compiti)
    for nome in range(numerocompiti):
        libro = esecutore(registro, compiti[nome], libro, dati, impostazioni)
        if iniziale:
            print(f"{nome+1}/{numerocompiti}: ho calcolato {registro[compiti[nome]].prodotto}")
    return libro


@dataclass
class Intervallo:
    parametro: str
    intervallo: Risultato
    asintotico: bool
    bayesiano: bool
    copertura: Risultato | float | None


@dataclass
class SuntoTest:
    statistica: float
    pvalue: float
    accettazione: bool


@dataclass
class Test:
    parametro: str
    metodo: str
    decisione: Risultato
    asintotico: bool
    alfa: Risultato | None
    beta: Risultato | None
    bayesiano: bool = False


def invalida(libro: Libro, chiavi: set[str]):
    for chiave in chiavi:
        if chiave in libro.contenuto:
            libro.tempototale -= libro.contenuto[chiave].tempo
            libro.contenuto.pop(chiave, None)


def riesegui(libro: Libro, dati, rieseguendi: set[str], registro: dict, impostazioni: dict):
    controllaimpostazioni(impostazioni)
    invalida(libro, rieseguendi)
    libro = eseguitutto(registro, list(rieseguendi), libro, dati, impostazioni, False)
    return libro


class Sessione:
    def modifica(self, modificando: str, nuoviparametri: dict):
        self.impostazioni[modificando] = True
        for parametro in nuoviparametri:
            self.impostazioni[parametro] = nuoviparametri[parametro]
        self.libro = riesegui(self.libro, self.dati, {modificando}, REGISTRO, self.impostazioni)

    def __init__(self, inserimento: SessioneGrezza, impostazioni: dict):
        self.dati = inserimento
        self.impostazioni = impostazioni
        print("Calcolo delle statistiche sull'allenamento in corso")
        tempoiniziale = time()
        compiti = list()
        for chiave in impostazioni:
            if chiave in REGISTRO and impostazioni[chiave]:
                compiti.append(chiave)
        self.libro = Libro(id_=inserimento.id_, contenuto={}, tempototale=0.0)
        eseguitutto(REGISTRO, compiti, self.libro, inserimento, impostazioni, True)
        self.libro.tempototale = time()-tempoiniziale
        print("\033[36mHo calcolato tutto!\033[0m")


def monitora_(nome: str):
    def decoratore(funzione):
        def confezione(libro: Libro, dati: list[Sessione], impostazioni: dict) -> Risultato:
            avvio = time()
            try:
                valore = funzione(libro, dati, impostazioni)
                return Risultato(nome=nome, valore=valore, errore=None, tempo=time() - avvio if valore is not None else None)
            except Exception as e:
                print(f"\033[35mNon ho potuto calcolare {FALDONE[nome].prodotto} perché {e}\033[0m")
                return Risultato(nome=nome, valore=None, errore=e, tempo=None)
        return confezione
    return decoratore


class Generali:
    def perioda(self, inizio, fine):
        pass

    def modifica(self, modificando: str, nuoviparametri: dict):
        self.impostazioni[modificando] = True
        for parametro in nuoviparametri:
            self.impostazioni[parametro] = nuoviparametri[parametro]
        self.libro = riesegui(self.libro, self.dati, {modificando}, FALDONE, self.impostazioni)

    def __init__(self, impostazioni: dict, periodo=None):
        elencosessioni = elencasessioni()
        if periodo is None:
            self.dati = [leggifile(sessione["path"]) for sessione in elencosessioni].sort(key=lambda s: s.dati.metadati.data)
        else:
            self.dati = periodo
        self.impostazioni = impostazioni
        print("Calcolo delle statistiche sull'allenamento in corso")
        tempoiniziale = time()
        compiti = ["ds"]
        for chiave in impostazioni:
            if chiave in FALDONE and impostazioni[chiave]:
                compiti.append(chiave)
        self.libro = Libro(id_=None, contenuto={}, tempototale=0.0)
        eseguitutto(REGISTRO, compiti, self.libro, self.dati, impostazioni, True)
        self.libro.tempototale = time() - tempoiniziale
        print("\033[36mHo calcolato tutto!\033[0m")


@dataclass
class ConfrontoPeriodi:
    periodi: tuple[tuple[date, date], tuple[date, date]]
    descrittive: tuple[dict[str, Risultato], dict[str, Risultato]]
    intervallihotelling: tuple[Risultato, Risultato]
    intervallit: tuple[Risultato, Risultato]
    intervallivarianze: tuple[Risultato, Risultato]
    anova: tuple[Risultato, Risultato]
    levene: tuple[Risultato, Risultato]
    clustering: tuple[Risultato, Risultato]
    chiquadratotempo: tuple[Risultato, Risultato]
    chiquadratofrecce: tuple[Risultato, Risultato]
