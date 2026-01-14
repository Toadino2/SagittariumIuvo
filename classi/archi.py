from dataclasses import dataclass
from typing import Any


@dataclass
class ArcoOlimpico:
    nomesetup : str = ""
    nomemirino: str = ""
    modelloarco: str = ""
    libbraggio: int = 0
    pivotcorda: Any = ""
    allungo: float = 0
    tipocorda: str = ""
    estensione: float = 0.0
    puntoincocco: Any = ""
    tiller: Any = ""
    occhiopin: float = 0.0
    occhiocorda: float = 0.0
    molla: Any = ""
    centershotbottone: Any = ""
    settaggiobottone: Any = ""
    note: Any = ""

    def modifica(self, dizionario: dict):
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


@dataclass
class ArcoNudo:
    nomesetup: str = ""
    nomemirino: str = ""
    modelloarco: str = ""
    libbraggio: int = 0
    pivotcorda: Any = ""
    allungo: float = 0
    tipocorda: str = ""
    estensione: float = 0.0
    puntoincocco: Any = ""
    tiller: Any = ""
    occhiopin: float = 0.0
    occhiocorda: float = 0.0
    molla: Any = ""
    centershotbottone: Any = ""
    settaggiobottone: Any = ""
    note: Any = ""

    def modifica(self, dizionario: dict):
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


@dataclass
class ArcoCompound:
    nomesetup: str = ""
    nomemirino: str = ""
    modelloarco: str = ""
    libbraggio: int = 0
    pivotcorda: Any = ""
    allungo: float = 0
    tipocorda: str = ""
    estensione: float = 0.0
    puntoincocco: Any = ""
    tiller: Any = ""
    occhiopin: float = 0.0
    occhiocorda: float = 0.0
    molla: Any = ""
    centershotbottone: Any = ""
    settaggiobottone: Any = ""
    note: Any = ""

    def modifica(self, dizionario: dict):
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
