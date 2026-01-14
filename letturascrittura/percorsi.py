from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_root() -> Path:
    return _PROJECT_ROOT


def fileimpostazionianalisi() -> Path:
    return project_root() / "ImpostazioniAnalisi.pkl"


def fileimpostazionigenerali() -> Path:
    return project_root() / "ImpostazioniGenerali.pkl"


def filearchi() -> Path:
    return project_root() / "ImpostazioniArchi.pkl"


def cartelladati() -> Path:
    return project_root() / "dati"


def filegenerale() -> Path:
    return cartelladati() / "Generali.pkl"


def cartellasessioni() -> Path:
    return cartelladati() / "sessioni"


def cartellagrafici() -> Path:
    return cartelladati() / "grafici"


def cartellasessione(id_: int) -> Path:
    return cartellasessioni() / str(id_)


def filesessione(id_: int) -> Path:
    return cartellasessione(id_) / "sessione.pkl"


# Ma questi servono davvero?!
def cartellaanalisi(id_: int) -> Path:
    return cartellasessione(id_) / "analisi"


def risultatianalisi(id_: int, chiave: str) -> Path:
    return cartellaanalisi(id_) / f"{chiave}.pkl"


def cartellagraficisessione(id_: int) -> Path:
    return cartellagrafici() / str(id_)


def filegrafico(id_: int, nome: str, estensione="png") -> Path:
    return cartellagraficisessione(id_) / f"{nome}.{estensione}"


def assicuracartella(percorso: Path) -> None:
    percorso.mkdir(parents=True, exist_ok=True)


def assicuracartellesessione(id_: int) -> None:
    assicuracartella(cartellasessione(id_))
    assicuracartella(cartellaanalisi(id_))
    assicuracartella(cartellagraficisessione(id_))
