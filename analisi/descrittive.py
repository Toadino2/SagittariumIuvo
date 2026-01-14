import numpy as np
import scipy.stats as st
from classi.dati import Libro, SessioneGrezza, registra, monitora, REGISTRO
import miomodulo


@registra("mp", set(), "media dei punteggi", REGISTRO)
@monitora("mp")
def mp(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    if dati.dati.taglia < 1:
        return None
    return np.mean(dati.dati.punteggi)


@registra("mirp", set(), "media degli IRP", REGISTRO)
@monitora("mirp")
def mirp(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    if dati.dati.taglia < 1:
        return None
    return np.mean(dati.dati.irp)


@registra("mpv", set(), "media dei punteggi per ogni volée", REGISTRO)
@monitora("mpv")
def mpv(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> list[float] | None:
    punteggi = dati.dati.punteggi
    volée = dati.dati.idvolée
    if dati.dati.volée < 2:
        return None
    return [np.mean(punteggi[volée == v]) for v in range(dati.dati.volée)]


@registra("mirpv", set(), "media degli IRP per ogni volée", REGISTRO)
@monitora("mirpv")
def mirpv(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> list[float] | None:
    irp = dati.dati.irp
    volée = dati.dati.idvolée
    if dati.dati.volée < 2:
        return None
    return [np.mean(irp[volée == v]) for v in range(dati.dati.volée)]


@registra("mc", set(), "media delle coordinate", REGISTRO)
@monitora("mc")
def mc(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> tuple[float, float] | None:
    if dati.dati.taglia < 1:
        return None
    return np.mean(dati.dati.xy, axis=1)


@registra("mcv", set(), "media delle coordinate per ogni volée", REGISTRO)
@monitora("mcv")
def mcv(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> list[tuple[float, float]] | None:
    coordinate = dati.dati.xy
    volée = dati.dati.idvolée
    if dati.dati.volée < 2:
        return None
    return [np.mean(coordinate[volée == v, :], axis=1) for v in range(dati.dati.volée)]


@registra("ma", set(), "angolo medio", REGISTRO)
@monitora("ma")
def ma(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    angoli = dati.dati.angoli
    return st.circmean(angoli)


@registra("mav", set(), "angolo medio per ogni volée", REGISTRO)
@monitora("mav")
def mav(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> list[float] | None:
    angoli = dati.dati.angoli
    volée = dati.dati.idvolée
    if dati.dati.volée < 2:
        return None
    return [st.circmean(angoli[volée == v]) for v in range(dati.dati.volée)]


@registra("vp", set(), "varianza dei punteggi", REGISTRO)
@monitora("vp")
def vp(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    if dati.dati.taglia < 2:
        return None
    return np.var(dati.dati.punteggi, ddof=1)


@registra("virp", set(), "varianza degli IRP", REGISTRO)
@monitora("virp")
def virp(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    if dati.dati.taglia < 2:
        return None
    return np.var(dati.dati.irp, ddof=1)


@registra("virpv", set(), "varianza degli IRP per ogni volée", REGISTRO)
@monitora("virpv")
def virpv(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> list[float] | None:
    irp = dati.dati.irp
    volée = dati.dati.idvolée
    if dati.dati.volée < 2 or any(len(volée[volée == v]) < 2 for v in range(dati.dati.volée)):
        return None
    return [np.var(irp[volée == v], ddof=1) for v in range(dati.dati.volée)]


@registra("vpv", set(), "varianza dei punteggi per ogni volée", REGISTRO)
@monitora("vpv")
def vpv(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> list[float] | None:
    punteggi = dati.dati.punteggi
    volée = dati.dati.idvolée
    if dati.dati.volée < 2 or any(len(volée[volée == v]) < 2 for v in range(dati.dati.volée)):
        return None
    return [np.var(punteggi[volée == v], ddof=1) for v in range(dati.dati.volée)]


@registra("vc", set(), "matrice di covarianze delle coordinate", REGISTRO)
@monitora("vc")
def vc(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> np.ndarray | None:
    if dati.dati.taglia < 2:
        return None
    return np.cov(dati.dati.xy, rowvar=True, ddof=1)


@registra("vcv", set(), "matrici di covarianze delle coordinate per ogni volée", REGISTRO)
@monitora("vcv")
def vcv(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> list[np.ndarray] | None:
    coordinate = dati.dati.xy
    volée = dati.dati.idvolée
    if dati.dati.volée < 2 or any(len(volée[volée == v]) < 2 for v in range(dati.dati.volée)):
        return None
    return [np.cov(coordinate[volée == v], rowvar=True, ddof=1) for v in range(dati.dati.volée)]


def varianzaangoli(angoli: np.ndarray) -> float:
    return (np.mean(np.cos(angoli)) ** 2 + np.mean(np.sin(angoli)) ** 2) ** 0.5


@registra("va", set(), "lunghezza risultante media", REGISTRO)
@monitora("va")
def va(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    if dati.dati.taglia < 2:
        return None
    return varianzaangoli(dati.dati.angoli)


def autocorrelazioni(osservazioni: np.ndarray | list):
    autocovarianze = np.correlate(osservazioni - np.mean(osservazioni), osservazioni - np.mean(osservazioni),
                        mode="full")
    return autocovarianze[autocovarianze.size//2+1:]/autocovarianze[autocovarianze.size//2]


@registra("vav", set(), "lunghezza risultante media per ogni volée", REGISTRO)
@monitora("vav")
def vav(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> list[float] | None:
    angoli = dati.dati.angoli
    volée = dati.dati.idvolée
    if dati.dati.volée < 2:
        return None
    return [varianzaangoli(angoli[volée == v]) for v in range(dati.dati.volée)]


@registra("cor", set(), "correlazione tra ascisse e ordinate", REGISTRO)
@monitora("cor")
def cor(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    coordinate = dati.dati.xy
    if dati.dati.taglia < 2:
        return None
    return np.corrcoef(coordinate[:, 0], coordinate[:, 1])[0, 1]


@registra("cv", set(), "correlazione tra ascisse e ordinate per ogni volée", REGISTRO)
@monitora("cv")
def cv(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> list[np.ndarray] | None:
    coordinate = dati.dati.xy
    volée = dati.dati.idvolée
    if dati.dati.volée < 2:
        return None
    return [np.corrcoef(coordinate[volée == v, 0], coordinate[volée == v, 1])[0, 1] for v in range(dati.dati.volée)]


@registra("rp", {"mpv"}, "regressione dei punteggi come serie storica", REGISTRO)
@monitora("rp")
def rp(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> np.ndarray | None:
    if dati.dati.ordine:
        if dati.dati.taglia < 2:
            return None
        punteggi = dati.dati.punteggi
        return np.polyfit([i for i in range(1, dati.dati.taglia+1)], punteggi, 1)
    else:
        if dati.dati.volée < 2:
            return None
        punteggimedi = libro.contenuto["mpv"].valore
        return np.polyfit([i for i in range(1, dati.dati.volée+1)], punteggimedi, 1)


@registra("rirp", {"mirpv"}, "regressione degli IRP come serie storica", REGISTRO)
@monitora("rirp")
def rirp(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> np.ndarray | None:
    if dati.dati.ordine:
        if dati.dati.taglia < 2:
            return None
        irp = dati.dati.irp
        return np.polyfit([i for i in range(1, dati.dati.taglia+1)], irp, 1)
    else:
        if dati.dati.volée < 2:
            return None
        irpmedi = libro.contenuto["mirpv"].valore
        return np.polyfit([i for i in range(1, dati.dati.volée+1)], irpmedi, 1)


@registra("rc", {"mcv"}, "regressione delle ascisse e delle ordinate come serie storiche", REGISTRO)
@monitora("rc")
def rc(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> tuple[np.ndarray, np.ndarray] | None:
    if dati.dati.ordine:
        if dati.dati.taglia < 2:
            return None
        coordinate = dati.dati.xy
        return (np.polyfit([i for i in range(1, dati.dati.taglia+1)], coordinate[:, 0], 1),
                np.polyfit([i for i in range(1, dati.dati.taglia+1)], coordinate[:, 1], 1))
    else:
        if dati.dati.volée < 2:
            return None
        coordinatemedie = libro.contenuto["mcv"].valore
        return (np.polyfit([i for i in range(1, dati.dati.volée+1)], [media[0] for media in coordinatemedie], 1),
                np.polyfit([i for i in range(1, dati.dati.volée+1)], [media[1] for media in coordinatemedie], 1))


@registra("ap", {"mpv"}, "autocorrelazioni dei punteggi come serie storica", REGISTRO)
@monitora("ap")
def ap(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> np.ndarray | None:
    if dati.dati.ordine:
        if dati.dati.taglia < 2:
            return None
        punteggi = dati.dati.punteggi
        return autocorrelazioni(punteggi)
    else:
        if dati.dati.volée < 2:
            return None
        punteggimedi = libro.contenuto["mpv"].valore
        return autocorrelazioni(punteggimedi)


@registra("airp", {"mirpv"}, "autocorrelazioni degli IRP come serie storica", REGISTRO)
@monitora("airp")
def airp(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> np.ndarray | None:
    if dati.dati.ordine:
        if dati.dati.taglia < 2:
            return None
        irp = dati.dati.irp
        return autocorrelazioni(irp)
    else:
        if dati.dati.volée < 2:
            return None
        irpmedi = libro.contenuto["mirpv"].valore
        return autocorrelazioni(irpmedi)


@registra("ac", {"mcv"}, "autocorrelazioni di ascisse e ordinate come serie storiche", REGISTRO)
@monitora("ac")
def ac(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> tuple[np.ndarray, np.ndarray] | None:
    if dati.dati.ordine:
        if dati.dati.taglia < 2:
            return None
        coordinate = dati.dati.xy
        return autocorrelazioni(coordinate[:, 0]), autocorrelazioni(coordinate[:, 1])
    else:
        if dati.dati.volée < 2:
            return None
        coordinatemedie = libro.contenuto["mcv"].valore
        return autocorrelazioni([media[0] for media in coordinatemedie]), autocorrelazioni([media[1] for media in coordinatemedie])


@registra("aa", {"mav"}, "autocorrelazioni circolari positive e negative dei dati angolari", REGISTRO)
@monitora("aa")
def aa(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> tuple[np.ndarray, np.ndarray] | None:
    if dati.dati.ordine:
        if dati.dati.taglia < 2:
            return None
        angoli = dati.dati.angoli
        autocorrelazionipositive, autocorrelazioninegative = miomodulo.autocorrelazioniangolari(angoli, dati.dati.taglia)
        return np.array(autocorrelazionipositive), np.array(autocorrelazioninegative)
    else:
        if dati.dati.volée < 2:
            return None
        angolimedi = libro.contenuto["mav"].valore
        autocorrelazionipositive, autocorrelazioninegative = miomodulo.autocorrelazioniangolari(angolimedi, dati.dati.taglia)
        return np.array(autocorrelazionipositive), np.array(autocorrelazioninegative)
