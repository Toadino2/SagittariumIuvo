from classi.dati import registra, monitora, REGISTRO, Libro, SessioneGrezza, Test, SuntoTest, Intervallo
import miomodulo
import scipy.stats as st
import numpy as np


def _th_(taglia: int, ascisse: list[float], ordinate: list[float], alfa: float):
    return miomodulo.testhotelling(taglia, ascisse, ordinate, alfa)


@monitora("th_")
def th_(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> SuntoTest | None:
    coordinate = dati.dati.xy
    test = _th_(dati.dati.taglia, coordinate[:, 0].tolist(), coordinate[:, 1].tolist(), impostazioni["ath"])
    return SuntoTest(test.statistica, test.pvalue, test.accettazione)


@monitora("bth")
def bth(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    return miomodulo.betatesthotelling(dati.dati.taglia, st.f.ppf(1-impostazioni["ath"], 2, dati.dati.taglia-2),
                                       impostazioni["ith"], impostazioni["dbth"], libro.contenuto["vc"].valore[0],
                                       libro.contenuto["vc"].valore[1])


@registra("th", {"lbc", "tn", "vc"}, "test di Hotelling sulle coordinate", REGISTRO)
@monitora("th")
def th(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> Test | None:
    if dati.dati.taglia < 3:
        return None
    test = th_(libro, dati, impostazioni)
    if impostazioni["ce"]:
        if test.valore.accettazione:
            beta = bth(libro, dati, impostazioni)
        else:
            beta = None
    else:
        beta = None
    return Test("media bivariata", "Hotelling", test, False, impostazioni["ath"], beta)


@monitora("thc_")
def thc_(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> SuntoTest | None:
    coordinate = dati.dati.xy
    cluster = coordinate[libro.contenuto["cl"].valore == impostazioni["clusterattuale"], :]
    if cluster[cluster == impostazioni["clusterattuale"]].shape[0] < 3:
        return None
    test = _th_(cluster.shape[0], cluster[:, 0].tolist(), cluster[:, 1].tolist(), impostazioni["ath"])
    return SuntoTest(test.statistica, test.pvalue, test.accettazione)


@monitora("bthc")
def bthc(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    cluster = libro.contenuto["cl"].valore
    tagliacluster = cluster[cluster == impostazioni["clusterattuale"]].shape
    return miomodulo.betatesthotelling(tagliacluster, st.f.ppf(1-impostazioni["ath"], 2, tagliacluster-2),
                                       impostazioni["ith"], impostazioni["dbth"], libro.contenuto["vc"].valore[0],
                                       libro.contenuto["vc"].valore[1])


@registra("thc", {"cl"}, "test di Hotelling sulle coordinate per ciascun cluster", REGISTRO)
@monitora("thc")
def thc(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> list[Test] | None:
    cluster = libro.contenuto["cl"].valore
    test = [None for _ in range(max(cluster)+1)]
    for indice in range(max(cluster)+1):
        impostazioni["clusterattuale"] = indice
        testcluster = thc_(libro, dati, impostazioni)
        betacluster = bthc(libro, dati, impostazioni)
        test[indice] = Test(f"media bivariata del cluster {indice}", "Hotelling", testcluster, False,
                            1-impostazioni["ath"], betacluster)
    del impostazioni["clusterattuale"]
    return test


def costanteih(coordinate: np.ndarray, impostazioni: dict, medie: np.ndarray) -> float | None:
    n = coordinate.shape[0]
    if impostazioni["cdih"] == "n" or impostazioni["cdih"] == "d":
        return (2*(n-1)/(n*(n-2)))*st.f.ppf(impostazioni["cih"], 2, n-2)
    elif impostazioni["cdih"] == "s":
        return miomodulo.bootstrapstazionariohotelling(n, coordinate[:, 0].tolist(), coordinate[:, 1].tolist(),
                                                       impostazioni["gsih"], impostazioni["bsih"],
                                                       medie[0], medie[1], impostazioni["cih"])
    elif impostazioni["cdih"] == "m":
        return miomodulo.bootstrapmobilehotelling(n, coordinate[:, 0].tolist(), coordinate[:, 1].tolist(),
                                                  impostazioni["lbmih"], impostazioni["bmih"], medie[0],
                                                  medie[1], impostazioni["cih"])


def ih_(varianze: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    return np.linalg.eigh(varianze)


@monitora("_ih")
def _ih(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                         float, np.ndarray | None] | None:
    if impostazioni["cdih"] == "d":
        regressione = libro.contenuto["rc"].valore
        va, vo, c, _ = miomodulo.detrenda(dati.dati.xy[:, 0], dati.dati.xy[:, 1],
                                          regressione[0][0], regressione[0][1],
                                          regressione[1][0], regressione[1][1])
        autovalori, autovettori = ih_(np.array([[va, c], [c, vo]]))
        media = np.array([regressione[0][0]+regressione[0][1], regressione[1][0]+regressione[1][1]])
        costante = costanteih(dati.dati.xy, impostazioni, media)
        return autovalori, autovettori, media, costante, regressione
    else:
        autovalori, autovettori = ih_(libro.contenuto["vc"].valore)
        media = libro.contenuto["mc"].valore
        costante = costanteih(dati.dati.xy, impostazioni, media)
        return autovalori, autovettori, media, costante, None


@monitora("cih")
def cih(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    if impostazioni["cdih"] == "n" or impostazioni["cdih"] == "d":
        return 1-impostazioni["cih"]
    elif impostazioni["cdih"] == "s":
        return 1-miomodulo.alfaverobootstrapstazionariohotelling(dati.dati.taglia, impostazioni["vabsih"],
                                                                 impostazioni["vobsih"], impostazioni["gsih"],
                                                                 impostazioni["bsih"], 1-impostazioni["cih"],
                                                                 impostazioni["ibsih"])
    elif impostazioni["cdih"] == "m":
        return 1-miomodulo.alfaverobootstrapmobilehotelling(dati.dati.taglia, impostazioni["vabmih"],
                                                            impostazioni["vobmih"], impostazioni["lbmih"],
                                                            impostazioni["bmih"], 1-impostazioni["cih"],
                                                            impostazioni["ibmih"])


@registra("ih", {"mc", "vc", "rc"}, "regione di confidenza per la media bivariata", REGISTRO)
@monitora("ih")
def ih(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> Intervallo | None:
    if dati.dati.taglia < 3:
        return None
    intervallo = Intervallo("media bivariata", _ih(libro, dati, impostazioni), False, False,
                            cih(libro, dati, impostazioni))
    return intervallo


@monitora("cihc")
def cihc(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    return 1-impostazioni["cih"]


@monitora("ihc_")
def ihc_(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                          float, np.ndarray | None] | None:
    coordinate = dati.dati.xy
    cluster = coordinate[libro.contenuto["cl"].valore == impostazioni["clusterattuale"], :]
    if cluster[cluster == impostazioni["clusterattuale"]].shape[0] < 3:
        return None
    autovalori, autovettori = ih_(np.cov(cluster))
    media = np.mean(cluster, axis=1)
    costante = costanteih(dati.dati.xy, impostazioni, media)
    return autovalori, autovettori, media, costante, None


@registra("ihc", {"cl"}, "regione di confidenza per la media bivariata di ogni cluster", REGISTRO)
@monitora("ihc")
def ihc(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> list[Intervallo] | None:
    cluster = libro.contenuto["cl"].valore
    intervalli = [None for _ in range(max(cluster) + 1)]
    for indice in range(max(cluster) + 1):
        impostazioni["clusterattuale"] = indice
        intervallocluster = ihc_(libro, dati, impostazioni)
        intervalli[indice] = Intervallo(f"media bivariata del cluster {indice}", intervallocluster,
                                        False, False, cihc(libro, dati, impostazioni))
    del impostazioni["clusterattuale"]
    return intervalli
