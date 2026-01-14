from classi.dati import registra, monitora, REGISTRO, Libro, SessioneGrezza, Test, SuntoTest
import numpy as np
import scipy.stats as st


@monitora("fd_")
def fd_(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> SuntoTest | None:
    return SuntoTest(impostazioni["statistica"], impostazioni["pvalue"], impostazioni["accettazione"])


@registra("fd", {"vc"}, "test di Hotelling sulla difettosità delle frecce", REGISTRO)
@monitora("fd")
def fd(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> dict[int, Test] | None:
    frecce = dati.dati.idfrecce
    if dati.dati.frecce < 2 or any(len(frecce[frecce == f]) < 2 for f in range(dati.dati.frecce)):
        return None
    coordinate = dati.dati.xy
    difettosità = miomodulo.freccedifettose(coordinate[:, 0], coordinate[:, 1], dati.dati.idfrecce.tolist(),
                                            dati.dati.taglia, impostazioni["ahdc"], dati.dati.frecce)
    matricecovarianze = libro.contenuto["vc"].valore
    risultati = dict()
    for f in range(dati.dati.frecce):
        impostazioni["statistica"] = difettosità[f].statistica
        impostazioni["pvalue"] = difettosità[f].pvalue
        impostazioni["accettazione"] = difettosità[f].accettazione
        if impostazioni["ce"] and impostazioni["accettazione"]:
            beta = miomodulo.betahotellingduecampioni(impostazioni["ibhdc"], matricecovarianze[0, 0], matricecovarianze[1, 1],
                        matricecovarianze[0, 1], dati.dati.taglia, impostazioni["dbhdc"], dati.dati.frecce, impostazioni["ahdc"])
        else:
            beta = None
        risultati[f] = Test(f"difettosità in media della freccia {f}", "Hotelling",
                            fd_(libro, dati, impostazioni), False, impostazioni["ahdc"], beta)
    del impostazioni["statistica"], impostazioni["pvalue"], impostazioni["accettazione"]
    return risultati


@monitora("vad")
def vad(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> SuntoTest | None:
    freccia = impostazioni["freccia"]
    difettanda = dati.dati.xy[:, 0][dati.dati.idfrecce == freccia]
    altre = dati.dati.xy[:, 0][dati.dati.idfrecce == freccia]
    statistica = np.var(difettanda)/np.var(altre)
    pvalue = st.f.sf(statistica, difettanda.shape-1, altre.shape-1)
    return SuntoTest(statistica, pvalue, pvalue >= impostazioni["avd"])


@monitora("vod")
def vod(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> SuntoTest | None:
    freccia = impostazioni["freccia"]
    difettanda = dati.dati.xy[:, 1][dati.dati.idfrecce == freccia]
    altre = dati.dati.xy[:, 1][dati.dati.idfrecce == freccia]
    statistica = np.var(difettanda)/np.var(altre)
    pvalue = st.f.sf(statistica, difettanda.shape-1, altre.shape-1)
    return SuntoTest(statistica, pvalue, pvalue >= impostazioni["avd"])


@monitora("bvd")
def bvd(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    return miomodulo.betavarianzeduecampioni(impostazioni["ibvdc"], impostazioni["dbvdc"],
                                             dati.dati.taglia, dati.dati.frecce,
                                             st.f.ppf(1-impostazioni["avd"], int(dati.dati.taglia/dati.dati.frecce),
                                                      dati.dati.taglia-int(dati.dati.taglia/dati.dati.frecce)))


@registra("vd", set(), "test sulle varianze a due campioni sulle frecce", REGISTRO)
@monitora("vd")
def vd(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> dict[int, tuple[Test, Test]] | None:
    nfrecce = dati.dati.frecce
    risultati = dict()
    for f in range(nfrecce):
        impostazioni["freccia"] = f
        testascisse = vad(libro, dati, impostazioni)
        testordinate = vod(libro, dati, impostazioni)
        if impostazioni["ce"]:
            if testascisse.valore.accettazione:
                betaascisse = bvd(libro, dati, impostazioni)
            else:
                betaascisse = None
            if testordinate.valore.accettazione:
                betaordinate = bvd(libro, dati, impostazioni)
            else:
                betaordinate = None
        else:
            betaascisse, betaordinate = None, None
        risultati[f] = (Test(f"difettosità in varianza delle ascisse della freccia {f}",
                             "Cochran e Snedecor", testascisse, False, impostazioni["avd"], betaascisse),
                        Test(f"difettosità in varianza delle ordinate della freccia {f}",
                             "Cochran e Snedecor", testordinate, False, impostazioni["avd"], betaordinate))
    return risultati
