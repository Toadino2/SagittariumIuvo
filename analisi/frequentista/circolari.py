from classi.dati import registra, monitora, REGISTRO, Libro, SessioneGrezza, Test, SuntoTest, Intervallo
import scipy.stats as st
import pycircstat as pcs
import math


@monitora("ua_")
def ua_(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> SuntoTest | None:
    statistica = 2*dati.dati.taglia*libro.contenuto["va"].valore
    pvalue = st.chi2.sf(statistica, df=2)
    return SuntoTest(statistica, pvalue, pvalue >= impostazioni["aua"])


@monitora("aua")
def aua(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    return miomodulo.alfaverorayleigh(impostazioni["iaua"], st.chi2.ppf(1-impostazioni["aua"], df=2), dati.dati.taglia)


@monitora("bua")
def bua(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    return miomodulo.betarayleigh(impostazioni["ibua"], impostazioni["kbua"],
                                  dati.dati.taglia, st.chi2.ppf(1-impostazioni["aua"], df=2))


@registra("ua", {"va"}, "uniformità dei dati angolari", REGISTRO)
@monitora("ua")
def ua(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> Test | None:
    if dati.dati.taglia < 2:
        return None
    test = ua_(libro, dati, impostazioni)
    if impostazioni["ce"]:
        if test.valore.accettazione:
            beta, alfa = bua(libro, dati, impostazioni), None
        else:
            beta, alfa = None, aua(libro, dati, impostazioni)
    else:
        beta, alfa = None, None
    return Test("uniformità degli angoli", "Rayleigh", test, True, alfa, beta)


@registra("k", set(), "parametro di concentrazione per gli angoli", REGISTRO)
@monitora("k")
def k(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    if dati.dati.taglia < 2:
        return None
    return pcs.kappa(dati.dati.angoli)[0]


@monitora("avm_")
def avm_(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> SuntoTest | None:
    test = miomodulo.testvonmises(dati.dati.angoli.tolist(), dati.dati.taglia, libro.contenuto["ma"].valore,
                                  libro.contenuto["k"].valore, st.chi2.ppf(1-impostazioni["atvm"], df=2))
    return SuntoTest(test.statistica, test.pvalue, test.accettazione)


@monitora("aavm")
def aavm(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    return miomodulo.alfaveroaffidabilitavonmises(impostazioni["iaavm"], libro.contenuto["k"].valore,
                                                  dati.dati.taglia, st.chi2.ppf(1-impostazioni["atvm"], df=2))


@monitora("bavm")
def bavm(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    return miomodulo.betaaffidabilitavonmises(impostazioni["ibavm"], impostazioni["ubavm"],
                                              dati.dati.taglia, st.chi2.ppf(1 - impostazioni["atvm"], df=2),
                                              impostazioni["dcavm"], impostazioni["kbavm"])


@registra("avm", {"ma", "k"}, "test di goodness of fit per la von Mises", REGISTRO)
@monitora("avm")
def avm(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> Test | None:
    if dati.dati.taglia < 2:
        return None
    test = avm_(libro, dati, impostazioni)
    if impostazioni["ce"]:
        if test.valore.accettazione:
            beta, alfa = bavm(libro, dati, impostazioni), None
        else:
            beta, alfa = None, aavm(libro, dati, impostazioni)
    else:
        beta, alfa = None, None
    return Test("goodness of fit della von Mises", "Cox", test, True, alfa, beta)


@monitora("iam_")
def iam_(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> tuple[float, float]:
    quantile = st.chi2.ppf(impostazioni["cam"], df=1)
    kappa = libro.contenuto["k"].valore
    r = libro.contenuto["va"].valore
    frecce = dati.dati.frecce
    angolomedio = libro.contenuto["ma"].valore
    n2r2 = frecce**2*r**2
    if impostazioni["iamf"]:
        distanza = math.acos((1 - quantile)/(2*kappa*r*frecce))
        return angolomedio-distanza, angolomedio+distanza
    elif r < 2 / 3:
        distanza = math.acos((2*frecce*(2*n2r2-frecce*quantile)/(n2r2*(4*frecce-quantile)))**0.5)
        return angolomedio-distanza, angolomedio+distanza
    else:
        distanza = math.acos((frecce**2-(frecce**2-n2r2)*math.exp(quantile))**0.5/n2r2)
        return angolomedio-distanza, angolomedio+distanza


@monitora("aiam")
def aiam(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    return miomodulo.alfaverointervalloangolomedio(impostazioni["iaiam"], libro.contenuto["k"].valore,
                                                   dati.dati.taglia, impostazioni["iamf"],
                                                   st.chi2.ppf(1-impostazioni["cam"], df=1))


@registra("iam", {"ma", "k", "va"}, "intervallo di confidenza per l'angolo medio", REGISTRO)
@monitora("iam")
def iam(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> Intervallo | None:
    if dati.dati.taglia < 2:
        return None
    intervallo = iam_(libro, dati, impostazioni)
    if dati.dati.taglia < 50:
        if impostazioni["ce"]:
            alfa = aiam(libro, dati, impostazioni)
        else:
            alfa = None
    else:
        alfa = 1-impostazioni["cam"]
    return Intervallo("angolo medio", intervallo, True, False, 1-alfa)


@monitora("ik_")
def ik_(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> tuple[float, float] | None:
    return miomodulo.intervallokappa(dati.dati.angoli, impostazioni["iik"], impostazioni["cik"], dati.dati.taglia)


@monitora("aik")
def aik(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    return miomodulo.alfaverointervallokappa(impostazioni["iaik"], impostazioni["cik"],
                                             libro.contenuto["k"].valore, impostazioni["iik"], dati.dati.taglia)


@registra("ik", set(), "intervallo di confidenza bootstrap per il parametro di concentrazione", REGISTRO)
@monitora("ik")
def ik(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> Intervallo | None:
    if dati.dati.taglia < 2:
        return None
    intervallo = ik_(libro, dati, impostazioni)
    if dati.dati.taglia < 50:
        if impostazioni["ce"]:
            alfa = aik(libro, dati, impostazioni)
        else:
            alfa = None
    else:
        alfa = 1-impostazioni["cik"]
    return Intervallo("parametro di concentrazione", intervallo, True, False, 1-alfa)
