from classi.dati import registra, monitora, REGISTRO, Libro, SessioneGrezza, Intervallo, Test, SuntoTest
import scipy.stats as st


@monitora("iva")
def iva(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> tuple[float, float]:
    varianza = libro.contenuto["vc"].valore[0, 0]
    n = dati.dati.taglia
    return (n-1)*varianza/st.chi2.ppf(1-(1-impostazioni["cva"])/2, df=n-1), (n-1)*varianza/st.chi2.ppf((1-impostazioni["cva"])/2, df=n-1)


@monitora("ivo")
def ivo(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> tuple[float, float]:
    varianza = libro.contenuto["vc"].valore[1, 1]
    n = dati.dati.taglia
    return (n-1)*varianza/st.chi2.ppf(1-(1-impostazioni["cvo"])/2, df=n-1), (n-1)*varianza/st.chi2.ppf((1-impostazioni["cvo"])/2, df=n-1)


@registra("_iv", {"vc"}, "intervalli di confidenza per le varianze di ascisse e ordinate", REGISTRO)
@monitora("_iv")
def _iv(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> tuple[Intervallo, Intervallo] | None:
    ascisse, ordinate = iva(libro, dati, impostazioni), ivo(libro, dati, impostazioni)
    intervalloascisse = Intervallo("varianza delle ascisse", ascisse, False, False, 1-impostazioni["cva"])
    intervalloordinate = Intervallo("varianza delle ordinate", ordinate, False, False, 1-impostazioni["cvo"])
    return intervalloascisse, intervalloordinate


@monitora("tva")
def tva(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> SuntoTest | None:
    statistica = (dati.dati.taglia-1)*libro.contenuto["vc"].valore[0, 0]/impostazioni["inva"]
    alternativa = impostazioni["a_va"]
    frecce = dati.dati.taglia
    if alternativa == ">":
        pvalue = 1-st.chi2.cdf(statistica, df=frecce - 1)
    elif alternativa == "=":
        pvalue = 2*min([st.chi2.cdf(statistica, df=frecce-1), 1-st.chi2.cdf(statistica, df=frecce-1)])
    elif alternativa == "<":
        pvalue = st.chi2.cdf(statistica, df=frecce - 1)
    return SuntoTest(statistica, pvalue, pvalue >= impostazioni["ava"])


@monitora("tvo")
def tvo(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> SuntoTest | None:
    statistica = (dati.dati.taglia-1)*libro.contenuto["vc"].valore[1, 1]/impostazioni["invo"]
    alternativa = impostazioni["a_vo"]
    frecce = dati.dati.taglia
    if alternativa == ">":
        pvalue = 1-st.chi2.cdf(statistica, df=frecce - 1)
    elif alternativa == "=":
        pvalue = 2*min([st.chi2.cdf(statistica, df=frecce-1), 1-st.chi2.cdf(statistica, df=frecce-1)])
    else:
        pvalue = st.chi2.cdf(statistica, df=frecce - 1)
    return SuntoTest(statistica, pvalue, pvalue >= impostazioni["avo"])


@monitora("btva")
def btva(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    alternativa = impostazioni["a_va"]
    iterazioni = impostazioni["ibtva"]
    frecce = dati.dati.taglia
    distanza = dati.metadati.distanza
    alfa = impostazioni["ava"]
    ipotesinulla = impostazioni["inva"]
    if alternativa == ">":
        return miomodulo.betatestvarianze(iterazioni, frecce, True, False, distanza,
                                          st.chi2.ppf(1 - alfa, df=frecce - 1),
                                          0, ipotesinulla)
    elif alternativa == "=":
        return miomodulo.betatestvarianze(iterazioni, frecce, False, True, distanza,
                                          st.chi2.ppf(alfa / 2, df=frecce - 1),
                                          st.chi2.ppf(1 - alfa / 2, df=frecce - 1), ipotesinulla)
    elif alternativa == "<":
        return miomodulo.betatestvarianze(iterazioni, frecce, False, False, distanza, st.chi2.ppf(alfa, df=frecce - 1),
                                          0, ipotesinulla)


@monitora("btvo")
def btvo(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    alternativa = impostazioni["a_vo"]
    iterazioni = impostazioni["ibtvo"]
    frecce = dati.dati.taglia
    distanza = dati.metadati.distanza
    alfa = impostazioni["avo"]
    ipotesinulla = impostazioni["invo"]
    if alternativa == ">":
        return miomodulo.betatestvarianze(iterazioni, frecce, True, False, distanza,
                                          st.chi2.ppf(1 - alfa, df=frecce - 1),
                                          0, ipotesinulla)
    elif alternativa == "=":
        return miomodulo.betatestvarianze(iterazioni, frecce, False, True, distanza,
                                          st.chi2.ppf(alfa / 2, df=frecce - 1),
                                          st.chi2.ppf(1 - alfa / 2, df=frecce - 1), ipotesinulla)
    elif alternativa == "<":
        return miomodulo.betatestvarianze(iterazioni, frecce, False, False, distanza, st.chi2.ppf(alfa, df=frecce - 1),
                                          0, ipotesinulla)


@registra("tv", {"vc"}, "test d'ipotesi sulle varianze delle ascisse e delle ordinate", REGISTRO)
@monitora("tv")
def tv(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> tuple[Test, Test] | None:
    ascisse, ordinate = tva(libro, dati, impostazioni), tvo(libro, dati, impostazioni)
    if impostazioni["ce"]:
        if ascisse.valore.accettazione:
            betaascisse = btva(libro, dati, impostazioni)
        else:
            betaascisse = None
        if ordinate.valore.accettazione:
            betaordinate = btvo(libro, dati, impostazioni)
        else:
            betaordinate = None
    else:
        betaascisse, betaordinate = None, None
    testascisse = Test("varianza delle ascisse", "Cochran e Snedecor", ascisse, False, 1-impostazioni["atva"], betaascisse)
    testordinate = Test("varianza delle ordinate", "Cochran e Snedecor", ordinate, False, 1-impostazioni["atvo"], betaordinate)
    return testascisse, testordinate
