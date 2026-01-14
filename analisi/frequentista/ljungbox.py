from classi.dati import registra, monitora, REGISTRO, Libro, SessioneGrezza, Test, Risultato, SuntoTest
import miomodulo
import scipy.stats as st


@monitora("vlbca")
def vlbca(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> SuntoTest:
    cose = miomodulo.ljungbox(libro.contenuto["ac"].valore[0].tolist(), dati.dati.taglia,
                                         impostazioni["hlbc"], impostazioni["albc"])
    return SuntoTest(cose.statistica, cose.pvalue, cose.accettazione)


@monitora("vlbco")
def vlbco(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> SuntoTest:
    cose = miomodulo.ljungbox(libro.contenuto["ac"].valore[1].tolist(), dati.dati.taglia,
                              impostazioni["hlbc"], impostazioni["albc"])
    return SuntoTest(cose.statistica, cose.pvalue, cose.accettazione)


@monitora("albc")
def albc(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float:
    if dati.dati.taglia < 50:
        return miomodulo.alfaveroljungbox(dati.dati.taglia, impostazioni["hlbc"],
                                          st.chi2.ppf(1-impostazioni["albc"], df=impostazioni["hlbc"]),
                                          impostazioni["ilbc"])
    else:
        return 1-impostazioni["albc"]


@monitora("blbc")
def blbc(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float:
    return miomodulo.betaljungbox(dati.dati.taglia, impostazioni["hlbc"],
                                  st.chi2.ppf(1-impostazioni["albc"], df=impostazioni["hlbc"]),
                                  impostazioni["ilbc"])


@registra("lbc", {"ac"}, "test di Ljung-Box sulle ascisse e sulle ordinate", REGISTRO)
@monitora("lbc")
def lbc(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> tuple[Test, Test] | None:
    if dati.dati.taglia < 3:
        return None
    ljungboxascisse, ljungboxordinate = vlbca(libro, dati, impostazioni), vlbco(libro, dati, impostazioni)
    if impostazioni["ce"]:
        if ljungboxascisse.valore.accettazione:
            betaascisse, alfaascisse = blbc(libro, dati, impostazioni), None
        else:
            betaascisse, alfaascisse = None, albc(libro, dati, impostazioni)
        if ljungboxordinate.valore.accettazione:
            betaordinate, alfaordinate = blbc(libro, dati, impostazioni), None
        else:
            betaordinate, alfaordinate = None, albc(libro, dati, impostazioni)
    else:
        betaascisse, alfaascisse, betaordinate, alfaordinate = None, None, None, None
    testascisse = Test("indipendenza delle ascisse", "Ljung-Box", ljungboxascisse, True, betaascisse, alfaascisse)
    testordinate = Test("indipendenza delle ordinate", "Ljung-Box", ljungboxordinate, True, betaordinate, alfaordinate)
    return testascisse, testordinate


@monitora("lbirp")
def lbirp(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> SuntoTest | None:
    ljungbox = miomodulo.ljungboxnorme(libro.contenuto["airp"].valore.tolist(), dati.dati.taglia,
                                       impostazioni["hlbirp"], impostazioni["albirp"])
    return SuntoTest(ljungbox.statistica, ljungbox.pvalue, ljungbox.accettazione)


@monitora("blbirp")
def blbirp(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    return miomodulo.betaljungbox(dati.dati.taglia, impostazioni["hlbirp"],
                                  st.chi2.ppf(1 - impostazioni["albirp"], df=impostazioni["hlbirp"]),
                                  impostazioni["ilbirp"])


@monitora("albirp")
def albirp(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    if dati.dati.taglia < 50:
        return miomodulo.alfaveroljungbox(dati.dati.taglia, impostazioni["hlbirp"],
                                          st.chi2.ppf(1-impostazioni["albirp"], df=impostazioni["hlbirp"]),
                                          impostazioni["ilbirp"])
    else:
        return 1-impostazioni["albirp"]


@registra("irpd", {"airp"}, "test di Ljung-Box sugli IRP", REGISTRO)
@monitora("irpd")
def irpd(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> Test | None:
    if dati.dati.taglia < 3:
        return None
    ljungboxirp = lbirp(libro, dati, impostazioni)
    if impostazioni["ce"]:
        if ljungboxirp.valore.accettazione:
            beta, alfa = blbirp(libro, dati, impostazioni), None
        else:
            beta, alfa = None, albirp(libro, dati, impostazioni)
    else:
        beta, alfa = None, None
    return Test("indipendenza degli IRP", "Ljung-Box", lbirp, True, alfa, beta)
