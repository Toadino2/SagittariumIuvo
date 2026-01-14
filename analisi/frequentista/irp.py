from classi.dati import registra, monitora, REGISTRO, Libro, SessioneGrezza, Test, Intervallo, SuntoTest
import scipy.stats as st


@monitora("iirp_")
def iirp_(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> tuple[float, float] | None:
    m = libro.contenuto["mirp"].valore
    v = libro.contenuto["virp"].valore
    n = dati.dati.taglia
    l = ((v/n)**0.5)*st.t.ppf((1-impostazioni["ciirp"])/2, df=n-1)
    return m+l, m-l


@registra("iirp", {"mirp", "virp"}, "intervallo di confidenza per la media degli IRP", REGISTRO)
@monitora("iirp")
def iirp(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> Intervallo | None:
    if dati.dati.taglia < 2:
        return None
    intervallo = iirp_(libro, dati, impostazioni)
    return Intervallo("media degli IRP", intervallo, False, False, impostazioni["ciirp"])


@monitora("tirp_")
def tirp_(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> SuntoTest | None:
    test = st.ttest_1samp(dati.dati.irp, alternative=impostazioni["a_tirp"], popmean=impostazioni["mtirp"])
    return SuntoTest(test.statistic, test.pvalue, test.pvalue >= impostazioni["atirp"])


@monitora("btirp")
def btirp(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    alternativa = impostazioni["a_tirp"]
    frecce = dati.dati.taglia
    iterazioni = impostazioni["ibtirp"]
    media = libro.contenuto["mirp"].valore
    distanza = impostazioni["dbtirp"]
    varianza = libro.contenuto["virp"].valore
    alfa = impostazioni["atirp"]
    if alternativa == ">":
        return miomodulo.betatestnorme(frecce, True, False, iterazioni, media, distanza, varianza,
                                       st.t.ppf(1 - alfa, df=frecce-1))
    elif alternativa == "=":
        return miomodulo.betatestnorme(frecce, False, True, iterazioni, media, distanza, varianza,
                                       st.t.ppf(1 - alfa / 2, df=frecce-1))
    elif alternativa == "<":
        return miomodulo.betatestnorme(frecce, False, False, iterazioni, media, distanza, varianza,
                                       st.t.ppf(alfa, df=frecce-1))


@registra("tirp", {"mirp", "virp"}, "t-test sugli IRP", REGISTRO)
@monitora("tirp")
def tirp(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> Test | None:
    if dati.dati.taglia < 2:
        return None
    test = tirp_(libro, dati, impostazioni)
    if impostazioni["ce"]:
        if test.valore.accettazione:
            beta = btirp(libro, dati, impostazioni)
        else:
            beta = None
    else:
        beta = None
    return Test("media degli IRP", "t-test", test, False, impostazioni["atirp"], beta)
