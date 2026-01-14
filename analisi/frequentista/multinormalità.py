import numpy as np
from classi.dati import Test, SuntoTest, Risultato, registra, monitora, REGISTRO, Libro, SessioneGrezza
import scipy.stats as st


@monitora("thz")
def thz(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> SuntoTest | None:
    test = miomodulo.testhenzezirkler(dati.dati.xy[:, 0].tolist(), dati.dati.xy[:, 1].tolist(),
                                      dati.dati.taglia, impostazioni["atn"])
    return SuntoTest(test.statistica, test.pvalue, test.accettazione)


@monitora("ahz")
def ahz(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    if dati.dati.taglia < 50:
        return 1-impostazioni["atn"]
    else:
        return miomodulo.alfaverohenzezirkler(impostazioni["iatn"], dati.dati.taglia, impostazioni["atn"])


@monitora("bhz")
def bhz(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    return miomodulo.betatesthenzezirkler(impostazioni["dbtn"], dati.dati.taglia, impostazioni["ibtn"], impostazioni["gtbtn"],
                                          impostazioni["aabtn"], impostazioni["aobtn"], impostazioni["dcbtn"],
                                          impostazioni["atn"])


@monitora("tm")
def tm(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> SuntoTest | None:
    test = miomodulo.testmardia(dati.dati.xy[:, 0].tolist(), dati.dati.xy[:, 1].tolist(),
                                dati.dati.taglia, impostazioni["aatn"], impostazioni["actn"])
    return SuntoTest(test.statistica, test.pvalue, test.accettazione)


@monitora("bm")
def bm(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    return miomodulo.betatestmardia(impostazioni["ibtn"], dati.dati.taglia, impostazioni["dbtn"],
                                    impostazioni["aabtn"], impostazioni["aobtn"], impostazioni["dcbtn"],
                                    impostazioni["gtbtn"], st.chi2.ppf(1-impostazioni["aatn"], df=4), st.norm.ppf(1-impostazioni["actn"]))


@monitora("am")
def am(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    if dati.dati.taglia < 50:
        return 1-impostazioni["atn"]
    else:
        return miomodulo.alfaveromardia(impostazioni["iatn"], dati.dati.taglia,
                                        st.chi2.ppf(1-impostazioni["atn"], df=4), st.norm.ppf(1-impostazioni["atn"]))


@registra("tn", set(), "test di binormalità sulle coordinate", REGISTRO)
@monitora("tn")
def tn(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> Test | None:
    if dati.dati.taglia < 3:
        return None
    if impostazioni["ttn"] == "h":
        test = thz(libro, dati, impostazioni)
        if impostazioni["ce"]:
            if test.valore.accettazione:
                beta, alfa = bhz(libro, dati, impostazioni), None
            else:
                beta, alfa = None, ahz(libro, dati, impostazioni)
        else:
            beta, alfa = None, None
        return Test("binormalità", "Henze-Zirkler", test, True, alfa, beta)
    elif impostazioni["ttn"] == "m":
        test = tm(libro, dati, impostazioni)
        if impostazioni["ce"]:
            if test.valore.accettazione:
                beta, alfa = bm(libro, dati, impostazioni), None
            else:
                beta, alfa = None, am(libro, dati, impostazioni)
        else:
            beta, alfa = None, None
        return Test("binormalità", "Mardia", test, True, alfa, beta)


@monitora("irpn_")
def irpn_(libro: Libro, dati:SessioneGrezza, impostazioni: dict) -> SuntoTest | None:
    sw = st.shapiro(dati.dati.irp)
    return SuntoTest(sw.statistic, sw.pvalue, sw.pvalue >= impostazioni["airpn"])


@monitora("birpn")
def birpn(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    alternativa = impostazioni["abirpn"]
    iterazioni = impostazioni["ibirpn"]
    frecce = dati.dati.taglia
    alfa = impostazioni["airpn"]
    accettazioni = 0
    if alternativa == "l":
        for _ in range(iterazioni):
            if st.shapiro(np.random.laplace(size=frecce)).pvalue > alfa:
                accettazioni += 1
    elif alternativa == "a":
        for _ in range(iterazioni):
            if st.shapiro(st.skewnorm.rvs(impostazioni["asbirpn"], size=frecce)).pvalue > alfa:
                accettazioni += 1
    elif alternativa == "u":
        for _ in range(iterazioni):
            if st.shapiro(np.random.uniform(size=frecce)).pvalue > alfa:
                accettazioni += 1
    elif alternativa == "m":
        for _ in range(iterazioni):
            if st.shapiro(np.array([np.random.normal(-impostazioni["dcbirpn"] / 2, 1.0) if np.random.uniform(0, 1, 1) >= 0.5
                                    else np.random.normal(impostazioni["dcbirpn"] / 2, 1.0) for _ in
                                    range(frecce)])).pvalue > alfa:
                accettazioni += 1
    elif alternativa == "n":
        for _ in range(iterazioni):
            if st.shapiro(np.random.lognormal(size=frecce)).pvalue > alfa:
                accettazioni += 1
    elif alternativa == "t":
        for _ in range(iterazioni):
            if st.shapiro(np.random.standard_t(df=impostazioni["gtbirpn"], size=frecce)).pvalue > alfa:
                accettazioni += 1
    return accettazioni / iterazioni


@monitora("airpn_")
def airpn_(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    accettazioni = 0
    for _ in range(impostazioni["iairpn"]):
        if st.shapiro(np.random.normal(size=dati.dati.taglia)).pvalue > impostazioni["airpn"]:
            accettazioni += 1
    return 1 - accettazioni / impostazioni["iairpn"]


@registra("irpn", set(), "test di Shapiro-Wilk", REGISTRO)
@monitora("irpn")
def irpn(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> Test | None:
    if dati.dati.taglia < 2:
        return None
    sw = irpn_(libro, dati, impostazioni)
    if impostazioni["ce"]:
        if sw.valore.accettazione:
            beta, alfa = birpn(libro, dati, impostazioni), None
        else:
            beta, alfa = None, airpn_(libro, dati, impostazioni)
    else:
        beta, alfa = None, None
    return Test("normalità degli IRP", "Shapiro-Wilk", sw, True, alfa, beta)
