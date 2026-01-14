from classi.dati import registra, monitora, REGISTRO, Libro, SessioneGrezza, Test, SuntoTest, Intervallo, Sessione
from letturascrittura.memoria import leggifile
from letturascrittura.percorsi import filesessione
from manipolazione.estrattori import intervallocircolare
import pycircstat as pcs
import numpy as np


def prioricircolari(noninformativa: bool, apiacere: bool, mu: float, kappa: float, alfa: float,
                    beta: float, allenamenti: int, id_: int) -> tuple:
    if noninformativa:
        return 0, 0.2, 2, 1
    elif apiacere:
        return mu, kappa, alfa, beta
    else:
        try:
            if allenamenti < id_:
                raise IndexError
            sessionipriori: list[Sessione] = [leggifile(filesessione(id_ - p - 1)) for p in range(allenamenti)]
            medie = [sessione.libro.contenuto["ma"].valore for sessione in sessionipriori]
            concentrazioni = [sessione.libro.contenuto["k"].valore for sessione in sessionipriori]
        except IndexError:
            return 0, 0.2, 2, 1
        return (pcs.mean(np.array(medie)), pcs.kappa(np.array(medie))[0],
                np.mean(concentrazioni)**2/np.var(concentrazioni), np.mean(concentrazioni)/np.var(concentrazioni))


@monitora("tkamb")
def tkamb(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    fittato = impostazioni["fittatokappa"]
    if impostazioni["atkamb"] == ">":
        return np.sum(fittato <= impostazioni["intkamb"])/fittato.shape
    elif impostazioni["atkamb"] == "=":
        return np.sum(impostazioni["intkamb"]-impostazioni["rtkamb"] <= fittato <= impostazioni["intkamb"]+impostazioni["rtkamb"])/fittato.shape
    elif impostazioni["atkamb"] == "<":
        return np.sum(fittato >= impostazioni["intkamb"])/fittato.shape


@monitora("i_amb")
def i_amb(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> tuple[float, float] | None:
    return intervallocircolare(impostazioni["fittatomu"], impostazioni["ciamb"])


@monitora("ikamb")
def ikamb(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> tuple[float, float] | None:
    return np.quantile(impostazioni["fittatokappa"], (1-impostazioni["cikamb"])/2), np.quantile(impostazioni["fittatokappa"], 1-(1-impostazioni["cikamb"])/2)


@registra("amb", set(), "inferenza bayesiana sull'angolo medio", REGISTRO)
@monitora("amb")
def amb(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> tuple[float | None,
                                                                                    float | None, Intervallo | None, Intervallo | None] | None:
    if dati.dati.taglia < 2:
        return None
    if impostazioni["ambv"]:
        mu0, _, a0, b0 = prioricircolari(impostazioni["niamb"], impostazioni["apamb"], impostazioni["ap_amb"],
                                         impostazioni["mapamb"], impostazioni["kapamb"],
                                         impostazioni["aapamb"], impostazioni["bapamb"], dati.id_)
        risultati = miomodulo.angolomediobayesiano(mu0, a0, b0, dati.dati.angoli.tolist(), dati.dati.taglia)
        emme = risultati[0]
        beta = risultati[1]
        a = risultati[2]
        b = risultati[3]
        fittatokappa = np.random.gamma(a, 1/b, size=10000)
        fittatomu = np.random.vonmises(emme, beta * fittatokappa, size=10000)
    else:
        if impostazioni["ambg"] and impostazioni["apamb"] > 0:
            idattuale = dati.id_
            sessionipriori: list[Sessione] = [leggifile(filesessione(idattuale - p - 1)) for p in
                                              range(impostazioni["aphb"])]
            modello = cmdstanpy.CmdStanModel(exe_file="Angolomediogerarchico.exe")
            fittato = modello.sample(data={"Ncorrente": dati.dati.taglia, "angolicorrenti": dati.dati.angoli.tolist(),
                                           "priori": impostazioni["apamb"],
                                           "Npriori": [sessione.dati.dati.taglia for sessione in sessionipriori],
                                           "gruppipriori": [*[i for _ in range(sessionipriori[i].dati.dati.taglia)] for i in range(len(sessionipriori))],
                                           "angolipriori": [*sessione.dati.dati.angoli for sessione in sessionipriori],
                                           "c": 2, "d": 1, "xi": 2, "zeta": 1}, chains=impostazioni["ncamb"], iter_sampling=impostazioni["iamb"])
        else:
            mu0, kappa0, alfa, beta = prioricircolari(impostazioni["niamb"], impostazioni["apamb"], impostazioni["ap_amb"],
                                         impostazioni["mapamb"], impostazioni["kapamb"],
                                         impostazioni["aapamb"], impostazioni["bapamb"], dati.id_)
            modello = cmdstanpy.CmdStanModel(exe_file="Angolomedio.exe")
            fittato = (modello.sample(
                data={"N": dati.dati.taglia, "y": dati.dati.angoli,
                      "mu0": mu0, "kappa0": kappa0, "alpha": alfa,
                      "beta": beta}, chains=impostazioni["ncamb"], iter_sampling=impostazioni["iamb"]))
        fittatomu = fittato.stan_variable("mu")
        fittatokappa = fittato.stan_variable("kappa")
    impostazioni["fittatokappa"] = fittatokappa
    impostazioni["fittatomu"] = fittatomu
    if impostazioni["tkamb"]:
        testkappa = tkamb(libro, dati, impostazioni)
    else:
        testkappa = None
    if impostazioni["kamb"]:
        kappa = np.mean(fittatokappa)
    else:
        kappa = None
    if impostazioni["i_amb"]:
        intervallo = Intervallo("angolo medio", i_amb(libro, dati, impostazioni), False, True, impostazioni["ciamb"])
    else:
        intervallo = None
    if impostazioni["ikamb"]:
        intervallokappa = Intervallo("parametro di concentrazione", ikamb(libro, dati, impostazioni), False, True,
                                      impostazioni["cikamb"])
    else:
        intervallokappa = None
    del impostazioni["fittatomu"], impostazioni["fittatokappa"]
    return testkappa, kappa, intervallo, intervallokappa


def priorimisturevonmises(noninformativa: bool, allenamentipriori: int, componenti: int, id_: int) -> tuple:
    if noninformativa:
        return ([1 / componenti for _ in range(componenti)], [1 for _ in range(componenti)],
                np.linspace(-np.pi, np.pi, componenti, endpoint=False).tolist(), [0.5 for _ in range(componenti)],
                [0.5 for _ in range(componenti)], None)
    else:
        try:
            if allenamentipriori < id_:
                raise IndexError
            sessionipriori: list[Sessione] = [leggifile(filesessione(id_ - p - 1)) for p in range(allenamentipriori)]
            tuttiangoli = [*(sessione.dati.dati.angoli.tolist()) for sessione in sessionipriori]
        except IndexError:
            return 0, 0.2, 2, 1
        return miomodulo.misturevonmisesvariazionali([1 / componenti for _ in range(componenti)],
                                                     [1 for _ in range(componenti)],
                                                     np.linspace(-np.pi, np.pi, componenti, endpoint=False).tolist(),
                                                     [0.5 for _ in range(componenti)],
                                                     [0.5 for _ in range(componenti)], componenti,
                                                     tuttiangoli, len(tuttiangoli))


@registra("mvm", set(), "modello a mistura di von Mises bayesiano", REGISTRO)
@monitora("mvm")
def mvm(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> tuple | None:
    if dati.dati.taglia < 2:
        return None
    if impostazioni["cfmvm"]:
        alfa0, beta0, m0, a0, b0, _ = priorimisturevonmises(impostazioni["nimvm"], impostazioni["apmvm"],
                                                            impostazioni["cm_mvm"], dati.id_)
        m, beta, a, b, alfa, _, assegnazioni = miomodulo.misturevonmisesvariazionali(alfa0, beta0, m0, a0, b0,
                                                                                     impostazioni["cm_mvm"],
                                                                                     dati.dati.angoli.tolist(),
                                                                                     dati.dati.taglia)
        lambdasimulati = [np.random.gamma(shape, rate, size=10000) for shape, rate in zip(a, b)]
        musimulati = [np.random.vonmises(media, fattore*varianza, size=10000)
                      for media, fattore, varianza in zip(m, beta, lambdasimulati)]
        return ([intervallocircolare(campione, impostazioni["cmmvm"]) for campione in musimulati],
                [(np.quantile(campione, (1 - impostazioni["ckmvm"]) / 2),
                  np.quantile(campione, 1 - (1 - impostazioni["ckmvm"]) / 2))
                 for campione in lambdasimulati], list(zip(m, alfa)), assegnazioni)
    else:
        bic = list()
        parametri = list()
        etichette = list()
        for componenti in range(2, impostazioni["cm_mvm"]+1):
            alfa0, beta0, m0, a0, b0, _ = priorimisturevonmises(impostazioni["nimvm"],
                                                                impostazioni["apmvm"], componenti, dati.id_)
            m, beta, a, b, alfa, bicsingolo, assegnazioni = miomodulo.misturevonmisesvariazionali(alfa0, beta0, m0, a0, b0,
                                                                                                  componenti,
                                                                                                  dati.dati.angoli.tolist(),
                                                                                                  dati.dati.taglia)
            parametri.append([m, beta, a, b, alfa])
            bic.append(bicsingolo[0])
            etichette.append(assegnazioni)
        G = bic.index(min(bic))
        scelti = parametri[G]
        lambdasimulati = [np.random.gamma(shape, 1/rate, size=10000) for shape, rate in zip(scelti[2], scelti[3])]
        musimulati = [np.random.vonmises(media, fattore * varianza, size=10000)
                      for media, fattore, varianza in zip(scelti[0], scelti[1], lambdasimulati)]
        return ([intervallocircolare(campione, impostazioni["cmmvm"]) for campione in musimulati],
                [(np.quantile(campione, (1 - impostazioni["ckmvm"]) / 2),
                  np.quantile(campione, 1 - (1 - impostazioni["ckmvm"]) / 2))
                 for campione in lambdasimulati], [(mediasingola, alfasingolo/sum(scelti[4])) for mediasingola,
                alfasingolo in zip(scelti[0], scelti[4])], etichette[G])
