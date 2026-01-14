from classi.dati import registra, monitora, REGISTRO, Test, Intervallo, Libro, SessioneGrezza, Sessione
from letturascrittura.memoria import leggifile
from letturascrittura.percorsi import filesessione
import numpy as np
import scipy.stats as st


def parametripriori(noninformativa: bool, allenamenti: int, apiacere: bool, mu0: list[float],
                    lambda0: list[list[float]], a0: float, b0: float, eta: float, id_: int):
    if noninformativa:
        return [0, 0], [[10, 0], [0, 10]], 0.3, 1, 1
    elif apiacere:
        return mu0, lambda0, a0, b0, eta
    else:
        try:
            if allenamenti < id_:
                raise IndexError
            sessionipriori: list[Sessione] = [leggifile(filesessione(id_ - p - 1)) for p in range(allenamenti)]
            mediepriori = [sessione.libro.contenuto["mc"].valore for sessione in sessionipriori]
            varianzepriori = [np.diag(sessione.libro.contenuto["vc"].valore) for sessione in sessionipriori]
            correlazionipriori = [sessione.libro.contenuto["cor"].valore for sessione in sessionipriori]
        except IndexError:
            return [0, 0], [[10, 0], [0, 10]], 0.3, 1, 1
        mu0 = np.mean(np.array(mediepriori), axis=0)
        lambda0 = np.cov(np.array(mediepriori).T)
        b_ = np.mean(varianzepriori * np.log(varianzepriori)) * np.mean(varianzepriori) * np.mean(
            np.log(varianzepriori))
        a_ = np.mean(varianzepriori) / b_
        b0 = len(varianzepriori) / (len(varianzepriori) - 1) * b_
        a0 = a_ - 1 / len(varianzepriori) * (3 * a_ - 2 / 3 * a_ / (1 + a_) - 0.8 * a_ / ((1 + a_) ** 2))
        if all([correlazione < 0.2 for correlazione in correlazionipriori]):
            eta = 2
        elif any([0.2 < correlazione < 0.7 for correlazione in correlazionipriori]):
            eta = 1
        else:
            eta = 0.5
        return mu0, lambda0, a0, 1.0/b0, eta


@monitora("thb")
def thb(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    if impostazioni["thb"]:
        maschera = (impostazioni["fittato"][:, 0] ** 2 + impostazioni["fittato"][:, 1] ** 2) <= impostazioni["rhb"] ** 2
        probabilitànulla = np.sum(maschera) / impostazioni["fittato"].shape[0]
    else:
        probabilitànulla = None
    return probabilitànulla


@monitora("i_hb")
def i_hb(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> tuple | None:
    if impostazioni["i_hb"]:
        if impostazioni["iphb"]:
            kernel = st.gaussian_kde(impostazioni["fittato"].T)
            grigliax = np.linspace(-2, 2, 1000)
            grigliay = np.linspace(-2, 2, 1000)
            x, y = np.meshgrid(grigliax, grigliay)
            posizioni = np.vstack([x.ravel(), y.ravel()])
            z = kernel(posizioni).reshape(x.shape)
            zordinata = np.sort(z.ravel())[::-1]
            cumulata = np.cumsum(zordinata)
            cumulata /= cumulata[-1]
            livello = zordinata[np.searchsorted(cumulata, impostazioni["chb"])]
        else:
            mediafittato = np.mean(impostazioni["fittato"], axis=0)
            covarianzefittato = np.cov(impostazioni["fittato"], rowvar=False, bias=False)
            covarianzeinverse = np.linalg.inv(covarianzefittato)
            x, y = np.meshgrid(np.linspace(-2, 2, 1000), np.linspace(-2, 2, 1000))
            differenze = np.stack([x.ravel(), y.ravel()], axis=1) - mediafittato
            z = np.einsum("ij,jk,ik->i", differenze, covarianzeinverse, differenze).reshape(x.shape)
            livello = st.chi2.ppf(impostazioni["chb"], df=2)
    else:
        x = None
        y = None
        z = None
        livello = None
    return x, y, z, livello


@registra("hb", set(), "inferenza bayesiana sulla media bivariata", REGISTRO)
@monitora("hb")
def hb(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> tuple[Test | None, Intervallo | None] | None:
    if dati.dati.taglia < 3:
        return None
    if impostazioni["hbg"] and impostazioni["aphb"] > 0:
        idattuale = dati.id_
        sessionipriori: list[Sessione] = [leggifile(filesessione(idattuale - p - 1)) for p in
                                          range(impostazioni["aphb"])]
        numerositàpriori = [sessione.dati.dati.taglia for sessione in sessionipriori]
        osservazionipriori = np.array([*(np.unstack(sessione.dati.dati.xy, axis=0)) for sessione in sessionipriori])
        indicipriori = [*[p for _ in range(sessionipriori[p].dati.dati.taglia)] for p in range(impostazioni["aphb"])]
        modello = cmdstanpy.CmdStanModel(exe_file="Hotellingbayesianogerarchico.exe")
        fittato = modello.sample(data={"G": impostazioni["aphb"], "N": numerositàpriori, "y_prior": osservazionipriori,
                                       "prior_group": indicipriori, "N_current": dati.dati.taglia,
                                       "y": dati.dati.xy},
                                 chains=impostazioni["nchb"], iter_sampling=impostazioni["ihb"]).stan_variable("mu")
    else:
        mu0, lambda0, a0, b0, eta = parametripriori(impostazioni["nihb"], impostazioni["aphb"], impostazioni["phb"],
                                                    impostazioni["mphb"], impostazioni["lphb"], impostazioni["a_phb"],
                                                    impostazioni["bphb"], impostazioni["ephb"], dati.id_)
        modello = cmdstanpy.CmdStanModel(exe_file="Hotellingbayesiano.exe")
        fittato = modello.sample(data={"N": dati.dati.taglia, "y": dati.dati.xy, "mu0": mu0, "lambda0": lambda0,
                                       "a0": a0, "b0": b0, "eta": eta},
                                 chains=impostazioni["nchb"], iter_sampling=impostazioni["ihb"]).stan_variable("mu")
    impostazioni["fittato"] = fittato
    if impostazioni["thb"]:
        test = thb(libro, dati, impostazioni)
    else:
        test = None
    if impostazioni["i_hb"]:
        intervallo = i_hb(libro, dati, impostazioni)
    else:
        intervallo = None
    del impostazioni["fittato"]
    test_ = Test("media bivariata", "MCMC", test, False, None, None, True)
    intervallo_ = Intervallo("media bivariata", intervallo, False, True, impostazioni["chb"])
    return test_, intervallo_
