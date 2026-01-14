import numpy as np
from classi.dati import registra, monitora, REGISTRO, Libro, SessioneGrezza, Test, Intervallo, SuntoTest
from analisi.bayesiana.mediebivariate import parametripriori
from applicazione.SagittariumIuvo import Sessione
from letturascrittura.memoria import leggifile
from letturascrittura.percorsi import filesessione


@monitora("tvab")
def tvab(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    ascisse = impostazioni["fittato"][:, 0]
    alternativaascisse = impostazioni["a_vab"]
    ipotesinullaascisse = impostazioni["invab"]
    ropeascisse = impostazioni["rvab"]
    if alternativaascisse == ">":
        return len(ascisse[ascisse <= ipotesinullaascisse]) / len(ascisse)
    elif alternativaascisse == "=":
        return (len(ascisse[ipotesinullaascisse - ropeascisse <= ascisse <= ipotesinullaascisse + ropeascisse])
                       / len(ascisse))
    elif alternativaascisse == "<":
        return len(ascisse[ascisse >= ipotesinullaascisse]) / len(ascisse)


@monitora("tvob")
def tvob(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    ordinate = impostazioni["fittato"][:, 1]
    alternativaordinate = impostazioni["a_vob"]
    ipotesinullaordinate = impostazioni["invob"]
    ropeordinate = impostazioni["rvob"]
    if alternativaordinate == ">":
        return len(ordinate[ordinate <= ipotesinullaordinate]) / len(ordinate)
    elif alternativaordinate == "=":
        return (len(ordinate[ipotesinullaordinate - ropeordinate <= ordinate <= ipotesinullaordinate + ropeordinate])/ len(ordinate))
    elif alternativaordinate == "<":
        return len(ordinate[ordinate >= ipotesinullaordinate]) / len(ordinate)


@monitora("ivab")
def ivab(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> Intervallo | None:
    ascisse = impostazioni["fittato"][:, 0]
    alfaascisse = 1-impostazioni["cvab"]
    intervalloascisse = (np.quantile(ascisse, alfaascisse / 2), np.quantile(ascisse, 1 - alfaascisse / 2))
    return Intervallo("varianza delle ascisse", intervalloascisse, False, True, impostazioni["cvab"])


@monitora("ivob")
def ivob(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> Intervallo | None:
    ordinate = impostazioni["fittato"][:, 1]
    alfaordinate = 1-impostazioni["cvob"]
    intervalloordinate = [np.quantile(ordinate, alfaordinate / 2), np.quantile(ordinate, 1 - alfaordinate / 2)]
    return Intervallo("varianza delle ordinate", intervalloordinate, False, True, impostazioni["cvob"])


@registra("vb", set(), "inferenza bayesiana sulle varianze delle coordinate", REGISTRO)
@monitora("vb")
def vb(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> tuple[tuple[float | None, Intervallo | None],
                                                                  tuple[float | None, Intervallo | None]] | None:
    if dati.dati.taglia < 2:
        return None
    if impostazioni["vbg"]:
        idattuale = dati.id_
        sessionipriori: list[Sessione] = [leggifile(filesessione(idattuale-p-1)) for p in range(impostazioni["apvb"])]
        numerositàpriori = [sessione.dati.dati.taglia for sessione in sessionipriori]
        datipriori = [*(sessione.dati.dati.xy) for sessione in sessionipriori]
        modello = cmdstanpy.CmdStanModel(exe_file="Varianzebayesianegerarchiche.exe")
        campione = modello.sample(data={"G": impostazioni["apvb"], "Npriori": numerositàpriori,
                                        "datipriori": datipriori,
                                        "group_id": [*[i for _ in range(numerositàpriori[i])] for i in range(impostazioni["apvb"])],
                                        "Ncorrente": dati.dati.taglia, "daticorrenti": dati.dati.xy},
                                 chains=impostazioni["ncvb"], iter_sampling=impostazioni["ivb"])
    else:
        mu0, sigma0, a, b, eta = parametripriori(impostazioni["nivb"], impostazioni["apvb"], impostazioni["ap_vb"],
                                                 impostazioni["mapvb"], impostazioni["sapvb"],
                                                 impostazioni["aapvb"], impostazioni["bapvb"],
                                                 impostazioni["eapvb"], dati.id_)
        modello = cmdstanpy.CmdStanModel(exe_file="Varianzebayesiane.exe")
        campione = modello.sample(
            data={"N": dati.dati.taglia, "dati": dati.dati.xy, "mu0": mu0, "Sigma0": sigma0, "a": a, "b": b,
                  "eta": eta}, chains=impostazioni["ncvb"], iter_sampling=impostazioni["ivb"])
    fittato = np.square(campione.stan_variable("sigma"))
    impostazioni["fittato"] = fittato
    if impostazioni["tvb"]:
        testascisse, testordinate = tvab(libro, dati, impostazioni), tvob(libro, dati, impostazioni)
    else:
        testascisse, testordinate = None, None
    if impostazioni["i_vb"]:
        intervalloascisse, intervalloordinate = ivab(libro, dati, impostazioni), ivob(libro, dati, impostazioni)
    else:
        intervalloascisse, intervalloordinate = None, None
    del impostazioni["fittato"]
    return (testascisse, testordinate), (intervalloascisse, intervalloordinate)
