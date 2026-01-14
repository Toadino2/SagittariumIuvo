import numpy as np
from letturascrittura.memoria import leggifile
from letturascrittura.percorsi import filesessione
from classi.dati import registra, monitora, REGISTRO, Test, Intervallo, SessioneGrezza, Libro, Sessione


def parametrinorme(noninformativa: bool, apiacere: bool, alfaapiacere: float, betaapiacere: float,
                   alfa2apiacere: float, beta2apiacere: float, id_: int, allenamenti: int) -> tuple:
    if noninformativa:
        return 1.0, 1.0, 4.0, 3.0
    elif apiacere:
        return alfaapiacere, betaapiacere, alfa2apiacere, beta2apiacere
    else:
        try:
            if allenamenti < id_:
                raise IndexError
            sessionipriori: list[Sessione] = [leggifile(filesessione(id_ - p - 1)) for p in range(allenamenti)]
            theta = [sessione.libro.contenuto["mirp"].valore/10 for sessione in sessionipriori]
        except IndexError:
            return [0, 0], [[10, 0], [0, 10]], 0.3, 1, 1
        mediatheta = np.mean(theta)
        varianzatheta = np.var(theta)
        alfa = mediatheta * (mediatheta * (1 - mediatheta) / varianzatheta - 1)
        beta = (1 - mediatheta) * (mediatheta * (1 - mediatheta) / varianzatheta - 1)
        beta2 = len(theta)/(len(theta)+1)*np.mean(theta*np.log(theta))-np.mean(theta)*np.mean(np.log(theta))
        alfa2 = np.mean(theta/beta2)
        return alfa, beta, alfa2, 1.0/beta2


@monitora("tirpb")
def tirpb(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> float | None:
    alternativa = impostazioni["a_irpb"]
    fittato = impostazioni["fittato"]
    media = impostazioni["mirpb"]
    rope = impostazioni["rirpb"]
    if alternativa == ">":
        return len(fittato[fittato <= media]) / len(fittato)
    elif alternativa == "=":
        return len(fittato[media - rope <= fittato <= media + rope]) / len(fittato)
    elif alternativa == "<":
        return len(fittato[fittato >= media]) / len(fittato)


@monitora("i_irpb")
def i_irpb(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> tuple[float] | None:
    alternativa = impostazioni["a_irpb"]
    fittato = impostazioni["fittato"]
    media = impostazioni["mirpb"]
    rope = impostazioni["rirpb"]
    credibilità = impostazioni["ciirpb"]
    if alternativa == ">":
        return np.quantile(fittato, (1 - credibilità) / 2), np.quantile(fittato, 1-(1-credibilità) / 2)
    elif alternativa == "=":
        return np.quantile(fittato, (1 - credibilità) / 2), np.quantile(fittato, 1-(1-credibilità) / 2)
    elif alternativa == "<":
        return np.quantile(fittato, (1 - credibilità) / 2), np.quantile(fittato, 1-(1-credibilità) / 2)




@registra("irpb", set(), "inferenza bayesiana sulla media degli IRP", REGISTRO)
@monitora("irpb")
def normebayesiane(libro: Libro, dati: SessioneGrezza, impostazioni: dict) -> tuple[Test | None, Intervallo | None] | None:
    if dati.dati.taglia < 3:
        return None
    if impostazioni["irpbg"] or (idattuale := dati.id_) < (ap := impostazioni["apirpb"]):
        sessionipriori: list[Sessione] = [leggifile(filesessione(idattuale - p - 1)) for p in range(ap)]
        arraypriori = [*sessione.dati.dati.irp for sessione in sessionipriori]
        tagliepriori = [sessione.dati.dati.taglia for sessione in sessionipriori]
        modello = cmdstanpy.CmdStanModel(exe_file="Normebayesianegerarchiche.exe")
        fittato = modello.sample(data={"G": ap, "N": sum(tagliepriori), "y": arraypriori,
                                       "group_id": [*[i for _ in range(tagliepriori[i])] for i in range(ap)],
                                       "Ncorrente": dati.dati.taglia,
                                       "ycorrente": dati.dati.irp.tolist()},
                                 chains=impostazioni["ncirpb"], iter_sampling=impostazioni["iirpb"]).stan_variable("mucorrente")
    else:
        alfa, beta, alfa2, beta2 = parametrinorme(impostazioni["niirpb"], impostazioni["ap_irpb"], impostazioni["aapirpb"],
                                                  impostazioni["bapirpb"], impostazioni["a2apirpb"], impostazioni["b2apirpb"],
                                                  idattuale, ap)
        modello = cmdstanpy.CmdStanModel(exe_file="Normebayesiane.exe")
        fittato = modello.sample(
            data={"N": dati.dati.taglia, "y": dati.dati.irp.tolist(),
                  "alpha": alfa, "beta": beta, "alpha2": alfa2,
                  "beta2": beta2}, chains=impostazioni["ncirpb"], iter_sampling=impostazioni["iirpb"]).stan_variable("mu")
    impostazioni["fittato"] = fittato
    if impostazioni["thb"]:
        test = tirpb(libro, dati, impostazioni)
    else:
        test = None
    if impostazioni["i_hb"]:
        intervallo = i_irpb(libro, dati, impostazioni)
    else:
        intervallo = None
    del impostazioni["fittato"]
    test_ = Test("media degli IRP", "MCMC", test, False, None, None, True)
    intervallo_ = Intervallo("media degli IRP", intervallo, False, True, impostazioni["ciirpb"])
    return test_, intervallo_
