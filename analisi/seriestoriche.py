from classi.dati import registra, monitora_, FALDONE, Libro, Sessione, SessioneGrezza, Test, SuntoTest, ConfrontoPeriodi, Dati
import numpy as np
import pycircstat as pcs
import miomodulo
from datetime import date
from scipy.linalg import fractional_matrix_power
from sklearn.kernel_ridge import KernelRidge
from scipy.stats import f as fisher, chi2
from analisi.descrittive import autocorrelazioni, rc, ac
from analisi.frequentista.mediebivariate import th, ih
from analisi.frequentista.irp import tirp, iirp
from analisi.frequentista.varianze import iva, ivo
from analisi.clustering import cl


@registra("ds", set(), "Indici temporali", FALDONE)
@monitora_("ds")
def ds(libro: Libro, dati: list[Sessione], impostazioni: dict) -> np.ndarray:
    return np.array([(s.dati.metadati.data-dati[0].dati.metadati.data).days for s in dati])


@registra("ns", set(), "Serie storica delle dimensioni delle sessioni", FALDONE)
@monitora_("ns")
def ns(libro: Libro, dati: list[Sessione], impostazioni: dict) -> np.ndarray:
    return np.array([[s.dati.dati.taglia for s in dati], libro.contenuto["ds"].valore])


@registra("mps", {"ds"}, "Serie storica dei punteggi medi", FALDONE)
@monitora_("mps")
def mps(libro: Libro, dati: list[Sessione], impostazioni: dict) -> np.ndarray:
    return np.array([[media, libro.contenuto["ds"].valore[s]] for s in range(len(dati)) if (media := dati[s].libro.contenuto["mp"].valore) is not None])


@registra("mmps", {"mps"}, "Medie mobili dei punteggi medi", FALDONE)
@monitora_("mmps")
def mmps(libro: Libro, dati: list[Sessione], impostazioni: dict) -> np.ndarray:
    return np.convolve(libro.contenuto["mps"].valore, np.ones(len(dati)), mode="valid")


@registra("mpfs", {"ds"}, "Serie storiche per ogni freccia dei punteggi medi", FALDONE)
@monitora_("mpfs")
def mpfs(libro: Libro, dati: list[Sessione], impostazioni: dict) -> dict[int, np.ndarray]:
    frecce = max(s.dati.dati.frecce for s in dati)+1
    return {f: np.array([[media, libro.contenuto["ds"].valore[s]] for s in range(len(dati)) if f < dati[s].dati.dati.frecce and (media := dati[s].libro.contenuto["mpf"].valore[f]) is not None]) for f in range(frecce)}


@registra("mirps", {"ds"}, "Serie storica degli IRP medi", FALDONE)
@monitora_("mirps")
def mirps(libro: Libro, dati: list[Sessione], impostazioni: dict) -> np.ndarray:
    return np.array([[media, libro.contenuto["ds"].valore[s]] for s in range(len(dati)) if (media := dati[s].libro.contenuto["mirp"].valore) is not None])


@registra("mmirps", {"mirps"}, "Medie mobili degli IRP medi", FALDONE)
@monitora_("mmirps")
def mmirps(libro: Libro, dati: list[Sessione], impostazioni: dict) -> np.ndarray:
    return np.convolve(libro.contenuto["mirps"].valore, np.ones(len(dati)), mode="valid")


@registra("mirpfs", {"ds"}, "Serie storiche per ogni freccia degli IRP medi", FALDONE)
@monitora_("mirpfs")
def mirpfs(libro: Libro, dati: list[Sessione], impostazioni: dict) -> dict[int, np.ndarray]:
    frecce = max(s.dati.dati.frecce for s in dati)+1
    return {f: np.array([[media, libro.contenuto["ds"].valore[s]] for s in range(len(dati)) if f < dati[s].dati.dati.frecce and (media := dati[s].libro.contenuto["mirpf"].valore[f]) is not None]) for f in range(frecce)}


@registra("virps", {"ds"}, "Serie storica delle varianze degli IRP", FALDONE)
@monitora_("virps")
def virps(libro: Libro, dati: list[Sessione], impostazioni: dict) -> np.ndarray:
    return np.array([[varianza, libro.contenuto["ds"].valore[s]] for s in range(len(dati)) if (varianza := dati[s].libro.contenuto["virp"].valore) is not None])


@registra("mcs", {"ds"}, "Serie storiche delle ascisse e delle ordinate", FALDONE)
@monitora_("mcs")
def mcs(libro: Libro, dati: list[Sessione], impostazioni: dict) -> np.ndarray:
    return np.array([[media[0], media[1], libro.contenuto["ds"].valore[s]] for s in range(len(dati)) if (media := dati[s].libro.contenuto["mc"].valore) is not None])


@registra("mmcs", {"mcs"}, "Medie mobili delle ascisse e delle ordinate", FALDONE)
@monitora_("mmcs")
def mmcs(libro: Libro, dati: list[Sessione], impostazioni: dict) -> np.ndarray:
    grezze = libro.contenuto["mcs"].valore
    return np.array([np.convolve(grezze[:, 0], np.ones(len(dati)), mode="valid"), np.convolve(grezze[:, 1], np.ones(len(dati), mode="valid"))]).T


@registra("mcfs", {"ds"}, "Serie storiche per ogni freccia delle ascisse e delle ordinate", FALDONE)
@monitora_("mcfs")
def mcfs(libro: Libro, dati: list[Sessione], impostazioni: dict) -> dict[int, np.ndarray]:
    frecce = max(s.dati.dati.frecce for s in dati)+1
    return {f: np.array([[media[0], media[1], libro.contenuto["ds"].valore[s]] for s in range(len(dati)) if f < dati[s].dati.dati.frecce and (media := dati[s].libro.contenuto["mcf"].valore[f]) is not None]) for f in range(frecce)}


@registra("vcs", {"ds"}, "Serie storiche delle matrici di covarianze", FALDONE)
@monitora_("vcs")
def vcs(libro: Libro, dati: list[Sessione], impostazioni: dict) -> np.ndarray:
    return np.array([[*(matrice.flatten()), libro.contenuto["ds"].valore[s]] for s in range(len(dati)) if (matrice := dati[s].libro.contenuto["vc"].valore) is not None])


@registra("vcfs", {"ds"}, "Serie storiche per ogni freccia delle matrici di covarianze", FALDONE)
@monitora_("vcfs")
def vcfs(libro: Libro, dati: list[Sessione], impostazioni: dict) -> dict[int, np.ndarray]:
    frecce = max(s.dati.dati.frecce for s in dati)+1
    return {f: np.array([[*(matrice.flatten()), libro.contenuto["ds"].valore[s]] for s in range(len(dati)) if f < dati[s].dati.dati.frecce and (matrice := dati[s].libro.contenuto["vcf"].valore) is not None]) for f in range(frecce)}


@registra("mas", {"ds"}, "Serie storica degli angoli medi", FALDONE)
@monitora_("mas")
def mas(libro: Libro, dati: list[Sessione], impostazioni: dict) -> np.ndarray:
    return np.array([[media, libro.contenuto["ds"].valore[s]] for s in range(len(dati)) if (media := dati[s].libro.contenuto["ma"].valore) is not None])


@registra("vas", {"ds"}, "Serie storica delle lunghezze risultanti medie", FALDONE)
@monitora_("vas")
def vas(libro: Libro, dati: list[Sessione], impostazioni: dict) -> np.ndarray:
    return np.array([[r, libro.contenuto["ds"].valore[s]] for s in range(len(dati)) if (r := dati[s].libro.contenuto["va"].valore) is not None])


@registra("ks", {"ds"}, "Serie storica dei parametri di concentrazioni", FALDONE)
@monitora_("ks")
def ks(libro: Libro, dati: list[Sessione], impostazioni: dict) -> np.ndarray:
    return np.array([[k, libro.contenuto["ds"].valore[s]] for s in range(len(dati)) if (k := dati[s].libro.contenuto["k"].valore) is not None])


@registra("mpc", set(), "Media complessiva dei punteggi", FALDONE)
@monitora_("mpc")
def mpc(libro: Libro, dati: list[Sessione], impostazioni: dict) -> float:
    return np.mean(np.concatenate([s.dati.dati.punteggi for s in dati]))


@registra("mirpc", set(), "Media complessiva degli IRP", FALDONE)
@monitora_("mirpc")
def mirpc(libro: Libro, dati: list[Sessione], impostazioni: dict) -> float:
    return np.mean(np.concatenate([s.dati.dati.irp for s in dati]))


@registra("mcc", set(), "Medie complessive di ascisse e ordinate", FALDONE)
@monitora_("mcc")
def mcc(libro: Libro, dati: list[Sessione], impostazioni: dict) -> np.ndarray:
    return np.mean(np.concatenate([s.dati.dati.xy for s in dati]), axis=0)


@registra("vcc", set(), "Matrice di covarianze complessiva di ascisse e ordinate", FALDONE)
@monitora_("vcc")
def vcc(libro: Libro, dati: list[Sessione], impostazioni: dict) -> np.ndarray:
    return np.var(np.concatenate([s.dati.dati.xy for s in dati]), axis=0)


@registra("vac", set(), "Lunghezza risultante media complessiva", FALDONE)
@monitora_("vac")
def vac(libro: Libro, dati: list[Sessione], impostazioni: dict) -> float:
    angoli = np.concatenate([s.dati.dati.angoli for s in dati])
    return (np.mean(np.cos(angoli))**2+np.mean(np.sin(angoli))**2)**0.5


@registra("kc", set(), "Parametro di concentrazione complessivo", FALDONE)
@monitora_("kc")
def kc(libro: Libro, dati: list[Sessione], impostazioni: dict) -> float:
    return pcs.kappa(np.concatenate([s.dati.dati.angoli for s in dati]))


@registra("mpc", set(), "Media complessiva dei punteggi per ogni freccia", FALDONE)
@monitora_("mpc")
def mpcf(libro: Libro, dati: list[Sessione], impostazioni: dict) -> dict[int, float]:
    frecce = max(s.dati.dati.frecce for s in dati) + 1
    return {f: np.mean(np.concatenate([s.dati.dati.punteggi[s.dati.dati.idfrecce == f] for s in dati])) for f in range(frecce)}


@registra("mirpc", set(), "Media complessiva degli IRP", FALDONE)
@monitora_("mirpc")
def mirpc(libro: Libro, dati: list[Sessione], impostazioni: dict) -> dict[int, float]:
    frecce = max(s.dati.dati.frecce for s in dati) + 1
    return {f: np.mean(np.concatenate([s.dati.dati.irp[s.dati.dati.idfrecce == f] for s in dati])) for f in range(frecce)}


@registra("mcc", set(), "Medie complessive di ascisse e ordinate", FALDONE)
@monitora_("mcc")
def mcc(libro: Libro, dati: list[Sessione], impostazioni: dict) -> dict[int, np.ndarray]:
    frecce = max(s.dati.dati.frecce for s in dati) + 1
    return {f: np.mean(np.concatenate([s.dati.dati.xy[s.dati.dati.idfrecce == f] for s in dati]), axis=0) for f in range(frecce)}


@registra("vcc", set(), "Matrice di covarianze complessiva di ascisse e ordinate", FALDONE)
@monitora_("vcc")
def vcc(libro: Libro, dati: list[Sessione], impostazioni: dict) -> dict[int, np.ndarray]:
    frecce = max(s.dati.dati.frecce for s in dati) + 1
    return {f: np.var(np.concatenate([s.dati.dati.xy[s.dati.dati.idfrecce == f] for s in dati]), axis=0) for f in range(frecce)}


@registra("vac", set(), "Lunghezza risultante media complessiva", FALDONE)
@monitora_("vac")
def vac(libro: Libro, dati: list[Sessione], impostazioni: dict) -> dict[int, float]:
    frecce = max(s.dati.dati.frecce for s in dati) + 1
    angoli = {f: np.concatenate([s.dati.dati.angoli[s.dati.dati.idfrecce == f] for s in dati]) for f in range(frecce)}
    return {f: (np.mean(np.cos(angoli[f]))**2+np.mean(np.sin(angoli[f]))**2)**0.5 for f in range(frecce)}


@registra("kc", set(), "Parametro di concentrazione complessivo", FALDONE)
@monitora_("kc")
def kc(libro: Libro, dati: list[Sessione], impostazioni: dict) -> dict[int, float]:
    frecce = max(s.dati.dati.frecce for s in dati) + 1
    return {f: pcs.kappa(np.concatenate([s.dati.dati.angoli[s.dati.dati.idfrecce == f] for s in dati])) for f in range(frecce)}


@registra("cps", set(), "Correlazione circolare-lineare (al quadrato) tra IRP e angoli", FALDONE)
@monitora_("cps")
def cps(libro: Libro, dati: list[Sessione], impostazioni: dict) -> np.ndarray:
    correlazioni = list()
    for s in dati:
        coseni = np.cos(s.dati.dati.angoli)
        seni = np.cos(s.dati.dati.angoli)
        rxc = np.corrcoef(s.dati.dati.irp, coseni)
        rxs = np.corrcoef(s.dati.dati.irp, seni)
        rcs = np.corrcoef(coseni, seni)
        correlazioni.append((rxc**2+rxs**2-2*rxc*rxs*rcs)/(1-rcs**2))
    return np.array(correlazioni)


def matricecovariate(tuttitempi: np.ndarray, tempo: np.ndarray, dati: list[Sessione], impostazioni: dict) -> np.ndarray:
    selezionandi = np.isin(tuttitempi, tempo)
    volée = np.array([s.dati.dati.volée for s in dati])[selezionandi]
    frecce = np.array([s.dati.dati.taglia for s in dati])[selezionandi] / volée
    distanza = np.array([s.dati.metadati.distanza for s in dati])[selezionandi]
    archi = (np.array(archi_ := [s.dati.metadati.arco for s in dati]) == np.unique(archi_)[:, None]).astype(int)[:,
            selezionandi]
    tag_ = [s.dati.metadati.tag for s in dati]
    _tag_ = np.unique(np.concatenate(tag_))
    tag = np.array([[_tag in gruppo for gruppo in tag_] for _tag in _tag_ if
                    impostazioni[f"ols_{_tag}"]])[:, selezionandi]
    spazio = np.array(
        [(impostazioni["dpa"] if s == 0 else dati[s + 1].dati.metadati.data - dati[s].dati.metadati.data for s in
         range(len(dati))])[:, selezionandi]
    concatenandi = [[1 for _ in range(len(dati))]]
    nomi = ["intercetta"]
    if impostazioni["tols"]: concatenandi.append(tempo); nomi.append("t")
    if impostazioni["vols"]: concatenandi.append(volée); nomi.append("v")
    if impostazioni["fols"]: concatenandi.append(frecce); nomi.append("f")
    if impostazioni["dols"]: concatenandi.append(distanza); nomi.append("d")
    if impostazioni["aols"]:
        concatenandi.append(archi)
        for arco in archi_: nomi.append(arco.nomesetup)
    if impostazioni["sols"]: concatenandi.append(spazio)
    if tag.shape[0] > 0:
        concatenandi.append(tag)
        for _tag in _tag_: nomi.append(_tag)
    return np.concatenate(concatenandi), nomi


@registra("ols", {"mirps", "virps", "mas", "vas", "cps", "mcs", "vcs", "ds"}, "Stimatori OLS", FALDONE)
@monitora_("ols")
def ols(libro: Libro, dati: list[Sessione], impostazioni: dict) -> dict[str, np.ndarray] | None:
    if len(dati) < 2:
        return None
    coefficienti = dict()
    speciali = {"mcso": libro.contenuto["mcs"].valore[:, 1],
                "vcso": libro.contenuto["vcs"].valore[:, 3], "vcsc": libro.contenuto["vcs"].valore[:, 1]}
    for variabile in ["mirps", "virps", "mas", "vas", "cps", "mcsa", "mcso", "vcsa", "vcso", "vcsc"]:
        if variabile in speciali:
            risposta = speciali[variabile]
        else:
            risposta = libro.contenuto[variabile].valore[:, 0]
        tempo = libro.contenuto[variabile].valore[:, -1]
        tuttitempi = libro.contenuto["ds"].valore
        covariate, _ = matricecovariate(tuttitempi, tempo, dati, impostazioni)[-[0], :]
        if covariate.shape[0] < 1: return None
        coefficienti[variabile] = np.linalg.inv(covariate @ covariate.T) @ covariate @ risposta.T
    return coefficienti


@monitora_("_lbt")
def _lbt(libro: Libro, dati: list[Sessione], impostazioni: dict) -> SuntoTest:
    serie = impostazioni["_lbt"]
    regressione = np.polyfit(serie[:, 1], serie[:, 0], 1)
    previsioni = serie[:, 1]*regressione[0]+regressione[1]
    residui = serie[:, 0]-previsioni
    autocorrelazioni_ = autocorrelazioni(residui)
    test = miomodulo.ljungbox(autocorrelazioni_.tolist(), len(dati), impostazioni["hlbt"], impostazioni["albt"])
    return SuntoTest(test.statistica, test.pvalue, test.accettazione)


@monitora_("lbt_")
def lbt_(libro: Libro, dati: list[Sessione], impostazioni: dict) -> SuntoTest:
    serie = impostazioni["_lbt"]
    regressione = miomodulo.regressionecircolare(serie[:, 1].tolist(), serie[:, 0].tolist(), serie.shape[0])
    previsioni = 2*np.arctan(regressione[0]*serie[:, 1]+regressione[1])
    residui = np.atan2(np.sin(serie[:, 0]-previsioni), np.cos(serie[:, 0]-previsioni))
    autocorrelazioni_ = autocorrelazioni(residui)
    test = miomodulo.ljungbox(autocorrelazioni_.tolist(), len(dati), impostazioni["hlbt"], impostazioni["albt"])
    return SuntoTest(test.statistica, test.pvalue, test.accettazione)


@registra("lbt", {"mps", "vps", "mas", "vas", "mcs", "vcs", "cps"}, "Test di Ljung-Box sulle variabili risposta", FALDONE)
@monitora_("lbt")
def lbt(libro: Libro, dati: list[Sessione], impostazioni: dict) -> dict[str, Test] | None:
    if len(dati) < 3:
        return None
    risposte = {variabile: libro.contenuto[variabile].valore for variabile in {"mps", "vps", "vas", "mcs", "vcs", "cps"}}
    test = dict()
    for risposta in risposte:
        if risposta == "mcs":
            impostazioni["_lbt"] = risposte[risposta][:, [0, 2]]
            test["mcsa"] = _lbt(libro, dati, impostazioni)
            impostazioni["_lbt"] = risposte[risposta][:, [1, 2]]
            test["mcso"] = _lbt(libro, dati, impostazioni)
        elif risposta == "vcs":
            impostazioni["_lbt"] = risposte[risposta][:, [0, 4]]
            test["vcsa"] = _lbt(libro, dati, impostazioni)
            impostazioni["_lbt"] = risposte[risposta][:, [3, 4]]
            test["vcso"] = _lbt(libro, dati, impostazioni)
            impostazioni["_lbt"] = risposte[risposta][:, [1, 4]]
            test["vcsc"] = _lbt(libro, dati, impostazioni)
        else:
            impostazioni["_lbt"] = risposte[risposta]
            test[risposta] = _lbt(libro, dati, impostazioni)
    impostazioni["_lbt"] = libro.contenuto["mas"].valore
    test["mas"] = lbt_(libro, dati, impostazioni)
    del impostazioni["_lbt"]
    return test


@registra("cm", {"ds"}, "Coefficienti di correlazione multipla e variance inflation factor", FALDONE)
@monitora_("cm")
def cm(libro: Libro, dati: list[Sessione], impostazioni: dict) -> dict[str, tuple[float, float]] | None:
    if len(dati) < 2:
        return None
    covariate, nomicovariate = matricecovariate(libro.contenuto["ds"].valore, libro.contenuto["ds"].valore, dati, impostazioni)
    if len(nomicovariate) < 1:
        return None
    coefficienti = dict()
    for covariata in range(len(nomicovariate)):
        esaminanda = covariate[covariata, :]
        confrontande = covariate[-[covariata], :]
        RXy = np.corrcoef(esaminanda, confrontande)
        RXX = np.corrcoef(confrontande, confrontande)
        coefficiente = RXy@np.linalg.pinv(RXX)@RXy.T
        if abs(coefficiente) < 1:
            coefficienti[nomicovariate[covariata]] = (coefficiente, 1/(1-coefficiente))
        else:
            coefficienti[nomicovariate[covariata]] = (1, float("inf"))
    return coefficienti


'''
@registra("ar", {"mirps", "virps", "mas", "vas", "cps", "mcs", "vcs", "ds"}, "Analisi della ridondanza", FALDONE)
@monitora_("ar")
def ar(libro: Libro, dati: list[Sessione], impostazioni: dict) -> dict[str, np.ndarray] | None:
    if len(dati) < 2:
        return None
    coefficienti = dict()
    speciali = {"mcso": libro.contenuto["mcs"].valore[:, 1],
                "vcso": libro.contenuto["vcs"].valore[:, 3], "vcsc": libro.contenuto["vcs"].valore[:, 1]}
    risposte = list()
    for variabile in ["mirps", "virps", "vas", "cps", "mcsa", "mcso", "vcsa", "vcso", "vcsc"]:
        if variabile in speciali and impostazioni[f"ar_{variabile}"]:
            risposta = speciali[variabile]
        elif impostazioni[f"ar_{variabile}"]:
            risposta = libro.contenuto[variabile].valore[:, 0]
        tempo = libro.contenuto[variabile].valore[:, -1]
        tuttitempi = libro.contenuto["ds"].valore
        X, _ = matricecovariate(tuttitempi, tempo, dati, impostazioni)[-[0], :]
        if X.shape[0]: return None
        radice = fractional_matrix_power(X.T@X, -0.5)
        P, _, _ = np.linalg.svd(radice@X.T@Y/len(dati)**0.5)
        B = radice@P[:, impostazioni["cpar"]]
        W = B.T@X.T@Y/len(dati)
        coefficienti[variabile] = np.linalg.inv(covariate @ covariate.T) @ covariate @ risposta.T
    return coefficienti
    

def aem():
    pass
    
def iols():
    pass

@registra("rnp", {"mirps", "virps", "mas", "vas", "cps", "mcs", "vcs", "ds"}, "Regressione kernel", FALDONE)
@monitora_("rnp")
def rnp(libro: Libro, dati: list[Sessione], impostazioni: dict):
    pass
'''


def ihcp(libro: Libro, dati: SessioneGrezza, impostazioni: dict):
    nuoveimpostazioni = {"cdih": impostazioni["cdihcp"], "gsih": impostazioni["gsihcp"],
                         "bsih": impostazioni["bsihcp"], "cih": impostazioni["cihcp"],
                         "lbmih": impostazioni["lbmih"], "bmih": impostazioni["bmihcp"],
                         "vabsih": impostazioni["vabsihcp"], "vobsih": impostazioni["vobsihcp"],
                         "ibsih": impostazioni["ibsihcp"], "vabmih": impostazioni["vabmihcp"],
                         "vobmih": impostazioni["vobmihcp"], "ibmih": impostazioni["ibmihcp"]}
    libro.contenuto["rc"] = rc(libro, dati, nuoveimpostazioni)
    return ih(libro, dati, nuoveimpostazioni)

'''
def thcp(libro: Libro, dati1: SessioneGrezza, dati2: SessioneGrezza, impostazioni: dict):
    nuoveimpostazioni = {"ce": impostazioni["cecp"], "ath": impostazioni["athcp"], "ith": impostazioni["ithcp"],
                         "dbth": impostazioni["dbthcp"]}
    libro.contenuto["ac"] = ac(libro, dati, nuoveimpostazioni)
    return th(libro, dati, nuoveimpostazioni)
'''

def itcp(libro: Libro, dati: SessioneGrezza, impostazioni: dict):
    nuoveimpostazioni = {"ciirp": impostazioni["ciirpcp"]}
    return iirp(libro, dati, nuoveimpostazioni)

'''
def ttcp(libro: Libro, dati: SessioneGrezza, impostazioni: dict):
    nuoveimpostazioni = {"ce": impostazioni["cecp"], "atirp": impostazioni["atirpcp"],
                         "a_tirp": impostazioni["a_tirpcp"], "ibtirp": impostazioni["ibtirpcp"],
                         "mirp": impostazioni["mirpcp"], "dbtirp": impostazioni["dbtirpcp"],
                         "mtirp": impostazioni["mtirpcp"]}
    return tirp(libro, dati, nuoveimpostazioni)
'''

def ivcp(libro: Libro, dati: SessioneGrezza, impostazioni: dict):
    nuoveimpostazioni = {"cva": impostazioni["cvacp"], "cvo": impostazioni["cvocp"]}
    return iva(libro, dati, nuoveimpostazioni), ivo(libro, dati, nuoveimpostazioni)


@monitora_("anovacp")
def anovacp(libro: Libro, dati: SessioneGrezza, impostazioni: dict):
    if f_ := dati.dati.idfrecce is None:
        return None
    xy = dati.dati.xy
    xyf = [xy[f_ == f] for f in range(dati.dati.frecce)]
    _xyf = np.array([np.mean(f, axis=1) for f in xyf])
    ssb = np.sum(np.square(_xyf-np.mean(xy, axis=1)))
    ssw = np.sum(np.square(np.array([xyf[f]-_xyf[f] for f in range(dati.dati.frecce)])))
    dfb = dati.dati.frecce-1
    dfw = dati.dati.taglia-dati.dati.frecce
    msb = ssb/dfb
    msw = ssw/dfw
    F = msb/msw
    p = fisher.sf(F, dfb, dfw)
    return {"SS": (ssb, ssw), "df": (dfb, dfw), "MS": (msb, msw), "F": F, "p": p}


@monitora_("lvcp")
def lvcp(libro: Libro, dati: SessioneGrezza, impostazioni: dict):
    if f_ := dati.dati.idfrecce is None:
        return None
    xy = dati.dati.xy
    xyf = [xy[f_ == f] for f in range(dati.dati.frecce)]
    _xyf = np.array([np.mean(f, axis=1) for f in xyf])
    zf = np.abs(xyf-_xyf)
    _zf = np.array([np.mean(f, axis=1) for f in zf])
    ssb = np.sum(np.square(_xyf-np.mean(zf.reshape(-1, 2), axis=1)))
    ssw = np.sum(np.square(np.array([zf[f]-_zf[f] for f in range(dati.dati.frecce)])))
    dfb = dati.dati.frecce-1
    dfw = dati.dati.taglia-dati.dati.frecce
    msb = ssb/dfb
    msw = ssw/dfw
    F = msb/msw
    p = fisher.sf(F, dfb, dfw)
    return {"SS": (ssb, ssw), "df": (dfb, dfw), "MS": (msb, msw), "F": F, "p": p}


def clcp(libro: Libro, dati: SessioneGrezza, impostazioni: dict):
    nuoveimpostazioni = {"icl": impostazioni["iclcp"], "ikcl": impostazioni["ikclcp"], "cmcl": impostazioni["cmcl"],
                         "scl": impostazioni["sclcp"], "cbcl": impostazioni["cbclcp"], "abcl": impostazioni["abclcp"],
                         "iicl": impostazioni["iiclcp"], "iecl": impostazioni["ieclcp"], "ncl": impostazioni["nclcp"],
                         "acl": impostazioni["aclcp"], "sccl": impostazioni["scclcp"], "cscl": impostazioni["cscl"],
                         "bicl": impostazioni["biclcp"], "fcl": impostazioni["fclcp"]}
    return cl(libro, dati, nuoveimpostazioni)


@monitora_("_cqtcp")
def _cqtcp(libro: Libro, dati: SessioneGrezza, impostazioni: dict):
    n_ = impostazioni["cqtcpn"]
    giorni = np.array([*[i for _ in range(n_[i])] for i in range(len(n_))])
    cluster = np.array(impostazioni["clucp"][0].valore)
    contingenze = np.array(
        [[np.sum(giorni == g_ & cluster == c_) for g_ in range(len(n_))] for c_ in range(np.max(cluster))])
    sommerighe = np.sum(contingenze, axis=1)
    sommecolonne = np.sum(contingenze, axis=0)
    n = sum(n_)
    chiquadrato = n * sum(sommerighe[i] * sommecolonne[j] * (
                (contingenze[i][j] / n - sommerighe[i] * sommecolonne[j]) / (sommerighe[i] * sommecolonne[j])) ** 2 for
                          i in range(contingenze.shape[0]) for j in range(contingenze.shape[1]))
    return SuntoTest(chiquadrato, p := chi2.sf(chiquadrato, df=(contingenze.shape[0]-1)*(contingenze.shape[1]-1)), p >= impostazioni["acqtcp"])


@monitora_("cqtcp")
def cqtcp(libro: Libro, dati: SessioneGrezza, impostazioni: dict):
    test = _cqtcp(libro, dati, impostazioni)
    return Test("Indipendenza tra tempo e cluster", "Test chi-quadrato", test, False, None, None)


@monitora_("_cqfcp")
def _cqfcp(libro: Libro, dati: SessioneGrezza, impostazioni: dict):
    frecce = dati.dati.idfrecce
    cluster = np.array(impostazioni["clucp"][0].valore)
    contingenze = np.array([[np.sum(frecce == f_ & cluster == c_) for f_ in range(dati.dati.frecce)] for c_ in range(np.max(cluster))])
    sommerighe = np.sum(contingenze, axis=1)
    sommecolonne = np.sum(contingenze, axis=0)
    n = dati.dati.taglia
    chiquadrato = n * sum(sommerighe[i] * sommecolonne[j] * (
            (contingenze[i][j] / n - sommerighe[i] * sommecolonne[j]) / (sommerighe[i] * sommecolonne[j])) ** 2 for
                          i in range(contingenze.shape[0]) for j in range(contingenze.shape[1]))
    return SuntoTest(chiquadrato, p := chi2.sf(chiquadrato, df=(contingenze.shape[0] - 1) * (contingenze.shape[1] - 1)),
                     p >= impostazioni["acqfcp"])


@monitora_("cqfcp")
def cqfcp(libro: Libro, dati: SessioneGrezza, impostazioni: dict):
    test = _cqfcp(libro, dati, impostazioni)
    return Test("Indipendenza tra le frecce e i cluster", "Test chi-quadrato", test, False, None, None)


def join_non_decreasing(lists):
    result = []
    for lst in lists:
        if not lst:
            continue
        if not result:
            result.extend(lst)
        else:
            offset = result[-1] + 1 - lst[0]
            result.extend(x + offset for x in lst)
    return result


@registra("cp", set(), "Confronto tra periodi", FALDONE)
@monitora_("cp")
def cp(libro: Libro, dati: list[Sessione], impostazioni: dict) -> ConfrontoPeriodi:
    periodo1, periodo2 = (impostazioni["pcp11"], impostazioni["pcp12"]), (impostazioni["pcp21"], impostazioni["pcp22"])
    primoperiodo_ = [s for s in dati if periodo1[0] < s.dati.metadati.data < periodo1[1]]
    secondoperiodo_ = [s for s in dati if periodo2[0] < s.dati.metadati.data < periodo2[1]]
    primadescrizione = {d: FALDONE[d].funzione(libro, primoperiodo_, impostazioni) for d in {"mpc", "mirpc", "mcc", "mac",
                                                                                            "vpc", "virpc", "vcc", "vac"}}
    secondadescrizione = {d: FALDONE[d].funzione(libro, secondoperiodo_, impostazioni) for d in {"mpc", "mirpc", "mcc", "mac",
                                                                                                "vpc", "virpc", "vcc", "vac"}}
    primolibro = Libro(id_=None, tempototale=0.0,
                       contenuto={"mp": primadescrizione["mpc"], "mirp": primadescrizione["mirpc"], "mc": primadescrizione["mcc"],
                                  "ma": primadescrizione["mac"], "vp": primadescrizione["vpc"], "vc": primadescrizione["vcc"],
                                  "va": primadescrizione["vac"]})
    secondolibro = Libro(id_=None, tempototale=0.0,
                         contenuto={"mp": secondadescrizione["mpc"], "mirp": secondadescrizione["mirpc"], "mc": secondadescrizione["mcc"],
                                    "ma": secondadescrizione["mac"], "vp": secondadescrizione["vpc"], "vc": secondadescrizione["vcc"],
                                    "va": secondadescrizione["vac"]})
    primoperiodo = SessioneGrezza(id_=None, metadati=None,
                                  dati=Dati(xy=np.concatenate([s.dati.dati.xy for s in primoperiodo_]),
                                            ordine=all(s.dati.dati.ordine for s in primoperiodo_),
                                            idv=join_non_decreasing([s.dati.dati.idvolée for s in primoperiodo_]),
                                            idf=np.concatenate([s.dati.dati.idfrecce for s in primoperiodo_]) if all(s.dati.dati.idfrecce is not None for s in primoperiodo_) else None))
    secondoperiodo = SessioneGrezza(id_=None, metadati=None,
                                  dati=Dati(xy=np.concatenate([s.dati.dati.xy for s in secondoperiodo_]),
                                            ordine=all(s.dati.dati.ordine for s in secondoperiodo_),
                                            idv=join_non_decreasing([s.dati.dati.idvolée for s in secondoperiodo_]),
                                            idf=np.concatenate([s.dati.dati.idfrecce for s in secondoperiodo_]) if all(
                                                s.dati.dati.idfrecce is not None for s in secondoperiodo_) else None))
    impostazioni["cqtcpn"] = [s.dati.dati.taglia for s in primoperiodo_]
    cqtcp1 = cqtcp(libro, primoperiodo, impostazioni)
    impostazioni["cqtcpn"] = [s.dati.dati.taglia for s in secondoperiodo_]
    cqtcp2 = cqtcp(libro, secondoperiodo, impostazioni)
    impostazioni["clucp"] = (clcp(libro, primoperiodo, impostazioni), clcp(libro, secondoperiodo, impostazioni))
    conf = ConfrontoPeriodi(periodi=(periodo1, periodo2),
                            descrittive=(primadescrizione, secondadescrizione),
                            intervallihotelling=(ihcp(libro, primoperiodo, impostazioni),
                                                 ihcp(libro, secondoperiodo, impostazioni)),
                            intervallit=(itcp(libro, primoperiodo, impostazioni),
                                         itcp(libro, secondoperiodo, impostazioni)),
                            intervallivarianze=(ivcp(libro, primoperiodo, impostazioni),
                                                ivcp(libro, secondoperiodo, impostazioni)),
                            anova=(anovacp(libro, primoperiodo, impostazioni),
                                   anovacp(libro, secondoperiodo, impostazioni)),
                            levene=(lvcp(libro, primoperiodo, impostazioni),
                                    lvcp(libro, secondoperiodo, impostazioni)),
                            clustering=impostazioni["clucp"],
                            chiquadratotempo=(cqtcp1, cqtcp2),
                            chiquadratofrecce=(cqfcp(libro, primoperiodo, impostazioni),
                                               cqfcp(libro, secondoperiodo, impostazioni)))
    del impostazioni["cqtcpn"], impostazioni["clucp"]
    return conf
