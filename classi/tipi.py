from classi.archi import ArcoOlimpico, ArcoNudo, ArcoCompound
from dataclasses import dataclass
from datetime import date
from enum import Enum
import numpy as np
from letturascrittura.memoria import leggifile
from letturascrittura.percorsi import cartellasessioni
from os import listdir


class TipoArco(Enum):
    OLIMPICO = ArcoOlimpico
    NUDO = ArcoNudo
    COMPOUND = ArcoCompound


@dataclass
class MetadatiSessione:
    tipoarco: TipoArco
    tag: set[str] | list[str]
    data: date
    arco: ArcoOlimpico | ArcoNudo | ArcoCompound
    distanza: int


def controlladati(dati: np.ndarray) -> np.ndarray:
    if dati.ndim != 2 or dati.shape[1] != 2:
        raise ValueError("I dati non sono un array nx2")
    if not np.isfinite(dati).all():
        raise ValueError("Dei dati sono infiniti")
    dati[np.all(dati == 0, axis=1)] += 0.00001
    return dati


def controllaindici(indici: np.ndarray) -> None:
    if not np.isfinite(indici).all():
        raise ValueError("Degli indici sono infiniti")
    if np.any(indici < 0):
        raise ValueError("Degli indici sono negativi")


def controllametadati(metadati: MetadatiSessione) -> None:
    if metadati.distanza <= 0:
        raise ValueError("Distanza negativa")
    if metadati.arco not in leggifile("ImpostazioniArco.pkl"):
        raise ValueError("Arco non definito")


sequenza = [0, 'crf', 'mpf', 'mcf', 'maf', 'vpf', 'vcf', 'vaf', 1, 'mp', 'mirp', 'mpv',
            'mirpv', 'mc', 'mcv', 'ma', 'mav', 'vp', 'virp', 'virpv', 'vpv', 'vc',
            'vcv', 'va', 'vav', 'cor', 'cv', 'rp', 'rirp', 'rc', 'ap', 'airp', 'ac',
            'aa', 2, 'cl', 'icl', 'ikcl', 'cmcl', 'scl', 'cbcl', 'abcl', 'iicl', 'iecl',
            'ncl', 'acl', 'sccl', 'cscl', 'bicl', 'fcl', 3, 'cva', 'cvo', '_iv', 'inva',
            'a_va', 'ava', 'invo', 'a_vo', 'avo', 'ibtva', 'tv', 4, 'tn', 'atn', 'iatn',
            'dbtn', 'ibtn', 'gtbtn', 'aabtn', 'aobtn', 'dcbtn', 'aatn', 'actn', 'ttn', 5,
            'ce', 6, 'irpn', 'airpn', 'abirpn', 'ibirpn', 'asbirpn', 'dcbirpn', 'gtbirpn',
            'iairpn', 7, 'th', 'ath', 'ith', 'dbth', 'ih', 'cdih', 'gsih', 'bsih', 'cih',
            'lbmih', 'bmih', 'vabsih', 'vobsih', 'ibsih', 'ibmih', 8, 'mg', 'smg', 'mi',
            'cmmi', 'gmi', 'rbmi', 'dvmi', 9, 'lbc', 'hlbc', 'albc', 'ilbc', 'lbirp', 'hlbirp',
            'albirp', 'ilbirp', 10, 'iirp', 'ciirp', 'tirp', 'atirp', 'a_tirp', 'mtirp', 'ibtirp', 'dbtirp',
            11, 'fd', 'ahdc', 'ibhdc', 'dbhdc', 'vd', 'avd', 'ibvdc', 'dbvdc', 12, 'ua', 'aua', 'iaua',
            'ibua', 'kbua', 13, 'avm', 'atvm', 'iaavm', 'ibavm', 'ubavm', 'dcavm', 'kbavm', 'iam',
            'iamf', 'iaiam', 'cam', 'ik', 'iik', 'cik', 'iaik', 14, 'vb', 'a_vab', 'invab', 'rvab',
            'a_vob', 'invob', 'rvob', 'cvab', 'cvob', 'vbg', 'apvb', 'ncvb', 'ivb', 'nivb',
            'ap_vb', 'mapvb', 'sapvb', 'aapvb', 'bapvb', 'eapvb', 'tvb', 'i_vb', 15, 'thb', 'rhb',
            'i_hb', 'iphb', 'chb', 'hb', 'hbg', 'aphb', 'nchb', 'ihb', 'nihb', 'phb', 'mphb',
            'lphb', 'a_phb', 'bphb', 'ephb', 16, 'a_irpb', 'mirpb', 'rirpb', 'ciirpb', 'irpb',
            'irpbg', 'apirpb', 'ncirpb', 'iirpb', 'niirpb', 'ap_irpb', 'aapirpb', 'bapirpb',
            'a2apirpb', 'b2apirpb', 17, 'atkamb', 'intkamb', 'rtkamb', 'ciamb', 'cikamb', 'ambv',
            'niamb', 'apamb', 'ap_amb', 'mapamb', 'kapamb', 'aapamb', 'bapamb', 'ambg', 'ncamb',
            'iamb', 'tkamb', 'kamb', 'i_amb', 'ikamb', 18, 'cfmvm', 'nimvm', 'apmvm', 'cm_mvm',
            'cmmvm', 'ckmvm']
significati = {"crf": "Correlazione ascisse-ordinate per ogni freccia",
               "mpf": "Media dei punteggi per ogni freccia",
               "mcf": "Media delle coordinate per ogni freccia",
               "maf": "Angolo medio per ogni freccia",
               "vpf": "Varianza dei punteggi per ogni freccia",
               "vcf": "Matrice di covarianze delle coordinate per ogni freccia",
               "vaf": "Lunghezza risultante media per ogni freccia",
               "mp": "Media dei punteggi",
               "mirp": "Media degli IRP",
               "mpv": "Media dei punteggi per ogni volée",
               "mirpv": "Media degli IRP per ogni volée",
               "mc": "Media delle coordinate",
               "mcv": "Media delle coordinate per ogni volée",
               "ma": "Angolo medio",
               "mav": "Angolo medio per ogni volée",
               "vp": "Varianza dei punteggi",
               "virp": "Varianza degli IRP",
               "virpv": "Varianza degli IRP per ogni volée",
               "vpv": "Varianza dei punteggi per ogni volée",
               "vc": "Matrice di covarianze delle coordinate",
               "vcv": "Matrice di covarianze delle coordinate per ogni volée",
               "va": "Lunghezza risultante media",
               "vav": "Lunghezza risultante media per ogni volée",
               "cor": "Correlazione tra ascisse e ordinate",
               "cv": "Correlazione tra ascisse e ordinate per ogni volée",
               "rp": "Regressione lineare sui punteggi",
               "rirp": "Regressione lineare sugli IRP",
               "rc": "Regressioni sulle coordinate",
               "ap": "Autocorrelazioni dei punteggi",
               "airp": "Autocorrelazioni degli IRP",
               "ac": "Autocorrelazioni delle coordinate",
               "aa": "Autocorrelazioni angolari",
               "cl": "Clustering",
               "icl": "Metodo di inizializzazione",
               "ikcl": "Iterazioni del metodo k-means o k-medoidi",
               "cmcl": "Numero massimo di componenti della mistura",
               "scl": "Metodo di selezione del modello",
               "cbcl": "Numero di campioni bootstrap",
               "abcl": "Alfa per il test bootstrap",
               "iicl": "Iterazioni del metodo di inizializzazione",
               "iecl": "Iterazioni dell'algoritmo di clustering",
               "ncl": "Componenti gaussiane",
               "acl": "Algoritmo di clustering",
               "sccl": "Soglia di convergenza nella regola di arresto",
               "cscl": "Criterio informativo",
               "bicl": "Numero di bootstrap interni nel doppio bootstrap",
               "fcl": "Numero di fold nella cross-validation",
               "cva": "Livello di confidenza per la varianza delle ascisse",
               "cvo": "Livello di confidenza per la varianza delle ordinate",
               "_iv": "Intervalli di confidenza per le varianze",
               "inva": "Ipotesi nulla per il test sulla varianza delle ascisse",
               "a_va": "Ipotesi alternativa per il test sulla varianza delle ascisse",
               "ava": "Alfa per il test sulla varianza delle ascisse",
               "invo": "Ipotesi nulla per il test sulla varianza delle ordinate",
               "a_vo": "Ipotesi alternativa per il test sulla varianza delle ordinate",
               "avo": "Alfa per il test sulla varianza delle ordinate",
               "ibtva": "Iterazioni del metodo MC per la stima del beta del test delle ascisse",
               "tv": "Test di ipotesi sulle varianze",
               "tn": "Test di multinormalità",
               "atn": "Alfa per il test di multinormalità",
               "iatn": "Iterazioni del metodo MC per la stima dell'alfa del test di multinormalità",
               "dbtn": "Distribuzione sotto l'ipotesi alternativa",
               "ibtn": "Iterazioni del metodo MC per la stima del beta del test di multinormalità",
               "gtbtn": "Gradi della t di Student sotto l'ipotesi alternativa",
               "aabtn": "Parametro di asimmetria della ascisse sotto l'ipotesi alternativa",
               "aobtn": "Parametro di asimmetria delle ordinate sotto l'ipotesi alternativa",
               "dcbtn": "Distanza delle componenti della mistura sotto l'ipotesi alternativa",
               "aatn": "Alfa per l'asimmetria del test di Mardia",
               "actn": "Alfa per la curtosi del test di Mardia",
               "ttn": "Tipo di test di multinormalità",
               "ce": "Stima delle probabilità di errore e delle coperture con metodi Monte Carlo",
               "irpn": "Test di Shapiro-Wilk sugli IRP",
               "airpn": "Alfa per il test di Shapiro-Wilk",
               "abirpn": "Ipotesi alternativa per il test di Shapiro-Wilk",
               "ibirpn": "Iterazioni del metodo MC per la stima del beta del test di Shapiro-Wilk",
               "asbirpn": "Parametro di asimmetria sotto l'ipotesi nulla",
               "dcbirpn": "Distanza delle componenti della mistura sotto l'ipotesi nulla",
               "gtbirpn": "Gradi della t di Student sotto l'ipotesi nulla",
               "iairpn": "Iterazioni del metodo MC per la stima dell'alfa del test di Shapiro-Wilk",
               "th": "Test di Hotelling",
               "ath": "Alfa del test di Hotelling",
               "ith": "Iterazioni del metodo MC per la stima del beta del test di Hotelling",
               "dbth": "Distanza dell'ipotesi alternativa",
               "ih": "Regione di confidenza bivariata",
               "cdih": "Metodo di controllo delle dipendenze",
               "gsih": "Probabilità della geometrica nel bootstrap stazionario",
               "bsih": "Blocchi del bootstrap stazionario",
               "cih": "Livello di confidenza della regione",
               "lbmih": "Lunghezza nel bootstrap a blocchi mobili",
               "bmih": "Blocchi nel bootstrap a blocchi mobili",
               "vabsih": "Varianza delle ascisse nel bootstrap stazionario",
               "vobsih": "Varianza delle ordinate nel bootstrap stazionario",
               "ibsih": "Iterazioni del bootstrap stazionario",
               "ibmih": "Iterazioni del bootstrap a blocchi mobili",
               "mg": "Mediana geometrica",
               "smg": "Soglia di convergenza della mediana geometrica",
               "mi": "Consigli sul mirino ideale",
               "cmmi": "Distanza tra la cocca e il mirino in metri",
               "gmi": "Angolo tra la freccia e il paglione",
               "rbmi": "Raggio del bersaglio in metri",
               "dvmi": "Distanza in verticale tra freccia e bersaglio",
               "lbc": "Test di Ljung-Box sulle coordinate",
               "hlbc": "Numero di ritardi nel test di Ljung-Box sulle coordinate",
               "albc": "Alfa del test di Ljung-Box sulle coordinate",
               "ilbc": "Iterazioni del metodo MC per la stima del beta del test di Ljung-Box sulle coordinate",
               "lbirp": "Test di Ljung-Box sugli IRP",
               "hlbirp": "Numero di ritardi nel test di Ljung-Box sugli IRP",
               "albirp": "Alfa del test di Ljung-Box sugli IRP",
               "iirp": "Iterazioni del metodo MC per la stima del beta del test di Ljung-Box sugli IRP",
               "ciirp": "Intervallo di confidenza per gli IRP",
               "tirp": "Test di ipotesi per gli IRP",
               "atirp": "Alfa per il test sugli IRP",
               "a_tirp": "Ipotesi alternativa del test sugli IRP",
               "mtirp": "Ipotesi nulla del test sugli IRP",
               "ibtirp": "Iterazioni del metodo MC per la stima del beta del test sugli IRP",
               "dbtirp": "Distanza dell'ipotesi alternativa",
               "fd": "Analisi della difettosità delle frecce in media",
               "ahdc": "Alfa per il test di Hotelling a due campioni",
               "ibhdc": "Iterazioni del metodo MC per la stima del beta del test di Hotelling a due campioni",
               "dbhdc": "Distanza dell'ipotesi alternativa del test di Hotelling a due campioni",
               "vd": "Analisi della difettosità delle frecce in varianza",
               "avd": "Alfa per il test sulle varianze a due campioni",
               "ibvdc": "Iterazioni del metodo MC per la stima del beta del test sulle varianze a due campioni",
               "dbvdc": "Distanza dell'ipotesi alternativa del test sulle varianze a due campioni",
               "ua": "Test di Rayleigh",
               "aua": "Alfa del test di Rayleigh",
               "iaua": "Iterazioni del metodo MC per la stima dell'alfa del test di Rayleigh",
               "ibua": "Iterazioni del metodo MC per la stima del beta del test di Rayleigh",
               "kbua": "Kappa sotto l'ipotesi alternativa del test di Rayleigh",
               "k": "Parametro di concentrazione circolare",
               "avm": "Test di goodness of fit per la von Mises",
               "atvm": "Alfa del test di goodness of fit per la von Mises",
               "iaavm": "Iterazioni del metodo MC per la stima dell'alfa del test di goodness of fit della von Mises",
               "ibavm": "Iterazioni del metodo MC per la stima del beta del test di goodness of fit della von Mises",
               "ubavm": "Uniformità sotto l'ipotesi alternativa",
               "dcavm": "Distanza tra le componenti della mistura sotto l'ipotesi alternativa",
               "kbavm": "Kappa sotto l'ipotesi nulla",
               "iam": "Intervallo di confidenza per l'angolo medio",
               "iamf": "Versione più rapida (meno coprente)",
               "iaiam": "Iterazioni del metodo MC per la stima della copertura dell'intervallo per l'angolo medio",
               "cam": "Livello di confidenza dell'intervallo per l'angolo medio",
               "ik": "Intervallo bootstrap per kappa",
               "iik": "Iterazioni bootstrap per l'intervallo per kappa",
               "cik": "Livello di confidenza per l'intervallo per kappa",
               "iaik": "Iterazioni del metodo MC per la stima della copertura dell'intervallo per kappa",
               "vb": "Inferenza bayesiana sulle varianze",
               "a_vab": "Ipotesi alternativa del test bayesiano sulle varianze delle ascisse",
               "invab": "Ipotesi nulla del test bayesiano sulle varianze delle ascisse",
               "rvab": "Raggio della ROPE per le varianze delle ascisse",
               "a_vob": "Ipotesi alternativa del test bayesiano sulle varianze delle ordinate",
               "invob": "Ipotesi nulla del test bayesiano sulle varianze delle ordinate",
               "rvob": "Raggio della ROPE per le varianze delle ordinate",
               "cvab": "Livello di credibilità dell'intervallo per le varianze delle ascisse",
               "cvob": "Livello di credibilità dell'intervallo per le varianze delle ordiante",
               "vbg": "Modello gerarchico",
               "apvb": "Allenamenti priori",
               "ncvb": "Numero di catene MCMC",
               "ivb": "Iterazioni MCMC",
               "nivb": "Priori non informative",
               "ap_vb": "Parametri priori a piacere",
               "mapvb": "Media informativa",
               "sapvb": "Matrice di covarianze informativa",
               "aapvb": "Parametro di shape informativo",
               "bapvb": "Parametro di rate informativo",
               "eapvb": "Parametro informativo della LKJ",
               "tvb": "Test bayesiano sulle varianze",
               "i_vb": "Intervallo di credibilità sulle varianze",
               "thb": "Test di Hotelling bayesiano",
               "rhb": "Raggio della ROPE per la media bivariata",
               "i_hb": "Regione di credibilità per la media bivariata",
               "iphb": "Regione di credibilità attraverso KDE",
               "chb": "Livello di credibilità per la media bivariata",
               "hb": "Inferenza bayesiana sulla media bivariata",
               "hbg": "Metodo bayesiano gerarchico",
               "aphb": "Allenamenti priori",
               "nchb": "Numero di catene MCMC",
               "ihb": "Numero di iterazioni MCMC",
               "nihb": "Priori non informative",
               "phb": "Parametri informativi a piacere",
               "mphb": "Media informativa per la media bivariata",
               "lphb": "Matrice di covarianze informativa per la media bivariata",
               "a_phb": "Parametro di shape informativo per la media bivariata",
               "bphb": "Parametro di rate informativo per la media bivariata",
               "ephb": "Parametro informativo della LKJ",
               "a_irpb": "Ipotesi alterantiva per il test bayesiano sugli IRP",
               "mirpb": "Ipotesi nulla per il test bayesiano sugli IRP",
               "rirpb": "Raggio della ROPE per il test bayesiano sugli IRP",
               "ciirpb": "Livello di credibilità per l'intervallo sugli IRP",
               "irpb": "Inferenza bayesiana sugli IRP",
               "irpbg": "Modello bayesiano gerarchico",
               "apirpb": "Allenamenti priori",
               "ncirpb": "Numero di catene MCMC",
               "iirpb": "Numero di iterazioni MCMC",
               "niirpb": "Priori non informative",
               "ap_irpb": "Parametri informativi a piacere",
               "aapirpb": "Parametro di shape informativo",
               "bapirpb": "Parametro di rate informativo",
               "a2apirpb": "Secondo parametro di shape informativo",
               "b2apirpb": "Secondo parametro di rate informativo",
               "amb": "Inferenza bayesiana sull'angolo medio",
               "atkamb": "Ipotesi alternativa per il test su kappa bayesiano",
               "intkamb": "Ipotesi nulla per il test su kappa bayesiano",
               "rtkamb": "Raggio della ROPE per il test su kappa bayesiano",
               "ciamb": "Livello di credibilità dell'intervallo sull'angolo medio",
               "cikamb": "Livello di credibilità dell'intervallo su kappa",
               "ambv": "Inferenza bayesiana sull'angolo medio tramite IV",
               "niamb": "Priori non informative",
               "apamb": "Allenamenti priori",
               "ap_amb": "Parametri informativi a piacere",
               "mapamb": "Angolo medio informativo",
               "kapamb": "Kappa informativo",
               "aapamb": "Parametro di shape informativo",
               "bapamb": "Parametro di rape informativo",
               "ambg": "Metodo bayesiano gerarchico",
               "ncamb": "Numero di catene MCMC",
               "iamb": "Numero di iterazioni MCMC",
               "tkamb": "Test bayesiano su kappa",
               "kamb": "Stima bayesiana di kappa",
               "i_amb": "Intervallo di credibilità per l'angolo medio",
               "ikamb": "Intervallo di credibilità per kappa",
               "cfmvm": "Numero di componenti fissate per la mistura di von Mises",
               "mvm": "Modello mistura bayesiano di von Mises",
               "nimvm": "Priori non informative",
               "apmvm": "Allenamenti priori",
               "cm_mvm": "Numero di componenti massime per la mistura",
               "cmmvm": "Livello di credibilità per gli intervalli dell'angolo medio",
               "ckmvm": "Livello di credibilità per gli intervalli su kappa"}
tipi = {"crf": bool, "mpf": bool, "mcf": bool, "maf": bool, "vpf": bool,
        "vcf": bool, "vaf": bool, "mp": bool, "mirp": bool, "mpv": bool,
        "mirpv": bool, "mc": bool, "mcv": bool, "ma": bool, "mav": bool,
        "vp": bool, "virp": bool, "virpv": bool, "vpv": bool, "vc": bool,
        "vcv": bool, "va": bool, "vav": bool, "cor": bool, "cv": bool,
        "rp": bool, "rirp": bool, "rc": bool, "ap": bool, "airp": bool,
        "ac": bool, "aa": bool, "cl": bool, "icl": int, "ikcl": int,
        "cmcl": int, "scl": str, "cbcl": int, "abcl": float, "iicl": int,
        "iecl": int, "ncl": bool, "acl": str, "sccl": float, "cscl": str,
        "bicl": int, "fcl": int, "cva": float, "cvo": float, "_iv": bool,
        "inva": float, "a_va": str, "ava": float, "invo": float, "a_vo": str,
        "avo": float, "ibtva": int, "tv": bool, "tn": bool, "atn": float, "iatn": int,
        "dbtn": str, "ibtn": int, "gtbtn": float, "aabtn": float, "aobtn": float,
        "dcbtn": float, "aatn": float, "actn": float, "ttn": str, "ce": bool, "irpn": bool,
        "airpn": float, "abirpn": str, "ibirpn": int, "asbirpn": float,
        "dcbirpn": float, "gtbirpn": float, "iairpn": int, "th": bool, "ath": float, "ith": int,
        "dbth": float, "ih": bool, "cdih": str, "gsih": float, "bsih": int, "cih": float,
        "lbmih": int, "bmih": int, "vabsih": float, "vobsih": float, "ibsih": int,
        "ibmih": int, "mg": bool, "smg": float, "mi": bool, "cmmi": float, "gmi": float, "rbmi": float,
        "dvmi": float, "lbc": bool, "hlbc": int, "albc": float, "ilbc": int, "lbirp": bool, "hlbirp": int,
        "albirp": float, "iirp": bool, "ciirp": float, "tirp": bool, "atirp": float, "a_tirp": str, "mtirp": float,
        "ibtirp": int, "dbtirp": float, "fd": bool, "ahdc": float, "ibhdc": int, "dbhdc": float,
        "vd": bool, "avd": float, "ibvdc": int, "dbvdc": float, "ua": bool, "aua": float, "iaua": int, "ibua": int,
        "kbua": float, "avm": bool, "atvm": float, "iaavm": int, "ibavm": int, "ubavm": bool, "dcavm": float,
        "kbavm": float, "iam": bool, "iamf": bool, "iaiam": int, "cam": float, "ik": bool, "iik": int,
        "cik": float, "iaik": int, "vb": bool, "a_vab": str, "invab": float, "rvab": float, "a_vob": str,
        "invob": float, "rvob": float, "cvab": float, "cvob": float, "vbg": bool, "apvb": int, "ncvb": int,
        "ivb": int, "nivb": bool, "ap_vb": bool, "mapvb": list[float], "sapvb": list[list[float]],
        "aapvb": float, "bapvb": float, "eapvb": float, "tvb": bool, "i_vb": bool, "thb": bool, "rhb": float,
        "i_hb": bool, "iphb": bool, "chb": float, "hb": bool, "hbg": bool, "aphb": int, "nchb": int,
        "ihb": int, "nihb": bool, "phb": bool, "mphb": list[float], "lphb": list[list[float]], "a_phb": float,
        "bphb": float, "ephb": float, "a_irpb": str, "mirpb": float, "rirpb": float, "ciirpb": float,
        "irpb": bool, "irpbg": bool, "apirpb": int, "ncirpb": int, "iirpb": int, "niirpb": bool,
        "ap_irpb": bool, "aapirpb": float, "bapirpb": float, "a2apirpb": float, "b2apirpb": float, "atkamb": str,
        "intkamb": float, "rtkamb": float, "ciamb": float, "cikamb": float, "ambv": bool, "niamb": bool,
        "apamb": int, "ap_amb": bool, "mapamb": float, "kapamb": float, "aapamb": float, "bapamb": float,
        "ambg": bool, "ncamb": int, "iamb": int, "tkamb": bool, "kamb": bool, "i_amb": bool, "ikamb": bool,
        "cfmvm": int, "nimvm": bool, "apmvm": int, "cm_mvm": int, "cmmvm": float, "ckmvm": float}
opzioni = {"icl": ["K-medie", "K-medoidi", "Casuale"],
           "scl": ["Tramite numero di mode", "Criterio informativo", "Bootstrap", "Doppio bootstrap", "Cross-validation"],
           "acl": ["EM", "SEM", "CEM"], "cscl": ["AIC", "BIC", "ICL"], "a_va": ["magiore", "diseguale", "minore"],
           "a_vo": ["maggiore, diseguale", "minore"], "dbtn": ["Laplace", "Normale asimmetrica", "Uniforme", "Mistura di gaussiane", "Lognormale", "t di Student"],
           "ttn": ["Henze-Zirkler", "Mardia"], "abirpn": ["maggiore", "diseguale", "minore"], "cdih": ["Nessuno", "Detrending", "Bootstrap stazionario", "Bootstrap a blocchi mobili"],
           "a_tirp": ["maggiore", "diseguale", "minore"], "a_vab": ["maggiore", "diseguale", "minore"],
           "a_vob": ["maggiore", "diseguale", "minore"], "a_irpb": ["maggiore", "diseguale", "minore"],
           "atkamb": ["maggiore", "diseguale", "minore"]}
vincoli = {"icl": ["K", "M", "C"], "ikcl": "+", "cmcl": "2", "scl": ["M", "C", "B", "D", "V"],
           "cbcl": "+", "abcl": "u", "iicl": "+", "iecl": "+", "acl": ["E", "S", "C"],
           "sccl": "+", "cscl": ["A", "B", "I"], "bicl": "+", "fcl": "2", "cva": "u",
           "cvo": "u", "inva": "+", "a_va": [">", "=", "<"], "ava": "u", "invo": "+",
           "a_vo": [">", "=", "<"], "avo": "u", "ibtva": "+", "ibtvo": "+", "atn": "u",
           "iatn": "+", "dbtn": ["L", "A", "U", "M", "N", "T"], "ibtn": "+", "gtbtn": "+",
           "aatn": "u", "actn": "u", "ttn": ["h", "m"], "airpn": "u", "abirpn": [">", "=", "<"],
           "ibirpn": "+", "dcbirpn": "+", "gtbirpn": "+", "iairpn": "+", "ath": "u", "ith": "+",
           "cdih": ["n", "d", "s", "m"], "gsih": "u", "bsih": "2", "cih": "u", "lbmih": "2",
           "bmih": "+", "vabsih": "+", "vobsih": "+", "ibsih": "+", "ibmih": "+", "smg": "+",
           "cmmi": "+", "rbmi": "+", "dvmi": "+", "hlbc": "+", "albc": "u", "ilbc": "+",
           "hlbirp": "+", "albirp": "u", "ciirp": "u", "atirp": "u", "a_tirp": [">", "=", "<"],
           "mtirp": "10", "ibtirp": "+", "dbtirp": "+", "ahdc": "u", "ibhdc": "+",
           "avd": "u", "ibvdc": "+", "aua": "u", "iaua": "+", "ibua": "+", "kbua": "+",
           "atvm": "u", "iaavm": "+", "ibavm": "+", "kbavm": "+", "iaiam": "+", "cam": "u", "iik": "+",
           "cik": "u", "iaik": "+", "a_vab": [">", "=", "<"], "invab": "+", "rvab": "+", "a_vob": [">", "=", "<"],
           "invob": "+", "rvob": "+", "cvab": "u", "cvob": "u", "apvb": "a", "ncvb": "+", "ivb": "+",
           "mapvb": "m", "sapvb": "s", "aapvb": "+", "bapvb": "+", "eapvb": "+", "rhb": "+", "chb": "u",
           "aphb": "a", "nchb": "+", "ihb": "+", "mphb": "m", "lphb": "s", "a_phb": "+", "bphb": "+",
           "ephb": "+", "a_irpb": [">", "=", "<"], "mirpb": "10", "rirpb": "+", "ciirpb": "u", "apirpb": "a",
           "ncirpb": "+", "iirpb": "+", "aapirpb": "+", "bapirpb": "+",
           "a2apirpb": "+", "b2apirpb": "+", "atkamb": [">", "=", "<"], "intkamb": "+", "rtkamb": "+",
           "ciamb": "u", "cikamb": "u", "apamb": "a", "kapamb": "+", "aapamb": "+", "bapamb": "+",
           "ncamb": "+", "iamb": "+", "cfmvm": "2", "apmvm": "a", "cm_mvm": "2", "cmmvm": "u", "ckmvm": "u"}


def controllaimpostazioni(impostazioni: dict):
    numerodate = len(listdir(cartellasessioni()))
    for chiave in tipi:
        if chiave not in impostazioni:
            raise IndexError("Manca una chiave")
        if not isinstance((impostazioni[chiave]), tipi[chiave]):
            raise IndexError("Tipo errato")
        if chiave in vincoli:
            if isinstance(vincoli[chiave], set):
                if impostazioni[chiave] not in vincoli[chiave]:
                    raise ValueError(f"Impostazione sbagliata: {chiave} è pari a {impostazioni[chiave]}")
            elif vincoli[chiave] == "+":
                if impostazioni[chiave] <= 0:
                    raise ValueError(f"Impostazione sbagliata: {chiave} è pari a {impostazioni[chiave]}")
            elif vincoli[chiave] == "2":
                if impostazioni[chiave] < 2:
                    raise ValueError(f"Impostazione sbagliata: {chiave} è pari a {impostazioni[chiave]}")
            elif vincoli[chiave] == "a":
                if impostazioni[chiave] > numerodate:
                    raise ValueError(f"Impostazione sbagliata: {chiave} è pari a {impostazioni[chiave]}")
            elif vincoli[chiave] == "m":
                if len(impostazioni[chiave]) != 2:
                    raise ValueError(f"Impostazione sbagliata: {chiave} è pari a {impostazioni[chiave]}")
            elif vincoli[chiave] == "s":
                if len((matrice := impostazioni[chiave])) != 2:
                    raise ValueError(f"Impostazione sbagliata: {chiave} è pari a {impostazioni[chiave]}")
                elif len(impostazioni[chiave][0]) != 2 or len(impostazioni[chiave][1]) != 2:
                    raise ValueError(f"Impostazione sbagliata: {chiave} è pari a {impostazioni[chiave]}")
                elif matrice[0][0]*matrice[1][1]-matrice[0][1]**2 < 0 or matrice[0][1] != matrice[1][0]:
                    raise ValueError(f"Impostazione sbagliata: {chiave} è pari a {impostazioni[chiave]}")
            elif vincoli[chiave] == "10":
                if impostazioni[chiave] <= 0 or impostazioni[chiave] >= 10:
                    raise ValueError(f"Impostazione sbagliata: {chiave} è pari a {impostazioni[chiave]}")
            else:
                raise KeyError(f"Vincolo non previsto per {chiave}")


# ols_{}
sequenza_ = ["ds", "ns", "mps", "mmps", "mpfs", "mirps", "mmirps", "mirpfs", "virps", "mcs", "mmcs", "mcfs",
             "vcs", "vcfs", "mas", "vas", "ks", "mpc", "mirpc", "mcc", "vcc", "vac", "kc", "mpcf",
             "cps", "dpa", "tols", "vols", "fols", "dols", "aols", "sols", "ols",
             "hlbt", "albt", "lbt", "cm", "cdihcp", "gsihcp", "bsihcp", "cihcp", "lbmih", "bmihcp", "vabsihcp",
             "vobsihcp", "ibsihcp", "vabmihcp", "vobmihcp", "ibmihcp", "ciirpcp", "cvacp", "cvocp", "iclcp",
             "ikclcp", "cmcl", "sclcp", "cbclcp", "abclcp", "iiclcp", "ieclcp", "nclcp", "aclcp", "scclcp",
             "cscl", "biclcp", "fclcp", "acqfcp", "acqtcp", "cp"]
significati_ = {"ds": "Calcolo degli indici della serie storica (automatico)",
                "ns": "Conteggio delle numerosità delle sessioni",
                "mps": "Punteggi medi per sessione",
                "mmps": "Medie mobili dei punteggi medi",
                "mpfs": "Punteggi medi per freccia e sessione",
                "mirps": "IRP medi per sessione",
                "mmirps": "Medie mobili degli IRP medi",
                "mirpfs": "IRP medi per freccia e sessione",
                "virps": "Varianze dei punteggi per sessione",
                "mcs": "Coordinate medie per sessione",
                "mmcs": "Medie mobili delle coordinate medie per sessione",
                "mcfs": "Coordinate medie per freccia e sessione",
                "vcs": "Matrici di covarianze delle coordinate per sessione",
                "vcfs": "Matrici di covarianze delle coordinate per freccia e sessione",
                "mas": "Angoli medi per sessione",
                "vas": "Lunghezze risultanti medie per sessione",
                "ks": "Parametri di concentrazione per sessione",
                "mpc": "Media dei punteggi complessiva",
                "mirpc": "Media degli IRP complessiva",
                "mcc": "Media delle coordinate complessiva",
                "vcc": "Matrice di covarianze complessiva",
                "vac": "Lunghezza risultante media complessiva",
                "kc": "Parametro di concentrazione complessivo",
                "cps": "Correlazione tra punteggi e angoli per sessione",
                "dpa": "Giorni passati dall'ultimo allenamento prima del primo allenamento registrato",
                "tols": "Uso del tempo come covariata",
                "vols": "Uso del numero di volée come covariata",
                "fols": "Uso del numero di frecce come covariata",
                "dols": "Uso della distanza dal bersaglio come covariata",
                "aols": "Uso dell'arco come covariata",
                "sols": "Uso degli intervalli tra allenamenti come covariata",
                "ols": "Calcolo degli stimatori OLS",
                "hlbt": "Numero di ritardi usati nel test di Ljung-Box",
                "albt": "Alfa del test di Ljung-Box",
                "lbt": "Test di Ljung-Box sui residui della regressione delle risposte sul tempo",
                "cm": "Calcolo dei coefficienti di correlazione multipla",
                "cdihcp": "Correzione delle dipendenze nella regione di confidenza",
                "gsihcp": "Parametro della geometrica nel bootstrap stazionario",
                "bsihcp": "Blocchi del bootstrap stazionario",
                "cihcp": "Confidenza della regione di confidenza",
                "lbmih": "Lunghezza del blocco nell'MBB",
                "bmihcp": "Blocchi dell'MBB",
                "vabsihcp": "Varianza delle ascisse nel bootstrap stazionario",
                "vobsihcp": "Varianza delle ordinate nel bootstrap stazionario",
                "ibsihcp": "Iterazioni del bootstrap stazionario",
                "vabmihcp": "Varianza delle ascisse nell'MBB",
                "vobmihcp": "Varianza delle ordinate nell'MBB",
                "ibmihcp": "Iterazioni dell'MBB",
                "ciirpcp": "Confidenza dell'intervallo di confidenza per gli IRP",
                "cvacp": "Confidenza dell'intervallo di confidenza per la varianza delle ascisse",
                "cvocp": "Confidenza dell'intervallo di confidneza per la varianza delle ordinate",
                "iclcp": "Inizializzazione del clustering",
                "ikclcp": "Iterazioni del metodo di inizializzazione del clustering",
                "cmcl": "Numero massimo di componenti del clustering",
                "sclcp": "Tecnica di selezione del modello del clustering",
                "cbclcp": "Numero di campioni bootstrap del clustering",
                "abclcp": "Alfa del bootstrap del clustering",
                "iiclcp": "Numero di starting point del clustering",
                "ieclcp": "Iterazioni dell'algoritmo di ottimizzazione del clustering",
                "nclcp": "Gaussianità delle componenti del clustering",
                "aclcp": "Algoritmo di ottimizzazione del clustering",
                "scclcp": "Soglia di convergenza del clustering",
                "cscl": "Criterio informativo per il clustering",
                "biclcp": "Numero di bootstrap interni nel doppio bootstrap del clustering",
                "fclcp": "Numero di fold nella cross-validation per il clustering",
                "acqfcp": "Alfa per il test chi-quadrato su frecce/cluster",
                "acqtcp": "Alfa per il test chi-quadrato su tempo/cluster",
                "cp": "Esecuzione del confronto tra periodi",
                "pcp11": "Data di inizio del primo periodo da confrontare",
                "pcp12": "Data di fine del primo periodo da confrontare",
                "pcp21": "Data di inizio del secondo periodo da confrontare",
                "pcp22": "Data di fine del primo periodo da confrontare"}
tipi_ = {"ds": bool, "ns": bool, "mps": bool, "mmps": bool, "mpfs": bool, "mirps": bool, "mmirps": bool,
         "mirpfs": bool, "virps": bool, "mcs": bool, "mmcs": bool, "mcfs": bool, "vcs": bool, "vcfs": bool,
         "mas": bool, "vas": bool, "ks": bool, "mpc": bool, "mirpc": bool, "mcc": bool, "vcc": bool, "vac": bool,
         "kc": bool, "mpcf": bool, "cps": bool, "dpa": int, "tols": bool, "vols": bool, "fols": bool,
         "dols": bool, "aols": bool, "sols": bool, "ols": bool, "hlbt": int, "albt": float, "lbt": bool,
         "cm": bool, "cdihcp": str, "gsihcp": float, "bsihcp": int, "cihcp": float, "lbmih": int,
         "bmihcp": int, "vabsihcp": float, "vobsihcp": float, "ibsihcp": int, "vabmihcp": float,
         "vobmihcp": float, "ibmihcp": int, "ciirpcp": float, "cvacp": float, "cvocp": float, "iclcp": str,
         "ikclcp": int, "cmcl": int, "sclcp": str, "cbclcp": int, "abclcp": float, "iiclcp": int,
         "ieclcp": int, "nclcp": bool, "aclcp": str, "scclcp": float, "cscl": str, "biclcp": int,
         "fclcp": int, "acqfcp": float, "acqtcp": float, "cp": bool}
opzioni_ = {"cdihcp": ["Nessuna", "Bootstrap stazionario", "Bootstrap a blocchi mobili", "Detrending"],
            "iclcp": ["K-means", "K-medoidi", "Randomica"], "sclcp": ["Mode", "Bootstrap", "Doppio bootstrap", "Criterio informativo", "Cross-validation"],
            "aclcp": ["EM", "SEM", "CEM"], "cscl": ["AIC", "BIC", "ICL"]}
vincoli_ = {"cdihcp": ["n", "s", "m", "d"], "iclcp": ["K", "M", "C"], "sclcp": ["M", "B", "D", "C", "V"],
            "aclcp": ["E", "S", "C"], "cscl": ["A", "B", "I"]}


def punteggia(dati: np.ndarray) -> np.ndarray:
    normedecuple = 10*np.linalg.norm(dati, axis=1)
    pavimenti = np.floor(normedecuple)
    bordi = (normedecuple == pavimenti)
    return np.clip(10-pavimenti+1*bordi, 0, 10).astype(int)


def irpa(dati: np.ndarray) -> np.ndarray:
    return 10*(1-np.linalg.norm(dati, axis=1))


def angola(dati: np.ndarray) -> np.ndarray:
    return np.atan2(dati[:, 1], dati[:, 1])
