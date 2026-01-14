from kivy.uix.screenmanager import Screen
from kivy.properties import DictProperty, NumericProperty
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.gridlayout import GridLayout
from kivy.uix.checkbox import CheckBox
from kivy.uix.textinput import TextInput
from kivy.uix.spinner import Spinner
from classi.tipi import controllaimpostazioni, significati as etichette, tipi, opzioni
from classi.dati import Sessione, SessioneGrezza, Libro, Dati, MetadatiSessione, riesegui, REGISTRO, Intervallo, Test
from letturascrittura.percorsi import filesessione
from letturascrittura.memoria import leggifile, scrivifile
from grafici.traduttori import traduci
from grafici.graficatori import *


funzionigrafiche = {"gd": graficodispersione, "gc": graficocluster, "gih": graficointervallohotelling,
                    "ghb": graficohotellingbayesiano, "gmsb": graficomediesubersaglio, "gp": graficopunteggi,
                    "gv": graficovolée, "ga": graficoangoli, "gac": graficiautocorrelazioni,
                    "gmg": graficomedianageometrica, "gic": graficointervallicluster, "gin": graficointervallonorme,
                    "giv": graficointervallivarianze, "givb": graficointervallovarianzebayesiano,
                    "giam": graficointervalloangolomedio, "gik": graficointervallokappa, "gamb": graficoangolomediobayesiano,
                    "gmvm": graficomisturevonmises}
etichettegrafici = {"gd": "dispersione", "gc": "cluster", "gih": "regione di confidenza bivariata",
                    "ghb": "regione di credibilità bivariata", "gmsb": "medie e regressioni bivariate",
                    "gp": "punteggi", "gv": "volée", "ga": "angoli", "gac": "autocorrelazioni",
                    "gmg": "mediana geometrica", "gic": "regioni di confidenza bivariate per i cluster",
                    "gin": "intervallo di confidenza degli IRP", "giv": "intervalli di confidenza delle varianze",
                    "givb": "intervallo di credibilità delle varianze", "giam": "intervallo di confidenza dell'angolo medio",
                    "gik": "intervallo di confidenza di kappa", "gamb": "intervallo di credibilità dell'angolo medio",
                    "gmvm": "cluster angolari"}
sequenza = ["mp", "mpv", "mpf", "mirp", "mirpv", "mirpf", "mc", "mcv", "mcf", "ma", "mav", "maf", "cor",
            "cv", "crf", "vp", "vpv", "vpf", "virp", "virpv", "virpf", "vc", "vcv", "vcf", "va", "vav",
            "vaf", "rp", "rirp", "rc", "gp", "gv", "gmsb", "ga", "ap", "airp", "ac", "aa", "gac", "cl",
            "gc", "gic", "_iv", "tv", "giv", "vb", "givb", "tn", "irpn", "lbc",
            "lbirp", "th", "ih", "gih", "hb", "ghb", "mg", "mi", "gmg", "iirp", "tirp", "gin", "irpb",
            "fd", "vd", "ua", "k", "avm", "iam", "giam", "ik", "gik", "amb", "gamb", "mvm", "gmvm"]
parametri = {"cl": {'icl', 'ikcl', 'cmcl', 'scl', 'cbcl', 'abcl', 'iicl', 'iecl',
                    'ncl', 'acl', 'sccl', 'cscl', 'bicl', 'fcl'},
             "_iv": {'cva', 'cvo'}, "tv": {'inva', 'a_va', 'ava', 'invo', 'a_vo', 'avo', 'ibtva'},
             "vb": {'a_vab', 'invab', 'rvab', 'a_vob', 'invob', 'rvob', 'cvab', 'cvob', 'vbg', 'apvb',
                    'ncvb', 'ivb', 'nivb', 'ap_vb', 'mapvb', 'sapvb', 'aapvb', 'bapvb', 'eapvb', 'tvb',
                    'i_vb'},
             "tn": {'atn', 'iatn', 'dbtn', 'ibtn', 'gtbtn', 'aabtn', 'aobtn', 'dcbtn', 'aatn', 'actn', 'ttn'},
             "irpn": {'airpn', 'abirpn', 'ibirpn', 'asbirpn', 'dcbirpn', 'gtbirpn', 'iairpn'},
             "lbc": {'hlbc', 'albc', 'ilbc'},
             "lbirp": {'hlbirp', 'albirp', 'ilbirp'},
             "th": {'ath', 'ith', 'dbth'},
             "ih": {'cdih', 'gsih', 'bsih', 'cih', 'lbmih', 'bmih', 'vabsih', 'vobsih', 'ibsih', 'ibmih'},
             "hb": {'thb', 'rhb', 'i_hb', 'iphb', 'chb', 'hbg', 'aphb', 'nchb', 'ihb', 'nihb', 'phb',
                    'mphb', 'lphb', 'a_phb', 'bphb', 'ephb'},
             "mg": {'smg'},
             "mi": {'cmmi', 'gmi', 'rbmi', 'dvmi'},
             "iirp": {'ciirp'},
             "tirp": {'atirp', 'a_tirp', 'mtirp', 'ibtirp', 'dbtirp'},
             "irpb": {'a_irpb', 'mirpb', 'rirpb', 'ciirpb', 'irpbg', 'apirpb', 'ncirpb', 'iirpb',
                      'niirpb', 'ap_irpb', 'aapirpb', 'bapirpb', 'a2apirpb', 'b2apirpb'},
             "fd": {'ahdc', 'ibhdc', 'dbhdc'},
             "vd": {'avd', 'ibvdc', 'dbvdc'},
             "ua": {'aua', 'iaua', 'ibua', 'kbua'},
             "avm": {'atvm', 'iaavm', 'ibavm', 'ubavm', 'dcavm', 'kbavm'},
             "iam": {'iamf', 'iaiam', 'cam'},
             "ik": {'iik', 'cik', 'iaik'},
             "amb": {'atkamb', 'intkamb', 'rtkamb', 'ciamb', 'cikamb', 'ambv',
                     'niamb', 'apamb', 'ap_amb', 'mapamb', 'kapamb', 'aapamb', 'bapamb', 'ambg', 'ncamb',
                     'iamb', 'tkamb', 'kamb', 'i_amb', 'ikamb'},
             "mvm": {'cfmvm', 'nimvm', 'apmvm', 'cm_mvm', 'cmmvm', 'ckmvm'}}


def mostra_grafico(self, nome_analisi: str, sessione: Sessione, immagine: Image):
    try:
        figura = funzionigrafiche[nome_analisi](sessione)
    except Exception as e:
        print(f"Errore nella graficazione: {e}")
    else:
        texture = traduci(figura)
        immagine.texture = texture


class PaginaDiario(Screen):
    session_id = NumericProperty(-1)
    results = DictProperty({})

    def mettisessione(self, id_):
        self.id_ = id_
        self.caricaanalisi()

    def caricaanalisi(self):
        self.analisi: Sessione = leggifile(filesessione(self.id_))
        self.scatola = self.ids.analizzatore
        self.scatola.add_widget(Label(text=f"Sessione del {self.analisi.dati.metadati.data}"))
        self.scatola.add_widget(Label(text=f"Svolta con l'arco {self.analisi.dati.metadati.arco} a {self.analisi.dati.metadati.distanza} metri"))
        grafico_dispersione = Image()
        mostra_grafico(self, "gd", self.analisi, grafico_dispersione)
        self.scatola.add_widget(grafico_dispersione)
        for dato in sequenza:
            if dato in self.analisi.libro.contenuto and self.analisi.libro.contenuto[dato].errore is not None:
                self.scatola.add_widget(Label(text=f"Errore nel calcolo di {etichette[dato]}: {self.analisi.libro.contenuto[dato].errore}"))
            elif dato in {"rp", "rirp", "rc", "ap", "airp", "ac", "aa", "cl"}:
                self.scatola.add_widget(Label(text=f"Tempo per {etichette[dato]}: {self.analisi.libro.contenuto[dato].tempo}"))
            elif dato[0] == "g":
                grigliagrafico = GridLayout(rows=1, columns=2)
                etichettagrafico = Label(text=f"Vuoi mostrare il grafico di {etichettegrafici[dato]}?")
                bottonegrafico = Button(text="Mostra")
                immaginegrafico = Image()
                bottonegrafico.bind(lambda _, dato=dato: self.mostragrafico(dato, immaginegrafico, etichettagrafico,
                                                                            bottonegrafico, grigliagrafico))
                grigliagrafico.add_widget(etichettagrafico)
                grigliagrafico.add_widget(bottonegrafico)
                self.scatola.add_widget(grigliagrafico)
            else:
                risultato = self.analisi.libro.contenuto[dato]
                if risultato.valore is not None:
                    griglia = GridLayout(rows=2, columns=1)
                    grigliarisultato = GridLayout(rows=1, columns=3)
                    griglia.add_widget(grigliarisultato)
                    grigliarisultato.add_widget(Label(text=etichette[dato]+":"))
                    if isinstance(risultato.valore, float):
                        grigliarisultato.add_widget(Label(text=str(round(risultato.valore, 3))))
                    elif isinstance(risultato.valore, int):
                        grigliarisultato.add_widget(Label(text=str(risultato.valore)))
                    elif isinstance(risultato.valore, dict) and 0 in risultato.valore and isinstance(risultato.valore[0], float):
                        grigliafrecce = GridLayout(rows=self.analisi.dati.dati.frecce, columns=2)
                        for f in range(self.analisi.dati.dati.frecce):
                            grigliafrecce.add_widget(Label(text=str(f)+":"))
                            grigliafrecce.add_widget(Label(text=str(round(risultato.valore[f], 3))))
                        grigliarisultato.add_widget(grigliafrecce)
                    elif isinstance(risultato.valore, list) and isinstance(risultato.valore[0], float):
                        grigliavolée = GridLayout(rows=self.analisi.dati.dati.volée, columns=2)
                        for v in range(self.analisi.dati.dati.volée):
                            grigliavolée.add_widget(Label(text=str(v)+":"))
                            grigliavolée.add_widget(Label(text=str(round(risultato.valore[v], 3))))
                        grigliarisultato.add_widget(grigliavolée)
                    elif isinstance(risultato.valore, np.ndarray):
                        grigliarisultato.add_widget(Label(text=str(risultato.valore.round(3))))
                    elif isinstance(risultato.valore, Intervallo):
                        intervallo = risultato.valore.intervallo.valore
                        copertura = risultato.valore.copertura
                        grigliaintervallo = GridLayout(rows=3, columns=2)
                        grigliaintervallo.add_widget(Label(text="Intervallo:"))
                        grigliaintervallo.add_widget(Label(text=str([round(intervallo[0], 3), round(intervallo[1], 3)])))
                        grigliaintervallo.add_widget(Label(text="Asintotico:"))
                        grigliaintervallo.add_widget(Label(text="Sì" if risultato.valore.asintotico else "No"))
                        grigliaintervallo.add_widget(Label(text="Livello di confidenza:"))
                        grigliaintervallo.add_widget(Label(text=str(100*round(copertura, 3))+"%" if copertura is not None else "Non stimata"))
                        grigliarisultato.add_widget(grigliaintervallo)
                    elif isinstance(risultato.valore, Test):
                        pvalue = risultato.valore.decisione.valore.pvalue
                        accettazione = risultato.valore.decisione.valore.accettazione
                        alfa = risultato.valore.alfa
                        beta = risultato.valore.beta
                        grigliatest = GridLayout(rows=4, columns=2)
                        grigliatest.add_widget(Label(text="p-value:"))
                        grigliatest.add_widget(Button(text=str(round(pvalue, 3)), background_normal="",
                                                      background_color=(0, 1, 0.2, 1) if accettazione else (1, 0, 0, 1)))
                        grigliatest.add_widget(Label(text="Asintotico:"))
                        grigliatest.add_widget(Label(text="Sì" if risultato.valore.asintotico else "No"))
                        if alfa.errore is not None:
                            grigliatest.add_widget(Label(text="Errore durante il metodo MC:"))
                            grigliatest.add_widget(Label(text=alfa.errore))
                        elif beta.errore is not None:
                            grigliatest.add_widget(Label(text="Errore nel metodo MC:"))
                            grigliatest.add_widget(Label(text=beta.errore))
                        elif alfa is not None:
                            grigliatest.add_widget(Label(text="Probabilità di\nerrore di I tipo:"))
                            grigliatest.add_widget(Label(text=str(100*round(alfa.valore, 3))+"%"))
                            grigliatest.add_widget(Label(text="Tempo del metodo MC:"))
                            grigliatest.add_widget(Label(text=str(round(alfa.tempo, 3))))
                        elif beta is not None:
                            grigliatest.add_widget(Label(text="Probabilità di\nerrore di II tipo:"))
                            grigliatest.add_widget(Label(text=str(100*round(beta.valore, 3))+"%"))
                            grigliatest.add_widget(Label(text="Tempo del metodo MC:"))
                            grigliatest.add_widget(Label(text=str(round(beta.tempo, 3))))
                        else:
                            grigliatest.add_widget(Label(text="Metodo MC non eseguito"))
                        grigliarisultato.add_widget(grigliatest)
                    elif dato == "_iv":
                        grigliavarianze = GridLayout(rows=4, columns=1)
                        grigliavarianze.add_widget(Label(text="Ascisse"))
                        intervallo1 = risultato.valore[0].intervallo.valore
                        copertura1 = risultato.valore[0].copertura
                        grigliaintervallo1 = GridLayout(rows=3, columns=2)
                        grigliaintervallo1.add_widget(Label(text="Intervallo:"))
                        grigliaintervallo1.add_widget(
                            Label(text=str([round(intervallo1[0], 3), round(intervallo1[1], 3)])))
                        grigliaintervallo1.add_widget(Label(text="Asintotico:"))
                        grigliaintervallo1.add_widget(Label(text="Sì" if risultato.valore[0].asintotico else "No"))
                        grigliaintervallo1.add_widget(Label(text="Livello di confidenza:"))
                        grigliaintervallo1.add_widget(Label(
                            text=str(100 * round(copertura1, 3)) + "%" if copertura1 is not None else "Non stimata"))
                        grigliavarianze.add_widget(grigliaintervallo1)
                        grigliavarianze.add_widget(Label(text="Ordinate"))
                        intervallo2 = risultato.valore[1].intervallo.valore
                        copertura2 = risultato.valore[1].copertura
                        grigliaintervallo2 = GridLayout(rows=3, columns=2)
                        grigliaintervallo2.add_widget(Label(text="Intervallo:"))
                        grigliaintervallo2.add_widget(
                            Label(text=str([round(intervallo2[0], 3), round(intervallo2[1], 3)])))
                        grigliaintervallo2.add_widget(Label(text="Asintotico:"))
                        grigliaintervallo2.add_widget(Label(text="Sì" if risultato.valore.asintotico else "No"))
                        grigliaintervallo2.add_widget(Label(text="Livello di confidenza:"))
                        grigliaintervallo2.add_widget(Label(
                            text=str(100 * round(copertura2, 3)) + "%" if copertura2 is not None else "Non stimata"))
                        grigliavarianze.add_widget(grigliaintervallo2)
                        grigliarisultato.add_widget(grigliavarianze)
                    elif dato == "tv":
                        grigliavarianze = GridLayout(rows=4, columns=1)
                        grigliavarianze.add_widget(Label(text="Ascisse"))
                        pvalue1 = risultato.valore[0].decisione.valore.pvalue
                        accettazione1 = risultato.valore[0].decisione.valore.accettazione
                        alfa1 = risultato.valore[0].alfa
                        beta1 = risultato.valore[0].beta
                        grigliatest1 = GridLayout(rows=4, columns=2)
                        grigliatest1.add_widget(Label(text="p-value:"))
                        grigliatest1.add_widget(Button(text=str(round(pvalue1, 3)), background_normal="",
                                                       background_color=(0, 1, 0.2, 1) if accettazione1 else (
                                                       1, 0, 0, 1)))
                        grigliatest1.add_widget(Label(text="Asintotico:"))
                        grigliatest1.add_widget(Label(text="Sì" if risultato.valore[0].asintotico else "No"))
                        if alfa1.errore is not None:
                            grigliatest1.add_widget(Label(text="Errore durante il metodo MC:"))
                            grigliatest1.add_widget(Label(text=alfa1.errore))
                        elif beta1.errore is not None:
                            grigliatest1.add_widget(Label(text="Errore nel metodo MC:"))
                            grigliatest1.add_widget(Label(text=beta1.errore))
                        elif alfa1 is not None:
                            grigliatest1.add_widget(Label(text="Probabilità di\nerrore di I tipo:"))
                            grigliatest1.add_widget(Label(text=str(100 * round(alfa1.valore, 3)) + "%"))
                            grigliatest1.add_widget(Label(text="Tempo del metodo MC:"))
                            grigliatest1.add_widget(Label(text=str(round(alfa1.tempo, 3))))
                        elif beta1 is not None:
                            grigliatest1.add_widget(Label(text="Probabilità di\nerrore di II tipo:"))
                            grigliatest1.add_widget(Label(text=str(100 * round(beta1.valore, 3)) + "%"))
                            grigliatest1.add_widget(Label(text="Tempo del metodo MC:"))
                            grigliatest1.add_widget(Label(text=str(round(beta1.tempo, 3))))
                        else:
                            grigliatest1.add_widget(Label(text="Metodo MC non eseguito"))
                        grigliavarianze.add_widget(grigliatest1)
                        grigliavarianze.add_widget(Label(text="Ordinate"))
                        pvalue2 = risultato.valore[1].decisione.valore.pvalue
                        accettazione2 = risultato.valore[1].decisione.valore.accettazione
                        alfa2 = risultato.valore[1].alfa
                        beta2 = risultato.valore[1].beta
                        grigliatest2 = GridLayout(rows=4, columns=2)
                        grigliatest2.add_widget(Label(text="p-value:"))
                        grigliatest2.add_widget(Button(text=str(round(pvalue2, 3)), background_normal="",
                                                       background_color=(0, 1, 0.2, 1) if accettazione2 else (
                                                       1, 0, 0, 1)))
                        grigliatest2.add_widget(Label(text="Asintotico:"))
                        grigliatest2.add_widget(Label(text="Sì" if risultato.valore.asintotico else "No"))
                        if alfa2.errore is not None:
                            grigliatest2.add_widget(Label(text="Errore durante il metodo MC:"))
                            grigliatest2.add_widget(Label(text=alfa2.errore))
                        elif beta2.errore is not None:
                            grigliatest2.add_widget(Label(text="Errore nel metodo MC:"))
                            grigliatest2.add_widget(Label(text=beta2.errore))
                        elif alfa2 is not None:
                            grigliatest2.add_widget(Label(text="Probabilità di\nerrore di I tipo:"))
                            grigliatest2.add_widget(Label(text=str(100 * round(alfa2.valore, 3)) + "%"))
                            grigliatest2.add_widget(Label(text="Tempo del metodo MC:"))
                            grigliatest2.add_widget(Label(text=str(round(alfa2.tempo, 3))))
                        elif beta2 is not None:
                            grigliatest2.add_widget(Label(text="Probabilità di\nerrore di II tipo:"))
                            grigliatest2.add_widget(Label(text=str(100 * round(beta2.valore, 3)) + "%"))
                            grigliatest2.add_widget(Label(text="Tempo del metodo MC:"))
                            grigliatest2.add_widget(Label(text=str(round(beta2.tempo, 3))))
                        else:
                            grigliatest2.add_widget(Label(text="Metodo MC non eseguito"))
                        grigliavarianze.add_widget(grigliatest2)
                        grigliarisultato.add_widget(grigliavarianze)
                    elif dato == "fd":
                        dizionario = risultato.valore
                        grigliadifetti = GridLayout(rows=len(dizionario), columns=2)
                        for f in dizionario:
                            grigliadifetti.add_widget(Label(text=str(f)+":"))
                            pvalue = risultato.valore[f].decisione.valore.pvalue
                            accettazione = risultato.valore[f].decisione.valore.accettazione
                            alfa = risultato.valore[f].alfa
                            beta = risultato.valore[f].beta
                            grigliatest = GridLayout(rows=4, columns=2)
                            grigliatest.add_widget(Label(text="p-value:"))
                            grigliatest.add_widget(Button(text=str(round(pvalue, 3)), background_normal="",
                                                          background_color=(0, 1, 0.2, 1) if accettazione else (
                                                          1, 0, 0, 1)))
                            grigliatest.add_widget(Label(text="Asintotico:"))
                            grigliatest.add_widget(Label(text="Sì" if risultato.valore[f].asintotico else "No"))
                            if alfa.errore is not None:
                                grigliatest.add_widget(Label(text="Errore durante il metodo MC:"))
                                grigliatest.add_widget(Label(text=alfa.errore))
                            elif beta.errore is not None:
                                grigliatest.add_widget(Label(text="Errore nel metodo MC:"))
                                grigliatest.add_widget(Label(text=beta.errore))
                            elif alfa is not None:
                                grigliatest.add_widget(Label(text="Probabilità di\nerrore di I tipo:"))
                                grigliatest.add_widget(Label(text=str(100 * round(alfa.valore, 3)) + "%"))
                                grigliatest.add_widget(Label(text="Tempo del metodo MC:"))
                                grigliatest.add_widget(Label(text=str(round(alfa.tempo, 3))))
                            elif beta is not None:
                                grigliatest.add_widget(Label(text="Probabilità di\nerrore di II tipo:"))
                                grigliatest.add_widget(Label(text=str(100 * round(beta.valore, 3)) + "%"))
                                grigliatest.add_widget(Label(text="Tempo del metodo MC:"))
                                grigliatest.add_widget(Label(text=str(round(beta.tempo, 3))))
                            else:
                                grigliatest.add_widget(Label(text="Metodo MC non eseguito"))
                            grigliadifetti.add_widget(grigliatest)
                        grigliarisultato.add_widget(grigliadifetti)
                    elif dato == "vd":
                        dizionario = risultato.valore
                        grigliadifetti = GridLayout(rows=4, columns=1)
                        grigliadifettiascisse = GridLayout(rows=len(dizionario), columns=2)
                        grigliadifettiordinate = GridLayout(rows=len(dizionario), columns=2)
                        for f in dizionario:
                            grigliadifettiascisse.add_widget(Label(text=str(f)+":"))
                            pvalue = risultato.valore[f][0].decisione.valore.pvalue
                            accettazione = risultato.valore[f][0].decisione.valore.accettazione
                            alfa = risultato.valore[f][0].alfa
                            beta = risultato.valore[f][0].beta
                            grigliatest = GridLayout(rows=4, columns=2)
                            grigliatest.add_widget(Label(text="p-value:"))
                            grigliatest.add_widget(Button(text=str(round(pvalue, 3)), background_normal="",
                                                          background_color=(0, 1, 0.2, 1) if accettazione else (
                                                          1, 0, 0, 1)))
                            grigliatest.add_widget(Label(text="Asintotico:"))
                            grigliatest.add_widget(Label(text="Sì" if risultato.valore[f][0].asintotico else "No"))
                            if alfa.errore is not None:
                                grigliatest.add_widget(Label(text="Errore durante il metodo MC:"))
                                grigliatest.add_widget(Label(text=alfa.errore))
                            elif beta.errore is not None:
                                grigliatest.add_widget(Label(text="Errore nel metodo MC:"))
                                grigliatest.add_widget(Label(text=beta.errore))
                            elif alfa is not None:
                                grigliatest.add_widget(Label(text="Probabilità di\nerrore di I tipo:"))
                                grigliatest.add_widget(Label(text=str(100 * round(alfa.valore, 3)) + "%"))
                                grigliatest.add_widget(Label(text="Tempo del metodo MC:"))
                                grigliatest.add_widget(Label(text=str(round(alfa.tempo, 3))))
                            elif beta is not None:
                                grigliatest.add_widget(Label(text="Probabilità di\nerrore di II tipo:"))
                                grigliatest.add_widget(Label(text=str(100 * round(beta.valore, 3)) + "%"))
                                grigliatest.add_widget(Label(text="Tempo del metodo MC:"))
                                grigliatest.add_widget(Label(text=str(round(beta.tempo, 3))))
                            else:
                                grigliatest.add_widget(Label(text="Metodo MC non eseguito"))
                            grigliadifettiascisse.add_widget(grigliatest)
                            grigliadifettiordinate.add_widget(Label(text=str(f)+":"))
                            pvalue_ = risultato.valore[f][1].decisione.valore.pvalue
                            accettazione_ = risultato.valore[f][1].decisione.valore.accettazione
                            alfa_ = risultato.valore[f][1].alfa
                            beta_ = risultato.valore[f][1].beta
                            grigliatest_ = GridLayout(rows=4, columns=2)
                            grigliatest_.add_widget(Label(text="p-value:"))
                            grigliatest_.add_widget(Button(text=str(round(pvalue_, 3)), background_normal="",
                                                           background_color=(0, 1, 0.2, 1) if accettazione_ else (
                                                           1, 0, 0, 1)))
                            grigliatest_.add_widget(Label(text="Asintotico:"))
                            grigliatest_.add_widget(Label(text="Sì" if risultato.valore[f][1].asintotico else "No"))
                            if alfa_.errore is not None:
                                grigliatest_.add_widget(Label(text="Errore durante il metodo MC:"))
                                grigliatest_.add_widget(Label(text=alfa_.errore))
                            elif beta_.errore is not None:
                                grigliatest_.add_widget(Label(text="Errore nel metodo MC:"))
                                grigliatest_.add_widget(Label(text=beta_.errore))
                            elif alfa_ is not None:
                                grigliatest_.add_widget(Label(text="Probabilità di\nerrore di I tipo:"))
                                grigliatest_.add_widget(Label(text=str(100 * round(alfa_.valore, 3)) + "%"))
                                grigliatest_.add_widget(Label(text="Tempo del metodo MC:"))
                                grigliatest_.add_widget(Label(text=str(round(alfa_.tempo, 3))))
                            elif beta is not None:
                                grigliatest_.add_widget(Label(text="Probabilità di\nerrore di II tipo:"))
                                grigliatest_.add_widget(Label(text=str(100 * round(beta_.valore, 3)) + "%"))
                                grigliatest_.add_widget(Label(text="Tempo del metodo MC:"))
                                grigliatest_.add_widget(Label(text=str(round(beta_.tempo, 3))))
                            else:
                                grigliatest_.add_widget(Label(text="Metodo MC non eseguito"))
                            grigliadifettiordinate.add_widget(grigliatest_)
                        grigliadifetti.add_widget(Label(text="Ascisse"))
                        grigliadifetti.add_widget(grigliadifettiascisse)
                        grigliadifetti.add_widget(Label(text="Ordinate"))
                        grigliadifetti.add_widget(grigliadifettiordinate)
                        grigliarisultato.add_widget(grigliadifetti)
                    elif dato == "mvm":
                        medie, kappa, componenti, assegnazioni = risultato.valore
                        grigliamisture = GridLayout(rows=3, columns=2)
                        grigliamisture.add_widget(Label(text="Angoli medi:"))
                        grigliamedie = GridLayout(rows=len(componenti), columns=2)
                        for c in range(len(componenti)):
                            grigliamedie.add_widget(Label(text=str(c)+":"))
                            grigliamedie.add_widget(Label(text=str([round(medie[c][0], 3), round(medie[c][1], 3)])))
                        grigliamisture.add_widget(grigliamedie)
                        grigliamisture.add_widget(Label(text="Kappa:"))
                        grigliakappa = GridLayout(rows=len(componenti), columns=2)
                        for c in range(len(componenti)):
                            grigliakappa.add_widget(Label(text=str(c)+":"))
                            grigliakappa.add_widget(Label(text=str([round(kappa[c][0], 3), round(kappa[c][1], 3)])))
                        grigliamisture.add_widget(grigliakappa)
                        grigliamisture.add_widget(Label(text="Pesi di mistura:"))
                        grigliapesi = GridLayout(rows=len(componenti), columns=2)
                        for c in range(len(componenti)):
                            grigliapesi.add_widget(Label(text=str(c)+":"))
                            grigliapesi.add_widget(Label(text=str(round(componenti[c][1]))))
                        grigliamisture.add_widget(grigliapesi)
                    elif dato == "irpb" or dato == "hb":
                        grigliabayesiana = GridLayout(rows=2, columns=2)
                        grigliabayesiana.add_widget(Label(text="Intervallo:"))
                        intervallo = risultato.valore[1].intervallo.valore
                        copertura = risultato.valore[1].copertura
                        grigliaintervallo = GridLayout(rows=2, columns=2)
                        grigliaintervallo.add_widget(Label(text="Intervallo:"))
                        grigliaintervallo.add_widget(
                            Label(text=str([round(intervallo[0], 3), round(intervallo[1], 3)])))
                        grigliaintervallo.add_widget(Label(text="Livello di credibilità:"))
                        grigliaintervallo.add_widget(Label(
                            text=str(100 * round(copertura, 3)) + "%" if copertura is not None else "Non stimata"))
                        grigliabayesiana.add_widget(grigliaintervallo)
                        grigliabayesiana.add_widget(Label(text="Probabilità di H0:"))
                        grigliabayesiana.add_widget(Label(text=str(round(risultato.valore[0], 3))))
                        grigliarisultato.add_widget(grigliabayesiana)
                    elif dato == "vb":
                        grigliabayesiana = GridLayout(rows=2, columns=1)
                        grigliavarianze = GridLayout(rows=4, columns=1)
                        grigliavarianze.add_widget(Label(text="Ascisse"))
                        intervallo1 = risultato.valore[1][0].intervallo.valore
                        copertura1 = risultato.valore[1][0].copertura
                        grigliaintervallo1 = GridLayout(rows=3, columns=2)
                        grigliaintervallo1.add_widget(Label(text="Intervallo:"))
                        grigliaintervallo1.add_widget(
                            Label(text=str([round(intervallo1[0], 3), round(intervallo1[1], 3)])))
                        grigliaintervallo1.add_widget(Label(text="Asintotico:"))
                        grigliaintervallo1.add_widget(Label(text="Sì" if risultato.valore[1][0].asintotico else "No"))
                        grigliaintervallo1.add_widget(Label(text="Livello di confidenza:"))
                        grigliaintervallo1.add_widget(Label(
                            text=str(100 * round(copertura1, 3)) + "%" if copertura1 is not None else "Non stimata"))
                        grigliavarianze.add_widget(grigliaintervallo1)
                        grigliavarianze.add_widget(Label(text="Ordinate"))
                        intervallo2 = risultato.valore[1][1].intervallo.valore
                        copertura2 = risultato.valore[1][1].copertura
                        grigliaintervallo2 = GridLayout(rows=3, columns=2)
                        grigliaintervallo2.add_widget(Label(text="Intervallo:"))
                        grigliaintervallo2.add_widget(
                            Label(text=str([round(intervallo2[0], 3), round(intervallo2[1], 3)])))
                        grigliaintervallo2.add_widget(Label(text="Asintotico:"))
                        grigliaintervallo2.add_widget(Label(text="Sì" if risultato.valore[1][1].asintotico else "No"))
                        grigliaintervallo2.add_widget(Label(text="Livello di confidenza:"))
                        grigliaintervallo2.add_widget(Label(
                            text=str(100 * round(copertura2, 3)) + "%" if copertura2 is not None else "Non stimata"))
                        grigliavarianze.add_widget(grigliaintervallo2)
                        grigliabayesiana.add_widget(grigliavarianze)
                        grigliavarianze_ = GridLayout(rows=4, columns=1)
                        grigliavarianze_.add_widget(Label(text="Ascisse"))
                        grigliavarianze_.add_widget(Label(text=f"Probabilità di H0: {risultato.valore[0][0]}"))
                        grigliavarianze_.add_widget(Label(text="Ordinate"))
                        grigliavarianze_.add_widget(Label(text=f"Probabilità di H0: {risultato.valore[0][1]}"))
                        grigliabayesiana.add_widget(grigliavarianze_)
                        grigliarisultato.add_widget(grigliabayesiana)
                    elif dato == "amb":
                        grigliabayesiana = GridLayout(rows=4, columns=2)
                        testkappa, kappa, intervallo, intervallokappa = risultato.valore
                        if testkappa is not None:
                            grigliabayesiana.add_widget(Label(text="Probabilità di H0 su kappa:"))
                            grigliabayesiana.add_widget(Label(text=str(100*round(testkappa, 3))+"%"))
                        if kappa is not None:
                            grigliabayesiana.add_widget(Label(text="Stima puntuale di kappa:"))
                            grigliabayesiana.add_widget(Label(text=str(round(kappa, 3))))
                        if intervallo is not None:
                            grigliabayesiana.add_widget(Label(text="Intervallo dell'angolo medio:"))
                            grigliabayesiana.add_widget(Label(text=str([round(intervallo.intervallo.valore[0], 3), round(intervallo.intervallo.valore[1], 3)])))
                        if intervallokappa is not None:
                            grigliabayesiana.add_widget(Label(text="Intervallo per kappa:"))
                            grigliabayesiana.add_widget(Label(text=str([round(intervallo.intervallo.valore[0], 3), round(intervallo.intervallo.valore[1], 3)])))
                        grigliarisultato.add_widget(grigliabayesiana)
                    else:
                        raise IndexError(f"Chiave non considerata: {dato}")
                    grigliarisultato.add_widget(Label(text=f"Tempo di calcolo:\n{self.analisi.libro.contenuto[dato].tempo}"))
                    if dato in parametri:
                        grigliaregolazione = GridLayout(rows=1, columns=3)
                        mutandi = parametri[dato]
                        grigliadescrizioni = GridLayout(rows=len(mutandi), columns=1)
                        grigliacampi = GridLayout(rows=len(mutandi), columns=1)
                        grigliaconfermatori = GridLayout(rows=len(mutandi), columns=1)
                        for chiave in mutandi:
                            grigliadescrizioni.add_widget(Label(text=etichette[chiave] + ":"))
                            if tipi[chiave] == int or tipi[chiave] == float:
                                campo = TextInput(text=self.impostazioni[chiave], multiline=False)
                            elif tipi[chiave] == bool:
                                campo = CheckBox()
                            elif tipi[chiave] == str:
                                scritte = opzioni[chiave]
                                campo = Spinner(text=scritte[0], values=scritte)
                            grigliacampi.add_widget(campo)
                            bottone = Button(text="Ricalcola")
                            bottone.bind(lambda _, dato=dato, chiave=chiave, campo=campo: self.nuovoparametro(dato, chiave, campo))
                        grigliaregolazione.add_widget(grigliadescrizioni)
                        grigliaregolazione.add_widget(grigliacampi)
                        grigliaregolazione.add_widget(grigliaconfermatori)
                    griglia.add_widget(grigliaregolazione)
                else:
                    griglia = GridLayout(rows=1, columns=2)
                    griglia.add_widget(Label(text=f"Non hai calcolato {etichette[dato].lower()}. Vuoi farlo?"))
                    calcolatore = Button(text="Calcola")
                    calcolatore.bind(lambda _: self.ricalcola(dato))
                self.scatola.add_widget(griglia)

    def cambiaparametro(self, nomeanalisi, parametro, valore):
        self.analisi.impostazioni[parametro] = valore.active if tipi[parametro] == bool else valore.text
        self.ricalcola(nomeanalisi)

    def ricalcola(self, nomeanalisi):
        try:
            self.analisi.modifica(nomeanalisi, self.analisi.impostazioni)
        except ValueError as ve:
            print(f"Un parametro è invalido: {ve}")
        else:
            scrivifile(filesessione(self.id_), self.analisi)
            self.caricaanalisi()

    def mostragrafico(self, nomeanalisi: str, immagine: Image, etichetta: Label, bottone: Button, griglia: GridLayout):
        mostra_grafico(self, nomeanalisi, self.analisi.libro, immagine)
        etichetta.text = f"Grafico di {etichettegrafici[nomeanalisi]}:"
        griglia.remove_widget(bottone)
        griglia.add_widget(immagine)

    def go_back(self):
        self.manager.current = "Diario delle sessioni"
