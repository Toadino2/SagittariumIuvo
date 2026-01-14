from classi.tipi import tipi, vincoli, tipi_, vincoli_
from letturascrittura.memoria import creafile
from kivy.app import App
from kivy.core.window import Window
from kivy.uix.gridlayout import GridLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.screenmanager import ScreenManager, FadeTransition
from kivy.uix.scrollview import ScrollView
from letturascrittura.percorsi import fileimpostazionianalisi, filearchi, filegenerale, fileimpostazionigenerali
from os.path import exists
from schermi.archi import ImpostazioniArchi
from schermi.annotazione import SchermataAnnotazione
from schermi.diario import DiarioSessioni
from schermi.generali import StatisticheGenerali
from schermi.impostazioni import SchermataImpostazioni
from schermi.manopole import ManopoleGenerali
from schermi.nuovo import NuovoAllenamento
from schermi.pagina import PaginaDiario
from schermi.parametri import SceltaParametri
from schermi.principale import MenùPrincipale


def mettibottoneindietro(funzione):
    Window.unbind(on_back_button=None)
    Window.bind(on_back_button=funzione)


def aggiungioggetti(aggiungendo: RelativeLayout | GridLayout | ScrollView, oggetti: tuple | list):
    for oggetto in oggetti:
        aggiungendo.add_widget(oggetto)


def rimuovioggetti(rimuovendo: RelativeLayout | GridLayout | ScrollView, oggetti: tuple | list):
    for oggetto in oggetti:
        rimuovendo.remove_widget(oggetto)


def impostazionipredefinite(t, v):
    impostazioni = dict()
    for chiave in t:
        if isinstance(t[chiave], int):
            impostazioni[chiave] = 1
        elif isinstance(t[chiave], float):
            impostazioni[chiave] = 0.5
        elif isinstance(t[chiave], str):
            elemento = v[chiave].pop()
            impostazioni[chiave] = elemento
            v[chiave].add(elemento)
        elif isinstance(t[chiave], bool):
            impostazioni[chiave] = True
    return impostazioni


class SagittariumApp(App):
    def build(self):
        sm = ScreenManager(transition=FadeTransition())
        sm.add_widget(MenùPrincipale(name="Menù principale"))
        sm.add_widget(NuovoAllenamento(name="Nuovo allenamento"))
        sm.add_widget(SchermataAnnotazione(name="Annotazione"))
        sm.add_widget(DiarioSessioni(name="Diario delle sessioni"))
        sm.add_widget(PaginaDiario(name="Pagina di diario"))
        sm.add_widget(StatisticheGenerali(name="Statistiche generali"))
        sm.add_widget(SchermataImpostazioni(name="Impostazioni"))
        sm.add_widget(ImpostazioniArchi(name="Impostazioni degli archi"))
        sm.add_widget(SceltaParametri(name="Impostazioni dell'applicazione e dell'analisi"))
        sm.add_widget(ManopoleGenerali(name="Impostazioni delle statistiche generali"))
        sm.current = "Menù principale"
        return sm


if __name__ == "__main__":
    if not exists(fileimpostazionianalisi()):
        creafile(fileimpostazionianalisi(), impostazionipredefinite(tipi, vincoli))
    if not exists(fileimpostazionigenerali()):
        creafile(fileimpostazionigenerali(), impostazionipredefinite(tipi_, vincoli_))
    if not exists(filearchi()):
        creafile(filearchi(), [])
    if not exists(filegenerale()):
        creafile(filegenerale(), None)
    SagittariumApp().run()
