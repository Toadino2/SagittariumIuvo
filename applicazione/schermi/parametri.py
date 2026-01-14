from kivy.uix.screenmanager import Screen
from kivy.properties import DictProperty
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox
from kivy.uix.spinner import Spinner
from letturascrittura.memoria import leggifile, scrivifile
from letturascrittura.percorsi import fileimpostazionianalisi
from classi.tipi import sequenza, significati, tipi, opzioni, vincoli, controllaimpostazioni

sezioni = ["statistiche descrittive per frecce", "statistiche descrittive generiche e per volée",
           "clustering sulle coordinate", "inferenza frequentista sulle varianze", "test di multinormalità",
           "metodi Monte Carlo", "test di Shapiro-Wilk", "inferenza frequentista sulla media bivariata",
           "mediana geometrica", "inferenza frequentista sugli IRP", "difettosità delle frecce",
           "test di Rayleigh", "inferenza frequentista sugli angoli", "inferenza bayesiana sulle varianze",
           "inferenza bayesiana sulla media bivariata", "inferenza bayesiana sugli IRP",
           "inferenza bayesiana sull'angolo medio", "clustering sugli angoli"]


class SceltaParametri(Screen):
    impostazioni = DictProperty({})

    def on_pre_enter(self):
        self.caricaimpostazioni()

    def caricaimpostazioni(self):
        self.impostazioni = leggifile(fileimpostazionianalisi())
        griglia = self.ids.lista_impostazioni
        self.campi = dict()
        for chiave in sequenza:
            if isinstance(chiave, int):
                griglia.add_widget(Label(text="Sezione:"))
                griglia.add_widget(Label(text=sezioni[chiave]))
            elif isinstance(chiave, str):
                griglia.add_widget(Label(text=significati[chiave]+":"))
                if tipi[chiave] == int or tipi[chiave] == float:
                    campo = TextInput(text=self.impostazioni[chiave], multiline=False)
                elif tipi[chiave] == bool:
                    campo = CheckBox()
                elif tipi[chiave] == str:
                    scritte = opzioni[chiave]
                    campo = Spinner(text=scritte[0], values=scritte)
                self.campi[chiave] = campo
                griglia.add_widget(self.campi[chiave])

    def salvaimpostazioni(self):
        nuoveimpostazioni = dict()
        for campo in self.campi:
            oggetto = self.campi[campo]
            if tipi[campo] == int:
                nuoveimpostazioni[campo] = int(oggetto.text)
            elif tipi[campo] == float:
                nuoveimpostazioni[campo] = float(oggetto.text)
            elif tipi[campo] == bool:
                nuoveimpostazioni[campo] = oggetto.active
            elif tipi[campo] == str:
                nuoveimpostazioni[campo] = vincoli[opzioni[campo].index(oggetto.text)]
        try:
            controllaimpostazioni(nuoveimpostazioni)
        except ValueError as ve:
            print(f"Un parametro è invalido: {ve}")
        else:
            self.impostazioni = nuoveimpostazioni
            scrivifile(fileimpostazionianalisi(), self.impostazioni)

    def torna(self):
        self.manager.current = "Impostazioni"
