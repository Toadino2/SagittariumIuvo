from kivy.uix.screenmanager import Screen
from kivy.properties import DictProperty
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox
from kivy.uix.spinner import Spinner
from letturascrittura.percorsi import fileimpostazionigenerali
from letturascrittura.memoria import leggifile, scrivifile
from classi.tipi import sequenza_, significati_, tipi_, vincoli_, opzioni_

sezioni = {}


class ManopoleGenerali(Screen):
    impostazioni = DictProperty({})

    def on_pre_enter(self):
        self.caricaimpostazioni()

    def caricaimpostazioni(self):
        self.impostazioni = leggifile(fileimpostazionigenerali())
        griglia = self.ids.lista_impostazioni
        self.campi = dict()
        for chiave in sequenza_:
            if isinstance(chiave, int):
                griglia.add_widget(Label(text="Sezione:"))
                griglia.add_widget(Label(text=sezioni[chiave]))
            elif isinstance(chiave, str):
                griglia.add_widget(Label(text=significati_[chiave] + ":"))
                if tipi_[chiave] == int or tipi_[chiave] == float:
                    campo = TextInput(text=self.impostazioni[chiave], multiline=False)
                elif tipi_[chiave] == bool:
                    campo = CheckBox()
                elif tipi_[chiave] == str:
                    scritte = opzioni_[chiave]
                    campo = Spinner(text=scritte[0], values=scritte)
                self.campi[chiave] = campo
                griglia.add_widget(self.campi[chiave])

    def salvaimpostazioni(self):
        nuoveimpostazioni = dict()
        for campo in self.campi:
            oggetto = self.campi[campo]
            if tipi_[campo] == int:
                nuoveimpostazioni[campo] = int(oggetto.text)
            elif tipi_[campo] == float:
                nuoveimpostazioni[campo] = float(oggetto.text)
            elif tipi_[campo] == bool:
                nuoveimpostazioni[campo] = oggetto.active
            elif tipi_[campo] == str:
                nuoveimpostazioni[campo] = vincoli_[opzioni_[campo].index(oggetto.text)]
        try:
            controllaimpostazioni(nuoveimpostazioni)
        except ValueError as ve:
            print(f"Un parametro è invalido: {ve}")
        else:
            self.impostazionigenerali = nuoveimpostazioni
            scrivifile(fileimpostazionigenerali(), self.impostazioni)

    def torna(self):
        self.manager.current = "Impostazioni"
