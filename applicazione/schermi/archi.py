from kivy.uix.screenmanager import Screen
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.properties import ListProperty, DictProperty, NumericProperty
from letturascrittura.memoria import leggifile, scrivifile
from letturascrittura.percorsi import filearchi
from classi.archi import ArcoOlimpico, ArcoNudo, ArcoCompound


class ImpostazioniArchi(Screen):
    archi = ListProperty([])            # lista di dict
    indice_selezionato = NumericProperty(-1)
    arco_corrente = DictProperty({})    # arco in modifica

    def on_pre_enter(self):
        self.carica_archi()

    def carica_archi(self):
        self.archi = leggifile(filearchi())
        container = self.ids.lista_archi
        container.clear_widgets()
        if len(self.archi) < 1:
            container.add_widget(Label(text="Nessun arco."))
        else:
            for i, arco in enumerate(self.archi):
                btn = Button(text=arco["nomesetup"], size_hint_y=None, height="40dp",
                             on_release=lambda _, idx=i: self.seleziona_arco(idx))
                container.add_widget(btn)
        self.nuovo_arco()

    def seleziona_arco(self, idx):
        self.indice_selezionato = idx
        self.arco_corrente = self.archi[idx]
        self.popola_arco()

    def nuovo_arco(self):
        self.indice_selezionato = -1
        self.arco_corrente = ArcoOlimpico()
        self.popola_arco()

    def popola_arco(self):
        attributi = self.arco_corrente.__dict__
        scatola = self.ids.lista_attributi
        for attributo in attributi:
            scatola.add_widget(Label(text=attributo+":"))
            scatola.add_widget(TextInput(text=attributi[attributo], multiline=False))

    def tipoarco(self, tipo):
        self.arco_corrente = {"Olimpico": ArcoOlimpico(), "Nudo": ArcoNudo(), "Compound": ArcoCompound()}[tipo]

    def salva_arco(self):
        if not self.arco_corrente.get("nomesetup"):
            return
        if self.indice_selezionato >= 0:
            self.archi[self.indice_selezionato] = self.arco_corrente
        else:
            self.archi.append(self.arco_corrente)
        scrivifile(filearchi(), self.archi)
        self.carica_archi()

    def cancella_arco(self):
        if self.indice_selezionato < 0:
            return
        arco = self.archi[self.indice_selezionato]
        self.archi.remove(arco)
        scrivifile(filearchi(), self.archi)
        self.carica_archi()

    def torna(self):
        self.manager.current = "Impostazioni"
