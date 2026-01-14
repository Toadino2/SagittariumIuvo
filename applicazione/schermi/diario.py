from kivy.uix.screenmanager import Screen
from kivy.properties import ListProperty, StringProperty
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from letturascrittura.memoria import elencasessioni, cancellasessione, scrivifile
from letturascrittura.percorsi import filegenerale
from applicazione.schermi.generali import Generali


class DiarioSessioni(Screen):
    sessions = ListProperty([])

    def on_pre_enter(self):
        self.caricasessioni()

    def caricasessioni(self):
        self.sessions = elencasessioni()
        container = self.ids.sessions_container
        container.clear_widgets()
        for s in self.sessions:
            row = BoxLayout(orientation="horizontal", size_hint_y=None, height="40dp", spacing=10)
            btn_open = Button(text=s["date"], on_release=lambda _, sid=s["id"]: self.open_session(sid))
            btn_delete = Button(text="Cancella", size_hint_x=None, width="90dp", on_release=lambda _, sid=s["id"]: self.delete_session(sid))
            row.add_widget(btn_open)
            row.add_widget(btn_delete)
            container.add_widget(row)

    def aprisessione(self, id_):
        dettaglio = self.manager.get_screen("Pagina di diario")
        dettaglio.mettisessione(id_)
        self.manager.current = "Pagina di diario"

    def toglisessione(self, id_):
        cancellasessione(id_)
        scrivifile(filegenerale(), Generali())
        self.caricasessioni()

    def torna(self):
        self.manager.current = "Menù principale"
