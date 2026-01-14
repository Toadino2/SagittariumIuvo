from kivy.uix.screenmanager import Screen


class SchermataImpostazioni(Screen):
    def vediarchi(self):
        self.manager.current = "Impostazioni degli archi"

    def vediimpostazioni(self):
        self.manager.current = "Impostazioni di applicazione e analisi"

    def vedigenerali(self):
        self.manager.current = "Impostazioni delle statistiche generali"

    def torna(self):
        self.manager.current = "Menù principale"
