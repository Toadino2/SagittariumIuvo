from kivy.uix.screenmanager import Screen
from kivy.properties import DictProperty


class StatisticheGenerali(Screen):
    results = DictProperty({})

    def on_pre_enter(self):
        self.caricaanalisi()

    def caricaanalisi(self):
        # TODO: services.load_global_statistics()
        self.results = {
            "media_globale": {"value": "9.85", "has_plot": False},
            "trend": {"value": None, "has_plot": True},
        }

    def ricalcola(self, nomeanalisi):
        # TODO: services.run_global_analysis([analysis_name])
        self.caricaanalisi()

    def mostragrafico(self, nomeanalisi):
        # TODO: services.get_global_plot(analysis_name)
        pass

    def go_back(self):
        self.manager.current = "Menù principale"
