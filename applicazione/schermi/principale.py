from kivy.uix.screenmanager import Screen


class MenùPrincipale(Screen):
    def esci(self):
        from kivy.app import App
        App.get_running_app().stop()

    def passa(self, screen_name):
        self.manager.current = screen_name
