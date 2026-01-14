import cv2
from kivy.uix.screenmanager import Screen
from plyer import filechooser
from classi.dati import SessioneGrezza, Sessione
from letturascrittura.memoria import leggifile, creafile
from letturascrittura.percorsi import filesessione, fileimpostazionianalisi
from roboflow import Roboflow
import supervision as sv


def istanziasessione(file: str | None=None):
    if file is not None:
        oggetto: SessioneGrezza = leggifile(file)
        impostazioni: dict = leggifile(fileimpostazionianalisi())
        sessione = Sessione(oggetto, impostazioni)
        creafile(filesessione(oggetto.id_), sessione)


class NuovoAllenamento(Screen):
    def torna(self):
        self.manager.current = "Menù principale"

    def file(self):
        filechooser.open_file(title="Seleziona file dati",
                              filters=[("File dati", "*.pkl;*.pickle;*.npz;*.csv"),
                                       ("Tutti i file", "*.*")],
                              on_selection=self.analizzafile)

    def analizzafile(self, selection):
        if not selection:
            return
        percorso = selection[0]
        print(f"File selezionato: {percorso}")
        istanziasessione(file=percorso)
        self.manager.current = "Diario delle sessioni"

    def analizzaimmagine(self, selection):
        if not selection:
            return
        percorso = selection[0]
        try:
            self.roboflow = Roboflow(api_key="eUmOoEWt4GStTIydwsM6")
        except ConnectionError:
            print("Manca la connessione ad Internet")
            self.manager.current = "Menù principale"
        else:
            project = self.roboflow.workspace().project("statisticallearningcolbrutti")
            model = project.version(6).model
            # Rendili regolabili
            result = model.predict(percorso, confidence=40, overlap=30).json()
            labels = [item["class"] for item in result["predictions"]]
            detections = sv.Detections.from_roboflow(result)
            label_annotator = sv.LabelAnnotator()
            bounding_box_annotator = sv.BoxAnnotator()
            image = cv2.imread(percorso)
            annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
            annotatore = self.manager.get_screen("Annotazione")
            annotatore.caricaimmagine(immagine=annotated_image, coordinate=result)
            self.manager.current = "Annotazione"

    def immagine(self):
        filechooser.open_file(title="Seleziona file immagine",
                              filters=[("File dati", "*.png;*.jpg"),
                                       ("Tutti i file", "*.*")],
                              on_selection=self.analizzaimmagine)

    def manuale(self):
        self.manager.current = "Annotazione"
