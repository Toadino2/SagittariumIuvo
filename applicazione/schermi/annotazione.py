from applicazione.schermi.nuovo import istanziasessione
from applicazione.schermi.generali import Generali
from classi.dati import Sessione, SessioneGrezza, Dati, MetadatiSessione
from kivy.uix.screenmanager import Screen
from kivy.properties import ListProperty, StringProperty, NumericProperty
from datetime import date, datetime
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ListProperty
from kivy.event import EventDispatcher
from kivy.graphics import Color, Line, Ellipse, Rectangle
import math
import numpy as np
from os import listdir
from letturascrittura.memoria import leggifile, creafile, scrivifile
from letturascrittura.percorsi import filearchi, filesessione, cartellasessioni, fileimpostazionianalisi, filegenerale


def disegnabersaglio(tela, posizione: tuple | None, x: float | int, y: float | int, larghezza: float | int):
    # Disegna un bersaglio dato un oggetto canvas di una certa larghezza.
    if posizione is None:
        posizione = (x, y)
    tela.add(Color(0, 3, 0, 1))
    tela.add(Rectangle(pos=posizione, size=(larghezza, larghezza)))
    tela.add(Color(3, 3, 3, 1))
    tela.add(Ellipse(pos=posizione, size=(larghezza, larghezza)))
    tela.add(Color(0, 0, 0, 1))
    tela.add(Line(ellipse=(x + larghezza / 20, y + larghezza / 20, larghezza * 9 / 10, larghezza * 9 / 10)))
    tela.add(Ellipse(pos=(x + larghezza / 10, y + larghezza / 10), size=(larghezza * 8 / 10, larghezza * 8 / 10)))
    tela.add(Color(3, 3, 3, 1))
    tela.add(Line(ellipse=(x + larghezza * 3 / 20, y + larghezza * 3 / 20, larghezza * 7 / 10, larghezza * 7 / 10)))
    tela.add(Color(0, 0, 3, 1))
    tela.add(Ellipse(pos=(x + larghezza * 4 / 20, y + larghezza * 4 / 20),
                     size=(larghezza * 6 / 10, larghezza * 6 / 10)))
    tela.add(Color(0, 0, 0, 1))
    tela.add(Line(ellipse=(x + larghezza * 5 / 20, y + larghezza * 5 / 20, larghezza * 5 / 10, larghezza * 5 / 10)))
    tela.add(Color(3, 0, 0, 1))
    tela.add(Ellipse(pos=(x + larghezza * 6 / 20, y + larghezza * 6 / 20),
                     size=(larghezza * 4 / 10, larghezza * 4 / 10)))
    tela.add(Color(0, 0, 0, 1))
    tela.add(Line(ellipse=(x + larghezza * 7 / 20, y + larghezza * 7 / 20, larghezza * 3 / 10, larghezza * 3 / 10)))
    tela.add(Color(3, 3, 0, 1))
    tela.add(Ellipse(pos=(x + larghezza * 8 / 20, y + larghezza * 8 / 20),
                     size=(larghezza * 2 / 10, larghezza * 2 / 10)))
    tela.add(Color(0, 0, 0, 1))
    tela.add(Line(ellipse=(x + larghezza * 9 / 20, y + larghezza * 9 / 20, larghezza / 10, larghezza / 10)))


class BersaglioToccabile(Widget):
    """
    Widget minimale:
    - disegna un bersaglio (cerchi concentrici + croce)
    - touch per aggiungere frecce
    - mantiene volley corrente (impostato dallo Screen)
    - conserva lista frecce per ridisegno
    """
    voléeattuale = NumericProperty(0)
    coordinate = ListProperty([])
    volée = ListProperty([])
    identificativi = ListProperty([])
    indice_selezionato = NumericProperty(-1)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_event_type("on_arrow_added")
        self.bind(pos=self._redraw, size=self._redraw)
        self.bind(coordinate=self._redraw, volée=self._redraw)
        self.bind(indice_selezionato=self._redraw)

    def mettivolée(self, v: int):
        self.voléeattuale = int(v)

    def mettifrecce(self, listacoordinate, listavolée, listaidentificativi):
        self.coordinate = list(listacoordinate)
        self.volée = list(listavolée)
        self.identificativi = list(listaidentificativi)

    def _center_radius(self):
        cx = self.x + self.width * 0.5
        cy = self.y + self.height * 0.5
        r = min(self.width, self.height) * 0.45
        return cx, cy, r

    def _to_normalized(self, x_px, y_px):
        cx, cy, r = self._center_radius()
        x = (x_px - cx) / r
        y = (y_px - cy) / r
        x = max(-1.2, min(1.2, x))
        y = max(-1.2, min(1.2, y))
        return float(x), float(y)

    def _to_pixels(self, x, y):
        cx, cy, r = self._center_radius()
        return cx + x * r, cy + y * r

    def _redraw(self, *args):
        self.canvas.clear()
        cx, cy, r = self._center_radius()
        disegnabersaglio(self.canvas, (cx-r/2, cy-r/2), cx, cy, r)
        for i, ((x, y), v) in enumerate(zip(self.coordinate, self.volée)):
            px, py = self._to_pixels(x, y)
            if i == self.indice_selezionato:
                Color(3, 1, 0, 1)
                r = 6
            else:
                Color(1, 0, 1, 1)
                r = 4
            Ellipse(pos=(px - r, py - r), size=(2 * r, 2 * r))

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return super().on_touch_down(touch)
        x, y = self._to_normalized(*touch.pos)
        for i, (fx, fy) in enumerate(self.coordinate):
            if (x - fx) ** 2 + (y - fy) ** 2 < 0.02 ** 2:
                self.indice_selezionato = i
                return True
        volée = int(self.voléeattuale)
        self.dispatch("on_arrow_added", x, y, volée)
        return True

    def on_arrow_added(self, x, y, volée):
        pass


def listaarchi():
    return leggifile(filearchi())


def sistemadata(s: str):
    dt = datetime.strptime(s.strip(), "%Y-%m-%d").date()
    return dt.isoformat()


def scegliid() -> int:
    return len(listdir(cartellasessioni()))


class SchermataAnnotazione(Screen):
    coordinate = ListProperty([])
    volée = ListProperty([])
    identificativi = ListProperty([])
    voléeattuale = NumericProperty(0)
    data = StringProperty(date.today().isoformat())
    arco = StringProperty("")
    tag = StringProperty("")
    status = StringProperty("")
    id_manual = StringProperty("")
    prossimo_id = NumericProperty(0)

    def caricaimmagine(self, immagine=None, coordinate=None, extra=None):
        self.immagine = immagine
        self.ids.img_cv.source = self.immagine
        self.ids.img_cv.reload()
        self.coordinate = list(coordinate)
        self.volée = [0] * len(coordinate)
        self.identificativi = list(range(len(coordinate)))
        self.prossimo_id = len(self.identificativi)
        self.ids.target.mettifrecce(self.coordinate, self.volée, self.identificativi)
        self.extra = extra or {}
        self.mettiimmagine()
        self.mostrafrecce()

    def on_pre_enter(self):
        try:
            archi = listaarchi()
            sp = self.ids.get("bow_spinner")
        except ValueError:
            print("Devi prima registrare un arco")
            self.manager.current = "Menù principale"
        else:
            if sp is not None:
                sp.values = archi
                if not self.arcoselezionato and archi:
                    self.selected_bow = archi[0]
                    sp.text = archi[0]
            tw = self.ids.get("target")
            if tw is not None:
                tw.mettivolée(self.voléeattuale)
                tw.mettifrecce(self.coordinate, self.volée, self.identificativi)

    def mettioggi(self):
        self.data = date.today().isoformat()
        self.status = ""

    def controlladata(self):
        try:
            self.data = sistemadata(self.data)
            self.status = ""
            return True
        except ValueError:
            self.status = "Data non valida. Usa formato YYYY-MM-DD."
            return False

    def selezionaarco(self, arco):
        self.arco = arco
        self.status = ""

    def nuovavolée(self):
        self.voléeattuale += 1
        tw = self.ids.get("target")
        if tw is not None:
            tw.mettivolée(self.voléeattuale)

    def annullafreccia(self):
        if not self.coordinate:
            return
        self.coordinate.pop()
        self.volée.pop()
        self.identificativi.pop()
        self.ids.target.indice_selezionato = -1
        tw = self.ids.get("target")
        if tw is not None:
            tw.mettifrecce(self.coordinate, self.volée, self.identificativi)

    def rimuovi_freccia_selezionata(self):
        idx = self.ids.target.indice_selezionato
        if idx < 0:
            return
        self.coordinate.pop(idx)
        self.volée.pop(idx)
        self.identificativi.pop(idx)
        self.ids.target.indice_selezionato = -1
        self.ids.target.set_existing_arrows(self.coordinate, self.volée, self.identificativi)
        self.status = "Freccia rimossa"

    def nuovafreccia(self, x, y, volée):
        if self.id_manual.strip() != "":
            try:
                id_freccia = int(self.id_manual)
            except ValueError:
                self.status = "ID freccia non valido"
                return
        else:
            id_freccia = self.prossimo_id
            self.prossimo_id += 1
        self.coordinate.append((x, y))
        self.volée.append(volée)
        self.identificativi.append(id_freccia)
        self.status = f"Freccia aggiunta (volée {volée}). Totale: {len(self.coordinate)}"

    def solosalvataggio(self):
        if not self.sistemadata():
            return
        payload = self._build_payload()
        creafile(filesessione(scegliid()), payload)
        self.status = "Sessione salvata (senza analisi)."
        self.torna()

    def analizza(self):
        if not self.sistemadata():
            return
        payload = self._build_payload()
        self.status = "Sessione salvata + analisi avviata."
        creafile(filesessione(scegliid()), Sessione(payload, leggifile(fileimpostazionianalisi())))
        scrivifile(filegenerale(), Generali())
        self.torna()

    def _build_payload(self):
        tag = [t.strip() for t in self.tag.split(",") if t.strip()]
        dati = Dati(np.array(self.coordinate), self.ids.ordinatore.active, np.array(self.volée), np.array(self.identificativi))
        metadati = MetadatiSessione(type(self.arco), tag, date.fromisoformat(sistemadata(self.data)), self.arco, int(self.ids.distanziatore.text))
        return SessioneGrezza(scegliid(), dati, metadati)

    def torna(self):
        self.manager.current = "Nuovo allenamento"
