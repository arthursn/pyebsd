from matplotlib.widgets import _SelectorWidget, RectangleSelector
from matplotlib.lines import Line2D
from matplotlib.path import Path
import numpy as np

__all__ = ["LassoSelector2", "RectangleSelector2"]


class LassoSelector2(_SelectorWidget):
    def __init__(self, ax, x, y, lineprops=None):
        super().__init__(ax, self.onselect, useblit=False, button=None)

        if lineprops is None:
            lineprops = dict()

        self.state_modifier_keys = dict(
            move=" ", accept="enter", clear="c", disconnect="escape"
        )

        self.verts = None
        self.line = Line2D([], [], **lineprops)
        self.ax.add_line(self.line)
        self._selection_artist = self.line

        self.data = list(zip(x, y))
        self.sel = np.ndarray(len(self.data), dtype=bool)
        self.sel[:] = False
        self.finished = False

        self._onselect = None

    def _press(self, event):
        if self.finished:
            self.line.set_data([[], []])
            self.verts = None
            self.finished = False

    def _release(self, event):
        if event.button == 1:
            if self.verts is None:
                self.verts = [self._get_data(event)]
            else:
                self.verts.append(self._get_data(event))
                self.line.set_data(list(zip(*self.verts)))
                self.update()
        elif event.button == 3:
            if self.verts is not None:
                self.accept()

    def onmove(self, event):
        if self.verts is None:
            return
        if not self.finished:
            xy = self.verts + [self._get_data(event)]
            self.line.set_data(list(zip(*xy)))
            self.update()

    def on_key_press(self, event):  # replace default action on_key_press
        if self.active:
            key = event.key or ""
            if key == self.state_modifier_keys["clear"]:
                self.clear()
                return
            if key == self.state_modifier_keys["accept"]:
                self.accept()
                return
            if key == self.state_modifier_keys["disconnect"]:
                self.clear()
                self.disconnect()
                return

    def onselect(self, verts):
        path = Path(verts)
        self.sel = path.contains_points(self.data)

        if self._onselect is not None:
            self._onselect()

    def accept(self):
        if len(self.verts) > 2 and not self.finished:
            xy = self.verts + [self.verts[0]]
            self.line.set_data(list(zip(*xy)))
            self.update()
            self.finished = True
            self.onselect(self.verts)
        else:
            self.clear()

    def clear(self):
        self.line.set_data([[], []])
        self.verts = None
        self.finished = True
        self.update()

    def disconnect(self):
        self.disconnect_events()


class RectangleSelector2(RectangleSelector):
    def __init__(self, ax, x, y, rectprops=None, aspect=None):
        RectangleSelector.__init__(
            self,
            ax,
            self.onselect,
            rectprops=rectprops,
            useblit=False,
            interactive=True,
        )

        self.state_modifier_keys = dict(
            move=" ",
            accept="enter",
            clear="c",
            disconnect="escape",
            square="shift",
            center="control",
        )

        self.data = list(zip(x, y))
        self.sel = []

        self._onselect = None

        if aspect is not None:
            if isinstance(aspect, (list, tuple, np.ndarray)):
                self.ratio = aspect[1] / aspect[0]
            elif type(aspect) == float:
                self.ratio = aspect
            elif type(aspect) == int:
                self.ratio = float(aspect)

    def _on_key_press(self, event):
        if self.active:
            key = event.key or ""
            if key == self.state_modifier_keys["disconnect"]:
                for artist in self.artists:
                    artist.set_visible(False)
                self.disconnect()
                return

    def _onmove(self, event):
        """on motion notify event if box/line is wanted"""
        # resize an existing shape
        if self.active_handle and not self.active_handle == "C":
            x1, x2, y1, y2 = self._extents_on_press

            if hasattr(self, "ratio"):
                center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]  # center
                dx = event.xdata - center[0]
                dy = event.ydata - center[1]
                if self.ratio < 1.0:
                    maxd = max(abs(dx), abs(dy / self.ratio))
                    if abs(dx) < maxd:
                        dx *= maxd / abs(dx)
                    if abs(dy) < maxd:
                        dy *= maxd / abs(dy / self.ratio)
                else:
                    maxd = max(abs(self.ratio * dx), abs(dy))
                    if abs(dx) < maxd:
                        dx *= maxd / abs(self.ratio * dx)
                    if abs(dy) < maxd:
                        dy *= maxd / abs(dy)
                x1, x2, y1, y2 = (
                    center[0] - dx,
                    center[0] + dx,
                    center[1] - dy,
                    center[1] + dy,
                )
            else:
                if self.active_handle in ["E", "W"] + self._corner_order:
                    x2 = event.xdata
                if self.active_handle in ["N", "S"] + self._corner_order:
                    y2 = event.ydata

        # move existing shape
        elif (
            "move" in self.state or self.active_handle == "C"
        ) and self._extents_on_press is not None:
            x1, x2, y1, y2 = self._extents_on_press
            dx = event.xdata - self.eventpress.xdata
            dy = event.ydata - self.eventpress.ydata
            x1 += dx
            x2 += dx
            y1 += dy
            y2 += dy

        # new shape
        else:
            center = [self.eventpress.xdata, self.eventpress.ydata]
            center_pix = [self.eventpress.x, self.eventpress.y]
            dx = (event.xdata - center[0]) / 2.0
            dy = (event.ydata - center[1]) / 2.0

            # rectangle with aspect ratio defined by the user
            if hasattr(self, "ratio"):
                dx_pix = abs(event.x - center_pix[0])
                dy_pix = abs(event.y - center_pix[1])
                if not dx_pix:
                    return
                if self.ratio < 1.0:
                    maxd = max(abs(dx_pix), abs(dy_pix / self.ratio))
                    if abs(dx_pix) < maxd:
                        dx *= maxd / (abs(dx_pix) + 1e-6)
                    if abs(dy_pix) < maxd:
                        dy *= maxd / (abs(dy_pix / self.ratio) + 1e-6)
                else:
                    maxd = max(abs(self.ratio * dx_pix), abs(dy_pix))
                    if abs(dx_pix) < maxd:
                        dx *= maxd / (abs(self.ratio * dx_pix) + 1e-6)
                    if abs(dy_pix) < maxd:
                        dy *= maxd / (abs(dy_pix) + 1e-6)

            # from center
            if "center" in self.state:
                dx *= 2
                dy *= 2

            # from corner
            else:
                center[0] += dx
                center[1] += dy

            x1, x2, y1, y2 = (
                center[0] - dx,
                center[0] + dx,
                center[1] - dy,
                center[1] + dy,
            )

        self.extents = x1, x2, y1, y2

    def onselect(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        verts = ([x1, y1], [x2, y1], [x2, y2], [x1, y2])
        path = Path(verts)
        self.sel = path.contains_points(self.data)
        if self._onselect is not None:
            self._onselect()

    def clear(self):
        self.set_visible(False)
        self.update()

    def disconnect(self):
        self.disconnect_events()
