from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qtpy.QtWidgets import QComboBox, QWidget

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked

if TYPE_CHECKING:
    from napari.layers import Surface


def _loader(layer: Surface) -> Any:
    """The Lodstone loader streaming this surface, if any."""
    return layer.metadata.get('lodstone_loader')


class QtStreamedLevelControl(QtWidgetControlsBase):
    """Show and lock the level of a surface streamed by Lodstone.

    Like the multiscale image control: "Auto (k)" follows the camera and
    names the level shown, and each level entry locks the stream to it.
    Hidden until the layer has a loader, which is attached after the layer
    is added, so it checks again at every data change.

    Parameters
    ----------
    parent : QWidget
        Parent widget.
    layer : napari.layers.Surface
        The surface layer.

    Attributes
    ----------
    level_combobox : QComboBox
        "Auto" plus one entry per streamed level.
    level_label : QtWrappedLabel
        Label for the combobox.
    """

    def __init__(self, parent: QWidget, layer: Surface) -> None:
        super().__init__(parent, layer)
        self._layer: Surface = layer
        self._populated_for: Any = None
        self.level_combobox = QComboBox(parent)
        self.level_combobox.setToolTip(
            'Level of the streamed surface: Auto follows the camera.'
        )
        self.level_label = QtWrappedLabel('resolution:', parent)
        self.level_combobox.currentIndexChanged.connect(
            self._on_combobox_changed
        )
        self._layer.events.data.connect(self._on_data_change)
        self._on_data_change()

    def _on_data_change(self, event=None) -> None:
        loader = _loader(self._layer)
        if loader is None:
            self.level_combobox.hide()
            self.level_label.hide()
            return
        if loader is not self._populated_for:
            self._populate(loader)
        self.level_combobox.show()
        self.level_label.show()
        self._update_auto_label(loader)

    def _populate(self, loader: Any) -> None:
        self._populated_for = loader
        with qt_signals_blocked(self.level_combobox):
            self.level_combobox.clear()
            self.level_combobox.addItem('Auto', None)
            for index, level in enumerate(loader.source.levels):
                self.level_combobox.addItem(
                    f'{index}: {len(level.chunks):,} chunks', index
                )
            locked = loader.locked_level
            self.level_combobox.setCurrentIndex(
                0 if locked is None else locked + 1
            )

    def _update_auto_label(self, loader: Any) -> None:
        shown = loader.target.level
        requested = loader.stream.requested_level
        if shown is None:
            label = 'Auto'
        elif requested is not None and requested != shown:
            label = f'Auto ({shown} → {requested})'
        else:
            label = f'Auto ({shown})'
        if self.level_combobox.itemText(0) != label:
            self.level_combobox.setItemText(0, label)

    def _on_combobox_changed(self, index: int) -> None:
        loader = _loader(self._layer)
        if loader is not None:
            loader.locked_level = self.level_combobox.itemData(index)

    def get_widget_controls(
        self,
    ) -> list[tuple[QtWrappedLabel, QWidget] | tuple[QWidget]]:
        return [(self.level_label, self.level_combobox)]
