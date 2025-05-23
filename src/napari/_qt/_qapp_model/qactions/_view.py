"""Qt 'View' menu Actions."""

import sys

from app_model.types import (
    Action,
    KeyCode,
    KeyMod,
    StandardKeyBinding,
    SubmenuItem,
    ToggleRule,
)

from napari._app_model.constants import MenuGroup, MenuId
from napari._qt.qt_main_window import Window
from napari._qt.qt_viewer import QtViewer
from napari.settings import get_settings
from napari.utils.translations import trans
from napari.viewer import Viewer, ViewerModel

# View submenus
VIEW_SUBMENUS = [
    (
        MenuId.MENUBAR_VIEW,
        SubmenuItem(submenu=MenuId.VIEW_AXES, title=trans._('Axes')),
    ),
    (
        MenuId.MENUBAR_VIEW,
        SubmenuItem(submenu=MenuId.VIEW_SCALEBAR, title=trans._('Scale Bar')),
    ),
]


# View actions
def _toggle_activity_dock(window: Window):
    window._status_bar._toggle_activity_dock()


def _get_current_fullscreen_status(window: Window):
    return window._qt_window.isFullScreen()


def _get_current_menubar_status(window: Window):
    return window._qt_window._toggle_menubar_visibility


def _get_current_play_status(qt_viewer: QtViewer):
    return bool(qt_viewer.dims.is_playing)


def _get_current_activity_dock_status(window: Window):
    return window._qt_window._activity_dialog.isVisible()


def _tooltip_visibility_toggle() -> None:
    settings = get_settings().appearance
    settings.layer_tooltip_visibility = not settings.layer_tooltip_visibility


def _get_current_tooltip_visibility() -> bool:
    return get_settings().appearance.layer_tooltip_visibility


def _fit_to_view(viewer: Viewer):
    viewer.fit_to_view()


def _zoom_in(viewer: Viewer):
    viewer.camera.zoom *= 1.5


def _zoom_out(viewer: Viewer):
    viewer.camera.zoom /= 1.5


def _toggle_canvas_ndim(viewer: ViewerModel):
    """Toggle the current canvas between 3D and 2D."""
    if viewer.dims.ndisplay == 2:
        viewer.dims.ndisplay = 3
    else:  # == 3
        viewer.dims.ndisplay = 2


Q_VIEW_ACTIONS: list[Action] = [
    Action(
        id='napari.window.view.toggle_fullscreen',
        title=trans._('Toggle Full Screen'),
        menus=[
            {
                'id': MenuId.MENUBAR_VIEW,
                'group': MenuGroup.NAVIGATION,
                'order': 1,
            }
        ],
        callback=Window._toggle_fullscreen,
        keybindings=[StandardKeyBinding.FullScreen],
        toggled=ToggleRule(get_current=_get_current_fullscreen_status),
    ),
    Action(
        id='napari.window.view.toggle_command_palette',
        title=trans._('Command Palette'),
        menus=[
            {
                'id': MenuId.MENUBAR_VIEW,
                'group': MenuGroup.NAVIGATION,
                'order': 2,
            }
        ],
        callback=Window._toggle_command_palette,
        keybindings=[
            {'primary': KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyP}
        ],
    ),
    Action(
        id='napari.window.view.toggle_menubar',
        title=trans._('Toggle Menubar Visibility'),
        menus=[
            {
                'id': MenuId.MENUBAR_VIEW,
                'group': MenuGroup.NAVIGATION,
                'order': 3,
                'when': sys.platform != 'darwin',
            }
        ],
        callback=Window._toggle_menubar_visible,
        keybindings=[
            {
                'win': KeyMod.CtrlCmd | KeyCode.KeyM,
                'linux': KeyMod.CtrlCmd | KeyCode.KeyM,
            }
        ],
        # TODO: add is_mac global context keys (rather than boolean here)
        enablement=sys.platform != 'darwin',
        status_tip=trans._('Show/Hide Menubar'),
        toggled=ToggleRule(get_current=_get_current_menubar_status),
    ),
    Action(
        id='napari.window.view.toggle_play',
        title=trans._('Toggle Play'),
        menus=[
            {
                'id': MenuId.MENUBAR_VIEW,
                'group': MenuGroup.NAVIGATION,
                'order': 3,
            }
        ],
        callback=Window._toggle_play,
        keybindings=[{'primary': KeyMod.CtrlCmd | KeyMod.Alt | KeyCode.KeyP}],
        toggled=ToggleRule(get_current=_get_current_play_status),
    ),
    Action(
        id='napari.viewer.fit_to_view',
        title=trans._('Fit to View'),
        menus=[
            {
                'id': MenuId.MENUBAR_VIEW,
                'group': MenuGroup.ZOOM,
                'order': 1,
            }
        ],
        callback=_fit_to_view,
        keybindings=[StandardKeyBinding.OriginalSize],
    ),
    Action(
        id='napari.viewer.camera.zoom_in',
        title=trans._('Zoom In'),
        menus=[
            {
                'id': MenuId.MENUBAR_VIEW,
                'group': MenuGroup.ZOOM,
                'order': 1,
            }
        ],
        callback=_zoom_in,
        keybindings=[StandardKeyBinding.ZoomIn],
    ),
    Action(
        id='napari.viewer.camera.zoom_out',
        title=trans._('Zoom Out'),
        menus=[
            {
                'id': MenuId.MENUBAR_VIEW,
                'group': MenuGroup.ZOOM,
                'order': 1,
            }
        ],
        callback=_zoom_out,
        keybindings=[StandardKeyBinding.ZoomOut],
    ),
    Action(
        id='napari.window.view.toggle_ndisplay',
        title=trans._('Toggle 2D/3D Camera'),
        menus=[
            {
                'id': MenuId.MENUBAR_VIEW,
                'group': MenuGroup.ZOOM,
                'order': 2,
            }
        ],
        callback=_toggle_canvas_ndim,
    ),
    Action(
        id='napari.window.view.toggle_activity_dock',
        title=trans._('Toggle Activity Dock'),
        menus=[
            {'id': MenuId.MENUBAR_VIEW, 'group': MenuGroup.RENDER, 'order': 11}
        ],
        callback=_toggle_activity_dock,
        toggled=ToggleRule(get_current=_get_current_activity_dock_status),
    ),
    # TODO: this could be made into a toggle setting Action subclass
    # using a similar pattern to the above ViewerToggleAction classes
    Action(
        id='napari.window.view.toggle_layer_tooltips',
        title=trans._('Toggle Layer Tooltips'),
        menus=[
            {
                'id': MenuId.MENUBAR_VIEW,
                'group': MenuGroup.RENDER,
                'order': 10,
            }
        ],
        callback=_tooltip_visibility_toggle,
        toggled=ToggleRule(get_current=_get_current_tooltip_visibility),
    ),
]
