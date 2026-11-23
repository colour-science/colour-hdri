"""
Network - Editor
================

Define the *Graph Editor* for visual node graph editing and processing.
"""

from __future__ import annotations

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import logging
import re
import traceback
import typing
from pathlib import Path

import colour.utilities.network

if typing.TYPE_CHECKING:
    from colour.hints import Any, Callable, List, Literal, Real, Tuple

from colour.utilities import (
    ExecutionNode,
    ExecutionPort,
    For,
    NodePassthrough,
    NodeSetGraphOutputPort,
    ParallelForThread,
    PortGraph,
    PortNode,
    attest,
    optional,
    validate_method,
)
from PySide6.QtCore import (
    QObject,
    QRunnable,
    QSettings,
    Qt,
    QThreadPool,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import colour_hdri.network.nodes

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "collect_colourscience_nodes",
    "exception_dialog",
    "confirmation_dialog",
    "WorkerSignals",
    "Worker",
    "LiteGraphWidget",
    "GraphEditor",
]

LOGGER = logging.getLogger(__name__)

HTML_INDEX = Path(__file__).parent / "resources" / "index.html"


def collect_colourscience_nodes() -> dict[str, type[ExecutionNode | PortNode]]:
    """
    Collect all *Colour* and *Colour - HDRI* node classes.

    Returns
    -------
    :class:`dict`
        Mapping of node names to node classes.
    """

    nodes = {
        "For": For,
        "ParallelForThread": ParallelForThread,
        "NodePassthrough": NodePassthrough,
    }

    for module in (colour.utilities.network, colour_hdri.network):
        for name in module.__all__:
            if not name.startswith("Node"):
                continue

            object_ = getattr(module, name)

            try:
                if issubclass(object_, (ExecutionNode, PortNode)):
                    nodes[name] = object_
            except TypeError:
                continue

    return nodes


COLOUR_SCIENCE_NODES: dict[str, type[ExecutionNode | PortNode]] = (
    collect_colourscience_nodes()
)


def exception_dialog(exception_info: Tuple) -> QMessageBox.StandardButton:
    """
    Display a critical exception dialog.

    Parameters
    ----------
    exception_info
        Tuple of exception type, value, and traceback message.

    Returns
    -------
    :class:`QMessageBox.Ok`
        Dialog result.
    """

    exception_type, value, message = exception_info

    LOGGER.critical(message)

    message_box = QMessageBox()
    message_box.setIcon(QMessageBox.Icon.Critical)
    message_box.setWindowTitle(f"{exception_type.__name__} : {value}")
    message_box.setText(f"{exception_type.__name__} : {value}")
    message_box.setDetailedText(message)

    return message_box.exec()  # pyright: ignore


def confirmation_dialog(message: str) -> QMessageBox.StandardButton:
    """
    Display a confirmation dialog for unsaved changes.

    Parameters
    ----------
    message
        Message to display in the dialog.

    Returns
    -------
    :class:`QMessageBox.StandardButton`
        Dialog result.
    """

    message_box = QMessageBox()
    message_box.setWindowTitle("Unsaved Changes")
    message_box.setText(message)
    message_box.setStandardButtons(
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    )
    message_box.setDefaultButton(QMessageBox.StandardButton.No)
    message_box.setIcon(QMessageBox.Icon.Warning)

    return message_box.exec()  # pyright: ignore


class WorkerSignals(QObject):
    """
    Define signals for :class:`Worker` thread communication.

    Attributes
    ----------
    started
        Signal emitted when the worker starts.
    ended
        Signal emitted when the worker ends.
    exception
        Signal emitted when an exception occurs.
    result
        Signal emitted with the result.
    progress
        Signal emitted with progress updates.
    """

    started = Signal()
    ended = Signal()
    exception = Signal(tuple)
    result = Signal(object)
    progress = Signal(int)


class Worker(QRunnable):
    """
    Implement a worker thread for executing functions asynchronously.

    Parameters
    ----------
    fn
        Function to execute.
    args
        Positional arguments for the function.
    kwargs
        Keyword arguments for the function.

    Attributes
    ----------
    signals
        Worker signals for thread communication.
    """

    def __init__(self, fn: Callable, *args: Any, **kwargs: Any) -> None:
        super().__init__()

        self._function = fn
        self._args = args
        self._kwargs = kwargs
        self.signals = WorkerSignals()

        self._kwargs["progress_callback"] = self.signals.progress

    def run(self) -> None:
        """Execute the worker function."""

        try:
            self.signals.started.emit()
            result = self._function(*self._args, **self._kwargs)
        except Exception:  # noqa: BLE001
            message = traceback.format_exc()

            LOGGER.critical(message)

            exception_type, value = sys.exc_info()[:2]
            self.signals.exception.emit((exception_type, value, message))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.ended.emit()


class LiteGraphWidget(QWidget):
    """
    Implement a *LiteGraph.js* based graph editing widget.

    Parameters
    ----------
    parent
        Parent widget.
    file_path
        Path to a graph file to load.
    developer_mode
        Whether to enable developer tools.

    Attributes
    ----------
    graph_changed
        Signal emitted when the graph changes.
    graph_loaded
        Signal emitted when a graph is loaded.
    graph_saved
        Signal emitted when a graph is saved.
    graph_process_started
        Signal emitted when graph processing starts.
    graph_process_ended
        Signal emitted when graph processing ends.
    graph_process_exception
        Signal emitted when graph processing raises an exception.
    node_process_started
        Signal emitted when a node starts processing.
    node_process_ended
        Signal emitted when a node ends processing.
    node_process_exception
        Signal emitted when a node processing raises an exception.
    """

    graph_changed = Signal()
    graph_loaded = Signal(bool)
    graph_saved = Signal(bool)
    graph_process_started = Signal()
    graph_process_ended = Signal()
    graph_process_exception = Signal()
    node_process_started = Signal(list)
    node_process_ended = Signal(list)
    node_process_exception = Signal(list)

    def __init__(
        self,
        parent: QWidget,
        file_path: str | None = None,
        developer_mode: bool = False,
    ) -> None:
        super().__init__()

        self._parent: GraphEditor = parent  # pyright: ignore
        self._file_path = None
        self.file_path = file_path
        self._developer_mode = developer_mode

        self._dirty = False

        self._web_channel = QWebChannel()
        self._web_channel.registerObject("backend", self)

        self._webview_QWebEngineView = QWebEngineView()
        self._process_graph_QPushButton = QPushButton("Process Graph")

        self._developer_tools_QWebEngineView = None
        if self._developer_mode:
            self._developer_tools_QWebEngineView = QWebEngineView()
            self._webview_QWebEngineView.page().setDevToolsPage(
                self._developer_tools_QWebEngineView.page()
            )

        self._setup_layout()
        self._setup_views()
        self._setup_signals()

    @property
    def file_path(self) -> str | None:
        """
        Getter and setter property for the file path.

        Parameters
        ----------
        value
            Value to set the file path with.

        Returns
        -------
        :class:`str`
            File path.
        """

        return self._file_path

    @file_path.setter
    def file_path(self, value: str | None) -> None:
        """Setter for the **self.file_path** property."""

        if value is None:
            return

        attest(
            isinstance(value, str),
            f'"file_path" property: "{value}" type is not "str"!',
        )

        self._file_path = value

    @property
    def dirty(self) -> bool:
        """
        Getter and setter property for the dirty state.

        Parameters
        ----------
        value
            Value to set the dirty state.

        Returns
        -------
        :class:`bool`
            Dirty state.
        """

        return self._dirty

    def _setup_layout(self) -> None:
        """Set up the widget layout."""

        layout_QVBoxLayout = QVBoxLayout()
        if self._developer_tools_QWebEngineView is not None:
            splitter = QSplitter(Qt.Orientation.Vertical)
            splitter.addWidget(self._webview_QWebEngineView)
            splitter.addWidget(self._developer_tools_QWebEngineView)
            layout_QVBoxLayout.addWidget(splitter)
        else:
            layout_QVBoxLayout.addWidget(self._webview_QWebEngineView)

        layout_QVBoxLayout.addWidget(self._process_graph_QPushButton)

        self.setLayout(layout_QVBoxLayout)

    def _setup_views(self) -> None:
        """Set up the web views."""

        self._webview_QWebEngineView.settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True
        )
        self._webview_QWebEngineView.load(QUrl.fromLocalFile(HTML_INDEX.resolve()))

    def _setup_signals(self) -> None:
        """Set up the signal connections."""

        self._webview_QWebEngineView.page().setWebChannel(self._web_channel)

        self._webview_QWebEngineView.loadFinished.connect(
            self._webview_QWebEngineView_on_page_loaded
        )

        self._process_graph_QPushButton.clicked.connect(
            self._process_graph_QPushButton_clicked
        )

        self.node_process_started.connect(self._on_node_process_started)
        self.node_process_ended.connect(self._on_node_process_ended)
        self.node_process_exception.connect(self._on_node_process_exception)

    def _webview_QWebEngineView_on_page_loaded(self) -> None:
        """Handle page loaded event."""

        self._register_colourscience_nodes()

        if self._file_path is not None:
            self.load(self._file_path)

    @Slot()
    def canvas_changed(self) -> None:
        """Handle canvas changed event from *LiteGraph.js*."""

        self._dirty = True
        self.graph_changed.emit()

    def _on_node_process_started(self, path: List[int]) -> None:
        """
        Handle node process started event.

        Parameters
        ----------
        path
            Node path in the graph hierarchy.
        """

        self._webview_QWebEngineView.page().runJavaScript(
            f"onNodeProcessStarted({path});"
        )

    def _on_node_process_ended(self, path: List[int]) -> None:
        """
        Handle node process ended event.

        Parameters
        ----------
        path
            Node path in the graph hierarchy.
        """

        self._webview_QWebEngineView.page().runJavaScript(
            f"onNodeProcessEnded({path});"
        )

    def _on_node_process_exception(self, path: List[int]) -> None:
        """
        Handle node process exception event.

        Parameters
        ----------
        path
            Node path in the graph hierarchy.
        """

        self._webview_QWebEngineView.page().runJavaScript(
            f"onNodeProcessException({path});"
        )

    def _process_graph_QPushButton_clicked(self) -> None:
        """Handle process graph button click."""

        self._webview_QWebEngineView.page().runJavaScript(
            "exportGraph();", self._process_graph_callback
        )

    def _process_graph_callback(self, result: str) -> None:
        """
        Handle graph export callback and process the graph.

        Parameters
        ----------
        result
            Exported graph JSON string.
        """

        if not hasattr(self._parent, "threadpool"):
            LOGGER.critical("Parent widget does not define a threadpool attribute!")
            return

        try:
            LOGGER.debug("Processing graph: %s", result)
            graph = self._build_graph(json.loads(result))

            worker = Worker(graph.process)

            worker.signals.started.connect(self._graph_process_started)
            worker.signals.ended.connect(self._graph_process_ended)
            worker.signals.exception.connect(self._graph_process_exception)

            self._parent.threadpool.start(worker)
        except Exception:  # noqa: BLE001
            exception_type, value = sys.exc_info()[:2]

            message = f"{traceback.format_exc()}"

            exception_dialog((exception_type, value, message))

    def _graph_process_started(self) -> None:
        """Handle graph process started event."""

        self.graph_process_started.emit()
        self._webview_QWebEngineView.page().runJavaScript("onGraphProcessStarted();")

    def _graph_process_ended(self) -> None:
        """Handle graph process ended event."""

        self.graph_process_ended.emit()

    def _graph_process_exception(self, exception_info: Tuple) -> None:
        """
        Handle graph process exception event.

        Parameters
        ----------
        exception_info
            Tuple of exception type, value, and traceback message.
        """

        self.graph_process_exception.emit()
        self._webview_QWebEngineView.page().runJavaScript("onGraphProcessException();")

        exception_dialog(exception_info)

    def _register_colourscience_nodes(self) -> None:
        """Register all *Colour* nodes with *LiteGraph.js*."""

        registry = ""
        for node_class in COLOUR_SCIENCE_NODES.values():
            node = node_class()
            class_name = node.__class__.__name__
            node_name = re.sub("^Node", "", class_name)
            title = " " if node_name == "Passthrough" else node_name
            description = node.description.replace('"', "'")
            category = node.category

            LOGGER.info('Registering "%s" node...', class_name)
            registry += f"function colourscience_{node_name}() {{"
            for name, port in node.input_ports.items():
                if isinstance(port, ExecutionPort):
                    registry += f'\tthis.addInput("{name}", "executionPort");'
                else:
                    registry += f'\tthis.addInput("{name}");'
            for name in node.output_ports:
                registry += f'\tthis.addOutput("{name}");'
            registry += "\n"
            registry += "\tthis.properties = {"
            registry += f'\t\tpythonClassName : "{class_name}"'
            registry += "\t};"
            registry += "\n"
            if node_name == "Passthrough":
                registry += "\tthis.flags = {"
                registry += "\t\tcollapsed : true"
                registry += "\t};"
                registry += "\n"
            registry += "};"
            registry += "\n"
            registry += f'colourscience_{node_name}.title = "{title}";'
            registry += f'colourscience_{node_name}.desc = "{description}";'
            registry += "\n"
            registry += (
                f'LiteGraph.registerNodeType("{category}/{node_name}", '
                f"colourscience_{node_name});"
            )
            registry += "\n"

        registry += "sortRegisteredNodeTypes();"

        self._webview_QWebEngineView.page().runJavaScript(registry)

    def _listener_on_process_started(self, *args: Any) -> None:
        """Emit node process started signal."""

        _node, path = args
        self.node_process_started.emit(path)

    def _listener_on_process_ended(self, *args: Any) -> None:
        """Emit node process ended signal."""

        _node, path = args
        self.node_process_ended.emit(path)

    def _listener_on_process_exception(self, *args: Any) -> None:
        """Emit node process exception signal."""

        _node, path = args
        self.node_process_exception.emit(path)

    def _build_graph(
        self,
        lg_graph: dict,
        path: list | None = None,
        graph: PortGraph | None = None,
    ) -> PortGraph:
        """
        Build a :class:`PortGraph` from *LiteGraph.js* graph data.

        Parameters
        ----------
        lg_graph
            *LiteGraph.js* graph data dictionary.
        path
            Current node path in the graph hierarchy.
        graph
            Existing :class:`PortGraph` to populate.

        Returns
        -------
        :class:`PortGraph`
            Built port graph.
        """

        if path is None:
            path = []

        def _lg_node_by_id(id_: int) -> dict | None:
            """Return the *Litegraph* node with given id."""

            return next(
                iter([node for node in lg_graph["nodes"] if node["id"] == id_]), None
            )

        graph = optional(graph, PortGraph())

        lg_node_to_node, lg_node_to_type, constants = {}, {}, {}

        # Nodes
        for lg_node in lg_graph["nodes"]:
            lg_node_to_type[lg_node["id"]] = lg_node["type"]

            node = None
            if (name := lg_node["properties"].get("pythonClassName")) is not None:
                node_class = COLOUR_SCIENCE_NODES[name]
                node = node_class(f"{lg_node.get('title', name)} ({lg_node['id']})")
            elif lg_node["type"] == "graph/subgraph":
                sub_graph = PortGraph(name=lg_node.get("title"))

                for input_ in lg_node.get("inputs", []):
                    name = input_["name"]
                    if name == "execution_input":
                        sub_graph.add_input_port(name, "executionPort")
                    else:
                        sub_graph.add_input_port(name)

                for output in lg_node.get("outputs", []):
                    name = output["name"]
                    if name == "execution_output":
                        sub_graph.add_output_port(name, "executionPort")
                    else:
                        sub_graph.add_output_port(name)

                node = self._build_graph(
                    lg_node["subgraph"], [lg_node["id"]], sub_graph
                )
            elif lg_node["type"].startswith("basic/"):
                value = lg_node["properties"]["value"]

                if lg_node["type"] == "basic/array":
                    value = eval(value)  # noqa: S307

                constants[lg_node["id"]] = value

            if node is not None:
                lg_node_to_node[lg_node["id"]] = node
                node.on_process_started.add_listener(
                    lambda x, y=[*path, lg_node["id"]]: (
                        self._listener_on_process_started(x, y)
                    )
                )
                node.on_process_ended.add_listener(
                    lambda x, y=[*path, lg_node["id"]]: self._listener_on_process_ended(
                        x, y
                    )
                )
                node.on_process_exception.add_listener(
                    lambda x, y=[*path, lg_node["id"]]: (
                        self._listener_on_process_exception(x, y)
                    )
                )

                graph.add_node(node)

        # Edges
        for link in lg_graph["links"]:
            _id, origin_id, origin_port_index, target_id, target_port_index, _type = (
                link
            )

            origin_node = lg_node_to_node.get(origin_id)
            target_node = lg_node_to_node.get(target_id)

            if lg_node_to_type[origin_id] == "graph/input":
                origin_node = graph
            elif lg_node_to_type[target_id] == "graph/output":
                # NOTE: Connections with outputs are skipped as they create DAG
                # cycles: A sub-graph is a node, connecting its inputs to the
                # sub-graph nodes and their outputs to the sub-graph outputs
                # create the cycles.
                # A special node that can set the sub-graph relevant output is
                # thus inserted instead of the connection.

                # TODO: Handle case where constant is connected.

                if origin_node is None:
                    continue

                node_set_graph_output_port = NodeSetGraphOutputPort()
                node_set_graph_output_port.set_input(
                    "name",
                    _lg_node_by_id(target_id)["properties"]["name"],  # pyright: ignore
                )
                origin_node.connect(
                    list(origin_node.output_ports)[origin_port_index],
                    node_set_graph_output_port,
                    "value",
                )
                graph.add_node(node_set_graph_output_port)
                continue

            if origin_id in constants and target_node is not None:
                target_node.set_input(
                    list(target_node.input_ports)[target_port_index],
                    constants[origin_id],
                )
            elif origin_node is not None and target_node is not None:
                if origin_node == graph:
                    origin_port = _lg_node_by_id(origin_id)["properties"]["name"]  # pyright: ignore
                else:
                    origin_port = list(origin_node.output_ports)[origin_port_index]

                target_port = list(target_node.input_ports)[target_port_index]
                origin_node.connect(origin_port, target_node, target_port)

        if self._developer_mode:
            graph.to_graphviz().write_svg(f"graph_{graph.name}.svg")

        return graph

    def load(self, file_path: str | None = None) -> None:
        """
        Load a graph from a file.

        Parameters
        ----------
        file_path
            Path to the graph file to load.
        """

        if file_path is None:
            file_path = self._file_path

        if file_path is None:
            return

        self._file_path = file_path

        try:
            with open(self._file_path) as lgson_file:
                LOGGER.info('Opening "%s" file...', self._file_path)

                self._webview_QWebEngineView.page().runJavaScript(
                    f"importGraph('{lgson_file.read()}');"
                )

                self._file_path = file_path
                self._dirty = False

                self.graph_loaded.emit(True)
        except Exception as exception:  # noqa: BLE001
            message = f'Error reading "lgson" file: {exception}'
            LOGGER.critical(message)

            QMessageBox.critical(None, "Error", message)
            self.graph_loaded.emit(False)

    def save(self, file_path: str | None = None) -> None:
        """
        Save the graph to a file.

        Parameters
        ----------
        file_path
            Path to save the graph file to.
        """

        if file_path is None:
            file_path = self._file_path

        if file_path is None:
            LOGGER.warning("No file path was set!")
            return

        self._file_path = file_path
        self._webview_QWebEngineView.page().runJavaScript(
            "exportGraph();",
            lambda result: self._save_callback(result),
        )

    def _save_callback(self, result: str) -> None:
        """
        Handle save callback and write graph to file.

        Parameters
        ----------
        result
            Exported graph JSON string.
        """

        if self._file_path is None:
            self.graph_saved.emit(False)
            return

        try:
            with open(self._file_path, "w") as lgson_file:
                LOGGER.info('Saving "%s" file...', self._file_path)

                result = result.replace("'", "\\'")
                lgson_file.write(result)
                self._dirty = False

                self.graph_saved.emit(True)
        except Exception as exception:  # noqa: BLE001
            message = f'Error saving "lgson" file: {exception}'
            LOGGER.critical(message)

            QMessageBox.critical(None, "Error", message)

            self.graph_saved.emit(False)

    def undo(self) -> None:
        """Undo the last action."""

        self._webview_QWebEngineView.page().runJavaScript("undo();")

    def redo(self) -> None:
        """Redo the last undone action."""

        self._webview_QWebEngineView.page().runJavaScript("redo();")

    def copy(self) -> None:
        """Copy selected nodes to clipboard."""

        self._webview_QWebEngineView.page().runJavaScript("copy();")

    def paste(self) -> None:
        """Paste nodes from clipboard."""

        self._webview_QWebEngineView.page().runJavaScript("paste();")

    def stash(self) -> None:
        """Stash the current graph to settings."""

        if not hasattr(self._parent, "settings"):
            LOGGER.critical("Parent widget does not define a settings attribute!")
            return

        self._webview_QWebEngineView.page().runJavaScript(
            "exportGraph();", self._stash_callback
        )

    def _stash_callback(self, result: str) -> None:
        """
        Handle stash callback and store graph in settings.

        Parameters
        ----------
        result
            Exported graph JSON string.
        """

        LOGGER.debug("Stash: %s", result)

        self._parent.settings.setValue("graph_stash", result)

    def unstash(self) -> None:
        """Restore the stashed graph from settings."""

        if not hasattr(self._parent, "settings"):
            LOGGER.critical("Parent widget does not define a settings attribute!")
            return

        if (graph_stash := self._parent.settings.value("graph_stash")) is not None:
            LOGGER.debug("Stash: %s", graph_stash)

            graph_stash = graph_stash.replace("'", "\\'")
            self._webview_QWebEngineView.page().runJavaScript(
                f"importGraph('{graph_stash}');"
            )

    def select_all(self) -> None:
        """Select all nodes in the graph."""

        self._webview_QWebEngineView.page().runJavaScript("selectAll();")

    def align_selected_nodes_to_grid(self) -> None:
        """Align selected nodes to the grid."""

        self._webview_QWebEngineView.page().runJavaScript("alignSelectedNodesToGrid();")

    def offset_selected_nodes(self, offset: Tuple[Real, Real]) -> None:
        """
        Offset selected nodes by given amount.

        Parameters
        ----------
        offset
            Offset as (x, y) tuple.
        """

        self._webview_QWebEngineView.page().runJavaScript(
            f"offsetSelectedNodes([{offset[0]}, {offset[1]}]);"
        )


class GraphEditor(QMainWindow):
    """
    Implement the main *Graph Editor* application window.

    Parameters
    ----------
    developer_mode
        Whether to enable developer tools.
    """

    def __init__(self, developer_mode: bool = False) -> None:
        super().__init__()

        self._developer_mode = developer_mode

        self._settings = QSettings("colour-science", "GraphEditor")
        self._recent_files_count = 5

        LOGGER.info("Settings location: %s", self._settings.fileName())

        self._threadpool = QThreadPool()

        self.setWindowTitle("Graph Editor")
        self.resize(1280, 720)

        self._tab_widget_QTabWidget = QTabWidget()

        self._setup_menu()
        self._setup_layout()
        self._setup_views()
        self._setup_signals()

    @property
    def settings(self) -> QSettings:
        """
        Getter and setter property for the settings.

        Returns
        -------
        :class:`QSettings`
        """

        return self._settings

    @property
    def threadpool(self) -> QThreadPool:
        """
        Getter and setter property for the threadpool.

        Returns
        -------
        :class:`QThreadpool`
        """

        return self._threadpool

    def _setup_menu(self) -> None:
        """Set up the menu bar."""

        # "File" menu
        file_menu = self.menuBar().addMenu("&File")

        new_action = QAction("New", self)
        new_action.triggered.connect(self._new_triggered)
        new_action.setShortcut(QKeySequence("Ctrl+N"))
        file_menu.addAction(new_action)

        open_action = QAction("Open...", self)
        open_action.triggered.connect(self._open_triggered)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        self._recent_files_menu = file_menu.addMenu("Recent Files")
        self._update_recent_files_menu()

        file_menu.addSeparator()

        save_action = QAction("Save", self)
        save_action.triggered.connect(self._save_triggered)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        file_menu.addAction(save_action)

        save_as_action = QAction("Save as...", self)
        save_as_action.triggered.connect(self._save_as_triggered)
        save_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        file_menu.addAction(save_as_action)

        # "Edit" menu
        edit_menu = self.menuBar().addMenu("&Edit")

        undo_action = QAction("Undo", self)
        undo_action.triggered.connect(self._undo_triggered)
        undo_action.setShortcut(QKeySequence("Ctrl+Z"))
        edit_menu.addAction(undo_action)

        redo_action = QAction("Redo", self)
        redo_action.triggered.connect(self._redo_triggered)
        redo_action.setShortcut(QKeySequence("Ctrl+Shift+Z"))
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()

        copy_action = QAction("Copy", self)
        copy_action.triggered.connect(self._copy_triggered)
        copy_action.setShortcut(QKeySequence("Ctrl+C"))
        edit_menu.addAction(copy_action)

        paste_action = QAction("Paste", self)
        paste_action.triggered.connect(self._paste_triggered)
        paste_action.setShortcut(QKeySequence("Ctrl+V"))
        edit_menu.addAction(paste_action)

        edit_menu.addSeparator()

        stash_action = QAction("Stash", self)
        stash_action.triggered.connect(self._stash_triggered)
        edit_menu.addAction(stash_action)

        unstash_action = QAction("Pop Stash", self)
        unstash_action.triggered.connect(self._unstash_triggered)
        edit_menu.addAction(unstash_action)

        edit_menu.addSeparator()

        select_all_action = QAction("Select All", self)
        select_all_action.triggered.connect(self._select_all_triggered)
        select_all_action.setShortcut(QKeySequence("Ctrl+A"))
        edit_menu.addAction(select_all_action)

        edit_menu.addSeparator()

        align_selected_nodes_to_grid_action = QAction(
            "Align Selected Nodes to Grid", self
        )
        align_selected_nodes_to_grid_action.triggered.connect(
            self._align_selected_nodes_to_grid_action
        )
        edit_menu.addAction(align_selected_nodes_to_grid_action)

        edit_menu.addSeparator()

        nudge_selected_nodes_left_action = QAction("Nudge Selected Nodes Left", self)
        nudge_selected_nodes_left_action.setShortcut(QKeySequence("Left"))
        nudge_selected_nodes_left_action.triggered.connect(
            lambda _: self._nudge_selected_nodes_action("Left")
        )
        edit_menu.addAction(nudge_selected_nodes_left_action)

        nudge_selected_nodes_right_action = QAction("Nudge Selected Nodes Right", self)
        nudge_selected_nodes_right_action.setShortcut(QKeySequence("Right"))
        nudge_selected_nodes_right_action.triggered.connect(
            lambda _: self._nudge_selected_nodes_action("Right")
        )
        edit_menu.addAction(nudge_selected_nodes_right_action)

        nudge_selected_nodes_up_action = QAction("Nudge Selected Nodes Up", self)
        nudge_selected_nodes_up_action.setShortcut(QKeySequence("Up"))
        nudge_selected_nodes_up_action.triggered.connect(
            lambda _: self._nudge_selected_nodes_action("Up")
        )
        edit_menu.addAction(nudge_selected_nodes_up_action)

        nudge_selected_nodes_down_action = QAction("Nudge Selected Nodes Down", self)
        nudge_selected_nodes_down_action.setShortcut(QKeySequence("Down"))
        nudge_selected_nodes_down_action.triggered.connect(
            lambda _: self._nudge_selected_nodes_action("Down")
        )
        edit_menu.addAction(nudge_selected_nodes_down_action)

    def _setup_layout(self) -> None:
        """Set up the main window layout."""

        self.setCentralWidget(self._tab_widget_QTabWidget)

    def _setup_views(self) -> None:
        """Set up the tab widget views."""

        self._tab_widget_QTabWidget.setTabsClosable(True)
        self._tab_widget_QTabWidget.setMovable(True)

        self._add_new_graph()

    def _setup_signals(self) -> None:
        """Set up the signal connections."""

        self._tab_widget_QTabWidget.tabCloseRequested.connect(
            self._tab_widget_QTabWidget_tabCloseRequested
        )

    def _set_window_title(self) -> None:
        """Set the window title based on current file path."""

        title = "Graph Editor"

        current_widget = self._tab_widget_QTabWidget.currentWidget()
        if current_widget is not None:
            file_path = current_widget.file_path  # pyright: ignore
            if file_path is not None:
                title = f"{title} - {file_path}"

        self.setWindowTitle(title)

    def _tab_widget_QTabWidget_tabCloseRequested(self, index: int) -> None:
        """
        Handle tab close request.

        Parameters
        ----------
        index
            Index of the tab to close.
        """

        if self._tab_widget_QTabWidget.widget(index).dirty and (  # pyright: ignore
            confirmation_dialog(
                "The graph contains unsaved changes.\n\n"
                "Do you want to close without saving?"
            )
            != QMessageBox.StandardButton.Yes
        ):
            return

        self._tab_widget_QTabWidget.removeTab(index)

        if self._tab_widget_QTabWidget.count() == 0:
            self._add_new_graph()

    def _new_triggered(self) -> None:
        """Handle new graph action."""

        self._add_new_graph()

    def _open_triggered(self) -> None:
        """Handle open graph action."""

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Graph", "", "Litegraph (*.lgson)"
        )

        if not file_path:
            return

        self._add_new_graph(file_path)

    def _save_triggered(self) -> None:
        """Handle save graph action."""

        file_path = self._tab_widget_QTabWidget.currentWidget().file_path  # pyright: ignore
        if file_path is None:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Graph", "", "Litegraph (*.lgson)"
            )

            if not file_path:
                return

        self._tab_widget_QTabWidget.currentWidget().save(file_path)  # pyright: ignore

    def _save_as_triggered(self) -> None:
        """Handle save as graph action."""

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Graph", "", "Litegraph (*.lgson)"
        )

        if file_path:
            self._tab_widget_QTabWidget.currentWidget().save(file_path)  # pyright: ignore

    def _undo_triggered(self) -> None:
        """Handle undo action."""

        self._tab_widget_QTabWidget.currentWidget().undo()  # pyright: ignore

    def _redo_triggered(self) -> None:
        """Handle redo action."""

        self._tab_widget_QTabWidget.currentWidget().redo()  # pyright: ignore

    def _copy_triggered(self) -> None:
        """Handle copy action."""

        self._tab_widget_QTabWidget.currentWidget().copy()  # pyright: ignore

    def _paste_triggered(self) -> None:
        """Handle paste action."""

        self._tab_widget_QTabWidget.currentWidget().paste()  # pyright: ignore

    def _stash_triggered(self) -> None:
        """Handle stash action."""

        self._tab_widget_QTabWidget.currentWidget().stash()  # pyright: ignore

    def _unstash_triggered(self) -> None:
        """Handle unstash action."""

        if self._tab_widget_QTabWidget.currentWidget().dirty and (  # pyright: ignore
            confirmation_dialog(
                "The graph contains unsaved changes.\n\n"
                "Do you want to pop the stash without saving?"
            )
            != QMessageBox.StandardButton.Yes
        ):
            return

        self._tab_widget_QTabWidget.currentWidget().unstash()  # pyright: ignore

    def _select_all_triggered(self) -> None:
        """Handle select all action."""

        self._tab_widget_QTabWidget.currentWidget().select_all()  # pyright: ignore

    def _align_selected_nodes_to_grid_action(self) -> None:
        """Handle align selected nodes to grid action."""

        self._tab_widget_QTabWidget.currentWidget().align_selected_nodes_to_grid()  # pyright: ignore

    def _nudge_selected_nodes_action(
        self, direction: Literal["Left", "Right", "Up", "Down"] | str
    ) -> None:
        """
        Handle nudge selected nodes action.

        Parameters
        ----------
        direction
            Direction to nudge the selected nodes.
        """

        direction = validate_method(direction, ("Left", "Right", "Up", "Down"))

        if direction == "left":
            offset = [-10, 0]
        elif direction == "right":
            offset = [10, 0]
        elif direction == "up":
            offset = [0, -10]
        # direction == "down"
        else:
            offset = [0, 10]

        self._tab_widget_QTabWidget.currentWidget().offset_selected_nodes(offset)  # pyright: ignore

    def _add_new_graph(
        self, file_path: str | None = None, graph_name: str = "Untitled"
    ) -> None:
        """
        Add a new graph tab.

        Parameters
        ----------
        file_path
            Path to the graph file to load.
        graph_name
            Name for the graph tab.
        """

        if file_path is not None:
            # Selecting existing graph if found
            for index in range(self._tab_widget_QTabWidget.count()):
                if self._tab_widget_QTabWidget.widget(index).file_path == file_path:  # pyright: ignore
                    self._tab_widget_QTabWidget.setCurrentIndex(index)
                    return

            # Closing initial graph if not dirty
            if (
                self._tab_widget_QTabWidget.count() == 1
                and self._tab_widget_QTabWidget.currentWidget().file_path is None  # pyright: ignore
                and not self._tab_widget_QTabWidget.currentWidget().dirty  # pyright: ignore
            ):
                self._tab_widget_QTabWidget.removeTab(0)

        litegraph_widget = LiteGraphWidget(self, file_path, self._developer_mode)

        litegraph_widget.graph_changed.connect(self._graph_changed)
        litegraph_widget.graph_loaded.connect(self._graph_loaded)
        litegraph_widget.graph_saved.connect(self._graph_saved)

        index = self._tab_widget_QTabWidget.addTab(litegraph_widget, graph_name)
        self._tab_widget_QTabWidget.setCurrentIndex(index)

    def _set_current_tab_text(self) -> None:
        """Set the current tab text based on file path and dirty state."""

        current_index = self._tab_widget_QTabWidget.currentIndex()

        if current_index == -1:
            return

        tab_name = "Untitled"

        file_path = self._tab_widget_QTabWidget.currentWidget().file_path  # pyright: ignore

        if file_path is not None:
            tab_name = Path(file_path).name

        if self._tab_widget_QTabWidget.currentWidget().dirty:  # pyright: ignore
            tab_name = f"{tab_name} *"

        self._tab_widget_QTabWidget.setTabText(current_index, tab_name)

    def _graph_changed(self) -> None:
        """Handle graph changed event."""

        self._set_current_tab_text()

    def _graph_loaded(self, status: bool) -> None:
        """
        Handle graph loaded event.

        Parameters
        ----------
        status
            Whether the graph was loaded successfully.
        """

        current_index = self._tab_widget_QTabWidget.currentIndex()

        if current_index == -1:
            return

        file_path = self._tab_widget_QTabWidget.currentWidget().file_path  # pyright: ignore

        if status:
            self._set_current_tab_text()
            self._tab_widget_QTabWidget.setTabToolTip(current_index, file_path)
            self._add_to_recent_files(file_path)
        else:
            self._tab_widget_QTabWidget.removeTab(current_index)
            self._remove_from_recent_files(file_path)

    def _graph_saved(self, status: bool) -> None:
        """
        Handle graph saved event.

        Parameters
        ----------
        status
            Whether the graph was saved successfully.
        """

        current_index = self._tab_widget_QTabWidget.currentIndex()

        if current_index == -1:
            return

        file_path = self._tab_widget_QTabWidget.currentWidget().file_path  # pyright: ignore

        if status:
            self._set_current_tab_text()
            self._add_to_recent_files(file_path)

    def _update_recent_files_menu(self) -> None:
        """Update the recent files menu."""

        self._recent_files_menu.clear()

        recent_files: list = self._settings.value("recent_files", [], type=list)  # pyright: ignore

        if not recent_files:
            no_recent_action = QAction("No Recent Files", self)
            no_recent_action.setEnabled(False)
            self._recent_files_menu.addAction(no_recent_action)
            return

        for file_path in recent_files:
            action = QAction(file_path, self)
            action.triggered.connect(
                lambda _, path=file_path: self._open_recent_file(path)
            )
            self._recent_files_menu.addAction(action)

    def _add_to_recent_files(self, file_path: str) -> None:
        """
        Add a file path to the recent files list.

        Parameters
        ----------
        file_path
            Path to the file to add.
        """

        recent_files: list = self._settings.value("recent_files", [], type=list)  # pyright: ignore

        if file_path in recent_files:
            recent_files.remove(file_path)

        recent_files.insert(0, file_path)

        recent_files = recent_files[: self._recent_files_count]

        self._settings.setValue("recent_files", recent_files)

        self._update_recent_files_menu()

    def _remove_from_recent_files(self, file_path: str) -> None:
        """
        Remove a file path from the recent files list.

        Parameters
        ----------
        file_path
            Path to the file to remove.
        """

        recent_files: list = self._settings.value("recent_files", [], type=list)  # pyright: ignore

        if file_path in recent_files:
            recent_files.remove(file_path)

        recent_files = recent_files[: self._recent_files_count]

        self._settings.setValue("recent_files", recent_files)

        self._update_recent_files_menu()

    def _open_recent_file(self, file_path: str) -> None:
        """
        Open a file from the recent files list.

        Parameters
        ----------
        file_path
            Path to the file to open.
        """

        self._add_new_graph(file_path)


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format=(
            "%(asctime)s [%(levelname)8s] [Thread ID: %(thread)d] %(name)s: %(message)s"
        ),
    )

    application = QApplication(sys.argv)
    application.setStyle("Fusion")
    graph_editor = GraphEditor(True)
    graph_editor.show()

    sys.exit(application.exec())
