from __future__ import annotations

import os
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

from spa_core.audio import MEDIA_FILE_FILTER, iter_audio_files, play_audio, read_wav, stop_audio
from spa_core.export import export_segment_audit_csv
from spa_core.models import SpaResult, SpaSettings, SpaSignal
from spa_core.qc_schema import QC_AUDIT_GROUPS, QC_AUDIT_METADATA
from spa_core.segmentation import run_spa


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
QC_AUDIT_COLORS = {
    "Environmental noise": "#2f80ed",
    "Competing speech": "#9b51e0",
    "Volume unstable": "#f2994a",
    "Clipping": "#d64545",
    "Reverberation/echo": "#00a676",
    "Platform effects": "#00a3a3",
    "Temporal discontinuities": "#6c757d",
    "Any non-task related content": "#b75d1e",
}
BAMBOO_PASSAGE = (
    "Bamboo walls are getting to be very popular. They are strong, easy to use, and good-looking. "
    "They provide a good background and can create a look of a Japanese garden. Bamboo is a grass, "
    "and is one of the most rapidly growing grasses in the world. Many varieties of bamboo are grown "
    "in Asia, although it is also grown in America. Last year we bought a new home and have been "
    "working on the flower garden. In a few more days, we will be done with the bamboo wall in our "
    "garden. We have really enjoyed the project."
)

_MPL_CONFIG_DIR = PROJECT_ROOT / ".matplotlib"
_MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))


try:
    from PySide6.QtCore import QPoint, Qt, QTimer
    from PySide6.QtGui import QColor, QKeySequence, QPainter, QPen, QShortcut
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSlider,
        QStackedLayout,
        QStackedWidget,
        QVBoxLayout,
        QWidget,
    )
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except Exception as exc:  # pragma: no cover - import-time GUI dependency check
    raise RuntimeError("PySide6 and matplotlib are required for the SPA GUI. Install requirements.txt first.") from exc


PAGE_SETTINGS = 0
PAGE_LOAD = 1
PAGE_BOUNDARIES = 2
PAGE_REVIEW = 3
PAGE_AUDIT = 4
PAGE_SAVE = 5

MAX_DISPLAY_POINTS = 25000


class CrosshairOverlay(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.position: QPoint | None = None
        self.playback_x: int | None = None
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.hide()

    def set_position(self, x: float, y: float) -> None:
        next_position = QPoint(int(round(x)), int(round(y)))
        if self.position == next_position and self.isVisible():
            return
        self.position = next_position
        self.show()
        self.raise_()
        self.update()

    def clear(self) -> None:
        if self.position is None:
            return
        self.position = None
        if self.playback_x is None:
            self.hide()
        self.update()

    def set_playback_x(self, x: float) -> None:
        next_x = int(round(x))
        if self.playback_x == next_x and self.isVisible():
            return
        self.playback_x = next_x
        self.show()
        self.raise_()
        self.update()

    def clear_playback(self) -> None:
        if self.playback_x is None:
            return
        self.playback_x = None
        if self.position is None:
            self.hide()
        self.update()

    def paintEvent(self, event) -> None:  # pragma: no cover - visual Qt drawing
        if self.position is None and self.playback_x is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        if self.playback_x is not None:
            playback_shadow = QPen(QColor(255, 255, 255, 220))
            playback_shadow.setWidth(5)
            painter.setPen(playback_shadow)
            painter.drawLine(self.playback_x, 0, self.playback_x, self.height())
            playback_pen = QPen(QColor("#c43c39"))
            playback_pen.setWidth(2)
            painter.setPen(playback_pen)
            painter.drawLine(self.playback_x, 0, self.playback_x, self.height())
        if self.position is None:
            return
        shadow = QPen(QColor(255, 255, 255, 180))
        shadow.setWidth(3)
        painter.setPen(shadow)
        painter.drawLine(self.position.x(), 0, self.position.x(), self.height())
        painter.drawLine(0, self.position.y(), self.width(), self.position.y())
        pen = QPen(QColor("#111111"))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(self.position.x(), 0, self.position.x(), self.height())
        painter.drawLine(0, self.position.y(), self.width(), self.position.y())


class SettingsDialog(QDialog):
    def __init__(self, settings: SpaSettings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("SPA Settings")
        self.setMinimumWidth(360)
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.curve_combo = QComboBox()
        self.curve_combo.addItems(["adaptive", "static"])
        self.curve_combo.setCurrentText("adaptive" if settings.adaptive else "static")

        self.sd_spin = QDoubleSpinBox()
        self.sd_spin.setRange(0.0, 20.0)
        self.sd_spin.setDecimals(2)
        self.sd_spin.setValue(settings.sd_multiplier)

        self.speech_spin = QDoubleSpinBox()
        self.speech_spin.setRange(0.0, 5000.0)
        self.speech_spin.setDecimals(1)
        self.speech_spin.setSuffix(" ms")
        self.speech_spin.setValue(settings.speech_threshold_ms)

        self.pause_spin = QDoubleSpinBox()
        self.pause_spin.setRange(0.0, 5000.0)
        self.pause_spin.setDecimals(1)
        self.pause_spin.setSuffix(" ms")
        self.pause_spin.setValue(settings.pause_threshold_ms)

        self.full_file_check = QCheckBox("Analyze full file")
        self.full_file_check.setChecked(settings.use_full_file)
        self.fixed_segments_check = QCheckBox("Split into 5-second audit segments and skip SPA")
        self.fixed_segments_check.setChecked(settings.use_fixed_segments)
        self.fixed_segments_check.toggled.connect(self.update_control_state)

        form.addRow("Threshold curve", self.curve_combo)
        form.addRow("SD multiplier", self.sd_spin)
        form.addRow("Speech minimum", self.speech_spin)
        form.addRow("Pause minimum", self.pause_spin)
        form.addRow("Analysis", self.full_file_check)
        form.addRow("Fixed segments", self.fixed_segments_check)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.update_control_state()

    def update_control_state(self) -> None:
        use_spa = not self.fixed_segments_check.isChecked()
        for widget in (self.curve_combo, self.sd_spin, self.speech_spin, self.pause_spin, self.full_file_check):
            widget.setEnabled(use_spa)

    def settings(self) -> SpaSettings:
        return SpaSettings(
            speech_threshold_ms=self.speech_spin.value(),
            pause_threshold_ms=self.pause_spin.value(),
            sd_multiplier=self.sd_spin.value(),
            threshold_mode="automatic",
            adaptive=self.curve_combo.currentText() == "adaptive",
            variable_mode="ONE TIME",
            iterations=1,
            use_full_file=self.full_file_check.isChecked() or self.fixed_segments_check.isChecked(),
            use_fixed_segments=self.fixed_segments_check.isChecked(),
            fixed_segment_seconds=5.0,
        )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SPA Segmentation")
        self.resize(1480, 900)

        self.settings = SpaSettings(use_full_file=True, use_fixed_segments=True)
        self.queue: list[Path] = []
        self.queue_index = -1
        self.signal: SpaSignal | None = None
        self.result: SpaResult | None = None
        self.noise_region: tuple[int, int] | None = None
        self.analysis_region: tuple[int, int] | None = None
        self.boundary_mode = "noise"
        self.boundary_clicks: list[int] = []
        self.review_qc_data: dict[str, object] = {"selected_effects": []}
        self.review_qc_checks: list[tuple[str, QCheckBox, QCheckBox]] = []
        self.review_zoom_region: tuple[int, int] | None = None
        self.review_zoom_clicks: list[int] = []
        self.review_zoom_selecting = False
        self.review_qc_issue_key: tuple[str, str] | None = None
        self.review_qc_issue_clicks: list[int] = []
        self.review_qc_issue_edit_index: int | None = None
        self._updating_review_qc_checks = False
        self.audit_by_segment_number: dict[int, dict[str, object]] = {}
        self.audit_checks: list[tuple[str, QCheckBox, QCheckBox]] = []
        self.audit_index = 0
        self.audit_zoom_region: tuple[int, int] | None = None
        self.audit_zoom_clicks: list[int] = []
        self.audit_zoom_selecting = False
        self.audit_issue_key: tuple[str, str] | None = None
        self.audit_issue_clicks: list[int] = []
        self.audit_issue_edit_index: int | None = None
        self.review_return_audit_index: int | None = None
        self._updating_audit_checks = False
        self.loaded_from_saved_output = False
        self.progress_dirty = False
        self.last_saved_path: Path | None = None
        self.output_run_dir = OUTPUT_ROOT
        self._updating_boundaries = False
        self.playback_widgets: list[tuple[QSlider, QLabel]] = []
        self.playback_active = False
        self.playback_range: tuple[int, int] | None = None
        self.playback_current_sample = 0
        self.playback_started_sample = 0
        self.playback_started_at = 0.0
        self.playback_label = "Playback"
        self.playback_speed = 1.0
        self.playback_button: QPushButton | None = None
        self.playback_button_play_text = ""
        self.playback_button_stop_text = ""
        self._updating_playback_widgets = False
        self._playback_slider_dragging = False
        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(50)
        self.playback_timer.timeout.connect(self.update_playback_progress)
        self.space_shortcut: QShortcut | None = None

        self._build_ui()
        self._apply_styles()
        self._show_page(PAGE_SETTINGS)

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.top_bar = QFrame()
        top_layout = QHBoxLayout(self.top_bar)
        top_layout.setContentsMargins(16, 10, 16, 10)
        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(self.go_back)
        self.session_label = QLabel("")
        self.session_label.setWordWrap(True)
        self.files_button = QPushButton("Files")
        self.files_button.clicked.connect(self.go_to_load_page)
        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self.open_settings_dialog)
        top_layout.addWidget(self.back_button)
        top_layout.addWidget(self.session_label, stretch=1)
        top_layout.addWidget(self.files_button)
        top_layout.addWidget(self.settings_button)
        layout.addWidget(self.top_bar)

        self.space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.space_shortcut.setContext(Qt.ApplicationShortcut)
        self.space_shortcut.activated.connect(self.toggle_current_page_playback)

        self.stack = QStackedWidget()
        layout.addWidget(self.stack, stretch=1)
        self.setCentralWidget(root)

        self.settings_page = self._build_settings_page()
        self.load_page = self._build_load_page()
        self.boundary_page = self._build_boundary_page()
        self.review_page = self._build_review_page()
        self.audit_page = self._build_audit_page()
        self.save_page = self._build_save_page()
        for page in (
            self.settings_page,
            self.load_page,
            self.boundary_page,
            self.review_page,
            self.audit_page,
            self.save_page,
        ):
            self.stack.addWidget(page)

    def _build_settings_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(80, 60, 80, 60)
        outer.setSpacing(24)
        title = QLabel("SPA Settings")
        title.setObjectName("PageTitle")
        subtitle = QLabel("These settings apply to the whole session. They can be changed later from the top bar.")
        subtitle.setObjectName("SubtleText")
        outer.addWidget(title)
        outer.addWidget(subtitle)

        box = QGroupBox("Core Segmentation Settings")
        form = QFormLayout(box)
        form.setSpacing(14)
        self.start_curve_combo = QComboBox()
        self.start_curve_combo.addItems(["adaptive", "static"])
        self.start_sd_spin = QDoubleSpinBox()
        self.start_sd_spin.setRange(0.0, 20.0)
        self.start_sd_spin.setDecimals(2)
        self.start_sd_spin.setValue(self.settings.sd_multiplier)
        self.start_speech_spin = QDoubleSpinBox()
        self.start_speech_spin.setRange(0.0, 5000.0)
        self.start_speech_spin.setDecimals(1)
        self.start_speech_spin.setSuffix(" ms")
        self.start_speech_spin.setValue(self.settings.speech_threshold_ms)
        self.start_pause_spin = QDoubleSpinBox()
        self.start_pause_spin.setRange(0.0, 5000.0)
        self.start_pause_spin.setDecimals(1)
        self.start_pause_spin.setSuffix(" ms")
        self.start_pause_spin.setValue(self.settings.pause_threshold_ms)
        self.start_full_file_check = QCheckBox("Analyze full file")
        self.start_full_file_check.setChecked(self.settings.use_full_file)
        self.start_fixed_segments_check = QCheckBox("Split into 5-second audit segments and skip SPA")
        self.start_fixed_segments_check.setChecked(self.settings.use_fixed_segments)
        self.start_fixed_segments_check.toggled.connect(self.update_start_settings_state)
        form.addRow("Threshold curve", self.start_curve_combo)
        form.addRow("SD multiplier", self.start_sd_spin)
        form.addRow("Speech minimum", self.start_speech_spin)
        form.addRow("Pause minimum", self.start_pause_spin)
        form.addRow("Analysis", self.start_full_file_check)
        form.addRow("Fixed segments", self.start_fixed_segments_check)
        outer.addWidget(box)

        continue_button = QPushButton("Continue To Files")
        continue_button.setObjectName("PrimaryButton")
        continue_button.clicked.connect(self.accept_start_settings)
        outer.addWidget(continue_button, alignment=Qt.AlignRight)
        outer.addStretch(1)
        self.update_start_settings_state()
        return page

    def update_start_settings_state(self) -> None:
        use_spa = not self.start_fixed_segments_check.isChecked()
        for widget in (
            self.start_curve_combo,
            self.start_sd_spin,
            self.start_speech_spin,
            self.start_pause_spin,
            self.start_full_file_check,
        ):
            widget.setEnabled(use_spa)

    def _build_load_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(48, 36, 48, 36)
        outer.setSpacing(18)
        title = QLabel("Load Audio")
        title.setObjectName("PageTitle")
        outer.addWidget(title)

        actions = QHBoxLayout()
        single = QPushButton("Select Audio File")
        multiple = QPushButton("Select Multiple Audio Files")
        folder = QPushButton("Select Folder")
        for button in (single, multiple, folder):
            button.setMinimumHeight(42)
            actions.addWidget(button)
        single.clicked.connect(self.select_single_file)
        multiple.clicked.connect(self.select_multiple_files)
        folder.clicked.connect(self.select_folder)
        outer.addLayout(actions)

        output_box = QGroupBox("Output Folder")
        output_layout = QHBoxLayout(output_box)
        self.output_folder_label = QLabel(str(self.output_run_dir))
        self.output_folder_label.setWordWrap(True)
        browse_output = QPushButton("Choose Output Folder")
        browse_output.clicked.connect(self.select_output_folder)
        output_layout.addWidget(self.output_folder_label, stretch=1)
        output_layout.addWidget(browse_output)
        outer.addWidget(output_box)

        self.queue_label = QLabel("No files queued")
        self.queue_label.setObjectName("SubtleText")
        self.queue_list = QListWidget()
        self.queue_list.setMinimumHeight(360)
        self.queue_list.itemDoubleClicked.connect(lambda item: self.open_selected_queue_file())
        outer.addWidget(self.queue_label)
        outer.addWidget(self.queue_list, stretch=1)

        queue_actions = QHBoxLayout()
        open_selected = QPushButton("Open Selected File")
        open_selected.clicked.connect(self.open_selected_queue_file)
        next_button = QPushButton("Start Next Pending File")
        next_button.setObjectName("PrimaryButton")
        next_button.clicked.connect(self.start_next_queued_file)
        queue_actions.addWidget(open_selected)
        queue_actions.addStretch(1)
        queue_actions.addWidget(next_button)
        outer.addLayout(queue_actions)
        return page

    def create_playback_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("PlaybackBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(10)
        label = QLabel("Playback: --")
        label.setMinimumWidth(280)
        slider = QSlider(Qt.Horizontal)
        slider.setEnabled(False)
        slider.setRange(0, 1)
        slider.setValue(0)
        slider.sliderPressed.connect(lambda slider=slider: self.on_playback_slider_pressed(slider))
        slider.sliderMoved.connect(lambda value, slider=slider: self.on_playback_slider_moved(slider, value))
        slider.sliderReleased.connect(lambda slider=slider: self.on_playback_slider_released(slider))
        layout.addWidget(label)
        layout.addWidget(slider, stretch=1)
        self.playback_widgets.append((slider, label))
        return bar

    def create_passage_box(self) -> QGroupBox:
        box = QGroupBox("Bamboo Passage")
        layout = QVBoxLayout(box)
        passage = QLabel(BAMBOO_PASSAGE)
        passage.setObjectName("PassageText")
        passage.setWordWrap(True)
        layout.addWidget(passage)
        return box

    def _build_boundary_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(24, 18, 24, 18)
        outer.setSpacing(12)

        header = QHBoxLayout()
        self.boundary_title = QLabel("Select Boundaries")
        self.boundary_title.setObjectName("PageTitle")
        self.boundary_hint = QLabel("")
        self.boundary_hint.setObjectName("SubtleText")
        header.addWidget(self.boundary_title)
        header.addWidget(self.boundary_hint, stretch=1)
        outer.addLayout(header)

        content = QHBoxLayout()
        self.boundary_figure = Figure(figsize=(9, 6), tight_layout=True)
        self.boundary_canvas = FigureCanvas(self.boundary_figure)
        self.boundary_ax_norm = self.boundary_figure.add_subplot(211)
        self.boundary_ax_raw = self.boundary_figure.add_subplot(212, sharex=self.boundary_ax_norm)
        self.boundary_canvas.mpl_connect("button_press_event", self.on_boundary_click)
        self.boundary_canvas.mpl_connect("motion_notify_event", self.on_boundary_motion)
        self.boundary_canvas.mpl_connect("figure_leave_event", self.on_boundary_leave)
        canvas_stack_widget = QWidget()
        canvas_stack = QStackedLayout(canvas_stack_widget)
        canvas_stack.setContentsMargins(0, 0, 0, 0)
        canvas_stack.setStackingMode(QStackedLayout.StackAll)
        canvas_stack.addWidget(self.boundary_canvas)
        self.boundary_crosshair = CrosshairOverlay()
        canvas_stack.addWidget(self.boundary_crosshair)
        plot_area = QWidget()
        plot_layout = QVBoxLayout(plot_area)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(8)
        plot_layout.addWidget(canvas_stack_widget, stretch=1)
        plot_layout.addWidget(self.create_playback_bar())
        content.addWidget(plot_area, stretch=1)

        side = QFrame()
        side.setObjectName("SidePanel")
        side.setFixedWidth(390)
        side_layout = QVBoxLayout(side)
        side_layout.setSpacing(12)
        self.boundary_status = QLabel("")
        self.boundary_status.setWordWrap(True)
        side_layout.addWidget(self.boundary_status)

        target_box = QGroupBox("Click Target")
        target_layout = QVBoxLayout(target_box)
        noise_button = QPushButton("Select Noise Boundaries")
        analysis_button = QPushButton("Select Analysis Boundaries")
        noise_button.clicked.connect(lambda: self.set_boundary_mode("noise"))
        analysis_button.clicked.connect(lambda: self.set_boundary_mode("analysis"))
        self.analysis_select_button = analysis_button
        target_layout.addWidget(noise_button)
        target_layout.addWidget(analysis_button)
        side_layout.addWidget(target_box)

        self.boundary_controls_box = QGroupBox("Boundary Adjustments")
        control_layout = QGridLayout(self.boundary_controls_box)
        self.boundary_widgets = {}
        for row, (key, label) in enumerate(
            [
                ("noise_start", "Noise start"),
                ("noise_end", "Noise end"),
                ("analysis_start", "Analysis start"),
                ("analysis_end", "Analysis end"),
            ]
        ):
            slider = QSlider(Qt.Horizontal)
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setSuffix(" s")
            spin.setRange(0.0, 1.0)
            slider.valueChanged.connect(lambda value, name=key: self.on_boundary_slider_changed(name, value))
            spin.valueChanged.connect(lambda value, name=key: self.on_boundary_spin_changed(name, value))
            control_layout.addWidget(QLabel(label), row, 0)
            control_layout.addWidget(slider, row, 1)
            control_layout.addWidget(spin, row, 2)
            self.boundary_widgets[key] = (slider, spin)
        side_layout.addWidget(self.boundary_controls_box)

        play_box = QGroupBox("Playback")
        play_layout = QVBoxLayout(play_box)
        self.boundary_play_full_button = QPushButton("Play Full Audio")
        self.boundary_play_noise_button = QPushButton("Play Noise Selection")
        self.boundary_play_analysis_button = QPushButton("Play Analysis Region")
        self.boundary_play_full_button.clicked.connect(lambda: self.play_full_audio(self.boundary_play_full_button))
        self.boundary_play_noise_button.clicked.connect(lambda: self.play_noise_region(self.boundary_play_noise_button))
        self.boundary_play_analysis_button.clicked.connect(lambda: self.play_analysis_region(self.boundary_play_analysis_button))
        for button in (self.boundary_play_full_button, self.boundary_play_noise_button, self.boundary_play_analysis_button):
            play_layout.addWidget(button)
        side_layout.addWidget(play_box)

        self.boundary_run_button = QPushButton("Confirm And Run SPA")
        self.boundary_run_button.setObjectName("PrimaryButton")
        self.boundary_run_button.clicked.connect(self.boundary_next_or_run_spa)
        side_layout.addWidget(self.boundary_run_button)
        side_layout.addStretch(1)
        content.addWidget(side)
        outer.addLayout(content, stretch=1)
        return page

    def _build_review_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(24, 18, 24, 18)
        outer.setSpacing(12)
        top = QHBoxLayout()
        title = QLabel("Review Segmentation")
        title.setObjectName("PageTitle")
        self.review_counts = QLabel("")
        self.review_counts.setObjectName("MetricText")
        top.addWidget(title)
        top.addWidget(self.review_counts, stretch=1)
        outer.addLayout(top)
        outer.addWidget(self.create_passage_box())
        self.review_zoom_status = QLabel("")
        self.review_zoom_status.setObjectName("SubtleText")
        outer.addWidget(self.review_zoom_status)

        self.review_figure = Figure(figsize=(10, 6), tight_layout=True)
        self.review_canvas = FigureCanvas(self.review_figure)
        self.review_canvas.mpl_connect("button_press_event", self.on_review_plot_click)
        self.review_ax_norm = self.review_figure.add_subplot(211)
        self.review_ax_raw = self.review_figure.add_subplot(212, sharex=self.review_ax_norm)
        content = QHBoxLayout()
        content.setSpacing(18)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)
        review_stack_widget = QWidget()
        review_stack = QStackedLayout(review_stack_widget)
        review_stack.setContentsMargins(0, 0, 0, 0)
        review_stack.setStackingMode(QStackedLayout.StackAll)
        review_stack.addWidget(self.review_canvas)
        self.review_playback_overlay = CrosshairOverlay()
        review_stack.addWidget(self.review_playback_overlay)
        left_layout.addWidget(review_stack_widget, stretch=1)
        left_layout.addWidget(self.create_playback_bar())
        content.addWidget(left, stretch=1)

        review_qc_panel = QFrame()
        review_qc_panel.setObjectName("SidePanel")
        review_qc_panel.setFixedWidth(430)
        review_qc_layout = QVBoxLayout(review_qc_panel)
        review_qc_layout.setContentsMargins(16, 14, 16, 14)
        review_qc_layout.setSpacing(10)
        review_qc_title = QLabel("File-Level QC")
        review_qc_title.setObjectName("MetricText")
        review_qc_layout.addWidget(review_qc_title)
        review_visibility_buttons = QHBoxLayout()
        self.review_qc_show_all_button = QPushButton("Show All")
        self.review_qc_hide_all_button = QPushButton("Hide All")
        for button, tooltip in (
            (self.review_qc_show_all_button, "Show all file-level QC windows on the plots"),
            (self.review_qc_hide_all_button, "Hide all file-level QC windows on the plots"),
        ):
            button.setObjectName("CompactButton")
            button.setToolTip(tooltip)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.review_qc_show_all_button.clicked.connect(lambda: self.set_all_review_qc_visibility(True))
        self.review_qc_hide_all_button.clicked.connect(lambda: self.set_all_review_qc_visibility(False))
        review_visibility_buttons.addWidget(self.review_qc_show_all_button)
        review_visibility_buttons.addWidget(self.review_qc_hide_all_button)
        review_qc_layout.addLayout(review_visibility_buttons)

        review_issue_box = QGroupBox("Issue Boundary")
        review_issue_layout = QVBoxLayout(review_issue_box)
        self.review_qc_status = QLabel("Check a QC artifact to mark the whole file or a large region.")
        self.review_qc_status.setWordWrap(True)
        self.review_qc_status.setObjectName("SubtleText")
        self.review_qc_combo = QComboBox()
        self.review_qc_window_list = QListWidget()
        self.review_qc_window_list.setMaximumHeight(88)
        self.review_qc_window_list.currentRowChanged.connect(self.on_review_qc_window_selected)
        review_issue_buttons = QHBoxLayout()
        self.review_qc_whole_file_button = QPushButton("Whole File")
        self.review_qc_add_button = QPushButton("Add/Reset")
        self.review_qc_play_button = QPushButton("Play")
        self.review_qc_confirm_button = QPushButton("Confirm")
        for button, tooltip in (
            (self.review_qc_whole_file_button, "Mark the selected artifact as present throughout the full source file"),
            (self.review_qc_add_button, "Start a new issue window, or reset the in-progress window"),
            (self.review_qc_play_button, "Play the in-progress or selected issue window"),
            (self.review_qc_confirm_button, "Confirm and save this issue window"),
        ):
            button.setObjectName("CompactButton")
            button.setToolTip(tooltip)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        review_window_buttons = QHBoxLayout()
        self.review_qc_edit_button = QPushButton("Edit Window")
        self.review_qc_remove_button = QPushButton("Remove Window")
        for button, tooltip in (
            (self.review_qc_edit_button, "Edit the selected saved file-level window"),
            (self.review_qc_remove_button, "Remove the selected saved file-level window"),
        ):
            button.setObjectName("CompactButton")
            button.setToolTip(tooltip)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.review_qc_whole_file_button.clicked.connect(self.mark_review_qc_whole_file)
        self.review_qc_add_button.clicked.connect(self.edit_selected_review_qc_boundary)
        self.review_qc_play_button.clicked.connect(lambda: self.play_selected_review_qc_boundary(self.review_qc_play_button))
        self.review_qc_confirm_button.clicked.connect(self.confirm_review_qc_boundary)
        self.review_qc_combo.currentIndexChanged.connect(self.on_review_qc_combo_changed)
        self.review_qc_edit_button.clicked.connect(self.edit_selected_review_qc_window)
        self.review_qc_remove_button.clicked.connect(self.remove_selected_review_qc_window)
        review_issue_buttons.addWidget(self.review_qc_whole_file_button)
        review_issue_buttons.addWidget(self.review_qc_add_button)
        review_issue_buttons.addWidget(self.review_qc_play_button)
        review_issue_buttons.addWidget(self.review_qc_confirm_button)
        review_window_buttons.addWidget(self.review_qc_edit_button)
        review_window_buttons.addWidget(self.review_qc_remove_button)
        review_issue_layout.addWidget(self.review_qc_status)
        review_issue_layout.addWidget(self.review_qc_combo)
        review_issue_layout.addWidget(self.review_qc_window_list)
        review_issue_layout.addLayout(review_issue_buttons)
        review_issue_layout.addLayout(review_window_buttons)
        review_qc_layout.addWidget(review_issue_box)

        review_qc_container = QWidget()
        review_qc_checks_layout = QVBoxLayout(review_qc_container)
        review_qc_checks_layout.setContentsMargins(0, 0, 0, 0)
        review_qc_checks_layout.setSpacing(10)
        self.review_qc_checks = []
        for gui_name, effects in QC_AUDIT_GROUPS:
            group = QGroupBox(gui_name)
            group_layout = QVBoxLayout(group)
            group_layout.setSpacing(4)
            for effect in effects:
                row = QHBoxLayout()
                row.setSpacing(6)
                check = QCheckBox(effect)
                check.toggled.connect(lambda checked, gui_name=gui_name, check=check: self.on_review_qc_check_toggled(gui_name, check, checked))
                visible = QCheckBox("Show")
                visible.setObjectName("VisibilityCheck")
                visible.setToolTip("Show or hide this file-level artifact's marked windows on the plots")
                visible.setChecked(False)
                visible.setEnabled(False)
                visible.toggled.connect(lambda checked, gui_name=gui_name, check=check: self.on_review_qc_visibility_toggled(gui_name, check, checked))
                self.review_qc_checks.append((gui_name, check, visible))
                row.addWidget(check, stretch=1)
                row.addWidget(visible)
                group_layout.addLayout(row)
            review_qc_checks_layout.addWidget(group)
        review_qc_checks_layout.addStretch(1)
        review_qc_scroll = QScrollArea()
        review_qc_scroll.setWidgetResizable(True)
        review_qc_scroll.setWidget(review_qc_container)
        review_qc_layout.addWidget(review_qc_scroll, stretch=1)
        content.addWidget(review_qc_panel)
        outer.addLayout(content, stretch=1)

        actions = QHBoxLayout()
        self.review_play_full_button = QPushButton("Play Full Audio")
        self.review_play_analysis_button = QPushButton("Play Analysis Region")
        self.review_zoom_select_button = QPushButton("Select Zoom")
        self.review_zoom_play_button = QPushButton("Play Zoom")
        self.review_zoom_reset_button = QPushButton("Reset Zoom")
        self.review_return_segment_button = QPushButton("Return To Segment")
        self.review_segment_combo = QComboBox()
        self.review_open_segment_button = QPushButton("Open Segment")
        self.review_next_button = QPushButton("Confirm Segmentation")
        self.review_next_button.setObjectName("PrimaryButton")
        self.review_play_full_button.clicked.connect(lambda: self.play_full_audio(self.review_play_full_button))
        self.review_play_analysis_button.clicked.connect(lambda: self.play_analysis_region(self.review_play_analysis_button))
        self.review_zoom_select_button.clicked.connect(self.start_review_zoom_selection)
        self.review_zoom_play_button.clicked.connect(lambda: self.play_current_review_zoom(self.review_zoom_play_button))
        self.review_zoom_reset_button.clicked.connect(self.reset_review_zoom)
        self.review_return_segment_button.clicked.connect(self.return_to_review_segment)
        self.review_open_segment_button.clicked.connect(self.open_review_selected_segment)
        self.review_next_button.clicked.connect(self.confirm_segmentation)
        actions.addWidget(self.review_play_full_button)
        actions.addWidget(self.review_play_analysis_button)
        actions.addWidget(self.review_zoom_select_button)
        actions.addWidget(self.review_zoom_play_button)
        actions.addWidget(self.review_zoom_reset_button)
        actions.addWidget(self.review_return_segment_button)
        actions.addWidget(QLabel("Segment"))
        actions.addWidget(self.review_segment_combo, stretch=1)
        actions.addWidget(self.review_open_segment_button)
        actions.addStretch(1)
        actions.addWidget(self.review_next_button)
        outer.addLayout(actions)
        return page

    def _build_audit_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(36, 24, 36, 24)
        outer.setSpacing(12)
        self.audit_title = QLabel("Audit Segment")
        self.audit_title.setObjectName("PageTitle")
        self.audit_meta = QLabel("")
        self.audit_meta.setObjectName("MetricText")
        self.audit_zoom_status = QLabel("")
        self.audit_zoom_status.setObjectName("SubtleText")
        outer.addWidget(self.audit_title)
        outer.addWidget(self.audit_meta)
        outer.addWidget(self.audit_zoom_status)

        content = QHBoxLayout()
        content.setSpacing(18)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)
        passage_box = self.create_passage_box()
        passage_box.setMaximumHeight(145)
        left_layout.addWidget(passage_box)

        self.audit_figure = Figure(figsize=(10, 6), tight_layout=True)
        self.audit_canvas = FigureCanvas(self.audit_figure)
        self.audit_canvas.mpl_connect("button_press_event", self.on_audit_plot_click)
        audit_grid = self.audit_figure.add_gridspec(2, 1, height_ratios=[1.0, 1.15])
        self.audit_ax = self.audit_figure.add_subplot(audit_grid[0])
        self.audit_ax_spec = self.audit_figure.add_subplot(audit_grid[1], sharex=self.audit_ax)
        audit_stack_widget = QWidget()
        audit_stack_widget.setMinimumHeight(430)
        audit_stack = QStackedLayout(audit_stack_widget)
        audit_stack.setContentsMargins(0, 0, 0, 0)
        audit_stack.setStackingMode(QStackedLayout.StackAll)
        audit_stack.addWidget(self.audit_canvas)
        self.audit_playback_overlay = CrosshairOverlay()
        audit_stack.addWidget(self.audit_playback_overlay)
        left_layout.addWidget(audit_stack_widget, stretch=1)
        left_layout.addWidget(self.create_playback_bar())
        content.addWidget(left, stretch=1)

        right_panel = QFrame()
        right_panel.setObjectName("SidePanel")
        right_panel.setFixedWidth(430)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(16, 14, 16, 14)
        right_layout.setSpacing(10)
        qc_title = QLabel("QC Audit")
        qc_title.setObjectName("MetricText")
        right_layout.addWidget(qc_title)
        visibility_buttons = QHBoxLayout()
        self.audit_show_all_button = QPushButton("Show All")
        self.audit_hide_all_button = QPushButton("Hide All")
        for button, tooltip in (
            (self.audit_show_all_button, "Show all saved QC windows on the plots"),
            (self.audit_hide_all_button, "Hide all saved QC windows on the plots"),
        ):
            button.setObjectName("CompactButton")
            button.setToolTip(tooltip)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.audit_show_all_button.clicked.connect(lambda: self.set_all_audit_visibility(True))
        self.audit_hide_all_button.clicked.connect(lambda: self.set_all_audit_visibility(False))
        visibility_buttons.addWidget(self.audit_show_all_button)
        visibility_buttons.addWidget(self.audit_hide_all_button)
        right_layout.addLayout(visibility_buttons)

        issue_box = QGroupBox("Issue Boundary")
        issue_layout = QVBoxLayout(issue_box)
        self.audit_issue_status = QLabel("Check a QC artifact to set where it occurs.")
        self.audit_issue_status.setWordWrap(True)
        self.audit_issue_status.setObjectName("SubtleText")
        self.audit_issue_combo = QComboBox()
        self.audit_issue_window_list = QListWidget()
        self.audit_issue_window_list.setMaximumHeight(88)
        self.audit_issue_window_list.currentRowChanged.connect(self.on_audit_issue_window_selected)
        issue_buttons = QHBoxLayout()
        self.audit_issue_edit_button = QPushButton("Add/Reset")
        self.audit_issue_play_button = QPushButton("Play")
        self.audit_issue_confirm_button = QPushButton("Confirm")
        for button, tooltip in (
            (self.audit_issue_edit_button, "Start a new window, or reset the in-progress window"),
            (self.audit_issue_play_button, "Play the in-progress or selected issue window"),
            (self.audit_issue_confirm_button, "Confirm and save this issue window"),
        ):
            button.setObjectName("CompactButton")
            button.setToolTip(tooltip)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        window_buttons = QHBoxLayout()
        self.audit_issue_window_edit_button = QPushButton("Edit Window")
        self.audit_issue_window_remove_button = QPushButton("Remove Window")
        for button, tooltip in (
            (self.audit_issue_window_edit_button, "Edit the selected saved window"),
            (self.audit_issue_window_remove_button, "Remove the selected saved window"),
        ):
            button.setObjectName("CompactButton")
            button.setToolTip(tooltip)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.audit_issue_edit_button.clicked.connect(self.edit_selected_issue_boundary)
        self.audit_issue_play_button.clicked.connect(lambda: self.play_selected_issue_boundary(self.audit_issue_play_button))
        self.audit_issue_confirm_button.clicked.connect(self.confirm_audit_issue_boundary)
        self.audit_issue_combo.currentIndexChanged.connect(self.on_audit_issue_combo_changed)
        self.audit_issue_window_edit_button.clicked.connect(self.edit_selected_issue_window)
        self.audit_issue_window_remove_button.clicked.connect(self.remove_selected_issue_window)
        issue_buttons.addWidget(self.audit_issue_edit_button)
        issue_buttons.addWidget(self.audit_issue_play_button)
        issue_buttons.addWidget(self.audit_issue_confirm_button)
        window_buttons.addWidget(self.audit_issue_window_edit_button)
        window_buttons.addWidget(self.audit_issue_window_remove_button)
        issue_layout.addWidget(self.audit_issue_status)
        issue_layout.addWidget(self.audit_issue_combo)
        issue_layout.addWidget(self.audit_issue_window_list)
        issue_layout.addLayout(issue_buttons)
        issue_layout.addLayout(window_buttons)
        right_layout.addWidget(issue_box)

        qc_container = QWidget()
        qc_layout = QVBoxLayout(qc_container)
        qc_layout.setContentsMargins(0, 0, 0, 0)
        qc_layout.setSpacing(10)
        self.audit_checks = []
        for gui_name, effects in QC_AUDIT_GROUPS:
            group = QGroupBox(gui_name)
            group_layout = QVBoxLayout(group)
            group_layout.setSpacing(4)
            for effect in effects:
                row = QHBoxLayout()
                row.setSpacing(6)
                check = QCheckBox(effect)
                check.toggled.connect(lambda checked, gui_name=gui_name, check=check: self.on_audit_check_toggled(gui_name, check, checked))
                visible = QCheckBox("Show")
                visible.setObjectName("VisibilityCheck")
                visible.setToolTip("Show or hide this artifact's marked windows on the plots")
                visible.setChecked(False)
                visible.setEnabled(False)
                visible.toggled.connect(lambda checked, gui_name=gui_name, check=check: self.on_audit_visibility_toggled(gui_name, check, checked))
                self.audit_checks.append((gui_name, check, visible))
                row.addWidget(check, stretch=1)
                row.addWidget(visible)
                group_layout.addLayout(row)
            qc_layout.addWidget(group)
        qc_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(qc_container)
        right_layout.addWidget(scroll, stretch=1)

        content.addWidget(right_panel)
        outer.addLayout(content, stretch=1)

        actions = QHBoxLayout()
        self.audit_play_button = QPushButton("Play Segment")
        self.audit_zoom_select_button = QPushButton("Select Zoom")
        self.audit_zoom_play_button = QPushButton("Play Zoom")
        self.audit_zoom_reset_button = QPushButton("Reset Zoom")
        self.audit_review_button = QPushButton("View Segmentation")
        self.audit_segment_combo = QComboBox()
        self.audit_go_segment_button = QPushButton("Go")
        speed_label = QLabel("Speed")
        self.audit_speed_combo = QComboBox()
        for label, value in (
            ("0.50x", 0.50),
            ("0.75x", 0.75),
            ("1.00x", 1.00),
            ("1.25x", 1.25),
            ("1.50x", 1.50),
            ("2.00x", 2.00),
        ):
            self.audit_speed_combo.addItem(label, value)
        self.audit_speed_combo.setCurrentText("1.00x")
        previous = QPushButton("Previous Segment")
        next_button = QPushButton("Next Segment")
        next_button.setObjectName("PrimaryButton")
        self.audit_play_button.clicked.connect(lambda: self.play_current_audit_segment(self.audit_play_button))
        self.audit_zoom_select_button.clicked.connect(self.start_audit_zoom_selection)
        self.audit_zoom_play_button.clicked.connect(lambda: self.play_current_audit_zoom(self.audit_zoom_play_button))
        self.audit_zoom_reset_button.clicked.connect(self.reset_audit_zoom)
        self.audit_review_button.clicked.connect(self.view_segmentation_from_audit)
        self.audit_go_segment_button.clicked.connect(self.go_to_audit_selected_segment)
        self.audit_speed_combo.currentIndexChanged.connect(self.on_playback_speed_changed)
        previous.clicked.connect(self.previous_audit_segment)
        next_button.clicked.connect(self.next_audit_segment)
        self.audit_previous_button = previous
        self.audit_next_button = next_button
        actions.addWidget(self.audit_play_button)
        actions.addWidget(self.audit_zoom_select_button)
        actions.addWidget(self.audit_zoom_play_button)
        actions.addWidget(self.audit_zoom_reset_button)
        actions.addWidget(self.audit_review_button)
        actions.addWidget(QLabel("Segment"))
        actions.addWidget(self.audit_segment_combo, stretch=1)
        actions.addWidget(self.audit_go_segment_button)
        actions.addWidget(speed_label)
        actions.addWidget(self.audit_speed_combo)
        actions.addStretch(1)
        actions.addWidget(previous)
        actions.addWidget(next_button)
        outer.addLayout(actions)
        return page

    def _build_save_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(64, 50, 64, 50)
        outer.setSpacing(18)
        title = QLabel("Save Output")
        title.setObjectName("PageTitle")
        self.save_summary = QLabel("")
        self.save_summary.setWordWrap(True)
        self.save_path_label = QLabel("")
        self.save_path_label.setObjectName("SubtleText")
        outer.addWidget(title)
        outer.addWidget(self.save_summary)
        outer.addWidget(self.save_path_label)
        save = QPushButton("Confirm And Save")
        save.setObjectName("PrimaryButton")
        save.clicked.connect(self.save_current_file)
        outer.addWidget(save, alignment=Qt.AlignRight)
        outer.addStretch(1)
        return page

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #f7f8f8;
                color: #1f2428;
                font-size: 14px;
            }
            QFrame#SidePanel, QGroupBox {
                background: #ffffff;
                border: 1px solid #d7dcdf;
                border-radius: 6px;
            }
            QGroupBox {
                margin-top: 12px;
                padding: 12px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
            QLabel#PageTitle {
                font-size: 28px;
                font-weight: 700;
            }
            QLabel#MetricText {
                font-size: 17px;
                font-weight: 600;
            }
            QLabel#SubtleText {
                color: #59646b;
            }
            QLabel#PassageText {
                color: #2f363b;
                line-height: 130%;
            }
            QScrollArea {
                border: 0;
                background: transparent;
            }
            QPushButton {
                background: #ffffff;
                border: 1px solid #b9c2c8;
                border-radius: 6px;
                padding: 8px 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #eef4f5;
            }
            QPushButton#PrimaryButton {
                background: #146c75;
                color: #ffffff;
                border: 1px solid #146c75;
                padding: 10px 18px;
            }
            QPushButton#PrimaryButton:hover {
                background: #0f5961;
            }
            QPushButton#CompactButton {
                font-size: 12px;
                padding: 6px 8px;
                min-width: 0;
            }
            QCheckBox#VisibilityCheck {
                font-size: 11px;
                color: #59646b;
            }
            QFrame#PlaybackBar {
                background: #ffffff;
                border: 1px solid #d7dcdf;
                border-radius: 6px;
            }
            QFrame {
                background: #ffffff;
            }
            """
        )

    def accept_start_settings(self) -> None:
        self.settings = SpaSettings(
            speech_threshold_ms=self.start_speech_spin.value(),
            pause_threshold_ms=self.start_pause_spin.value(),
            sd_multiplier=self.start_sd_spin.value(),
            threshold_mode="automatic",
            adaptive=self.start_curve_combo.currentText() == "adaptive",
            variable_mode="ONE TIME",
            iterations=1,
            use_full_file=self.start_full_file_check.isChecked() or self.start_fixed_segments_check.isChecked(),
            use_fixed_segments=self.start_fixed_segments_check.isChecked(),
            fixed_segment_seconds=5.0,
        )
        self._show_page(PAGE_LOAD)

    def _show_page(self, page_index: int) -> None:
        if self.stack.currentIndex() != page_index:
            self.stop_playback()
        self.stack.setCurrentIndex(page_index)
        self.top_bar.setVisible(page_index != PAGE_SETTINGS)
        self.back_button.setEnabled(page_index not in (PAGE_LOAD, PAGE_SETTINGS))
        self.files_button.setEnabled(page_index not in (PAGE_LOAD, PAGE_SETTINGS))
        self.update_session_label()
        if page_index == PAGE_BOUNDARIES:
            self.prepare_boundary_page()
        elif page_index == PAGE_REVIEW:
            self.plot_review()
        elif page_index == PAGE_AUDIT:
            self.load_audit_segment()
        elif page_index == PAGE_SAVE:
            self.prepare_save_page()
        self.autosave_progress(page_index)

    def update_session_label(self) -> None:
        file_text = "No file loaded"
        if self.signal is not None:
            file_text = self.signal.path.name
        queue_text = ""
        if self.queue:
            if self.queue_index >= 0:
                queue_text = f"  File {self.queue_index + 1} of {len(self.queue)}"
            else:
                queue_text = f"  {len(self.queue)} files queued"
        self.session_label.setText(f"{file_text}{queue_text}  Output: {self.output_run_dir}")
        if hasattr(self, "output_folder_label"):
            self.output_folder_label.setText(str(self.output_run_dir))

    def go_back(self) -> None:
        page = self.stack.currentIndex()
        if page == PAGE_BOUNDARIES:
            self.autosave_progress(page)
            self._show_page(PAGE_LOAD)
        elif page == PAGE_REVIEW:
            if self.current_result_is_fixed():
                self.autosave_progress(page)
                self._show_page(PAGE_LOAD)
            else:
                self._show_page(PAGE_BOUNDARIES)
        elif page == PAGE_AUDIT:
            if self.audit_index > 0:
                if not self.ensure_audit_progress_saved_for_navigation():
                    return
                self.audit_index -= 1
                self.clear_audit_zoom()
                self.clear_audit_issue_selection()
                self.load_audit_segment()
            else:
                if not self.ensure_audit_progress_saved_for_navigation():
                    return
                self.review_return_audit_index = 0
                self._show_page(PAGE_REVIEW)
        elif page == PAGE_SAVE:
            if self.result is not None and len(self.audit_segments()) > 0:
                self._show_page(PAGE_AUDIT)
            else:
                self._show_page(PAGE_REVIEW)

    def go_to_load_page(self) -> None:
        self.autosave_progress(self.stack.currentIndex())
        self.refresh_queue_display()
        self._show_page(PAGE_LOAD)

    def open_settings_dialog(self) -> None:
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec() != QDialog.Accepted:
            return
        new_settings = dialog.settings()
        changed = new_settings != self.settings
        self.settings = new_settings
        if changed and self.signal is not None:
            self.mark_progress_dirty()
            self.result = None
            self.audit_by_segment_number = {}
            self.audit_index = 0
            self.review_return_audit_index = None
            if self.settings.use_fixed_segments:
                self.prepare_fixed_segment_result()
                self._show_page(PAGE_REVIEW)
            else:
                self.noise_region = None
                self.analysis_region = (0, len(self.signal.raw_audio)) if self.settings.use_full_file else None
                self._show_page(PAGE_BOUNDARIES)

    def select_single_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Audio File", str(Path.home()), MEDIA_FILE_FILTER)
        if path:
            self.set_queue([Path(path)])

    def select_multiple_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Audio Files", str(Path.home()), MEDIA_FILE_FILTER)
        if paths:
            self.set_queue([Path(path) for path in paths])

    def select_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select folder of audio/media files", str(Path.home()))
        if not directory:
            return
        paths = iter_audio_files(Path(directory))
        if not paths:
            self.warn("No audio files", "No files with readable audio streams were found in that folder.")
            return
        self.set_queue(paths, default_output_base=Path(directory))

    def select_output_folder(self) -> None:
        start_dir = self.output_run_dir if self.output_run_dir.exists() else self.output_run_dir.parent
        directory = QFileDialog.getExistingDirectory(self, "Select output folder", str(start_dir))
        if not directory:
            return
        self.output_run_dir = Path(directory)
        if self.signal is None:
            self.queue_index = -1
        self.refresh_queue_display()
        self.update_session_label()

    def set_queue(self, paths: list[Path], default_output_base: Path | None = None) -> None:
        self.queue = paths
        self.queue_index = -1
        self.output_run_dir = self.default_output_dir_for_queue(paths, default_output_base)
        self.refresh_queue_display()
        self.update_session_label()

    def refresh_queue_display(self) -> None:
        self.queue_list.clear()
        for index, path in enumerate(self.queue, start=1):
            status = self.queue_status_for_file(path)
            self.queue_list.addItem(f"{index}. [{status}] {path.name}")
        if 0 <= self.queue_index < self.queue_list.count():
            self.queue_list.setCurrentRow(self.queue_index)
        if not self.queue:
            self.queue_label.setText("No files queued")
            return
        pending = sum(1 for path in self.queue if self.queue_status_for_file(path) != "DONE")
        in_progress = sum(1 for path in self.queue if self.queue_status_for_file(path) == "IN PROGRESS")
        done = len(self.queue) - pending
        self.queue_label.setText(f"{len(self.queue)} file(s) queued  |  {pending} pending  |  {in_progress} in progress  |  {done} already saved")

    def start_next_queued_file(self) -> None:
        if not self.queue:
            self.warn("No files", "Select an audio file or queue first.")
            return
        next_index = self.next_pending_index()
        if next_index is None:
            self.warn("Queue complete", "Every queued audio file already has a segment CSV in the output folder.")
            self.refresh_queue_display()
            return
        self.queue_index = next_index
        self.load_best_current_file()

    def open_selected_queue_file(self) -> None:
        if not self.queue:
            self.warn("No files", "Select an audio file or queue first.")
            return
        selected_index = self.queue_list.currentRow()
        if selected_index < 0 or selected_index >= len(self.queue):
            self.warn("No file selected", "Select a file in the queue first.")
            return
        self.queue_index = selected_index
        path = self.queue[self.queue_index]
        if self.should_resume_progress(path) and self.load_progress_current_file():
            return
        if self.output_exists_for_file(path) and self.load_saved_current_file():
            return
        self.load_current_file()

    def next_pending_index(self) -> int | None:
        for index in range(self.queue_index + 1, len(self.queue)):
            if self.queue_status_for_file(self.queue[index]) != "DONE":
                return index
        return None

    def load_best_current_file(self) -> None:
        path = self.queue[self.queue_index]
        if self.should_resume_progress(path) and self.load_progress_current_file():
            return
        if self.output_exists_for_file(path) and self.load_saved_current_file():
            return
        self.load_current_file()

    def default_output_dir_for_queue(self, paths: list[Path], default_output_base: Path | None = None) -> Path:
        if default_output_base is not None:
            return default_output_base / "spa_outputs"
        if not paths:
            return OUTPUT_ROOT
        parents = [path.parent for path in paths]
        try:
            common_parent = Path(os.path.commonpath([str(parent) for parent in parents]))
        except ValueError:
            common_parent = parents[0]
        return common_parent / "spa_outputs"

    def output_path_for_file(self, path: Path) -> Path:
        return self.output_run_dir / f"{path.stem}_segments.csv"

    def metadata_dir(self) -> Path:
        return self.output_run_dir / "metadata_json"

    def metadata_path_for_file(self, path: Path) -> Path:
        return self.metadata_dir() / f"{path.stem}_segments_meta.json"

    def progress_dir(self) -> Path:
        return self.output_run_dir / "progress_json"

    def progress_path_for_file(self, path: Path) -> Path:
        return self.progress_dir() / f"{path.stem}_progress.json"

    def output_exists_for_file(self, path: Path) -> bool:
        return self.output_path_for_file(path).exists()

    def progress_exists_for_file(self, path: Path) -> bool:
        return self.progress_path_for_file(path).exists()

    def should_resume_progress(self, path: Path) -> bool:
        progress_path = self.progress_path_for_file(path)
        if not progress_path.exists():
            return False
        output_path = self.output_path_for_file(path)
        if not output_path.exists():
            return True
        return progress_path.stat().st_mtime > output_path.stat().st_mtime

    def queue_status_for_file(self, path: Path) -> str:
        if self.should_resume_progress(path):
            return "IN PROGRESS"
        if self.output_exists_for_file(path):
            return "DONE"
        if self.progress_exists_for_file(path):
            return "IN PROGRESS"
        return "PENDING"

    def clear_progress_for_file(self, path: Path) -> None:
        progress_path = self.progress_path_for_file(path)
        try:
            if progress_path.exists():
                progress_path.unlink()
        except OSError:
            pass

    def load_current_file(self) -> None:
        path = self.queue[self.queue_index]
        try:
            self.signal = read_wav(path)
        except Exception as exc:
            self.warn("Load failed", str(exc))
            self._show_page(PAGE_LOAD)
            return
        self.result = None
        self.noise_region = None
        self.analysis_region = (0, len(self.signal.raw_audio)) if self.settings.use_full_file else None
        self.boundary_mode = "noise"
        self.boundary_clicks = []
        self.review_qc_data = {"selected_effects": []}
        self.clear_review_qc_issue_selection()
        self.clear_review_zoom()
        self.audit_by_segment_number = {}
        self.audit_index = 0
        self.review_return_audit_index = None
        self.clear_audit_zoom()
        self.clear_audit_issue_selection()
        self.loaded_from_saved_output = False
        self.progress_dirty = False
        self.last_saved_path = None
        if self.settings.use_fixed_segments:
            self.prepare_fixed_segment_result()
            self._show_page(PAGE_REVIEW)
        else:
            self._show_page(PAGE_BOUNDARIES)

    def load_progress_current_file(self) -> bool:
        path = self.queue[self.queue_index]
        progress_path = self.progress_path_for_file(path)
        if not progress_path.exists():
            return False
        try:
            with progress_path.open("r", encoding="utf-8") as handle:
                progress = json.load(handle)
            if not isinstance(progress, dict):
                raise ValueError("Progress file is not a JSON object.")
            self.signal = read_wav(path)
            self.settings = self.settings_from_json(progress.get("settings"))
            self.noise_region = self.region_from_progress(progress.get("noise_region_samples"))
            self.analysis_region = self.region_from_progress(progress.get("analysis_region_samples"))
            if self.analysis_region is not None:
                self.signal.analysis_region = self.analysis_region
            if self.noise_region is not None:
                self.signal.noise_region = self.noise_region
            self.boundary_mode = str(progress.get("boundary_mode") or "noise")
            if self.boundary_mode not in {"noise", "analysis"}:
                self.boundary_mode = "noise"
            self.boundary_clicks = self.sample_list_from_progress(progress.get("boundary_clicks"))
            self.review_qc_data = self.saved_review_qc_data({"file_level_qc": progress.get("file_level_qc")})
            self.audit_by_segment_number = self.audit_from_progress(progress.get("audit_by_segment_number"))
            self.audit_index = max(0, int(progress.get("audit_index") or 0))
            self.review_return_audit_index = self.optional_int_from_progress(progress.get("review_return_audit_index"))
            self.review_qc_issue_key = self.qc_key_from_progress(progress.get("review_qc_issue_key"))
            self.review_qc_issue_clicks = self.sample_list_from_progress(progress.get("review_qc_issue_clicks"))
            self.review_qc_issue_edit_index = self.optional_int_from_progress(progress.get("review_qc_issue_edit_index"))
            self.review_zoom_region = self.region_from_progress(progress.get("review_zoom_region"))
            self.review_zoom_clicks = self.sample_list_from_progress(progress.get("review_zoom_clicks"))
            self.review_zoom_selecting = bool(progress.get("review_zoom_selecting", False))
            self.audit_zoom_region = self.region_from_progress(progress.get("audit_zoom_region"))
            self.audit_zoom_clicks = self.sample_list_from_progress(progress.get("audit_zoom_clicks"))
            self.audit_zoom_selecting = bool(progress.get("audit_zoom_selecting", False))
            self.audit_issue_key = self.qc_key_from_progress(progress.get("audit_issue_key"))
            self.audit_issue_clicks = self.sample_list_from_progress(progress.get("audit_issue_clicks"))
            self.audit_issue_edit_index = self.optional_int_from_progress(progress.get("audit_issue_edit_index"))
            self.result = self.result_from_progress(progress.get("result"))
            if self.result is not None:
                self.analysis_region = self.result.analysis_region
                self.noise_region = self.result.noise_region
        except Exception as exc:
            self.warn("Progress could not be opened", f"{progress_path}\n\n{exc}")
            return False

        self.loaded_from_saved_output = False
        self.progress_dirty = False
        self.last_saved_path = None
        page_index = self.valid_progress_page(progress.get("current_page"))
        self._show_page(page_index)
        return True

    def mark_progress_dirty(self) -> None:
        self.progress_dirty = True
        self.loaded_from_saved_output = False

    def autosave_progress(self, page_index: int | None = None, force: bool = False) -> None:
        if self.signal is None:
            return
        page_index = self.stack.currentIndex() if page_index is None else page_index
        if page_index in (PAGE_SETTINGS, PAGE_LOAD):
            return
        if self.loaded_from_saved_output and not force and not self.progress_dirty:
            return
        if force:
            self.progress_dirty = True
            self.loaded_from_saved_output = False
        progress_path = self.progress_path_for_file(self.signal.path)
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress = {
            "version": 1,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "file_name": self.signal.path.name,
            "source_path": str(self.signal.path),
            "sample_rate": int(self.signal.sample_rate),
            "queue_index": int(self.queue_index),
            "current_page": int(page_index),
            "current_step": self.page_name(page_index),
            "settings": self.settings_to_json(self.settings),
            "noise_region_samples": self.region_to_json(self.noise_region),
            "analysis_region_samples": self.region_to_json(self.analysis_region or (self.result.analysis_region if self.result is not None else None)),
            "boundary_mode": self.boundary_mode,
            "boundary_clicks": [int(sample) for sample in self.boundary_clicks],
            "file_level_qc": self.review_qc_data,
            "review_qc_issue_key": self.qc_key_to_json(self.review_qc_issue_key),
            "review_qc_issue_clicks": [int(sample) for sample in self.review_qc_issue_clicks],
            "review_qc_issue_edit_index": self.review_qc_issue_edit_index,
            "review_zoom_region": self.region_to_json(self.review_zoom_region),
            "review_zoom_clicks": [int(sample) for sample in self.review_zoom_clicks],
            "review_zoom_selecting": bool(self.review_zoom_selecting),
            "audit_by_segment_number": self.audit_to_progress(),
            "audit_index": int(self.audit_index),
            "audit_issue_key": self.qc_key_to_json(self.audit_issue_key),
            "audit_issue_clicks": [int(sample) for sample in self.audit_issue_clicks],
            "audit_issue_edit_index": self.audit_issue_edit_index,
            "audit_zoom_region": self.region_to_json(self.audit_zoom_region),
            "audit_zoom_clicks": [int(sample) for sample in self.audit_zoom_clicks],
            "audit_zoom_selecting": bool(self.audit_zoom_selecting),
            "review_return_audit_index": self.review_return_audit_index,
            "result": self.result_to_progress(),
        }
        temp_path = progress_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(progress, handle, indent=2, sort_keys=True)
        temp_path.replace(progress_path)

    def page_name(self, page_index: int) -> str:
        return {
            PAGE_SETTINGS: "settings",
            PAGE_LOAD: "load",
            PAGE_BOUNDARIES: "boundaries",
            PAGE_REVIEW: "review",
            PAGE_AUDIT: "audit",
            PAGE_SAVE: "save",
        }.get(page_index, "unknown")

    def valid_progress_page(self, value: object) -> int:
        try:
            page_index = int(value)
        except (TypeError, ValueError):
            page_index = PAGE_BOUNDARIES if self.result is None else PAGE_REVIEW
        if page_index in (PAGE_SETTINGS, PAGE_LOAD):
            page_index = PAGE_BOUNDARIES if self.result is None else PAGE_REVIEW
        if self.result is None and page_index in (PAGE_REVIEW, PAGE_AUDIT, PAGE_SAVE):
            return PAGE_BOUNDARIES
        if page_index == PAGE_AUDIT and not self.audit_segments():
            return PAGE_SAVE
        return page_index if page_index in {PAGE_BOUNDARIES, PAGE_REVIEW, PAGE_AUDIT, PAGE_SAVE} else PAGE_BOUNDARIES

    def clear_stale_progress_for_file(self, path: Path) -> None:
        progress_path = self.progress_path_for_file(path)
        output_path = self.output_path_for_file(path)
        try:
            if progress_path.exists() and output_path.exists() and progress_path.stat().st_mtime <= output_path.stat().st_mtime:
                progress_path.unlink()
        except OSError:
            pass

    def settings_to_json(self, settings: SpaSettings) -> dict[str, object]:
        return {
            "speech_threshold_ms": float(settings.speech_threshold_ms),
            "pause_threshold_ms": float(settings.pause_threshold_ms),
            "sd_multiplier": float(settings.sd_multiplier),
            "threshold_mode": str(settings.threshold_mode),
            "adaptive": bool(settings.adaptive),
            "variable_mode": str(settings.variable_mode),
            "iterations": int(settings.iterations),
            "amplitude_increment": float(settings.amplitude_increment),
            "time_increment_ms": float(settings.time_increment_ms),
            "plot_graph": bool(settings.plot_graph),
            "use_full_file": bool(settings.use_full_file),
            "use_fixed_segments": bool(settings.use_fixed_segments),
            "fixed_segment_seconds": float(settings.fixed_segment_seconds),
            "passage_word_count": int(settings.passage_word_count),
        }

    def settings_from_json(self, value: object) -> SpaSettings:
        if not isinstance(value, dict):
            return self.settings
        base = SpaSettings()
        return SpaSettings(
            speech_threshold_ms=float(value.get("speech_threshold_ms", base.speech_threshold_ms)),
            pause_threshold_ms=float(value.get("pause_threshold_ms", base.pause_threshold_ms)),
            sd_multiplier=float(value.get("sd_multiplier", base.sd_multiplier)),
            threshold_mode=str(value.get("threshold_mode", base.threshold_mode)),
            adaptive=bool(value.get("adaptive", base.adaptive)),
            variable_mode=str(value.get("variable_mode", base.variable_mode)),
            iterations=int(value.get("iterations", base.iterations)),
            amplitude_increment=float(value.get("amplitude_increment", base.amplitude_increment)),
            time_increment_ms=float(value.get("time_increment_ms", base.time_increment_ms)),
            plot_graph=bool(value.get("plot_graph", base.plot_graph)),
            use_full_file=bool(value.get("use_full_file", base.use_full_file)),
            use_fixed_segments=bool(value.get("use_fixed_segments", base.use_fixed_segments)),
            fixed_segment_seconds=float(value.get("fixed_segment_seconds", base.fixed_segment_seconds)),
            passage_word_count=int(value.get("passage_word_count", base.passage_word_count)),
        )

    def result_to_progress(self) -> dict[str, object] | None:
        if self.result is None:
            return None
        return {
            "segmentation_mode": str(self.result.segmentation_mode),
            "settings": self.settings_to_json(self.result.settings),
            "analysis_region_samples": self.region_to_json(self.result.analysis_region),
            "noise_region_samples": self.region_to_json(self.result.noise_region),
            "speech_events_samples": self.events_to_json(self.result.speech_events_samples),
            "pause_events_samples": self.events_to_json(self.result.pause_events_samples),
            "fixed_events_samples": None if self.result.fixed_events_samples is None else self.events_to_json(self.result.fixed_events_samples),
        }

    def result_from_progress(self, value: object) -> SpaResult | None:
        if self.signal is None or not isinstance(value, dict):
            return None
        analysis_region = self.region_from_progress(value.get("analysis_region_samples")) or self.analysis_region
        if analysis_region is None:
            return None
        noise_region = self.region_from_progress(value.get("noise_region_samples")) or self.noise_region
        mode = str(value.get("segmentation_mode") or "spa")
        result_settings = self.settings_from_json(value.get("settings"))
        if mode == "fixed" or result_settings.use_fixed_segments:
            return self.result_from_progress_events(value, analysis_region, noise_region, replace(result_settings, use_fixed_segments=True), "fixed")
        if noise_region is not None:
            try:
                return run_spa(self.signal, settings=result_settings, noise_region=noise_region, analysis_region=analysis_region)
            except Exception:
                pass
        return self.result_from_progress_events(value, analysis_region, noise_region, result_settings, "spa")

    def result_from_progress_events(
        self,
        value: dict[str, object],
        analysis_region: tuple[int, int],
        noise_region: tuple[int, int] | None,
        settings: SpaSettings,
        mode: str,
    ) -> SpaResult:
        analysis_start, analysis_end = analysis_region
        fixed_value = value.get("fixed_events_samples")
        fixed_events = self.events_from_progress(fixed_value) if fixed_value is not None else None
        if mode == "fixed" and fixed_events is None:
            fixed_events = self.fixed_segment_events(analysis_end - analysis_start, self.signal.sample_rate if self.signal else 1)
        return SpaResult(
            settings=settings,
            signal_path=self.signal.path if self.signal is not None else Path(""),
            sample_rate=self.signal.sample_rate if self.signal is not None else 1,
            bit_depth=self.signal.bit_depth if self.signal is not None else 16,
            analysis_region=analysis_region,
            noise_region=noise_region,
            threshold_curve=np.full(max(1, analysis_end - analysis_start), np.nan),
            threshold_by_iteration={},
            speech_matrix=pd.DataFrame(),
            pause_matrix=pd.DataFrame(),
            total_matrix=pd.DataFrame(),
            speech_events_samples=self.events_from_progress(value.get("speech_events_samples")),
            pause_events_samples=self.events_from_progress(value.get("pause_events_samples")),
            fixed_events_samples=fixed_events,
            segmentation_mode=mode,
        )

    def events_to_json(self, events: np.ndarray | None) -> list[list[int]]:
        if events is None:
            return []
        return np.asarray(events, dtype=int).reshape(-1, 2).tolist()

    def events_from_progress(self, value: object) -> np.ndarray:
        if not isinstance(value, list) or not value:
            return np.empty((0, 2), dtype=int)
        rows: list[list[int]] = []
        for row in value:
            if isinstance(row, list) and len(row) >= 2:
                try:
                    rows.append([int(row[0]), int(row[1])])
                except (TypeError, ValueError):
                    continue
        return np.asarray(rows, dtype=int).reshape(-1, 2) if rows else np.empty((0, 2), dtype=int)

    def region_from_progress(self, value: object) -> tuple[int, int] | None:
        if self.signal is None or not isinstance(value, list) or len(value) != 2:
            return None
        try:
            return self.clean_region(int(value[0]), int(value[1]))
        except (TypeError, ValueError):
            return None

    def sample_list_from_progress(self, value: object) -> list[int]:
        if self.signal is None or not isinstance(value, list):
            return []
        samples = []
        for item in value:
            try:
                samples.append(max(0, min(len(self.signal.raw_audio), int(item))))
            except (TypeError, ValueError):
                continue
        return samples

    def optional_int_from_progress(self, value: object) -> int | None:
        try:
            return None if value is None else int(value)
        except (TypeError, ValueError):
            return None

    def qc_key_to_json(self, key: tuple[str, str] | None) -> list[str] | None:
        return [str(key[0]), str(key[1])] if key is not None else None

    def qc_key_from_progress(self, value: object) -> tuple[str, str] | None:
        if not isinstance(value, list) or len(value) != 2:
            return None
        return (str(value[0]), str(value[1]))

    def audit_to_progress(self) -> dict[str, object]:
        return {str(int(segment_number)): data for segment_number, data in self.audit_by_segment_number.items()}

    def audit_from_progress(self, value: object) -> dict[int, dict[str, object]]:
        if not isinstance(value, dict):
            return {}
        audits: dict[int, dict[str, object]] = {}
        for key, data in value.items():
            try:
                segment_number = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(data, dict):
                audits[segment_number] = data
        return audits

    def current_result_is_fixed(self) -> bool:
        return self.result is not None and self.result.segmentation_mode == "fixed"

    def prepare_fixed_segment_result(self) -> None:
        if self.signal is None:
            return
        total_samples = len(self.signal.raw_audio)
        analysis_region = (0, total_samples)
        self.signal.analysis_region = analysis_region
        self.noise_region = None
        self.analysis_region = analysis_region
        self.result = SpaResult(
            settings=replace(self.settings, use_fixed_segments=True),
            signal_path=self.signal.path,
            sample_rate=self.signal.sample_rate,
            bit_depth=self.signal.bit_depth,
            analysis_region=analysis_region,
            noise_region=None,
            threshold_curve=np.full(max(1, total_samples), np.nan),
            threshold_by_iteration={},
            speech_matrix=pd.DataFrame(),
            pause_matrix=pd.DataFrame(),
            total_matrix=pd.DataFrame(),
            speech_events_samples=np.empty((0, 2), dtype=int),
            pause_events_samples=np.empty((0, 2), dtype=int),
            fixed_events_samples=self.fixed_segment_events(total_samples, self.signal.sample_rate),
            segmentation_mode="fixed",
        )
        self.audit_by_segment_number = {}
        self.audit_index = 0
        self.review_return_audit_index = None
        self.clear_audit_zoom()
        self.clear_audit_issue_selection()
        self.clear_review_zoom()
        self.loaded_from_saved_output = False
        self.progress_dirty = False

    def fixed_segment_events(self, total_samples: int, sample_rate: int) -> np.ndarray:
        if total_samples <= 0:
            return np.empty((0, 2), dtype=int)
        duration_seconds = max(0.001, float(self.settings.fixed_segment_seconds))
        step = max(1, int(round(duration_seconds * sample_rate)))
        events = [[start, min(start + step, total_samples)] for start in range(0, total_samples, step)]
        return np.asarray(events, dtype=int).reshape(-1, 2)

    def load_saved_current_file(self) -> bool:
        path = self.queue[self.queue_index]
        output_path = self.output_path_for_file(path)
        try:
            signal = read_wav(path)
            saved = pd.read_csv(output_path)
            result, audit_by_segment_number = self.result_from_saved_segments(signal, saved)
            metadata = self.load_saved_metadata(path)
        except Exception as exc:
            self.warn("Saved output could not be opened", f"{output_path}\n\n{exc}")
            return False
        self.signal = signal
        self.result = result
        self.review_qc_data = self.saved_review_qc_data(metadata)
        self.noise_region = self.saved_noise_region(metadata, result)
        self.analysis_region = self.saved_analysis_region(metadata, result)
        self.apply_analysis_region_to_saved_result(self.result, self.analysis_region)
        self.result.noise_region = self.noise_region
        self.boundary_mode = "noise"
        self.boundary_clicks = []
        self.audit_by_segment_number = audit_by_segment_number
        self.audit_index = 0
        self.review_return_audit_index = None
        self.clear_audit_zoom()
        self.clear_audit_issue_selection()
        self.clear_review_qc_issue_selection()
        self.clear_review_zoom()
        self.apply_review_qc_to_segments()
        self.loaded_from_saved_output = True
        self.progress_dirty = False
        self.last_saved_path = output_path
        self.clear_stale_progress_for_file(path)
        self._show_page(PAGE_REVIEW)
        return True

    def load_saved_metadata(self, path: Path) -> dict[str, object]:
        metadata_path = self.metadata_path_for_file(path)
        if not metadata_path.exists():
            return {}
        with metadata_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}

    def saved_noise_region(self, metadata: dict[str, object], result: SpaResult) -> tuple[int, int] | None:
        if result.segmentation_mode == "fixed":
            return None
        region = self.region_from_metadata(metadata.get("noise_region_samples"))
        if region is not None:
            return region
        if len(result.pause_events_samples) > 0:
            start_rel, end_rel = result.pause_events_samples[0]
            analysis_start = result.analysis_region[0]
            return self.clean_region(analysis_start + int(start_rel), analysis_start + int(end_rel))
        return result.analysis_region

    def saved_analysis_region(self, metadata: dict[str, object], result: SpaResult) -> tuple[int, int]:
        region = self.region_from_metadata(metadata.get("analysis_region_samples"))
        return region if region is not None else result.analysis_region

    def apply_analysis_region_to_saved_result(self, result: SpaResult, analysis_region: tuple[int, int]) -> None:
        previous_start = result.analysis_region[0]
        next_start, next_end = analysis_region
        shift = previous_start - next_start
        if shift:
            if len(result.speech_events_samples) > 0:
                result.speech_events_samples = result.speech_events_samples + shift
            if len(result.pause_events_samples) > 0:
                result.pause_events_samples = result.pause_events_samples + shift
            if result.fixed_events_samples is not None and len(result.fixed_events_samples) > 0:
                result.fixed_events_samples = result.fixed_events_samples + shift
        result.analysis_region = analysis_region
        result.threshold_curve = np.full(max(1, next_end - next_start), np.nan)

    def region_from_metadata(self, value: object) -> tuple[int, int] | None:
        if self.signal is None or not isinstance(value, list) or len(value) != 2:
            return None
        try:
            return self.clean_region(int(value[0]), int(value[1]))
        except (TypeError, ValueError):
            return None

    def result_from_saved_segments(self, signal: SpaSignal, saved: pd.DataFrame) -> tuple[SpaResult, dict[int, dict[str, object]]]:
        required = {"segment_type", "onset_seconds_absolute", "offset_seconds_absolute"}
        missing = required - set(saved.columns)
        if missing:
            raise ValueError(f"Missing saved segment columns: {', '.join(sorted(missing))}")
        saved = saved.sort_values("onset_seconds_absolute")
        fs = signal.sample_rate
        speech_events: list[list[int]] = []
        pause_events: list[list[int]] = []
        fixed_events: list[list[int]] = []
        segment_audits: dict[int, dict[str, object]] = {}
        absolute_events: list[tuple[str, int, int, pd.Series]] = []
        for _, row in saved.iterrows():
            segment_type = str(row["segment_type"]).strip().lower()
            if segment_type not in {"speech", "pause", "fixed"}:
                continue
            start = int(round(float(row["onset_seconds_absolute"]) * fs))
            end = int(round(float(row["offset_seconds_absolute"]) * fs))
            start = max(0, min(len(signal.raw_audio), start))
            end = max(start + 1, min(len(signal.raw_audio), end))
            absolute_events.append((segment_type, start, end, row))
        if not absolute_events:
            raise ValueError("No segment rows were found in the saved CSV.")
        analysis_start = min(start for _, start, _, _row in absolute_events)
        analysis_end = max(end for _, _, end, _row in absolute_events)
        segment_number = 0
        for segment_type, start, end, row in absolute_events:
            segment_number += 1
            relative = [start - analysis_start, end - analysis_start]
            if segment_type == "speech":
                speech_events.append(relative)
            elif segment_type == "pause":
                pause_events.append(relative)
            else:
                fixed_events.append(relative)
            segment_audits[segment_number] = self.parse_saved_row_audit(row, fs)
        signal.analysis_region = (analysis_start, analysis_end)
        threshold_curve = np.full(max(1, analysis_end - analysis_start), np.nan)
        is_fixed_result = bool(fixed_events)
        result = SpaResult(
            settings=replace(self.settings, use_fixed_segments=is_fixed_result),
            signal_path=signal.path,
            sample_rate=signal.sample_rate,
            bit_depth=signal.bit_depth,
            analysis_region=(analysis_start, analysis_end),
            noise_region=None,
            threshold_curve=threshold_curve,
            threshold_by_iteration={},
            speech_matrix=pd.DataFrame(),
            pause_matrix=pd.DataFrame(),
            total_matrix=pd.DataFrame(),
            speech_events_samples=np.asarray(speech_events, dtype=int).reshape(-1, 2),
            pause_events_samples=np.asarray(pause_events, dtype=int).reshape(-1, 2),
            fixed_events_samples=np.asarray(fixed_events, dtype=int).reshape(-1, 2) if fixed_events else None,
            segmentation_mode="fixed" if is_fixed_result else "spa",
        )
        return result, segment_audits

    def parse_saved_row_audit(self, row: pd.Series, sample_rate: int) -> dict[str, object]:
        if "audit_json" in row.index:
            audit = self.parse_saved_audit_json(row.get("audit_json", ""))
            if audit.get("selected_effects"):
                return audit

        selected_effects = []
        for gui_name, effects in QC_AUDIT_GROUPS:
            if gui_name not in row.index:
                continue
            data = self.parse_saved_audit_group(row.get(gui_name))
            for effect in effects:
                regions = self.issue_region_jsons_from_time_value(data.get(effect, []), sample_rate)
                if not regions:
                    continue
                item = {"gui_name": gui_name, "effect": effect}
                item.update(self.qc_metadata_for_effect(gui_name, effect))
                item["visible"] = True
                item["issue_regions"] = regions
                selected_effects.append(item)
        return {"selected_effects": selected_effects}

    def parse_saved_audit_group(self, value: object) -> dict[str, object]:
        if not isinstance(value, str) or not value.strip():
            return {}
        try:
            data = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def issue_region_jsons_from_time_value(self, value: object, sample_rate: int) -> list[dict[str, float | int]]:
        if not isinstance(value, list) or not value:
            return []
        if len(value) >= 2 and all(isinstance(item, (int, float)) for item in value[:2]):
            region = self.issue_region_json_from_seconds(value[0], value[1], sample_rate)
            return [region] if region else []
        regions = []
        for time_span in value:
            if not isinstance(time_span, list) or len(time_span) < 2:
                continue
            region = self.issue_region_json_from_seconds(time_span[0], time_span[1], sample_rate)
            if region:
                regions.append(region)
        return regions

    def issue_region_json_from_seconds(self, start_seconds: object, end_seconds: object, sample_rate: int) -> dict[str, float | int]:
        try:
            start_seconds_float = float(start_seconds)
            end_seconds_float = float(end_seconds)
        except (TypeError, ValueError):
            return {}
        if end_seconds_float <= start_seconds_float:
            return {}
        start = int(round(start_seconds_float * sample_rate))
        end = int(round(end_seconds_float * sample_rate))
        if end <= start:
            return {}
        return {
            "onset_sample_absolute": int(start),
            "offset_sample_absolute": int(end),
            "onset_seconds_absolute": float(start / sample_rate),
            "offset_seconds_absolute": float(end / sample_rate),
            "duration_seconds": float((end - start) / sample_rate),
        }

    def parse_saved_audit_json(self, audit_json: object) -> dict[str, object]:
        if not isinstance(audit_json, str) or audit_json.strip().upper() == "NA" or not audit_json.strip():
            return {"selected_effects": []}
        try:
            data = json.loads(audit_json)
        except json.JSONDecodeError:
            return {"selected_effects": []}
        if not isinstance(data, dict):
            return {"selected_effects": []}
        data.setdefault("selected_effects", [])
        return data

    def prepare_boundary_page(self) -> None:
        if self.signal is None:
            self._show_page(PAGE_LOAD)
            return
        duration = len(self.signal.raw_audio) / self.signal.sample_rate
        for slider, spin in self.boundary_widgets.values():
            slider.setRange(0, len(self.signal.raw_audio))
            spin.setRange(0.0, duration)
        self.analysis_select_button.setEnabled(not self.settings.use_full_file)
        if self.settings.use_full_file:
            self.analysis_region = (0, len(self.signal.raw_audio))
        if self.noise_region is None:
            self.boundary_mode = "noise"
        elif not self.settings.use_full_file and self.analysis_region is None:
            self.boundary_mode = "analysis"
        self.boundary_run_button.setText("Next Page" if self.loaded_from_saved_output and self.result is not None else "Confirm And Run SPA")
        self.update_boundary_controls()
        self.plot_boundaries()
        self.set_playback_range(0, len(self.signal.raw_audio), "Full Audio", reset_position=True)

    def set_boundary_mode(self, mode: str) -> None:
        if mode == "analysis" and self.settings.use_full_file:
            return
        self.boundary_mode = mode
        self.boundary_clicks = []
        self.update_boundary_controls()
        self.plot_boundaries()
        self.autosave_progress(PAGE_BOUNDARIES, force=True)

    def boundary_next_or_run_spa(self) -> None:
        if self.loaded_from_saved_output and self.result is not None:
            self._show_page(PAGE_REVIEW)
            return
        if self.settings.use_fixed_segments:
            self.prepare_fixed_segment_result()
            self._show_page(PAGE_REVIEW)
            return
        self.confirm_boundaries_and_run()

    def on_boundary_click(self, event) -> None:
        if self.signal is None or event.inaxes not in (self.boundary_ax_norm, self.boundary_ax_raw) or event.xdata is None:
            return
        sample = self.seconds_to_sample(float(event.xdata))
        self.boundary_clicks.append(sample)
        if len(self.boundary_clicks) < 2:
            self.update_boundary_controls()
            self.plot_boundaries()
            self.autosave_progress(PAGE_BOUNDARIES, force=True)
            return
        start, end = sorted(self.boundary_clicks[:2])
        if end <= start:
            self.boundary_clicks = []
            self.warn("Invalid region", "The selected region has no duration.")
            return
        if self.boundary_mode == "noise":
            self.noise_region = (start, end)
            if not self.settings.use_full_file:
                self.boundary_mode = "analysis"
        else:
            self.analysis_region = (start, end)
        self.boundary_clicks = []
        self.mark_progress_dirty()
        self.result = None
        self.audit_by_segment_number = {}
        self.update_boundary_controls()
        self.plot_boundaries()
        self.autosave_progress(PAGE_BOUNDARIES, force=True)

    def on_boundary_motion(self, event) -> None:
        if event.inaxes not in (self.boundary_ax_norm, self.boundary_ax_raw) or event.xdata is None or event.ydata is None:
            self.boundary_crosshair.clear()
            return
        gui_event = getattr(event, "guiEvent", None)
        if gui_event is not None and hasattr(gui_event, "position"):
            position = gui_event.position()
            self.boundary_crosshair.set_position(position.x(), position.y())
        elif gui_event is not None and hasattr(gui_event, "pos"):
            position = gui_event.pos()
            self.boundary_crosshair.set_position(position.x(), position.y())
        else:
            self.boundary_crosshair.set_position(event.x, self.boundary_canvas.height() - event.y)

    def on_boundary_leave(self, event) -> None:
        self.boundary_crosshair.clear()

    def on_boundary_slider_changed(self, key: str, value: int) -> None:
        if self._updating_boundaries:
            return
        self.set_boundary_value(key, int(value))

    def on_boundary_spin_changed(self, key: str, value: float) -> None:
        if self._updating_boundaries:
            return
        self.set_boundary_value(key, self.seconds_to_sample(float(value)))

    def set_boundary_value(self, key: str, sample: int) -> None:
        if self.signal is None:
            return
        sample = max(0, min(len(self.signal.raw_audio), sample))
        if key.startswith("noise"):
            start, end = self.noise_region or (0, min(len(self.signal.raw_audio), self.signal.sample_rate // 2))
            start, end = (sample, end) if key.endswith("start") else (start, sample)
            self.noise_region = self.clean_region(start, end)
        else:
            start, end = self.analysis_region or (0, len(self.signal.raw_audio))
            start, end = (sample, end) if key.endswith("start") else (start, sample)
            self.analysis_region = self.clean_region(start, end)
        self.mark_progress_dirty()
        self.result = None
        self.audit_by_segment_number = {}
        self.update_boundary_controls()
        self.plot_boundaries()
        self.autosave_progress(PAGE_BOUNDARIES, force=True)

    def clean_region(self, start: int, end: int) -> tuple[int, int]:
        if self.signal is None:
            return (0, 0)
        start, end = sorted((start, end))
        end = max(start + 1, end)
        return max(0, start), min(len(self.signal.raw_audio), end)

    def update_boundary_controls(self) -> None:
        if self.signal is None:
            return
        self._updating_boundaries = True
        regions = {
            "noise": self.noise_region,
            "analysis": self.analysis_region,
        }
        for name, region_key, edge in (
            ("noise_start", "noise", 0),
            ("noise_end", "noise", 1),
            ("analysis_start", "analysis", 0),
            ("analysis_end", "analysis", 1),
        ):
            slider, spin = self.boundary_widgets[name]
            region = regions[region_key]
            enabled = region is not None and (region_key != "analysis" or not self.settings.use_full_file)
            if region_key == "analysis" and self.settings.use_full_file:
                enabled = False
            slider.setEnabled(enabled)
            spin.setEnabled(enabled)
            value = region[edge] if region is not None else 0
            slider.setValue(value)
            spin.setValue(value / self.signal.sample_rate)
        self._updating_boundaries = False
        if self.boundary_mode == "noise":
            self.boundary_status.setText("Click the start and end of a noise-only pause, then adjust with the sliders.")
        else:
            self.boundary_status.setText("Click the start and end of the analysis region, then adjust with the sliders.")
        if self.settings.use_full_file:
            self.boundary_hint.setText("Analysis region: full file")
        else:
            self.boundary_hint.setText("Noise region and analysis region are both required.")

    def plot_boundaries(self) -> None:
        if self.signal is None:
            return
        fs = self.signal.sample_rate
        x = np.arange(len(self.signal.raw_audio)) / fs
        display = self.display_indices(len(x))
        self.boundary_ax_norm.clear()
        self.boundary_ax_raw.clear()
        self.boundary_ax_norm.plot(
            x[display],
            self.signal.normalized_envelope[display],
            color="#2f80ed",
            linewidth=0.75,
            label="normalized",
        )
        self.boundary_ax_raw.plot(x[display], self.signal.raw_audio[display], color="#4a4f53", linewidth=0.55, label="original")
        self.draw_region(self.boundary_ax_norm, self.boundary_ax_raw, self.noise_region, "#2f80ed", "noise")
        self.draw_region(self.boundary_ax_norm, self.boundary_ax_raw, self.analysis_region, "#1f9d55", "analysis")
        for click in self.boundary_clicks:
            seconds = click / fs
            for ax in (self.boundary_ax_norm, self.boundary_ax_raw):
                ax.axvline(seconds, color="#f28e2b", linewidth=1.1)
        self.boundary_ax_norm.set_title(self.signal.path.name)
        self.boundary_ax_norm.set_ylabel("Normalized")
        self.boundary_ax_raw.set_ylabel("Original")
        self.boundary_ax_raw.set_xlabel("Time (seconds)")
        self.style_axes(self.boundary_ax_norm, self.boundary_ax_raw)
        self.boundary_crosshair.clear()
        self.boundary_canvas.draw_idle()
        self.update_playback_markers()

    def confirm_boundaries_and_run(self) -> None:
        if self.signal is None:
            return
        if self.noise_region is None:
            self.warn("Missing noise region", "Select a noise-only region before running SPA.")
            return
        if self.settings.use_full_file:
            analysis_region = (0, len(self.signal.raw_audio))
        else:
            if self.analysis_region is None:
                self.warn("Missing analysis region", "Select the analysis region before running SPA.")
                return
            analysis_region = self.analysis_region
        try:
            self.result = run_spa(
                self.signal,
                settings=self.settings,
                noise_region=self.noise_region,
                analysis_region=analysis_region,
            )
        except Exception as exc:
            self.warn("SPA failed", str(exc))
            return
        self.audit_by_segment_number = {}
        self.audit_index = 0
        self.review_return_audit_index = None
        self.clear_audit_zoom()
        self.clear_audit_issue_selection()
        self.clear_review_qc_issue_selection()
        self.clear_review_zoom()
        self.mark_progress_dirty()
        self.loaded_from_saved_output = False
        self._show_page(PAGE_REVIEW)

    def plot_review(self) -> None:
        if self.signal is None or self.result is None:
            return
        fs = self.signal.sample_rate
        analysis_start, analysis_end = self.result.analysis_region
        display_end = min(analysis_end, analysis_start + len(self.result.threshold_curve))
        envelope = self.signal.normalized_envelope[analysis_start:display_end]
        raw = self.signal.raw_audio[analysis_start:display_end]
        x = (analysis_start + np.arange(len(envelope))) / fs
        display = self.display_indices(len(x))
        threshold = self.result.threshold_curve[: len(envelope)]
        self.review_ax_norm.clear()
        self.review_ax_raw.clear()
        self.sync_review_qc_checks_from_data()
        self.review_ax_norm.plot(x[display], envelope[display], color="#2f80ed", linewidth=0.75, label="normalized")
        fixed_events = self.result.fixed_events_samples
        is_fixed = fixed_events is not None and len(fixed_events) > 0
        if not is_fixed:
            self.review_ax_norm.plot(x[display], threshold[display], color="#c43c39", linewidth=1.0, label="threshold")
        self.review_ax_raw.plot(x[display], raw[display], color="#4a4f53", linewidth=0.55, label="original")
        if is_fixed:
            boundaries = sorted(
                {
                    sample
                    for start, end in np.asarray(fixed_events, dtype=int)
                    for sample in (analysis_start + int(start), analysis_start + int(end))
                }
            )
            for idx, sample in enumerate(boundaries):
                for ax in (self.review_ax_norm, self.review_ax_raw):
                    ax.axvline(
                        sample / fs,
                        color="#1f2937",
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.68,
                        label="5-second boundary" if idx == 0 else None,
                    )
        else:
            for idx, (start, end) in enumerate(self.result.speech_events_samples):
                self.review_ax_raw.axvspan((analysis_start + start) / fs, (analysis_start + end) / fs, color="#1f9d55", alpha=0.24, label="speech" if idx == 0 else None)
            for idx, (start, end) in enumerate(self.result.pause_events_samples):
                self.review_ax_raw.axvspan((analysis_start + start) / fs, (analysis_start + end) / fs, color="#2f80ed", alpha=0.25, label="pause" if idx == 0 else None)
        self.draw_review_qc_regions()
        for click in self.review_zoom_clicks:
            for ax in (self.review_ax_norm, self.review_ax_raw):
                ax.axvline(click / fs, color="#f28e2b", linewidth=1.2)
        if self.review_zoom_region is not None:
            zoom_start, zoom_end = self.review_zoom_region
            for ax in (self.review_ax_norm, self.review_ax_raw):
                ax.axvspan(zoom_start / fs, zoom_end / fs, color="#2f80ed", alpha=0.10)
        self.review_ax_norm.set_title("Normalized signal with segment boundaries" if is_fixed else "Normalized signal and threshold")
        self.review_ax_raw.set_title("Original waveform with fixed 5-second segments" if is_fixed else "Original waveform with detected segments")
        self.review_ax_norm.set_ylabel("Normalized")
        self.review_ax_raw.set_ylabel("Original")
        self.review_ax_raw.set_xlabel("Time (seconds)")
        if self.review_zoom_region is not None:
            zoom_start, zoom_end = self.review_zoom_region
            self.review_ax_norm.set_xlim(zoom_start / fs, zoom_end / fs)
            self.review_ax_raw.set_xlim(zoom_start / fs, zoom_end / fs)
        elif len(x) > 0:
            self.review_ax_norm.set_xlim(x[0], x[-1])
            self.review_ax_raw.set_xlim(x[0], x[-1])
        self.style_axes(self.review_ax_norm, self.review_ax_raw)
        self.add_legend(self.review_ax_norm)
        self.add_legend(self.review_ax_raw)
        if is_fixed:
            self.review_counts.setText(
                f"{len(fixed_events)} fixed 5-second segments  |  SPA skipped  |  File {analysis_start / fs:.3f}s to {analysis_end / fs:.3f}s"
            )
        else:
            speech_count = len(self.result.speech_events_samples)
            pause_count = len(self.result.pause_events_samples)
            self.review_counts.setText(
                f"{speech_count} speech segments  |  {pause_count} pauses  |  Analysis {analysis_start / fs:.3f}s to {analysis_end / fs:.3f}s"
            )
        self.review_next_button.setText("Next Page" if self.loaded_from_saved_output else "Confirm Segmentation")
        self.update_review_segment_controls()
        self.update_review_qc_controls()
        self.update_review_zoom_controls()
        if self.review_zoom_region is not None:
            zoom_start, zoom_end = self.review_zoom_region
            self.set_playback_range(zoom_start, zoom_end, "Zoomed Review", reset_position=True)
        else:
            self.set_playback_range(analysis_start, analysis_end, "Analysis Region", reset_position=True)
        self.review_canvas.draw_idle()
        self.update_playback_markers()

    def clear_review_qc_issue_selection(self) -> None:
        self.review_qc_issue_key = None
        self.review_qc_issue_clicks = []
        self.review_qc_issue_edit_index = None

    def clear_review_zoom(self) -> None:
        self.review_zoom_region = None
        self.review_zoom_clicks = []
        self.review_zoom_selecting = False

    def update_review_zoom_controls(self) -> None:
        if not hasattr(self, "review_zoom_select_button"):
            return
        has_zoom = self.review_zoom_region is not None
        self.review_zoom_play_button.setEnabled(has_zoom)
        self.review_zoom_reset_button.setEnabled(has_zoom or bool(self.review_zoom_clicks) or self.review_zoom_selecting)
        if self.review_zoom_selecting:
            self.review_zoom_select_button.setText("Selecting Zoom")
            if len(self.review_zoom_clicks) == 0:
                self.review_zoom_status.setText("Click the zoom start time on either review plot.")
            elif self.signal is not None:
                seconds = self.review_zoom_clicks[0] / self.signal.sample_rate
                self.review_zoom_status.setText(f"Zoom start set at {seconds:.3f}s. Click the zoom end time.")
        elif has_zoom and self.signal is not None:
            start, end = self.review_zoom_region
            self.review_zoom_select_button.setText("Select Zoom")
            self.review_zoom_status.setText(f"Zoom {start / self.signal.sample_rate:.3f}s to {end / self.signal.sample_rate:.3f}s. Play Zoom or Reset Zoom.")
        else:
            self.review_zoom_select_button.setText("Select Zoom")
            self.review_zoom_status.setText("Optional: select a zoom window to inspect and play a smaller part of the full segmentation.")

    def start_review_zoom_selection(self) -> None:
        if self.signal is None or self.result is None:
            return
        self.stop_playback()
        self.review_zoom_region = None
        self.review_zoom_clicks = []
        self.review_zoom_selecting = True
        self.update_review_zoom_controls()
        self.plot_review()

    def reset_review_zoom(self) -> None:
        self.stop_playback()
        self.clear_review_zoom()
        self.plot_review()

    def play_current_review_zoom(self, button: QPushButton | None = None) -> None:
        if self.signal is None or self.result is None or self.review_zoom_region is None:
            return
        start, end = self.review_zoom_region
        self.toggle_playback(start, end, "Zoomed Review", button)

    def selected_review_qc_keys(self) -> list[tuple[str, str]]:
        return [(gui_name, check.text()) for gui_name, check, _visible in self.review_qc_checks if check.isChecked()]

    def visible_review_qc_keys(self) -> set[tuple[str, str]]:
        return {
            (gui_name, check.text())
            for gui_name, check, visible in self.review_qc_checks
            if check.isChecked() and visible.isChecked()
        }

    def current_review_qc_effect_map(self) -> dict[tuple[str, str], dict[str, object]]:
        data = self.review_qc_data if isinstance(self.review_qc_data, dict) else {"selected_effects": []}
        effects: dict[tuple[str, str], dict[str, object]] = {}
        for item in data.get("selected_effects", []):
            if not isinstance(item, dict):
                continue
            gui_name = str(item.get("gui_name", ""))
            effect = str(item.get("effect", ""))
            if gui_name and effect:
                effects[(gui_name, effect)] = dict(item)
        return effects

    def sync_review_qc_checks_from_data(self) -> None:
        if not hasattr(self, "review_qc_checks"):
            return
        selected = set(self.current_review_qc_effect_map())
        self._updating_review_qc_checks = True
        effect_map = self.current_review_qc_effect_map()
        for gui_name, check, visible in self.review_qc_checks:
            key = (gui_name, check.text())
            is_selected = key in selected
            check.setChecked(is_selected)
            visible.setEnabled(is_selected)
            visible.setChecked(is_selected and bool(effect_map.get(key, {}).get("visible", True)))
        self._updating_review_qc_checks = False

    def set_review_qc_effects(self, effects: list[dict[str, object]]) -> None:
        self.review_qc_data = {"selected_effects": effects}
        self.mark_progress_dirty()

    def ensure_review_qc_item(self, key: tuple[str, str]) -> dict[str, object]:
        effects = [item for item in self.review_qc_data.get("selected_effects", []) if isinstance(item, dict)]
        for item in effects:
            if (str(item.get("gui_name", "")), str(item.get("effect", ""))) == key:
                return item
        gui_name, effect = key
        item: dict[str, object] = {"gui_name": gui_name, "effect": effect, "visible": True}
        item.update(self.qc_metadata_for_effect(gui_name, effect))
        effects.append(item)
        self.set_review_qc_effects(effects)
        return item

    def remove_review_qc_item(self, key: tuple[str, str]) -> None:
        effects = [
            item
            for item in self.review_qc_data.get("selected_effects", [])
            if isinstance(item, dict) and (str(item.get("gui_name", "")), str(item.get("effect", ""))) != key
        ]
        self.set_review_qc_effects(effects)

    def review_qc_window_index(self) -> int | None:
        if not hasattr(self, "review_qc_window_list"):
            return None
        item = self.review_qc_window_list.currentItem()
        if item is None:
            return None
        value = item.data(Qt.UserRole)
        return int(value) if isinstance(value, int) else None

    def update_review_qc_combo(self, preferred_key: tuple[str, str] | None = None) -> None:
        if not hasattr(self, "review_qc_combo"):
            return
        current = preferred_key or self.review_qc_combo.currentData()
        selected_keys = self.selected_review_qc_keys()
        self.review_qc_combo.blockSignals(True)
        self.review_qc_combo.clear()
        for gui_name, effect in selected_keys:
            self.review_qc_combo.addItem(f"{gui_name}: {effect}", (gui_name, effect))
        if current in selected_keys:
            self.review_qc_combo.setCurrentIndex(selected_keys.index(current))
        self.review_qc_combo.blockSignals(False)
        self.review_qc_combo.setEnabled(bool(selected_keys))

    def update_review_qc_window_list(self, preferred_index: int | None = None) -> None:
        if not hasattr(self, "review_qc_window_list"):
            return
        current_index = preferred_index if preferred_index is not None else self.review_qc_window_index()
        selected_key = self.review_qc_combo.currentData()
        regions: list[tuple[int, int]] = []
        if selected_key in self.selected_review_qc_keys():
            regions = self.raw_issue_regions_from_entry(self.current_review_qc_effect_map().get(selected_key))
        self.review_qc_window_list.blockSignals(True)
        self.review_qc_window_list.clear()
        for index, (start, end) in enumerate(regions):
            label = f"Window {index + 1}: {start / self.signal.sample_rate:.3f}s to {end / self.signal.sample_rate:.3f}s" if self.signal is not None else f"Window {index + 1}"
            self.review_qc_window_list.addItem(label)
            item = self.review_qc_window_list.item(self.review_qc_window_list.count() - 1)
            item.setData(Qt.UserRole, index)
        if regions:
            next_index = current_index if current_index is not None and 0 <= current_index < len(regions) else len(regions) - 1
            self.review_qc_window_list.setCurrentRow(next_index)
        self.review_qc_window_list.blockSignals(False)

    def update_review_qc_controls(self, preferred_key: tuple[str, str] | None = None) -> None:
        if not hasattr(self, "review_qc_status"):
            return
        self.update_review_qc_combo(preferred_key or self.review_qc_issue_key)
        selected_key = self.review_qc_combo.currentData()
        selected_keys = self.selected_review_qc_keys()
        effect_map = self.current_review_qc_effect_map()
        self.update_review_qc_window_list(self.review_qc_issue_edit_index)
        selected_window_index = self.review_qc_window_index()
        has_selected_key = selected_key in selected_keys
        self.review_qc_whole_file_button.setEnabled(has_selected_key)
        self.review_qc_add_button.setEnabled(has_selected_key)
        self.review_qc_play_button.setEnabled(self.current_review_qc_play_range() is not None)
        self.review_qc_confirm_button.setEnabled(self.review_qc_issue_key is not None and len(self.review_qc_issue_clicks) >= 2)
        self.review_qc_edit_button.setEnabled(has_selected_key and selected_window_index is not None)
        self.review_qc_remove_button.setEnabled(has_selected_key and selected_window_index is not None)
        if self.review_qc_issue_key is not None:
            gui_name, effect = self.review_qc_issue_key
            if len(self.review_qc_issue_clicks) == 0:
                self.review_qc_status.setText(f"{gui_name}: {effect} needs a file-level window. Click start/end on either review plot, or choose Whole File.")
            elif len(self.review_qc_issue_clicks) == 1 and self.signal is not None:
                seconds = self.review_qc_issue_clicks[0] / self.signal.sample_rate
                self.review_qc_status.setText(f"{gui_name}: {effect} start set at {seconds:.3f}s. Click the issue end time.")
            else:
                action = "replace the selected window" if self.review_qc_issue_edit_index is not None else "add it"
                self.review_qc_status.setText(f"{gui_name}: {effect} window selected. Click Confirm to {action}, or Add/Reset to start over.")
            return
        if has_selected_key:
            entry = effect_map.get(selected_key)
            regions = self.raw_issue_regions_from_entry(entry)
            gui_name, effect = selected_key
            if regions and self.signal is not None:
                start, end = regions[-1]
                self.review_qc_status.setText(f"{gui_name}: {effect} has {len(regions)} file-level window(s). Latest {start / self.signal.sample_rate:.3f}s to {end / self.signal.sample_rate:.3f}s.")
            else:
                self.review_qc_status.setText(f"{gui_name}: {effect} needs Whole File or at least one window.")
        else:
            self.review_qc_status.setText("Check a QC artifact to mark the whole file or a large region.")

    def current_review_qc_play_range(self) -> tuple[int, int, str] | None:
        if self.signal is None:
            return None
        if self.review_qc_issue_key is not None and len(self.review_qc_issue_clicks) >= 2:
            start, end = sorted(self.review_qc_issue_clicks[:2])
            if end > start:
                _, effect = self.review_qc_issue_key
                return start, end, f"File-Level QC: {effect}"
        selected_key = self.review_qc_combo.currentData() if hasattr(self, "review_qc_combo") else None
        if selected_key in self.selected_review_qc_keys():
            regions = self.raw_issue_regions_from_entry(self.current_review_qc_effect_map().get(selected_key))
            if regions:
                _, effect = selected_key
                selected_index = self.review_qc_window_index()
                if selected_index is None or selected_index >= len(regions):
                    selected_index = len(regions) - 1
                start, end = regions[selected_index]
                return start, end, f"File-Level QC: {effect} window {selected_index + 1}"
        return None

    def play_selected_review_qc_boundary(self, button: QPushButton | None = None) -> None:
        play_range = self.current_review_qc_play_range()
        if play_range is None:
            self.warn("No window selected", "Select a saved file-level window, click both boundary points, or choose Whole File first.")
            return
        start, end, label = play_range
        self.toggle_playback(start, end, label, button)

    def on_review_qc_check_toggled(self, gui_name: str, check: QCheckBox, checked: bool) -> None:
        if self._updating_review_qc_checks:
            return
        key = (gui_name, check.text())
        visible = self.review_qc_visibility_check_for_key(key)
        if visible is not None:
            visible.setEnabled(checked)
            visible.setChecked(checked)
        if checked:
            item = self.ensure_review_qc_item(key)
            item["visible"] = True
            self.clear_review_qc_issue_selection()
            self.update_review_qc_controls(key)
            self.plot_review()
            self.autosave_progress(PAGE_REVIEW, force=True)
            return
        if self.review_qc_issue_key == key:
            self.clear_review_qc_issue_selection()
        self.remove_review_qc_item(key)
        self.update_review_qc_controls()
        self.plot_review()
        self.autosave_progress(PAGE_REVIEW, force=True)

    def on_review_qc_visibility_toggled(self, gui_name: str, check: QCheckBox, checked: bool) -> None:
        if self._updating_review_qc_checks:
            return
        if not check.isChecked():
            return
        self.set_review_qc_visibility((gui_name, check.text()), checked)
        self.plot_review()
        self.autosave_progress(PAGE_REVIEW, force=True)

    def review_qc_visibility_check_for_key(self, key: tuple[str, str]) -> QCheckBox | None:
        for gui_name, check, visible in self.review_qc_checks:
            if (gui_name, check.text()) == key:
                return visible
        return None

    def set_review_qc_visibility(self, key: tuple[str, str], visible_state: bool) -> None:
        item = self.ensure_review_qc_item(key)
        item["visible"] = bool(visible_state)

    def set_all_review_qc_visibility(self, visible_state: bool) -> None:
        if self.result is None:
            return
        self._updating_review_qc_checks = True
        for gui_name, check, visible in self.review_qc_checks:
            if check.isChecked():
                visible.setChecked(visible_state)
                self.set_review_qc_visibility((gui_name, check.text()), visible_state)
        self._updating_review_qc_checks = False
        self.update_review_qc_controls()
        self.plot_review()
        self.autosave_progress(PAGE_REVIEW, force=True)

    def on_review_qc_combo_changed(self) -> None:
        if self.review_qc_issue_key is None:
            self.update_review_qc_controls()
            self.plot_review()

    def on_review_qc_window_selected(self) -> None:
        if self.review_qc_issue_key is None:
            self.update_review_qc_controls()
            self.plot_review()

    def edit_selected_review_qc_boundary(self) -> None:
        key = self.review_qc_combo.currentData()
        if key not in self.selected_review_qc_keys():
            return
        self.start_review_qc_boundary_selection(key)

    def start_review_qc_boundary_selection(self, key: tuple[str, str]) -> None:
        self.stop_playback()
        self.ensure_review_qc_item(key)
        self.review_qc_issue_key = key
        self.review_qc_issue_clicks = []
        self.review_qc_issue_edit_index = None
        self.update_review_qc_controls(key)
        self.plot_review()
        self.autosave_progress(PAGE_REVIEW, force=True)

    def edit_selected_review_qc_window(self) -> None:
        key = self.review_qc_combo.currentData()
        selected_index = self.review_qc_window_index()
        if key not in self.selected_review_qc_keys() or selected_index is None:
            return
        regions = self.raw_issue_regions_from_entry(self.current_review_qc_effect_map().get(key))
        if selected_index >= len(regions):
            return
        self.stop_playback()
        self.review_qc_issue_key = key
        self.review_qc_issue_edit_index = selected_index
        self.review_qc_issue_clicks = [regions[selected_index][0], regions[selected_index][1]]
        self.update_review_qc_controls(key)
        self.plot_review()
        self.autosave_progress(PAGE_REVIEW, force=True)

    def remove_selected_review_qc_window(self) -> None:
        key = self.review_qc_combo.currentData()
        selected_index = self.review_qc_window_index()
        if key not in self.selected_review_qc_keys() or selected_index is None:
            return
        effects = [item for item in self.review_qc_data.get("selected_effects", []) if isinstance(item, dict)]
        for item in effects:
            if (str(item.get("gui_name", "")), str(item.get("effect", ""))) != key:
                continue
            regions = self.raw_issue_region_jsons_from_entry(item)
            if 0 <= selected_index < len(regions):
                del regions[selected_index]
            item["issue_regions"] = regions
            item["whole_file"] = self.region_list_covers_full_file(regions)
            item.pop("issue_region", None)
            break
        self.set_review_qc_effects(effects)
        if self.review_qc_issue_key == key:
            self.clear_review_qc_issue_selection()
        self.update_review_qc_controls(key)
        self.update_review_qc_window_list(max(0, selected_index - 1))
        self.plot_review()
        self.autosave_progress(PAGE_REVIEW, force=True)

    def mark_review_qc_whole_file(self) -> None:
        if self.signal is None:
            return
        key = self.review_qc_combo.currentData()
        if key not in self.selected_review_qc_keys():
            return
        item = self.ensure_review_qc_item(key)
        item["issue_regions"] = [self.issue_region_json_with_source(0, len(self.signal.raw_audio), "file_level")]
        item["whole_file"] = True
        self.clear_review_qc_issue_selection()
        self.update_review_qc_controls(key)
        self.plot_review()
        self.autosave_progress(PAGE_REVIEW, force=True)

    def confirm_review_qc_boundary(self) -> None:
        if self.signal is None or self.review_qc_issue_key is None or len(self.review_qc_issue_clicks) < 2:
            return
        start, end = sorted(self.review_qc_issue_clicks[:2])
        if end <= start:
            self.review_qc_issue_clicks = []
            self.warn("Invalid boundary", "The selected file-level QC boundary has no duration.")
            self.update_review_qc_controls()
            self.plot_review()
            return
        confirmed_key = self.review_qc_issue_key
        confirmed_index = self.set_review_qc_region_for_key(confirmed_key, (start, end))
        self.clear_review_qc_issue_selection()
        self.update_review_qc_controls(confirmed_key)
        self.update_review_qc_window_list(confirmed_index)
        self.plot_review()
        self.autosave_progress(PAGE_REVIEW, force=True)

    def set_review_qc_region_for_key(self, key: tuple[str, str], region: tuple[int, int]) -> int:
        item = self.ensure_review_qc_item(key)
        regions = self.raw_issue_region_jsons_from_entry(item)
        region_json = self.issue_region_json_with_source(region[0], region[1], "file_level")
        if self.review_qc_issue_edit_index is not None and 0 <= self.review_qc_issue_edit_index < len(regions):
            target_index = self.review_qc_issue_edit_index
            regions[target_index] = region_json
        else:
            target_index = len(regions)
            regions.append(region_json)
        item["issue_regions"] = regions
        item["whole_file"] = self.region_list_covers_full_file(regions)
        item.pop("issue_region", None)
        return target_index

    def on_review_plot_click(self, event) -> None:
        if self.signal is None:
            return
        if event.inaxes not in (self.review_ax_norm, self.review_ax_raw) or event.xdata is None:
            return
        if self.review_zoom_selecting:
            self.handle_review_zoom_click(float(event.xdata))
            return
        if self.review_qc_issue_key is None:
            return
        sample = self.seconds_to_sample(float(event.xdata))
        sample = max(0, min(len(self.signal.raw_audio), sample))
        if len(self.review_qc_issue_clicks) >= 2:
            self.review_qc_issue_clicks = []
        self.review_qc_issue_clicks.append(sample)
        self.update_review_qc_controls(self.review_qc_issue_key)
        self.plot_review()
        self.autosave_progress(PAGE_REVIEW, force=True)

    def handle_review_zoom_click(self, seconds: float) -> None:
        if self.signal is None or self.result is None:
            return
        analysis_start, analysis_end = self.result.analysis_region
        sample = self.seconds_to_sample(seconds)
        sample = max(analysis_start, min(analysis_end, sample))
        self.review_zoom_clicks.append(sample)
        if len(self.review_zoom_clicks) < 2:
            self.update_review_zoom_controls()
            self.plot_review()
            return
        zoom_start, zoom_end = sorted(self.review_zoom_clicks[:2])
        if zoom_end <= zoom_start:
            self.review_zoom_clicks = []
            self.warn("Invalid zoom", "The selected zoom window has no duration.")
            self.update_review_zoom_controls()
            return
        self.review_zoom_region = (zoom_start, zoom_end)
        self.review_zoom_clicks = []
        self.review_zoom_selecting = False
        self.plot_review()

    def raw_issue_region_jsons_from_entry(self, entry: dict[str, object] | None) -> list[dict[str, float | int | str]]:
        if self.signal is None or not isinstance(entry, dict):
            return []
        raw_regions = entry.get("issue_regions")
        if not isinstance(raw_regions, list):
            return [self.issue_region_json(start, end) for start, end in self.raw_issue_regions_from_entry(entry)]
        regions: list[dict[str, float | int | str]] = []
        for raw_region in raw_regions:
            region_tuple = self.raw_issue_region_tuple_from_json(raw_region)
            if region_tuple is None:
                continue
            region_json = self.issue_region_json(*region_tuple)
            if isinstance(raw_region, dict) and raw_region.get("source"):
                region_json["source"] = str(raw_region["source"])
            regions.append(region_json)
        return regions

    def issue_region_json_with_source(self, start: int, end: int, source: str) -> dict[str, float | int | str]:
        region = self.issue_region_json(start, end)
        region["source"] = source
        return region

    def region_list_covers_full_file(self, regions: list[dict[str, object]]) -> bool:
        if self.signal is None or len(regions) != 1:
            return False
        region = self.raw_issue_region_tuple_from_json(regions[0])
        return region == (0, len(self.signal.raw_audio))

    def draw_review_qc_regions(self) -> None:
        if self.signal is None or self.result is None:
            return
        fs = self.signal.sample_rate
        analysis_start, analysis_end = self.result.analysis_region
        effect_map = self.current_review_qc_effect_map()
        visible_keys = self.visible_review_qc_keys()
        selected_key = self.review_qc_combo.currentData() if hasattr(self, "review_qc_combo") else None
        selected_index = self.review_qc_window_index()
        for key, item in effect_map.items():
            if key not in visible_keys:
                continue
            color = self.audit_color_for_key(key)
            for start, end in self.raw_issue_regions_from_entry(item):
                start, end = max(analysis_start, start), min(analysis_end, end)
                if end <= start:
                    continue
                for ax in (self.review_ax_norm, self.review_ax_raw):
                    ax.axvspan(start / fs, end / fs, color=color, alpha=0.13, label="file-level QC")
        if selected_key in effect_map and selected_index is not None:
            regions = self.raw_issue_regions_from_entry(effect_map.get(selected_key))
            if 0 <= selected_index < len(regions):
                start, end = regions[selected_index]
                start, end = max(analysis_start, start), min(analysis_end, end)
                if end > start:
                    color = self.audit_color_for_key(selected_key)
                    for ax in (self.review_ax_norm, self.review_ax_raw):
                        ax.axvspan(start / fs, end / fs, facecolor=color, edgecolor="#101820", linewidth=1.6, alpha=0.28)
                        ax.axvline(start / fs, color="#101820", linewidth=1.2)
                        ax.axvline(end / fs, color="#101820", linewidth=1.2)
        if self.review_qc_issue_key is not None and len(self.review_qc_issue_clicks) >= 2:
            start, end = sorted(self.review_qc_issue_clicks[:2])
            color = self.audit_color_for_key(self.review_qc_issue_key)
            for ax in (self.review_ax_norm, self.review_ax_raw):
                ax.axvspan(start / fs, end / fs, facecolor=color, edgecolor="#101820", linewidth=1.6, alpha=0.30)
        for click in self.review_qc_issue_clicks:
            for ax in (self.review_ax_norm, self.review_ax_raw):
                ax.axvline(click / fs, color=self.audit_color_for_key(self.review_qc_issue_key), linewidth=1.5, linestyle="--")

    def unresolved_review_qc_keys(self) -> list[tuple[str, str]]:
        effect_map = self.current_review_qc_effect_map()
        unresolved = []
        for key in self.selected_review_qc_keys():
            if not self.raw_issue_regions_from_entry(effect_map.get(key)):
                unresolved.append(key)
        if self.review_qc_issue_key is not None and self.review_qc_issue_key not in unresolved:
            unresolved.append(self.review_qc_issue_key)
        return unresolved

    def ensure_review_qc_resolved(self) -> bool:
        unresolved = self.unresolved_review_qc_keys()
        if not unresolved:
            return True
        key = unresolved[0]
        if self.review_qc_issue_key != key:
            self.start_review_qc_boundary_selection(key)
        else:
            self.update_review_qc_controls(key)
            self.plot_review()
        gui_name, effect = key
        self.warn("File-level QC boundary required", f"Choose Whole File or set at least one window for {gui_name}: {effect}, or uncheck that QC option before moving on.")
        return False

    def saved_review_qc_data(self, metadata: dict[str, object]) -> dict[str, object]:
        data = metadata.get("file_level_qc")
        if not isinstance(data, dict):
            return {"selected_effects": []}
        effects = data.get("selected_effects")
        if not isinstance(effects, list):
            return {"selected_effects": []}
        return {"selected_effects": [item for item in effects if isinstance(item, dict)]}

    def apply_review_qc_to_segments(self) -> None:
        if self.signal is None or self.result is None:
            return
        segments = self.audit_segments()
        review_items = [item for item in self.review_qc_data.get("selected_effects", []) if isinstance(item, dict)]
        review_by_key = {
            (str(item.get("gui_name", "")), str(item.get("effect", ""))): item
            for item in review_items
            if item.get("gui_name") and item.get("effect")
        }
        for segment_number, segment in enumerate(segments, start=1):
            segment_start = int(segment["start"])
            segment_end = int(segment["end"])
            data = self.audit_by_segment_number.setdefault(segment_number, {"selected_effects": []})
            effects = self.segment_effects_without_file_level(data)
            effect_by_key = {
                (str(item.get("gui_name", "")), str(item.get("effect", ""))): item
                for item in effects
                if isinstance(item, dict)
            }
            for key, review_item in review_by_key.items():
                intersections = []
                for start, end in self.raw_issue_regions_from_entry(review_item):
                    overlap_start = max(segment_start, start)
                    overlap_end = min(segment_end, end)
                    if overlap_end > overlap_start:
                        intersections.append(self.issue_region_json_with_source(overlap_start, overlap_end, "file_level"))
                if not intersections:
                    continue
                item = effect_by_key.get(key)
                if item is None:
                    gui_name, effect = key
                    item = {"gui_name": gui_name, "effect": effect, "visible": bool(review_item.get("visible", True))}
                    item.update(self.qc_metadata_for_effect(gui_name, effect))
                    effects.append(item)
                existing_regions = self.raw_issue_region_jsons_from_entry(item)
                manual_regions = [region for region in existing_regions if not (isinstance(region, dict) and region.get("source") == "file_level")]
                item["issue_regions"] = self.deduplicate_region_jsons([*manual_regions, *intersections])
                item.pop("issue_region", None)
            data["selected_effects"] = [item for item in effects if self.raw_issue_regions_from_entry(item)]

    def segment_effects_without_file_level(self, data: dict[str, object]) -> list[dict[str, object]]:
        effects: list[dict[str, object]] = []
        for item in data.get("selected_effects", []):
            if not isinstance(item, dict):
                continue
            regions = self.raw_issue_region_jsons_from_entry(item)
            manual_regions = [region for region in regions if not (isinstance(region, dict) and region.get("source") == "file_level")]
            if manual_regions:
                next_item = dict(item)
                next_item["issue_regions"] = manual_regions
                next_item.pop("issue_region", None)
                effects.append(next_item)
        return effects

    def deduplicate_region_jsons(self, regions: list[dict[str, object]]) -> list[dict[str, object]]:
        deduped: list[dict[str, object]] = []
        index_by_region: dict[tuple[int, int], int] = {}
        for region in regions:
            region_tuple = self.raw_issue_region_tuple_from_json(region)
            if region_tuple is None:
                continue
            if region_tuple in index_by_region:
                existing_index = index_by_region[region_tuple]
                if isinstance(region, dict) and region.get("source") == "file_level":
                    deduped[existing_index] = region
                continue
            index_by_region[region_tuple] = len(deduped)
            deduped.append(region)
        return deduped

    def segment_label(self, index: int, segment: dict[str, object]) -> str:
        if self.signal is None:
            return f"Segment {index + 1}"
        start = float(segment["start"]) / self.signal.sample_rate
        end = float(segment["end"]) / self.signal.sample_rate
        segment_type = str(segment["segment_type"]).title()
        return f"{index + 1}: {segment_type} {start:.3f}s-{end:.3f}s"

    def populate_segment_combo(self, combo: QComboBox, selected_index: int | None = None) -> None:
        segments = self.audit_segments()
        combo.blockSignals(True)
        combo.clear()
        for index, segment in enumerate(segments):
            combo.addItem(self.segment_label(index, segment), index)
        if segments:
            next_index = selected_index if selected_index is not None else self.audit_index
            next_index = max(0, min(len(segments) - 1, next_index))
            combo.setCurrentIndex(next_index)
        combo.blockSignals(False)
        combo.setEnabled(bool(segments))

    def selected_segment_index_from_combo(self, combo: QComboBox) -> int | None:
        value = combo.currentData()
        return int(value) if isinstance(value, int) else None

    def update_review_segment_controls(self) -> None:
        if not hasattr(self, "review_segment_combo"):
            return
        segments = self.audit_segments()
        selected_index = self.review_return_audit_index if self.review_return_audit_index is not None else self.audit_index
        self.populate_segment_combo(self.review_segment_combo, selected_index)
        self.review_open_segment_button.setEnabled(bool(segments))
        has_return = self.review_return_audit_index is not None and bool(segments)
        self.review_return_segment_button.setEnabled(has_return)
        self.review_return_segment_button.setText(f"Return To Segment {self.review_return_audit_index + 1}" if has_return else "Return To Segment")

    def update_audit_segment_controls(self) -> None:
        if not hasattr(self, "audit_segment_combo"):
            return
        segments = self.audit_segments()
        self.populate_segment_combo(self.audit_segment_combo, self.audit_index)
        self.audit_go_segment_button.setEnabled(bool(segments))
        self.audit_review_button.setEnabled(bool(segments))

    def open_review_selected_segment(self) -> None:
        if not self.ensure_review_qc_resolved():
            return
        self.apply_review_qc_to_segments()
        index = self.selected_segment_index_from_combo(self.review_segment_combo)
        if index is None:
            return
        self.review_return_audit_index = None
        self.jump_to_audit_segment(index)

    def return_to_review_segment(self) -> None:
        if self.review_return_audit_index is None:
            return
        if not self.ensure_review_qc_resolved():
            return
        self.apply_review_qc_to_segments()
        index = self.review_return_audit_index
        self.review_return_audit_index = None
        self.jump_to_audit_segment(index)

    def go_to_audit_selected_segment(self) -> None:
        index = self.selected_segment_index_from_combo(self.audit_segment_combo)
        if index is None or index == self.audit_index:
            return
        self.jump_to_audit_segment(index)

    def view_segmentation_from_audit(self) -> None:
        if self.result is None:
            return
        if not self.ensure_audit_progress_saved_for_navigation():
            return
        self.review_return_audit_index = self.audit_index
        self.clear_audit_zoom()
        self.clear_audit_issue_selection()
        self._show_page(PAGE_REVIEW)

    def jump_to_audit_segment(self, index: int) -> None:
        segments = self.audit_segments()
        if not segments:
            return
        if self.stack.currentIndex() == PAGE_AUDIT:
            if not self.ensure_audit_progress_saved_for_navigation():
                return
        self.audit_index = max(0, min(len(segments) - 1, index))
        self.clear_audit_zoom()
        self.clear_audit_issue_selection()
        if self.stack.currentIndex() == PAGE_AUDIT:
            self.load_audit_segment()
            self.autosave_progress(PAGE_AUDIT)
        else:
            self._show_page(PAGE_AUDIT)

    def ensure_audit_progress_saved_for_navigation(self) -> bool:
        if self.audit_issue_key is not None and self.audit_issue_clicks:
            self.warn(
                "Issue window in progress",
                "Confirm the current issue window, or use Add/Reset to clear it, before leaving this segment.",
            )
            return False
        if not self.ensure_current_audit_resolved():
            return False
        self.store_current_audit()
        return True

    def confirm_segmentation(self) -> None:
        if self.result is None:
            return
        if not self.ensure_review_qc_resolved():
            return
        self.apply_review_qc_to_segments()
        if len(self.audit_segments()) == 0:
            self._show_page(PAGE_SAVE)
            return
        self.audit_index = 0
        self.clear_audit_zoom()
        self.clear_audit_issue_selection()
        self._show_page(PAGE_AUDIT)

    def load_audit_segment(self) -> None:
        if self.signal is None or self.result is None:
            return
        segments = self.audit_segments()
        if len(segments) == 0:
            self._show_page(PAGE_SAVE)
            return
        self.audit_index = max(0, min(self.audit_index, len(segments) - 1))
        segment_number = self.audit_index + 1
        data = self.audit_by_segment_number.get(segment_number, {"selected_effects": []})
        selected_effects = {
            (str(item.get("gui_name", "")), str(item.get("effect", "")))
            for item in data.get("selected_effects", [])
            if isinstance(item, dict)
        }
        visible_effects = {
            (str(item.get("gui_name", "")), str(item.get("effect", ""))): bool(item.get("visible", True))
            for item in data.get("selected_effects", [])
            if isinstance(item, dict)
        }
        legacy_issues = set(data.get("issues", []))
        self._updating_audit_checks = True
        for gui_name, check, visible in self.audit_checks:
            key = (gui_name, check.text())
            selected = key in selected_effects or check.text() in legacy_issues
            check.setChecked(selected)
            visible.setEnabled(selected)
            visible.setChecked(selected and visible_effects.get(key, True))
        self._updating_audit_checks = False
        self.clear_audit_issue_selection()
        self.audit_previous_button.setEnabled(self.audit_index > 0)
        self.audit_next_button.setText("Continue To Save" if self.audit_index == len(segments) - 1 else "Next Segment")
        self.update_audit_zoom_controls()
        self.update_audit_issue_controls()
        self.plot_audit_segment()

    def plot_audit_segment(self) -> None:
        if self.signal is None or self.result is None:
            return
        fs = self.signal.sample_rate
        segments = self.audit_segments()
        if not segments:
            return
        segment = segments[self.audit_index]
        start = int(segment["start"])
        end = int(segment["end"])
        segment_type = str(segment["segment_type"])
        audio = self.signal.raw_audio[start:end]
        x = np.arange(start, end) / fs
        display = self.display_indices(len(x))
        self.audit_ax.clear()
        self.audit_ax_spec.clear()
        self.audit_ax.plot(x[display], audio[display], color="#4a4f53", linewidth=0.65)
        self.audit_ax.set_title("Original waveform")
        self.audit_ax.set_ylabel("Original amplitude")
        self.audit_ax.grid(True, alpha=0.25)
        self.plot_audit_spectrogram(audio, start, fs)
        self.draw_audit_issue_regions()
        for click in self.audit_zoom_clicks:
            for ax in (self.audit_ax, self.audit_ax_spec):
                ax.axvline(click / fs, color="#f28e2b", linewidth=1.2)
        if self.audit_zoom_region is not None:
            zoom_start, zoom_end = self.audit_zoom_region
            for ax in (self.audit_ax, self.audit_ax_spec):
                ax.axvspan(zoom_start / fs, zoom_end / fs, color="#2f80ed", alpha=0.14)
            self.audit_ax.set_xlim(zoom_start / fs, zoom_end / fs)
        else:
            self.audit_ax.set_xlim(start / fs, end / fs)
        duration = (end - start) / fs
        self.audit_title.setText(f"Audit {segment_type.title()} Segment {self.audit_index + 1} of {len(segments)}")
        self.audit_meta.setText(f"Type {segment_type}  |  Onset {start / fs:.3f}s  |  Offset {end / fs:.3f}s  |  Duration {duration:.3f}s")
        if self.audit_zoom_region is not None:
            zoom_start, zoom_end = self.audit_zoom_region
            self.set_playback_range(zoom_start, zoom_end, f"Zoomed {segment_type.title()} Segment {self.audit_index + 1}", reset_position=True)
        else:
            self.set_playback_range(start, end, f"{segment_type.title()} Segment {self.audit_index + 1}", reset_position=True)
        self.update_audit_zoom_controls()
        self.update_audit_issue_controls()
        self.update_audit_segment_controls()
        self.audit_canvas.draw_idle()
        self.update_playback_markers()

    def current_audit_segment_bounds(self) -> tuple[int, int, str] | None:
        if self.signal is None or self.result is None:
            return None
        segments = self.audit_segments()
        if not segments:
            return None
        segment = segments[self.audit_index]
        return int(segment["start"]), int(segment["end"]), str(segment["segment_type"])

    def clear_audit_zoom(self) -> None:
        self.audit_zoom_region = None
        self.audit_zoom_clicks = []
        self.audit_zoom_selecting = False

    def clear_audit_issue_selection(self) -> None:
        self.audit_issue_key = None
        self.audit_issue_clicks = []
        self.audit_issue_edit_index = None

    def selected_audit_keys(self) -> list[tuple[str, str]]:
        return [(gui_name, check.text()) for gui_name, check, _visible in self.audit_checks if check.isChecked()]

    def visible_audit_keys(self) -> set[tuple[str, str]]:
        return {
            (gui_name, check.text())
            for gui_name, check, visible in self.audit_checks
            if check.isChecked() and visible.isChecked()
        }

    def current_audit_effect_map(self) -> dict[tuple[str, str], dict[str, object]]:
        segment_number = self.audit_index + 1
        data = self.audit_by_segment_number.get(segment_number, {"selected_effects": []})
        effects: dict[tuple[str, str], dict[str, object]] = {}
        for item in data.get("selected_effects", []):
            if not isinstance(item, dict):
                continue
            gui_name = str(item.get("gui_name", ""))
            effect = str(item.get("effect", ""))
            if gui_name and effect:
                effects[(gui_name, effect)] = dict(item)
        return effects

    def audit_color_for_key(self, key: tuple[str, str] | None) -> str:
        if key is None:
            return "#c43c39"
        return QC_AUDIT_COLORS.get(key[0], "#c43c39")

    def issue_region_from_entry(self, entry: dict[str, object] | None) -> tuple[int, int] | None:
        regions = self.issue_regions_from_entry(entry)
        return regions[-1] if regions else None

    def issue_regions_from_entry(self, entry: dict[str, object] | None) -> list[tuple[int, int]]:
        if self.signal is None or not isinstance(entry, dict):
            return []
        raw_regions = entry.get("issue_regions")
        if isinstance(raw_regions, list):
            regions = [self.issue_region_tuple_from_json(region) for region in raw_regions]
            return [region for region in regions if region is not None]
        legacy_region = self.issue_region_tuple_from_json(entry.get("issue_region"))
        return [legacy_region] if legacy_region is not None else []

    def raw_issue_region_tuple_from_json(self, region: object) -> tuple[int, int] | None:
        if self.signal is None or not isinstance(region, dict):
            return None
        try:
            if "onset_sample_absolute" in region and "offset_sample_absolute" in region:
                start = int(region["onset_sample_absolute"])
                end = int(region["offset_sample_absolute"])
            else:
                start = self.seconds_to_sample(float(region["onset_seconds_absolute"]))
                end = self.seconds_to_sample(float(region["offset_seconds_absolute"]))
        except (KeyError, TypeError, ValueError):
            return None
        start = max(0, min(len(self.signal.raw_audio), start))
        end = max(0, min(len(self.signal.raw_audio), end))
        if end <= start:
            return None
        return start, end

    def raw_issue_regions_from_entry(self, entry: dict[str, object] | None) -> list[tuple[int, int]]:
        if not isinstance(entry, dict):
            return []
        raw_regions = entry.get("issue_regions")
        if isinstance(raw_regions, list):
            regions = [self.raw_issue_region_tuple_from_json(region) for region in raw_regions]
            return [region for region in regions if region is not None]
        legacy_region = self.raw_issue_region_tuple_from_json(entry.get("issue_region"))
        return [legacy_region] if legacy_region is not None else []

    def issue_region_tuple_from_json(self, region: object) -> tuple[int, int] | None:
        if self.signal is None or not isinstance(region, dict):
            return None
        try:
            if "onset_sample_absolute" in region and "offset_sample_absolute" in region:
                start = int(region["onset_sample_absolute"])
                end = int(region["offset_sample_absolute"])
            else:
                start = self.seconds_to_sample(float(region["onset_seconds_absolute"]))
                end = self.seconds_to_sample(float(region["offset_seconds_absolute"]))
        except (KeyError, TypeError, ValueError):
            return None
        bounds = self.current_audit_segment_bounds()
        if bounds is None:
            return None
        segment_start, segment_end, _ = bounds
        start = max(segment_start, min(segment_end, start))
        end = max(segment_start, min(segment_end, end))
        if end <= start:
            return None
        return start, end

    def issue_region_jsons_from_entry(self, entry: dict[str, object] | None) -> list[dict[str, float | int | str]]:
        if self.signal is None or not isinstance(entry, dict):
            return []
        raw_regions = entry.get("issue_regions")
        if not isinstance(raw_regions, list):
            return [self.issue_region_json(start, end) for start, end in self.issue_regions_from_entry(entry)]
        regions: list[dict[str, float | int | str]] = []
        for raw_region in raw_regions:
            region_tuple = self.issue_region_tuple_from_json(raw_region)
            if region_tuple is None:
                continue
            region_json = self.issue_region_json(*region_tuple)
            if isinstance(raw_region, dict) and raw_region.get("source"):
                region_json["source"] = str(raw_region["source"])
            regions.append(region_json)
        return regions

    def issue_region_json(self, start: int, end: int) -> dict[str, float | int]:
        if self.signal is None:
            return {}
        return {
            "onset_sample_absolute": int(start),
            "offset_sample_absolute": int(end),
            "onset_seconds_absolute": float(start / self.signal.sample_rate),
            "offset_seconds_absolute": float(end / self.signal.sample_rate),
            "duration_seconds": float((end - start) / self.signal.sample_rate),
        }

    def update_audit_issue_combo(self, preferred_key: tuple[str, str] | None = None) -> None:
        if not hasattr(self, "audit_issue_combo"):
            return
        current = preferred_key or self.audit_issue_combo.currentData()
        selected_keys = self.selected_audit_keys()
        self.audit_issue_combo.blockSignals(True)
        self.audit_issue_combo.clear()
        for gui_name, effect in selected_keys:
            self.audit_issue_combo.addItem(f"{gui_name}: {effect}", (gui_name, effect))
        if current in selected_keys:
            self.audit_issue_combo.setCurrentIndex(selected_keys.index(current))
        self.audit_issue_combo.blockSignals(False)

    def selected_issue_window_index(self) -> int | None:
        if not hasattr(self, "audit_issue_window_list"):
            return None
        item = self.audit_issue_window_list.currentItem()
        if item is None:
            return None
        value = item.data(Qt.UserRole)
        return int(value) if isinstance(value, int) else None

    def update_audit_issue_window_list(self, preferred_index: int | None = None) -> None:
        if not hasattr(self, "audit_issue_window_list"):
            return
        current_index = preferred_index if preferred_index is not None else self.selected_issue_window_index()
        selected_key = self.audit_issue_combo.currentData()
        regions = []
        if selected_key in self.selected_audit_keys():
            regions = self.issue_regions_from_entry(self.current_audit_effect_map().get(selected_key))

        self.audit_issue_window_list.blockSignals(True)
        self.audit_issue_window_list.clear()
        for index, (start, end) in enumerate(regions):
            label = f"Window {index + 1}: {start / self.signal.sample_rate:.3f}s to {end / self.signal.sample_rate:.3f}s" if self.signal is not None else f"Window {index + 1}"
            self.audit_issue_window_list.addItem(label)
            item = self.audit_issue_window_list.item(self.audit_issue_window_list.count() - 1)
            item.setData(Qt.UserRole, index)
        if regions:
            next_index = current_index if current_index is not None and 0 <= current_index < len(regions) else len(regions) - 1
            self.audit_issue_window_list.setCurrentRow(next_index)
        self.audit_issue_window_list.blockSignals(False)

    def update_audit_issue_controls(self, preferred_key: tuple[str, str] | None = None) -> None:
        if not hasattr(self, "audit_issue_status"):
            return
        self.update_audit_check_enabled_states()
        self.update_audit_issue_combo(preferred_key or self.audit_issue_key)
        selected_key = self.audit_issue_combo.currentData()
        selected_keys = self.selected_audit_keys()
        effect_map = self.current_audit_effect_map()
        self.update_audit_issue_window_list(self.audit_issue_edit_index)
        selected_window_index = self.selected_issue_window_index()
        self.audit_issue_edit_button.setEnabled(selected_key in selected_keys)
        self.audit_issue_play_button.setEnabled(self.current_issue_boundary_play_range() is not None)
        self.audit_issue_confirm_button.setEnabled(self.audit_issue_key is not None and len(self.audit_issue_clicks) >= 2)
        self.audit_issue_window_edit_button.setEnabled(selected_key in selected_keys and selected_window_index is not None)
        self.audit_issue_window_remove_button.setEnabled(selected_key in selected_keys and selected_window_index is not None)
        if self.audit_issue_key is not None:
            gui_name, effect = self.audit_issue_key
            if len(self.audit_issue_clicks) == 0:
                saved_count = len(self.issue_regions_from_entry(effect_map.get(self.audit_issue_key)))
                suffix = f" {saved_count} window(s) already saved." if saved_count else ""
                self.audit_issue_status.setText(f"{gui_name}: {effect} needs a window. Click the issue start time on either plot, or uncheck it.{suffix}")
            elif len(self.audit_issue_clicks) == 1 and self.signal is not None:
                seconds = self.audit_issue_clicks[0] / self.signal.sample_rate
                self.audit_issue_status.setText(f"{gui_name}: {effect} start set at {seconds:.3f}s. Click the issue end time.")
            else:
                action = "replace the selected window" if self.audit_issue_edit_index is not None else "add it"
                self.audit_issue_status.setText(f"{gui_name}: {effect} window selected. Click Confirm to {action}, or Add/Reset to start over.")
            return
        if selected_key in selected_keys:
            entry = effect_map.get(selected_key)
            regions = self.issue_regions_from_entry(entry)
            gui_name, effect = selected_key
            if regions and self.signal is not None:
                start, end = regions[-1]
                self.audit_issue_status.setText(f"{gui_name}: {effect} has {len(regions)} window(s). Latest {start / self.signal.sample_rate:.3f}s to {end / self.signal.sample_rate:.3f}s. Use Add/Reset for another.")
            else:
                self.audit_issue_status.setText(f"{gui_name}: {effect} needs a window. Use Add/Reset or uncheck it.")
        else:
            self.audit_issue_status.setText("Check a QC artifact to set where it occurs.")

    def update_audit_check_enabled_states(self) -> None:
        pending_key = self.audit_issue_key
        for gui_name, check, visible in self.audit_checks:
            key = (gui_name, check.text())
            check.setEnabled(pending_key is None or key == pending_key)
            visible.setEnabled(check.isChecked() and (pending_key is None or key == pending_key))

    def current_issue_boundary_play_range(self) -> tuple[int, int, str] | None:
        if self.signal is None:
            return None
        if self.audit_issue_key is not None and len(self.audit_issue_clicks) >= 2:
            start, end = sorted(self.audit_issue_clicks[:2])
            if end > start:
                gui_name, effect = self.audit_issue_key
                return start, end, f"Issue Boundary: {effect}"
        selected_key = self.audit_issue_combo.currentData() if hasattr(self, "audit_issue_combo") else None
        if selected_key in self.selected_audit_keys():
            regions = self.issue_regions_from_entry(self.current_audit_effect_map().get(selected_key))
            if regions:
                _, effect = selected_key
                selected_index = self.selected_issue_window_index()
                if selected_index is None or selected_index >= len(regions):
                    selected_index = len(regions) - 1
                start, end = regions[selected_index]
                return start, end, f"Issue Boundary: {effect} window {selected_index + 1}"
        return None

    def play_selected_issue_boundary(self, button: QPushButton | None = None) -> None:
        play_range = self.current_issue_boundary_play_range()
        if play_range is None:
            self.warn("No window selected", "Select a saved issue window, or click both boundary points before playing an in-progress window.")
            return
        start, end, label = play_range
        self.toggle_playback(start, end, label, button)

    def draw_audit_issue_regions(self) -> None:
        if self.signal is None:
            return
        fs = self.signal.sample_rate
        effect_map = self.current_audit_effect_map()
        selected_keys = self.visible_audit_keys()
        for key in selected_keys:
            color = self.audit_color_for_key(key)
            for start, end in self.issue_regions_from_entry(effect_map.get(key)):
                for ax in (self.audit_ax, self.audit_ax_spec):
                    ax.axvspan(start / fs, end / fs, color=color, alpha=0.18)

        selected_key = self.audit_issue_combo.currentData() if hasattr(self, "audit_issue_combo") else None
        selected_index = self.selected_issue_window_index()
        if selected_key in self.selected_audit_keys() and selected_index is not None:
            regions = self.issue_regions_from_entry(effect_map.get(selected_key))
            if 0 <= selected_index < len(regions):
                start, end = regions[selected_index]
                color = self.audit_color_for_key(selected_key)
                for ax in (self.audit_ax, self.audit_ax_spec):
                    ax.axvspan(start / fs, end / fs, facecolor=color, edgecolor="#101820", linewidth=1.6, alpha=0.34)
                    ax.axvline(start / fs, color="#101820", linewidth=1.2)
                    ax.axvline(end / fs, color="#101820", linewidth=1.2)

        if self.audit_issue_key is not None and len(self.audit_issue_clicks) >= 2:
            start, end = sorted(self.audit_issue_clicks[:2])
            color = self.audit_color_for_key(self.audit_issue_key)
            for ax in (self.audit_ax, self.audit_ax_spec):
                ax.axvspan(start / fs, end / fs, facecolor=color, edgecolor="#101820", linewidth=1.6, alpha=0.30)
        for click in self.audit_issue_clicks:
            for ax in (self.audit_ax, self.audit_ax_spec):
                ax.axvline(click / fs, color=self.audit_color_for_key(self.audit_issue_key), linewidth=1.5, linestyle="--")

    def update_audit_zoom_controls(self) -> None:
        if not hasattr(self, "audit_zoom_select_button"):
            return
        has_zoom = self.audit_zoom_region is not None
        self.audit_zoom_play_button.setEnabled(has_zoom)
        self.audit_zoom_reset_button.setEnabled(has_zoom or bool(self.audit_zoom_clicks) or self.audit_zoom_selecting)
        if self.audit_zoom_selecting:
            self.audit_zoom_select_button.setText("Selecting Zoom")
            if len(self.audit_zoom_clicks) == 0:
                self.audit_zoom_status.setText("Click the zoom start time on either QC plot.")
            else:
                seconds = self.audit_zoom_clicks[0] / self.signal.sample_rate if self.signal is not None else 0.0
                self.audit_zoom_status.setText(f"Zoom start set at {seconds:.3f}s. Click the zoom end time.")
        elif has_zoom and self.signal is not None:
            start, end = self.audit_zoom_region
            self.audit_zoom_select_button.setText("Select Zoom")
            self.audit_zoom_status.setText(f"Zoom {start / self.signal.sample_rate:.3f}s to {end / self.signal.sample_rate:.3f}s. Play Zoom or Reset Zoom.")
        else:
            self.audit_zoom_select_button.setText("Select Zoom")
            self.audit_zoom_status.setText("Optional: select a zoom window to inspect and play a smaller part of this segment.")

    def start_audit_zoom_selection(self) -> None:
        if self.current_audit_segment_bounds() is None:
            return
        self.stop_playback()
        self.audit_zoom_region = None
        self.audit_zoom_clicks = []
        self.audit_zoom_selecting = True
        self.update_audit_zoom_controls()
        self.plot_audit_segment()

    def reset_audit_zoom(self) -> None:
        bounds = self.current_audit_segment_bounds()
        self.stop_playback()
        self.clear_audit_zoom()
        self.plot_audit_segment()
        if bounds is not None:
            start, end, segment_type = bounds
            self.set_playback_range(start, end, f"{segment_type.title()} Segment {self.audit_index + 1}", reset_position=True)

    def on_audit_plot_click(self, event) -> None:
        if self.signal is None:
            return
        if event.inaxes not in (self.audit_ax, self.audit_ax_spec) or event.xdata is None:
            return
        if self.audit_zoom_selecting:
            self.handle_audit_zoom_click(float(event.xdata))
        elif self.audit_issue_key is not None:
            self.handle_audit_issue_click(float(event.xdata))

    def handle_audit_zoom_click(self, seconds: float) -> None:
        bounds = self.current_audit_segment_bounds()
        if bounds is None:
            return
        segment_start, segment_end, _ = bounds
        sample = self.seconds_to_sample(seconds)
        sample = max(segment_start, min(segment_end, sample))
        self.audit_zoom_clicks.append(sample)
        if len(self.audit_zoom_clicks) < 2:
            self.update_audit_zoom_controls()
            self.plot_audit_segment()
            return
        zoom_start, zoom_end = sorted(self.audit_zoom_clicks[:2])
        if zoom_end <= zoom_start:
            self.audit_zoom_clicks = []
            self.warn("Invalid zoom", "The selected zoom window has no duration.")
            self.update_audit_zoom_controls()
            return
        self.audit_zoom_region = (zoom_start, zoom_end)
        self.audit_zoom_clicks = []
        self.audit_zoom_selecting = False
        self.plot_audit_segment()

    def handle_audit_issue_click(self, seconds: float) -> None:
        bounds = self.current_audit_segment_bounds()
        if bounds is None:
            return
        segment_start, segment_end, _ = bounds
        sample = self.seconds_to_sample(seconds)
        sample = max(segment_start, min(segment_end, sample))
        if len(self.audit_issue_clicks) >= 2:
            self.audit_issue_clicks = []
        self.audit_issue_clicks.append(sample)
        self.update_audit_issue_controls(self.audit_issue_key)
        self.plot_audit_segment()
        self.autosave_progress(PAGE_AUDIT, force=True)

    def on_audit_check_toggled(self, gui_name: str, check: QCheckBox, checked: bool) -> None:
        if self._updating_audit_checks:
            return
        key = (gui_name, check.text())
        visible = self.visibility_check_for_key(key)
        if visible is not None:
            visible.setEnabled(checked)
            visible.setChecked(checked)
        if checked:
            self.store_current_audit()
            if self.issue_region_from_entry(self.current_audit_effect_map().get(key)) is None:
                self.start_issue_boundary_selection(key)
            else:
                self.update_audit_issue_controls(key)
                self.plot_audit_segment()
            return
        if self.audit_issue_key == key:
            self.clear_audit_issue_selection()
        self.store_current_audit()
        self.update_audit_issue_controls()
        self.plot_audit_segment()

    def on_audit_visibility_toggled(self, gui_name: str, check: QCheckBox, checked: bool) -> None:
        if self._updating_audit_checks:
            return
        if not check.isChecked():
            return
        self.store_current_audit()
        self.plot_audit_segment()

    def set_all_audit_visibility(self, visible_state: bool) -> None:
        if self.result is None:
            return
        self._updating_audit_checks = True
        for _gui_name, check, visible in self.audit_checks:
            if check.isChecked():
                visible.setChecked(visible_state)
        self._updating_audit_checks = False
        self.store_current_audit()
        self.update_audit_issue_controls()
        self.plot_audit_segment()

    def visibility_check_for_key(self, key: tuple[str, str]) -> QCheckBox | None:
        for gui_name, check, visible in self.audit_checks:
            if (gui_name, check.text()) == key:
                return visible
        return None

    def on_audit_issue_combo_changed(self) -> None:
        if self.audit_issue_key is None:
            self.update_audit_issue_controls()

    def start_issue_boundary_selection(self, key: tuple[str, str]) -> None:
        self.stop_playback()
        self.audit_issue_key = key
        self.audit_issue_clicks = []
        self.audit_issue_edit_index = None
        self.update_audit_issue_controls(key)
        self.plot_audit_segment()
        self.autosave_progress(PAGE_AUDIT, force=True)

    def edit_selected_issue_boundary(self) -> None:
        key = self.audit_issue_combo.currentData()
        if key not in self.selected_audit_keys():
            return
        self.start_issue_boundary_selection(key)

    def on_audit_issue_window_selected(self) -> None:
        if self.audit_issue_key is None:
            self.update_audit_issue_controls()
            self.plot_audit_segment()

    def edit_selected_issue_window(self) -> None:
        key = self.audit_issue_combo.currentData()
        selected_index = self.selected_issue_window_index()
        if key not in self.selected_audit_keys() or selected_index is None:
            return
        regions = self.issue_regions_from_entry(self.current_audit_effect_map().get(key))
        if selected_index >= len(regions):
            return
        self.stop_playback()
        self.audit_issue_key = key
        self.audit_issue_edit_index = selected_index
        self.audit_issue_clicks = [regions[selected_index][0], regions[selected_index][1]]
        self.update_audit_issue_controls(key)
        self.plot_audit_segment()
        self.autosave_progress(PAGE_AUDIT, force=True)

    def remove_selected_issue_window(self) -> None:
        key = self.audit_issue_combo.currentData()
        selected_index = self.selected_issue_window_index()
        if key not in self.selected_audit_keys() or selected_index is None:
            return
        self.store_current_audit()
        segment_number = self.audit_index + 1
        data = self.audit_by_segment_number.setdefault(segment_number, {"selected_effects": []})
        effects = [item for item in data.get("selected_effects", []) if isinstance(item, dict)]
        for item in effects:
            if (str(item.get("gui_name", "")), str(item.get("effect", ""))) != key:
                continue
            regions = self.issue_region_jsons_from_entry(item)
            if 0 <= selected_index < len(regions):
                del regions[selected_index]
            item["issue_regions"] = regions
            item.pop("issue_region", None)
            break
        data["selected_effects"] = effects
        if self.audit_issue_key == key:
            self.clear_audit_issue_selection()
        next_index = max(0, selected_index - 1)
        self.update_audit_issue_controls(key)
        self.update_audit_issue_window_list(next_index)
        self.plot_audit_segment()
        self.autosave_progress(PAGE_AUDIT, force=True)

    def confirm_audit_issue_boundary(self) -> None:
        if self.signal is None or self.audit_issue_key is None or len(self.audit_issue_clicks) < 2:
            return
        start, end = sorted(self.audit_issue_clicks[:2])
        if end <= start:
            self.audit_issue_clicks = []
            self.warn("Invalid boundary", "The selected issue boundary has no duration.")
            self.update_audit_issue_controls()
            self.plot_audit_segment()
            return
        confirmed_index = self.set_issue_region_for_key(self.audit_issue_key, (start, end))
        confirmed_key = self.audit_issue_key
        self.clear_audit_issue_selection()
        self.update_audit_issue_controls(confirmed_key)
        self.update_audit_issue_window_list(confirmed_index)
        self.plot_audit_segment()
        self.autosave_progress(PAGE_AUDIT, force=True)

    def set_issue_region_for_key(self, key: tuple[str, str], region: tuple[int, int]) -> int:
        self.store_current_audit()
        segment_number = self.audit_index + 1
        data = self.audit_by_segment_number.setdefault(segment_number, {"selected_effects": []})
        effects = [item for item in data.get("selected_effects", []) if isinstance(item, dict)]
        for item in effects:
            if (str(item.get("gui_name", "")), str(item.get("effect", ""))) == key:
                item.update(self.qc_metadata_for_effect(*key))
                regions = self.issue_region_jsons_from_entry(item)
                if self.audit_issue_edit_index is not None and 0 <= self.audit_issue_edit_index < len(regions):
                    target_index = self.audit_issue_edit_index
                    regions[target_index] = self.issue_region_json(*region)
                else:
                    target_index = len(regions)
                    regions.append(self.issue_region_json(*region))
                item["issue_regions"] = regions
                item.pop("issue_region", None)
                data["selected_effects"] = effects
                return target_index
        gui_name, effect = key
        item = {"gui_name": gui_name, "effect": effect}
        item.update(self.qc_metadata_for_effect(gui_name, effect))
        item["visible"] = True
        item["issue_regions"] = [self.issue_region_json(*region)]
        effects.append(item)
        data["selected_effects"] = effects
        return 0

    def unresolved_audit_issue_keys(self) -> list[tuple[str, str]]:
        effect_map = self.current_audit_effect_map()
        unresolved = []
        for key in self.selected_audit_keys():
            if not self.issue_regions_from_entry(effect_map.get(key)):
                unresolved.append(key)
        if self.audit_issue_key is not None and self.audit_issue_key not in unresolved:
            unresolved.append(self.audit_issue_key)
        return unresolved

    def ensure_current_audit_resolved(self) -> bool:
        self.store_current_audit()
        unresolved = self.unresolved_audit_issue_keys()
        if not unresolved:
            return True
        key = unresolved[0]
        if self.audit_issue_key == key:
            self.update_audit_issue_controls(key)
            self.plot_audit_segment()
        else:
            self.start_issue_boundary_selection(key)
        gui_name, effect = key
        self.warn("Issue boundary required", f"Set at least one window for {gui_name}: {effect}, or uncheck that QC option before moving on.")
        return False

    def plot_audit_spectrogram(self, audio: np.ndarray, absolute_start_sample: int, sample_rate: int) -> None:
        audio = np.asarray(audio, dtype=float).reshape(-1)
        if audio.size < 16:
            self.audit_ax_spec.text(0.5, 0.5, "Segment too short for spectrogram", ha="center", va="center", transform=self.audit_ax_spec.transAxes)
            self.audit_ax_spec.set_xlabel("Time (seconds)")
            self.audit_ax_spec.set_ylabel("Frequency (Hz)")
            return

        window_seconds = 0.025
        target_nperseg = max(64, int(round(window_seconds * sample_rate)))
        nperseg = min(audio.size, target_nperseg)
        if nperseg < 16:
            nperseg = min(audio.size, 16)
        noverlap = min(nperseg - 1, int(round(nperseg * 0.75)))
        nfft = 1 << int(np.ceil(np.log2(max(256, nperseg))))
        frequencies, times, magnitude = scipy_signal.spectrogram(
            audio,
            fs=sample_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            detrend=False,
            scaling="spectrum",
            mode="magnitude",
        )
        if magnitude.size == 0:
            return
        db_values = 20.0 * np.log10(np.maximum(magnitude, np.finfo(float).eps))
        top = float(np.nanmax(db_values))
        db_values = np.maximum(db_values, top - 80.0)
        max_frequency = min(5500.0, sample_rate / 2.0)
        keep = frequencies <= max_frequency
        absolute_times = absolute_start_sample / sample_rate + times
        self.audit_ax_spec.pcolormesh(
            absolute_times,
            frequencies[keep],
            db_values[keep],
            shading="auto",
            cmap="magma",
        )
        self.audit_ax_spec.set_title("Spectrogram")
        self.audit_ax_spec.set_ylabel("Frequency (Hz)")
        self.audit_ax_spec.set_xlabel("Time (seconds)")
        self.audit_ax_spec.set_ylim(0, max_frequency)
        self.audit_ax_spec.grid(False)

    def store_current_audit(self) -> None:
        if self.result is None or len(self.audit_segments()) == 0:
            return
        segment_number = self.audit_index + 1
        existing = self.current_audit_effect_map()
        selected_effects = [
            self.stored_audit_effect(gui_name, check.text(), visible.isChecked(), existing.get((gui_name, check.text())))
            for gui_name, check, visible in self.audit_checks
            if check.isChecked()
        ]
        next_data = {
            "selected_effects": selected_effects,
        }
        if self.audit_by_segment_number.get(segment_number, {"selected_effects": []}) == next_data:
            self.audit_by_segment_number[segment_number] = next_data
            return
        self.audit_by_segment_number[segment_number] = next_data
        self.mark_progress_dirty()
        self.autosave_progress(PAGE_AUDIT, force=True)

    def stored_audit_effect(self, gui_name: str, effect: str, visible: bool, existing: dict[str, object] | None) -> dict[str, object]:
        item: dict[str, object] = {"gui_name": gui_name, "effect": effect}
        item.update(self.qc_metadata_for_effect(gui_name, effect))
        item["visible"] = bool(visible)
        regions = self.issue_region_jsons_from_entry(existing)
        if regions:
            item["issue_regions"] = regions
        return item

    def qc_metadata_for_effect(self, gui_name: str, effect: str) -> dict[str, str]:
        return dict(QC_AUDIT_METADATA.get((gui_name, effect), {}))

    def previous_audit_segment(self) -> None:
        if self.audit_index <= 0:
            return
        if not self.ensure_audit_progress_saved_for_navigation():
            return
        self.audit_index -= 1
        self.clear_audit_zoom()
        self.clear_audit_issue_selection()
        self.load_audit_segment()
        self.autosave_progress(PAGE_AUDIT)

    def next_audit_segment(self) -> None:
        if self.result is None:
            return
        if not self.ensure_audit_progress_saved_for_navigation():
            return
        if self.audit_index >= len(self.audit_segments()) - 1:
            self._show_page(PAGE_SAVE)
            return
        self.audit_index += 1
        self.clear_audit_zoom()
        self.clear_audit_issue_selection()
        self.load_audit_segment()
        self.autosave_progress(PAGE_AUDIT)

    def audit_segments(self) -> list[dict[str, object]]:
        if self.result is None:
            return []
        analysis_start = int(self.result.analysis_region[0])
        segments: list[dict[str, object]] = []
        fixed_events = self.result.fixed_events_samples
        if fixed_events is not None and len(fixed_events) > 0:
            for start, end in np.asarray(fixed_events, dtype=int):
                segments.append({"segment_type": "fixed", "start": analysis_start + int(start), "end": analysis_start + int(end)})
            segments.sort(key=lambda segment: int(segment["start"]))
            return segments
        for start, end in np.asarray(self.result.speech_events_samples, dtype=int):
            segments.append({"segment_type": "speech", "start": analysis_start + int(start), "end": analysis_start + int(end)})
        for start, end in np.asarray(self.result.pause_events_samples, dtype=int):
            segments.append({"segment_type": "pause", "start": analysis_start + int(start), "end": analysis_start + int(end)})
        segments.sort(key=lambda segment: (int(segment["start"]), 0 if segment["segment_type"] == "speech" else 1))
        return segments

    def prepare_save_page(self) -> None:
        if self.result is None:
            return
        fixed_events = self.result.fixed_events_samples
        if fixed_events is not None and len(fixed_events) > 0:
            summary = f"The output will include {len(fixed_events)} fixed 5-second segment rows with absolute timestamps."
        else:
            speech_count = len(self.result.speech_events_samples)
            pause_count = len(self.result.pause_events_samples)
            summary = f"The output will include {speech_count} speech rows and {pause_count} pause rows with absolute timestamps."
        self.save_summary.setText(f"Ready to save {self.signal.path.name if self.signal else ''}. {summary}")
        output_path = self.output_path_for_file(self.result.signal_path)
        self.save_path_label.setText(f"Output file: {output_path}")

    def save_current_file(self) -> None:
        if self.result is None:
            return
        self.apply_review_qc_to_segments()
        self.output_run_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_path_for_file(self.result.signal_path)
        try:
            self.last_saved_path = export_segment_audit_csv(self.result, output_path, self.audit_by_segment_number)
            self.save_current_metadata()
        except Exception as exc:
            self.warn("Save failed", str(exc))
            return
        self.clear_progress_for_file(self.result.signal_path)
        self.progress_dirty = False
        self.refresh_queue_display()
        next_index = self.next_pending_index()
        if next_index is not None:
            self.queue_index = next_index
            self.load_best_current_file()
            return
        self.signal = None
        self.result = None
        self.noise_region = None
        self.analysis_region = None
        self.audit_by_segment_number = {}
        self.review_qc_data = {"selected_effects": []}
        self.progress_dirty = False
        self.clear_review_qc_issue_selection()
        self.clear_review_zoom()
        self.review_return_audit_index = None
        self.queue_index = len(self.queue) - 1 if self.queue else -1
        self.refresh_queue_display()
        self._show_page(PAGE_LOAD)
        QMessageBox.information(self, "Saved", f"Saved output to {self.last_saved_path}")

    def save_current_metadata(self) -> None:
        if self.signal is None or self.result is None:
            return
        result_settings = self.result.settings
        metadata = {
            "file_name": self.signal.path.name,
            "source_path": str(self.signal.path),
            "sample_rate": int(self.signal.sample_rate),
            "segmentation_mode": str(self.result.segmentation_mode),
            "noise_region_samples": self.region_to_json(self.noise_region),
            "analysis_region_samples": self.region_to_json(self.result.analysis_region),
            "file_level_qc": self.review_qc_data,
            "settings": {
                "speech_threshold_ms": float(result_settings.speech_threshold_ms),
                "pause_threshold_ms": float(result_settings.pause_threshold_ms),
                "sd_multiplier": float(result_settings.sd_multiplier),
                "adaptive": bool(result_settings.adaptive),
                "use_full_file": bool(result_settings.use_full_file),
                "use_fixed_segments": bool(result_settings.use_fixed_segments),
                "fixed_segment_seconds": float(result_settings.fixed_segment_seconds),
            },
        }
        metadata_path = self.metadata_path_for_file(self.signal.path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

    def region_to_json(self, region: tuple[int, int] | None) -> list[int] | None:
        if region is None:
            return None
        return [int(region[0]), int(region[1])]

    def draw_region(self, ax_norm, ax_raw, region: tuple[int, int] | None, color: str, label: str) -> None:
        if self.signal is None or region is None:
            return
        fs = self.signal.sample_rate
        start, end = region
        for ax in (ax_norm, ax_raw):
            ax.axvspan(start / fs, end / fs, color=color, alpha=0.20, label=label)

    def style_axes(self, *axes) -> None:
        for ax in axes:
            ax.grid(True, alpha=0.25)
        self.add_legend(axes[0])

    def add_legend(self, ax) -> None:
        handles, labels = ax.get_legend_handles_labels()
        by_label = {label: handle for handle, label in zip(handles, labels) if label}
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    def playback_sample_rate(self) -> int:
        if self.signal is None:
            return 1
        return max(1, int(round(self.signal.sample_rate * self.playback_speed)))

    def on_playback_speed_changed(self) -> None:
        if not hasattr(self, "audit_speed_combo"):
            return
        value = self.audit_speed_combo.currentData()
        try:
            next_speed = float(value)
        except (TypeError, ValueError):
            next_speed = 1.0
        next_speed = max(0.25, min(4.0, next_speed))
        if np.isclose(next_speed, self.playback_speed):
            return
        if self.playback_active:
            self.update_playback_progress()
        self.playback_speed = next_speed
        if self.playback_active and self.playback_range is not None and self.signal is not None:
            start, end = self.playback_range
            if self.playback_current_sample >= end:
                self.stop_playback(finished=True)
            else:
                try:
                    stop_audio()
                    play_audio(self.signal.raw_audio[self.playback_current_sample:end], self.playback_sample_rate())
                except Exception as exc:
                    self.stop_playback()
                    self.warn("Playback failed", str(exc))
                    return
                self.playback_started_sample = self.playback_current_sample
                self.playback_started_at = time.monotonic()
        self.update_playback_widgets()

    def toggle_current_page_playback(self) -> None:
        if self.playback_active:
            self.stop_playback()
            return
        if self.signal is None:
            return
        page = self.stack.currentIndex()
        if page == PAGE_BOUNDARIES:
            self.start_current_range_or_default(self.boundary_play_full_button)
        elif page == PAGE_REVIEW:
            self.start_current_range_or_default(self.review_play_analysis_button)
        elif page == PAGE_AUDIT:
            self.start_current_range_or_default(self.audit_play_button)

    def start_current_range_or_default(self, default_button: QPushButton | None) -> None:
        if self.signal is None:
            return
        if self.playback_range is None:
            if default_button is self.boundary_play_full_button:
                self.play_full_audio(default_button)
            elif default_button is self.review_play_analysis_button:
                self.play_analysis_region(default_button)
            return
        start, end = self.playback_range
        button = self.button_for_playback_label(self.playback_label) or default_button
        play_text = button.text() if button is not None else ""
        stop_text = self.stop_text_for_button(play_text)
        self.start_playback(start, end, self.playback_label, button, play_text, stop_text)

    def button_for_playback_label(self, label: str) -> QPushButton | None:
        page = self.stack.currentIndex()
        if page == PAGE_BOUNDARIES:
            if label == "Full Audio":
                return self.boundary_play_full_button
            if label == "Noise Selection":
                return self.boundary_play_noise_button
            if label == "Analysis Region":
                return self.boundary_play_analysis_button
        elif page == PAGE_REVIEW:
            if label.startswith("File-Level QC:"):
                return self.review_qc_play_button
            if label.startswith("Zoomed Review"):
                return self.review_zoom_play_button
            if label == "Full Audio":
                return self.review_play_full_button
            if label == "Analysis Region":
                return self.review_play_analysis_button
        elif page == PAGE_AUDIT:
            if label.startswith("Issue Boundary:"):
                return self.audit_issue_play_button
            if label.startswith("Zoomed "):
                return self.audit_zoom_play_button
            return self.audit_play_button
        return None

    def play_full_audio(self, button: QPushButton | None = None) -> None:
        if self.signal is None:
            return
        self.toggle_playback(0, len(self.signal.raw_audio), "Full Audio", button)

    def play_noise_region(self, button: QPushButton | None = None) -> None:
        if self.noise_region is None:
            self.warn("No noise selection", "Select a noise region first.")
            return
        self.toggle_playback(*self.noise_region, label="Noise Selection", button=button)

    def play_analysis_region(self, button: QPushButton | None = None) -> None:
        if self.signal is None:
            return
        if self.result is not None:
            start, end = self.result.analysis_region
        elif self.analysis_region is not None:
            start, end = self.analysis_region
        elif self.settings.use_full_file:
            start, end = 0, len(self.signal.raw_audio)
        else:
            self.warn("No analysis region", "Select an analysis region first.")
            return
        self.toggle_playback(start, end, "Analysis Region", button)

    def play_current_audit_segment(self, button: QPushButton | None = None) -> None:
        if self.signal is None or self.result is None:
            return
        segments = self.audit_segments()
        if not segments:
            return
        segment = segments[self.audit_index]
        self.toggle_playback(
            int(segment["start"]),
            int(segment["end"]),
            f"{str(segment['segment_type']).title()} Segment {self.audit_index + 1}",
            button,
        )

    def play_current_audit_zoom(self, button: QPushButton | None = None) -> None:
        if self.signal is None or self.result is None or self.audit_zoom_region is None:
            return
        bounds = self.current_audit_segment_bounds()
        if bounds is None:
            return
        _, _, segment_type = bounds
        start, end = self.audit_zoom_region
        self.toggle_playback(
            start,
            end,
            f"Zoomed {segment_type.title()} Segment {self.audit_index + 1}",
            button,
        )

    def play_samples(self, start: int, end: int) -> None:
        self.start_playback(start, end, "Playback", None, "", "")

    def toggle_playback(self, start: int, end: int, label: str, button: QPushButton | None = None) -> None:
        if self.playback_active and button is not None and self.playback_button is button:
            self.stop_playback()
            return
        play_text = button.text() if button is not None else ""
        stop_text = self.stop_text_for_button(play_text)
        self.start_playback(start, end, label, button, play_text, stop_text)

    def start_playback(
        self,
        start: int,
        end: int,
        label: str,
        button: QPushButton | None,
        play_text: str,
        stop_text: str,
    ) -> None:
        if self.signal is None:
            return
        start = max(0, min(len(self.signal.raw_audio), int(start)))
        end = max(start, min(len(self.signal.raw_audio), int(end)))
        if end <= start:
            return
        self.stop_playback()
        self.set_playback_range(start, end, label, reset_position=False)
        if self.playback_current_sample < start or self.playback_current_sample >= end:
            self.playback_current_sample = start
        play_start = self.playback_current_sample
        try:
            play_audio(self.signal.raw_audio[play_start:end], self.playback_sample_rate())
        except Exception as exc:
            self.warn("Playback failed", str(exc))
            self.update_playback_widgets()
            return
        self.playback_active = True
        self.playback_range = (start, end)
        self.playback_label = label
        self.playback_started_sample = play_start
        self.playback_started_at = time.monotonic()
        self.playback_button = button
        self.playback_button_play_text = play_text
        self.playback_button_stop_text = stop_text
        if button is not None:
            button.setText(stop_text)
        self.playback_timer.start()
        self.update_playback_widgets()

    def stop_playback(self, finished: bool = False) -> None:
        if self.playback_active:
            try:
                stop_audio()
            except Exception:
                pass
        self.playback_timer.stop()
        self.playback_active = False
        if finished and self.playback_range is not None:
            self.playback_current_sample = self.playback_range[1]
        if self.playback_button is not None and self.playback_button_play_text:
            self.playback_button.setText(self.playback_button_play_text)
        self.playback_button = None
        self.playback_button_play_text = ""
        self.playback_button_stop_text = ""
        self.update_playback_widgets()

    def update_playback_progress(self) -> None:
        if not self.playback_active or self.signal is None or self.playback_range is None or self._playback_slider_dragging:
            return
        _, end = self.playback_range
        elapsed = time.monotonic() - self.playback_started_at
        self.playback_current_sample = self.playback_started_sample + int(round(elapsed * self.signal.sample_rate * self.playback_speed))
        if self.playback_current_sample >= end:
            self.stop_playback(finished=True)
            return
        self.update_playback_widgets()

    def set_playback_range(self, start: int, end: int, label: str, reset_position: bool = False) -> None:
        if self.signal is None:
            return
        start = max(0, min(len(self.signal.raw_audio), int(start)))
        end = max(start, min(len(self.signal.raw_audio), int(end)))
        if end <= start:
            return
        changed = self.playback_range != (start, end) or self.playback_label != label
        self.playback_range = (start, end)
        self.playback_label = label
        if reset_position or changed or self.playback_current_sample < start or self.playback_current_sample > end:
            self.playback_current_sample = start
        self.update_playback_widgets()

    def update_playback_widgets(self) -> None:
        if self.signal is None or self.playback_range is None:
            label_text = "Playback: --"
            enabled = False
            slider_max = 1
            slider_value = 0
        else:
            start, end = self.playback_range
            slider_max = max(1, end - start)
            slider_value = max(0, min(slider_max, self.playback_current_sample - start))
            elapsed_text = self.format_seconds(slider_value / self.signal.sample_rate)
            duration_text = self.format_seconds((end - start) / self.signal.sample_rate)
            absolute_text = self.format_seconds(self.playback_current_sample / self.signal.sample_rate)
            speed_text = f"  {self.playback_speed:.2f}x"
            label_text = f"{self.playback_label}: {elapsed_text} / {duration_text}  ({absolute_text}){speed_text}"
            enabled = True
        self._updating_playback_widgets = True
        for slider, label in self.playback_widgets:
            slider.setEnabled(enabled)
            slider.setRange(0, slider_max)
            if not self._playback_slider_dragging:
                slider.setValue(slider_value)
            label.setText(label_text)
        self._updating_playback_widgets = False
        self.update_playback_markers()

    def update_playback_markers(self) -> None:
        if self.signal is None or self.playback_range is None:
            for overlay_name in ("boundary_crosshair", "review_playback_overlay", "audit_playback_overlay"):
                overlay = getattr(self, overlay_name, None)
                if overlay is not None:
                    overlay.clear_playback()
            return
        seconds = self.playback_current_sample / self.signal.sample_rate
        self.update_playback_marker_for_axis(getattr(self, "boundary_crosshair", None), self.boundary_ax_raw, seconds)
        self.update_playback_marker_for_axis(getattr(self, "review_playback_overlay", None), self.review_ax_raw, seconds)
        self.update_playback_marker_for_axis(getattr(self, "audit_playback_overlay", None), self.audit_ax, seconds)

    def update_playback_marker_for_axis(self, overlay: CrosshairOverlay | None, axis, seconds: float) -> None:
        if overlay is None or axis is None:
            return
        x_min, x_max = axis.get_xlim()
        if seconds < min(x_min, x_max) or seconds > max(x_min, x_max):
            overlay.clear_playback()
            return
        axis_width_seconds = x_max - x_min
        if axis_width_seconds == 0:
            overlay.clear_playback()
            return
        axis_bbox = axis.get_window_extent()
        position_fraction = (seconds - x_min) / axis_width_seconds
        x_display = axis_bbox.x0 + position_fraction * axis_bbox.width
        canvas = axis.figure.canvas
        canvas_width = max(1, canvas.width())
        figure_display_width = max(1.0, axis.figure.bbox.width)
        display_to_widget_scale = canvas_width / figure_display_width
        x_widget = x_display * display_to_widget_scale
        x_widget = max(0, min(max(0, overlay.width() - 1), x_widget))
        overlay.set_playback_x(x_widget)

    def on_playback_slider_pressed(self, slider: QSlider) -> None:
        if self.playback_range is None:
            return
        self._playback_slider_dragging = True

    def on_playback_slider_moved(self, slider: QSlider, value: int) -> None:
        if self.playback_range is None or self.signal is None or self._updating_playback_widgets:
            return
        start, _ = self.playback_range
        self.playback_current_sample = start + int(value)
        self.update_playback_widgets()

    def on_playback_slider_released(self, slider: QSlider) -> None:
        if self.playback_range is None or self.signal is None:
            self._playback_slider_dragging = False
            return
        start, end = self.playback_range
        self.playback_current_sample = max(start, min(end, start + int(slider.value())))
        self._playback_slider_dragging = False
        if self.playback_active:
            if self.playback_current_sample >= end:
                self.stop_playback(finished=True)
            else:
                try:
                    stop_audio()
                    play_audio(self.signal.raw_audio[self.playback_current_sample:end], self.playback_sample_rate())
                except Exception as exc:
                    self.stop_playback()
                    self.warn("Playback failed", str(exc))
                    return
                self.playback_started_sample = self.playback_current_sample
                self.playback_started_at = time.monotonic()
        self.update_playback_widgets()

    def stop_text_for_button(self, play_text: str) -> str:
        if play_text.startswith("Play "):
            return play_text.replace("Play ", "Stop ", 1)
        return "Stop"

    def format_seconds(self, seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        if hours:
            return f"{hours:d}:{minutes:02d}:{secs:05.2f}"
        return f"{minutes:02d}:{secs:05.2f}"

    def seconds_to_sample(self, seconds: float) -> int:
        if self.signal is None:
            return 0
        return max(0, min(len(self.signal.raw_audio), int(round(seconds * self.signal.sample_rate))))

    def display_indices(self, length: int) -> slice:
        if length <= MAX_DISPLAY_POINTS:
            return slice(None)
        stride = int(np.ceil(length / MAX_DISPLAY_POINTS))
        return slice(None, None, max(1, stride))

    def closeEvent(self, event) -> None:  # pragma: no cover - Qt lifecycle hook
        self.autosave_progress(self.stack.currentIndex())
        self.stop_playback()
        event.accept()

    def warn(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
