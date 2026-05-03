from __future__ import annotations

import os
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from spa_core.audio import play_audio, read_wav, stop_audio
from spa_core.export import export_segment_audit_csv
from spa_core.models import SpaResult, SpaSettings, SpaSignal
from spa_core.segmentation import run_spa


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
QC_AUDIT_GROUPS = [
    ("Environmental noise", ["Traffic", "HVAC", "Pets", "TV (non-speech)"]),
    ("Competing speech", ["TV (speech)", "Other human speakers"]),
    ("Volume unstable", ["Volume too quiet", "Volume too loud"]),
    ("Clipping", ["Clipping/Saturation"]),
    ("Reverberation/echo", ["Reverb", "Echo"]),
    ("Platform effects", ["Muffled", "Compressed/robotic"]),
    ("Temporal discontinuities", ["Audio lagging", "Audio glitching", "Audio skipped/missed"]),
    ("Any non-task related content", ["Extra or filler word", "Missed word"]),
]
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
    from PySide6.QtGui import QColor, QPainter, QPen
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

        form.addRow("Threshold curve", self.curve_combo)
        form.addRow("SD multiplier", self.sd_spin)
        form.addRow("Speech minimum", self.speech_spin)
        form.addRow("Pause minimum", self.pause_spin)
        form.addRow("Analysis", self.full_file_check)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def settings(self) -> SpaSettings:
        return SpaSettings(
            speech_threshold_ms=self.speech_spin.value(),
            pause_threshold_ms=self.pause_spin.value(),
            sd_multiplier=self.sd_spin.value(),
            threshold_mode="automatic",
            adaptive=self.curve_combo.currentText() == "adaptive",
            variable_mode="ONE TIME",
            iterations=1,
            use_full_file=self.full_file_check.isChecked(),
        )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SPA Segmentation")
        self.resize(1480, 900)

        self.settings = SpaSettings()
        self.queue: list[Path] = []
        self.queue_index = -1
        self.signal: SpaSignal | None = None
        self.result: SpaResult | None = None
        self.noise_region: tuple[int, int] | None = None
        self.analysis_region: tuple[int, int] | None = None
        self.boundary_mode = "noise"
        self.boundary_clicks: list[int] = []
        self.audit_by_segment_number: dict[int, dict[str, object]] = {}
        self.audit_checks: list[tuple[str, QCheckBox]] = []
        self.audit_index = 0
        self.loaded_from_saved_output = False
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
        self.playback_button: QPushButton | None = None
        self.playback_button_play_text = ""
        self.playback_button_stop_text = ""
        self._updating_playback_widgets = False
        self._playback_slider_dragging = False
        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(50)
        self.playback_timer.timeout.connect(self.update_playback_progress)

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
        form.addRow("Threshold curve", self.start_curve_combo)
        form.addRow("SD multiplier", self.start_sd_spin)
        form.addRow("Speech minimum", self.start_speech_spin)
        form.addRow("Pause minimum", self.start_pause_spin)
        form.addRow("Analysis", self.start_full_file_check)
        outer.addWidget(box)

        continue_button = QPushButton("Continue To Files")
        continue_button.setObjectName("PrimaryButton")
        continue_button.clicked.connect(self.accept_start_settings)
        outer.addWidget(continue_button, alignment=Qt.AlignRight)
        outer.addStretch(1)
        return page

    def _build_load_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(48, 36, 48, 36)
        outer.setSpacing(18)
        title = QLabel("Load Audio")
        title.setObjectName("PageTitle")
        outer.addWidget(title)

        actions = QHBoxLayout()
        single = QPushButton("Select WAV")
        multiple = QPushButton("Select Multiple WAVs")
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

        self.review_figure = Figure(figsize=(10, 6), tight_layout=True)
        self.review_canvas = FigureCanvas(self.review_figure)
        self.review_ax_norm = self.review_figure.add_subplot(211)
        self.review_ax_raw = self.review_figure.add_subplot(212, sharex=self.review_ax_norm)
        review_stack_widget = QWidget()
        review_stack = QStackedLayout(review_stack_widget)
        review_stack.setContentsMargins(0, 0, 0, 0)
        review_stack.setStackingMode(QStackedLayout.StackAll)
        review_stack.addWidget(self.review_canvas)
        self.review_playback_overlay = CrosshairOverlay()
        review_stack.addWidget(self.review_playback_overlay)
        outer.addWidget(review_stack_widget, stretch=1)
        outer.addWidget(self.create_playback_bar())

        actions = QHBoxLayout()
        self.review_play_full_button = QPushButton("Play Full Audio")
        self.review_play_analysis_button = QPushButton("Play Analysis Region")
        self.review_next_button = QPushButton("Confirm Segmentation")
        self.review_next_button.setObjectName("PrimaryButton")
        self.review_play_full_button.clicked.connect(lambda: self.play_full_audio(self.review_play_full_button))
        self.review_play_analysis_button.clicked.connect(lambda: self.play_analysis_region(self.review_play_analysis_button))
        self.review_next_button.clicked.connect(self.confirm_segmentation)
        actions.addWidget(self.review_play_full_button)
        actions.addWidget(self.review_play_analysis_button)
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
        outer.addWidget(self.audit_title)
        outer.addWidget(self.audit_meta)

        content = QHBoxLayout()
        content.setSpacing(18)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)
        passage_box = self.create_passage_box()
        passage_box.setMaximumHeight(145)
        left_layout.addWidget(passage_box)

        self.audit_figure = Figure(figsize=(10, 5), tight_layout=True)
        self.audit_canvas = FigureCanvas(self.audit_figure)
        self.audit_ax = self.audit_figure.add_subplot(111)
        audit_stack_widget = QWidget()
        audit_stack_widget.setMinimumHeight(360)
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
                check = QCheckBox(effect)
                self.audit_checks.append((gui_name, check))
                group_layout.addWidget(check)
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
        previous = QPushButton("Previous Segment")
        next_button = QPushButton("Next Segment")
        next_button.setObjectName("PrimaryButton")
        self.audit_play_button.clicked.connect(lambda: self.play_current_audit_segment(self.audit_play_button))
        previous.clicked.connect(self.previous_audit_segment)
        next_button.clicked.connect(self.next_audit_segment)
        self.audit_previous_button = previous
        self.audit_next_button = next_button
        actions.addWidget(self.audit_play_button)
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
            use_full_file=self.start_full_file_check.isChecked(),
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
            self._show_page(PAGE_LOAD)
        elif page == PAGE_REVIEW:
            self._show_page(PAGE_BOUNDARIES)
        elif page == PAGE_AUDIT:
            if self.audit_index > 0:
                self.store_current_audit()
                self.audit_index -= 1
                self.load_audit_segment()
            else:
                self._show_page(PAGE_REVIEW)
        elif page == PAGE_SAVE:
            if self.result is not None and len(self.audit_segments()) > 0:
                self._show_page(PAGE_AUDIT)
            else:
                self._show_page(PAGE_REVIEW)

    def go_to_load_page(self) -> None:
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
            self.result = None
            self.audit_by_segment_number = {}
            self.audit_index = 0
            if self.settings.use_full_file:
                self.analysis_region = (0, len(self.signal.raw_audio))
            self._show_page(PAGE_BOUNDARIES)

    def select_single_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select WAV", str(Path.home()), "WAV files (*.wav *.WAV);;All files (*)")
        if path:
            self.set_queue([Path(path)])

    def select_multiple_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Select WAV files", str(Path.home()), "WAV files (*.wav *.WAV);;All files (*)")
        if paths:
            self.set_queue([Path(path) for path in paths])

    def select_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select folder of WAV files", str(Path.home()))
        if not directory:
            return
        paths = sorted(
            path
            for path in Path(directory).rglob("*")
            if path.is_file() and path.suffix.lower() in {".wav", ".wave"}
        )
        if not paths:
            self.warn("No WAV files", "No WAV files were found in that folder.")
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
            status = "DONE" if self.output_exists_for_file(path) else "PENDING"
            self.queue_list.addItem(f"{index}. [{status}] {path.name}")
        if 0 <= self.queue_index < self.queue_list.count():
            self.queue_list.setCurrentRow(self.queue_index)
        if not self.queue:
            self.queue_label.setText("No files queued")
            return
        pending = sum(1 for path in self.queue if not self.output_exists_for_file(path))
        done = len(self.queue) - pending
        self.queue_label.setText(f"{len(self.queue)} file(s) queued  |  {pending} pending  |  {done} already saved")

    def start_next_queued_file(self) -> None:
        if not self.queue:
            self.warn("No files", "Select a WAV file or queue first.")
            return
        next_index = self.next_pending_index()
        if next_index is None:
            self.warn("Queue complete", "Every queued WAV already has a segment CSV in the output folder.")
            self.refresh_queue_display()
            return
        self.queue_index = next_index
        self.load_current_file()

    def open_selected_queue_file(self) -> None:
        if not self.queue:
            self.warn("No files", "Select a WAV file or queue first.")
            return
        selected_index = self.queue_list.currentRow()
        if selected_index < 0 or selected_index >= len(self.queue):
            self.warn("No file selected", "Select a file in the queue first.")
            return
        self.queue_index = selected_index
        if self.output_exists_for_file(self.queue[self.queue_index]) and self.load_saved_current_file():
            return
        self.load_current_file()

    def next_pending_index(self) -> int | None:
        for index in range(self.queue_index + 1, len(self.queue)):
            if not self.output_exists_for_file(self.queue[index]):
                return index
        return None

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

    def output_exists_for_file(self, path: Path) -> bool:
        return self.output_path_for_file(path).exists()

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
        self.audit_by_segment_number = {}
        self.audit_index = 0
        self.loaded_from_saved_output = False
        self.last_saved_path = None
        self._show_page(PAGE_BOUNDARIES)

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
        self.noise_region = self.saved_noise_region(metadata, result)
        self.analysis_region = self.saved_analysis_region(metadata, result)
        self.apply_analysis_region_to_saved_result(self.result, self.analysis_region)
        self.result.noise_region = self.noise_region
        self.boundary_mode = "noise"
        self.boundary_clicks = []
        self.audit_by_segment_number = audit_by_segment_number
        self.audit_index = 0
        self.loaded_from_saved_output = True
        self.last_saved_path = output_path
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
        required = {"segment_type", "onset_seconds_absolute", "offset_seconds_absolute", "audit_json"}
        missing = required - set(saved.columns)
        if missing:
            raise ValueError(f"Missing saved segment columns: {', '.join(sorted(missing))}")
        saved = saved.sort_values("onset_seconds_absolute")
        fs = signal.sample_rate
        speech_events: list[list[int]] = []
        pause_events: list[list[int]] = []
        segment_audits: dict[int, dict[str, object]] = {}
        absolute_events: list[tuple[str, int, int, object]] = []
        for _, row in saved.iterrows():
            segment_type = str(row["segment_type"]).strip().lower()
            if segment_type not in {"speech", "pause"}:
                continue
            start = int(round(float(row["onset_seconds_absolute"]) * fs))
            end = int(round(float(row["offset_seconds_absolute"]) * fs))
            start = max(0, min(len(signal.raw_audio), start))
            end = max(start + 1, min(len(signal.raw_audio), end))
            absolute_events.append((segment_type, start, end, row.get("audit_json", "NA")))
        if not absolute_events:
            raise ValueError("No speech or pause rows were found in the saved CSV.")
        analysis_start = min(start for _, start, _, _ in absolute_events)
        analysis_end = max(end for _, _, end, _ in absolute_events)
        segment_number = 0
        for segment_type, start, end, audit_json in absolute_events:
            segment_number += 1
            relative = [start - analysis_start, end - analysis_start]
            if segment_type == "speech":
                speech_events.append(relative)
            else:
                pause_events.append(relative)
            segment_audits[segment_number] = self.parse_saved_audit_json(audit_json)
        signal.analysis_region = (analysis_start, analysis_end)
        threshold_curve = np.full(max(1, analysis_end - analysis_start), np.nan)
        result = SpaResult(
            settings=self.settings,
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
        )
        return result, segment_audits

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

    def boundary_next_or_run_spa(self) -> None:
        if self.loaded_from_saved_output and self.result is not None:
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
        self.result = None
        self.audit_by_segment_number = {}
        self.update_boundary_controls()
        self.plot_boundaries()

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
        self.result = None
        self.audit_by_segment_number = {}
        self.update_boundary_controls()
        self.plot_boundaries()

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
        self.review_ax_norm.plot(x[display], envelope[display], color="#2f80ed", linewidth=0.75, label="normalized")
        self.review_ax_norm.plot(x[display], threshold[display], color="#c43c39", linewidth=1.0, label="threshold")
        self.review_ax_raw.plot(x[display], raw[display], color="#4a4f53", linewidth=0.55, label="original")
        for idx, (start, end) in enumerate(self.result.speech_events_samples):
            self.review_ax_raw.axvspan((analysis_start + start) / fs, (analysis_start + end) / fs, color="#1f9d55", alpha=0.24, label="speech" if idx == 0 else None)
        for idx, (start, end) in enumerate(self.result.pause_events_samples):
            self.review_ax_raw.axvspan((analysis_start + start) / fs, (analysis_start + end) / fs, color="#2f80ed", alpha=0.25, label="pause" if idx == 0 else None)
        self.review_ax_norm.set_title("Normalized signal and threshold")
        self.review_ax_raw.set_title("Original waveform with detected segments")
        self.review_ax_norm.set_ylabel("Normalized")
        self.review_ax_raw.set_ylabel("Original")
        self.review_ax_raw.set_xlabel("Time (seconds)")
        self.style_axes(self.review_ax_norm, self.review_ax_raw)
        self.add_legend(self.review_ax_norm)
        self.add_legend(self.review_ax_raw)
        speech_count = len(self.result.speech_events_samples)
        pause_count = len(self.result.pause_events_samples)
        self.review_counts.setText(
            f"{speech_count} speech segments  |  {pause_count} pauses  |  Analysis {analysis_start / fs:.3f}s to {analysis_end / fs:.3f}s"
        )
        self.review_next_button.setText("Next Page" if self.loaded_from_saved_output else "Confirm Segmentation")
        self.set_playback_range(analysis_start, analysis_end, "Analysis Region", reset_position=True)
        self.review_canvas.draw_idle()
        self.update_playback_markers()

    def confirm_segmentation(self) -> None:
        if self.result is None:
            return
        if len(self.audit_segments()) == 0:
            self._show_page(PAGE_SAVE)
            return
        self.audit_index = 0
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
        legacy_issues = set(data.get("issues", []))
        for gui_name, check in self.audit_checks:
            check.setChecked((gui_name, check.text()) in selected_effects or check.text() in legacy_issues)
        self.audit_previous_button.setEnabled(self.audit_index > 0)
        self.audit_next_button.setText("Continue To Save" if self.audit_index == len(segments) - 1 else "Next Segment")
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
        self.audit_ax.plot(x[display], audio[display], color="#4a4f53", linewidth=0.65)
        self.audit_ax.set_xlabel("Time (seconds)")
        self.audit_ax.set_ylabel("Original amplitude")
        self.audit_ax.grid(True, alpha=0.25)
        duration = (end - start) / fs
        self.audit_title.setText(f"Audit {segment_type.title()} Segment {self.audit_index + 1} of {len(segments)}")
        self.audit_meta.setText(f"Type {segment_type}  |  Onset {start / fs:.3f}s  |  Offset {end / fs:.3f}s  |  Duration {duration:.3f}s")
        self.set_playback_range(start, end, f"{segment_type.title()} Segment {self.audit_index + 1}", reset_position=True)
        self.audit_canvas.draw_idle()
        self.update_playback_markers()

    def store_current_audit(self) -> None:
        if self.result is None or len(self.audit_segments()) == 0:
            return
        segment_number = self.audit_index + 1
        selected_effects = [
            {"gui_name": gui_name, "effect": check.text()}
            for gui_name, check in self.audit_checks
            if check.isChecked()
        ]
        self.audit_by_segment_number[segment_number] = {
            "selected_effects": selected_effects,
        }

    def previous_audit_segment(self) -> None:
        if self.audit_index <= 0:
            return
        self.store_current_audit()
        self.audit_index -= 1
        self.load_audit_segment()

    def next_audit_segment(self) -> None:
        if self.result is None:
            return
        self.store_current_audit()
        if self.audit_index >= len(self.audit_segments()) - 1:
            self._show_page(PAGE_SAVE)
            return
        self.audit_index += 1
        self.load_audit_segment()

    def audit_segments(self) -> list[dict[str, object]]:
        if self.result is None:
            return []
        analysis_start = int(self.result.analysis_region[0])
        segments: list[dict[str, object]] = []
        for start, end in np.asarray(self.result.speech_events_samples, dtype=int):
            segments.append({"segment_type": "speech", "start": analysis_start + int(start), "end": analysis_start + int(end)})
        for start, end in np.asarray(self.result.pause_events_samples, dtype=int):
            segments.append({"segment_type": "pause", "start": analysis_start + int(start), "end": analysis_start + int(end)})
        segments.sort(key=lambda segment: (int(segment["start"]), 0 if segment["segment_type"] == "speech" else 1))
        return segments

    def prepare_save_page(self) -> None:
        if self.result is None:
            return
        speech_count = len(self.result.speech_events_samples)
        pause_count = len(self.result.pause_events_samples)
        self.save_summary.setText(
            f"Ready to save {self.signal.path.name if self.signal else ''}. "
            f"The output will include {speech_count} speech rows and {pause_count} pause rows with absolute timestamps."
        )
        output_path = self.output_path_for_file(self.result.signal_path)
        self.save_path_label.setText(f"Output file: {output_path}")

    def save_current_file(self) -> None:
        if self.result is None:
            return
        self.output_run_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_path_for_file(self.result.signal_path)
        try:
            self.last_saved_path = export_segment_audit_csv(self.result, output_path, self.audit_by_segment_number)
            self.save_current_metadata()
        except Exception as exc:
            self.warn("Save failed", str(exc))
            return
        self.refresh_queue_display()
        next_index = self.next_pending_index()
        if next_index is not None:
            self.queue_index = next_index
            self.load_current_file()
            return
        self.signal = None
        self.result = None
        self.noise_region = None
        self.analysis_region = None
        self.audit_by_segment_number = {}
        self.queue_index = len(self.queue) - 1 if self.queue else -1
        self.refresh_queue_display()
        self._show_page(PAGE_LOAD)
        QMessageBox.information(self, "Saved", f"Saved output to {self.last_saved_path}")

    def save_current_metadata(self) -> None:
        if self.signal is None or self.result is None:
            return
        metadata = {
            "file_name": self.signal.path.name,
            "source_path": str(self.signal.path),
            "sample_rate": int(self.signal.sample_rate),
            "noise_region_samples": self.region_to_json(self.noise_region),
            "analysis_region_samples": self.region_to_json(self.result.analysis_region),
            "settings": {
                "speech_threshold_ms": float(self.settings.speech_threshold_ms),
                "pause_threshold_ms": float(self.settings.pause_threshold_ms),
                "sd_multiplier": float(self.settings.sd_multiplier),
                "adaptive": bool(self.settings.adaptive),
                "use_full_file": bool(self.settings.use_full_file),
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
            play_audio(self.signal.raw_audio[play_start:end], self.signal.sample_rate)
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
        self.playback_current_sample = self.playback_started_sample + int(round(elapsed * self.signal.sample_rate))
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
            label_text = f"{self.playback_label}: {elapsed_text} / {duration_text}  ({absolute_text})"
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
                    play_audio(self.signal.raw_audio[self.playback_current_sample:end], self.signal.sample_rate)
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

    def warn(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
