from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from spa_core.audio import export_speech_segments, play_audio, read_wav
from spa_core.export import export_excel
from spa_core.models import SpaSettings, SpaSignal
from spa_core.segmentation import run_spa


_MPL_CONFIG_DIR = Path(__file__).resolve().parents[1] / ".matplotlib"
_MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))


try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except Exception as exc:  # pragma: no cover - import-time GUI dependency check
    raise RuntimeError("PySide6 and matplotlib are required for the SPA GUI. Install requirements.txt first.") from exc


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SPA Python")
        self.resize(1400, 820)

        self.signal: SpaSignal | None = None
        self.result = None
        self.noise_region: tuple[int, int] | None = None
        self.analysis_region: tuple[int, int] | None = None
        self.manual_threshold: float | None = None
        self.selection_mode: str | None = None
        self.candidate_mode: str | None = None
        self.selection_candidate: tuple[int, int] | float | None = None
        self.pending_clicks: list[float] = []

        self.figure = Figure(figsize=(10, 7), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax_norm = self.figure.add_subplot(211)
        self.ax_raw = self.figure.add_subplot(212, sharex=self.ax_norm)
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)

        controls = self._build_controls()
        layout = QHBoxLayout()
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(controls)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self._set_status("Load a WAV file to begin.")
        self._plot_empty()

    def _build_controls(self) -> QWidget:
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        panel.setFixedWidth(320)
        outer = QVBoxLayout(panel)
        title = QLabel("SPA")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 34px; font-weight: 700; color: #d71920; background: #26dce0; padding: 10px;")
        outer.addWidget(title)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        outer.addWidget(self.status_label)

        form = QFormLayout()
        self.threshold_mode = QComboBox()
        self.threshold_mode.addItems(["automatic", "manual"])
        form.addRow("Threshold", self.threshold_mode)

        self.adaptive_check = QCheckBox("Adaptive")
        self.adaptive_check.setChecked(True)
        form.addRow("Adaptive", self.adaptive_check)

        self.sd_spin = QDoubleSpinBox()
        self.sd_spin.setRange(0.0, 20.0)
        self.sd_spin.setDecimals(2)
        self.sd_spin.setValue(3.0)
        form.addRow("SD multiplier", self.sd_spin)

        self.speech_spin = QDoubleSpinBox()
        self.speech_spin.setRange(0.0, 5000.0)
        self.speech_spin.setDecimals(1)
        self.speech_spin.setValue(25.0)
        self.speech_spin.setSuffix(" ms")
        form.addRow("Speech min", self.speech_spin)

        self.pause_spin = QDoubleSpinBox()
        self.pause_spin.setRange(0.0, 5000.0)
        self.pause_spin.setDecimals(1)
        self.pause_spin.setValue(250.0)
        self.pause_spin.setSuffix(" ms")
        form.addRow("Pause min", self.pause_spin)

        self.full_file_check = QCheckBox("Use full file")
        self.full_file_check.setChecked(False)
        form.addRow("Trim", self.full_file_check)
        outer.addLayout(form)

        buttons = [
            ("Load WAV", self.load_file),
            ("Select Noise Region", self.select_noise_region),
            ("Select Analysis Region", self.select_analysis_region),
            ("Select Manual Threshold", self.select_manual_threshold),
        ]
        for label, callback in buttons:
            button = QPushButton(label)
            button.clicked.connect(callback)
            button.setMinimumHeight(34)
            outer.addWidget(button)

        self.confirm_button = QPushButton("Confirm Selection")
        self.confirm_button.clicked.connect(self.confirm_selection)
        self.confirm_button.setMinimumHeight(34)
        self.confirm_button.setEnabled(False)
        outer.addWidget(self.confirm_button)

        self.redo_button = QPushButton("Redo Selection")
        self.redo_button.clicked.connect(self.redo_selection)
        self.redo_button.setMinimumHeight(34)
        self.redo_button.setEnabled(False)
        outer.addWidget(self.redo_button)

        action_buttons = [
            ("Run SPA", self.run_analysis),
            ("Play Full File", self.play_full_file),
            ("Play Analysis Region", self.play_analysis_region),
            ("Save Excel", self.save_excel),
            ("Export Speech Segments", self.export_segments),
        ]
        for label, callback in action_buttons:
            button = QPushButton(label)
            button.clicked.connect(callback)
            button.setMinimumHeight(34)
            outer.addWidget(button)

        outer.addStretch(1)
        return panel

    def _settings(self) -> SpaSettings:
        return SpaSettings(
            speech_threshold_ms=self.speech_spin.value(),
            pause_threshold_ms=self.pause_spin.value(),
            sd_multiplier=self.sd_spin.value(),
            threshold_mode=self.threshold_mode.currentText(),
            adaptive=self.adaptive_check.isChecked(),
            variable_mode="ONE TIME",
            iterations=1,
            amplitude_increment=1.0,
            time_increment_ms=25.0,
            use_full_file=self.full_file_check.isChecked(),
        )

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _warn(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)

    def _plot_empty(self) -> None:
        self.ax_norm.clear()
        self.ax_raw.clear()
        self.ax_norm.set_title("No file loaded")
        self.ax_norm.set_ylabel("Normalized amplitude")
        self.ax_raw.set_xlabel("Time (seconds)")
        self.ax_raw.set_ylabel("Original amplitude")
        self.canvas.draw_idle()

    def _plot_signal(self) -> None:
        if self.signal is None:
            self._plot_empty()
            return
        fs = self.signal.sample_rate
        x = np.arange(len(self.signal.raw_audio)) / fs
        self.ax_norm.clear()
        self.ax_raw.clear()
        self.ax_norm.plot(x, self.signal.normalized_envelope, color="#d6d000", linewidth=0.8, label="normalized")
        self.ax_raw.plot(x, self.signal.raw_audio, color="#6f6f6f", linewidth=0.55, label="original")
        self.ax_norm.set_title(f"{self.signal.path.name} - normalized signal")
        self.ax_raw.set_title("Original waveform")
        self.ax_norm.set_ylabel("Normalized amplitude")
        self.ax_raw.set_ylabel("Original amplitude")
        self.ax_raw.set_xlabel("Time (seconds)")
        self._style_axes()
        if self.noise_region is not None:
            self._draw_region(self.noise_region, "#4c78a8", "noise")
        if self.analysis_region is not None and not self.full_file_check.isChecked():
            self._draw_region(self.analysis_region, "#59a14f", "analysis")
        if self.selection_candidate is not None and isinstance(self.selection_candidate, tuple):
            self._draw_region(self.selection_candidate, "#f28e2b", f"proposed {self.candidate_mode}")
        if self.manual_threshold is not None:
            self.ax_norm.axhline(self.manual_threshold, color="#e15759", linewidth=1.3, label="manual threshold")
        if isinstance(self.selection_candidate, float):
            self.ax_norm.axhline(self.selection_candidate, color="#f28e2b", linewidth=1.3, label="proposed threshold")
        self._legend_if_needed(self.ax_norm)
        self._legend_if_needed(self.ax_raw)
        self.canvas.draw_idle()

    def _draw_region(self, region: tuple[int, int], color: str, label: str) -> None:
        if self.signal is None:
            return
        fs = self.signal.sample_rate
        start, end = sorted(region)
        for ax in (self.ax_norm, self.ax_raw):
            ax.axvspan(start / fs, end / fs, color=color, alpha=0.2, label=label)

    def _style_axes(self) -> None:
        for ax in (self.ax_norm, self.ax_raw):
            ax.grid(True, alpha=0.25)

    def _legend_if_needed(self, ax) -> None:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = {label: handle for handle, label in zip(handles, labels) if label}
            if by_label:
                ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    def _plot_result(self) -> None:
        if self.signal is None or self.result is None:
            self._plot_signal()
            return
        fs = self.signal.sample_rate
        start, end = self.result.analysis_region
        display_end = min(end, start + len(self.result.threshold_curve))
        envelope = self.signal.normalized_envelope[start:display_end]
        raw = self.signal.raw_audio[start:display_end]
        x = (start + np.arange(len(envelope))) / fs
        self.ax_norm.clear()
        self.ax_raw.clear()
        self.ax_norm.plot(x, envelope, color="#d6d000", linewidth=0.75, label="normalized")
        self.ax_norm.plot(x, self.result.threshold_curve, color="#e15759", linewidth=1.1, label="threshold")
        self.ax_raw.plot(x, raw, color="#6f6f6f", linewidth=0.55, label="original")
        for idx, (event_start, event_end) in enumerate(self.result.pause_events_samples):
            label = "pause" if idx == 0 else None
            for ax in (self.ax_norm, self.ax_raw):
                ax.axvspan((start + event_start) / fs, (start + event_end) / fs, color="#4c78a8", alpha=0.28, label=label)
        for idx, (event_start, event_end) in enumerate(self.result.speech_events_samples):
            label = "speech" if idx == 0 else None
            for ax in (self.ax_norm, self.ax_raw):
                ax.axvspan((start + event_start) / fs, (start + event_end) / fs, color="#59a14f", alpha=0.08, label=label)
        self.ax_norm.set_title(f"{self.signal.path.name} - normalized signal and threshold")
        self.ax_raw.set_title("Original waveform with detected pauses")
        self.ax_norm.set_ylabel("Normalized amplitude")
        self.ax_raw.set_ylabel("Original amplitude")
        self.ax_raw.set_xlabel("Time (seconds)")
        self._style_axes()
        self._legend_if_needed(self.ax_norm)
        self._legend_if_needed(self.ax_raw)
        self.canvas.draw_idle()

    def load_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load WAV file", str(Path.home()), "WAV files (*.wav *.WAV);;All files (*)")
        if not path:
            return
        try:
            self.signal = read_wav(path)
        except Exception as exc:
            self._warn("Load failed", str(exc))
            return
        self.result = None
        self.noise_region = None
        self.analysis_region = None
        self.manual_threshold = None
        self._clear_selection_state()
        self._set_status(f"Loaded {Path(path).name}. Select a noise region, then an analysis region, then run SPA.")
        self._plot_signal()

    def select_noise_region(self) -> None:
        if self.signal is None:
            self._warn("No file", "Load a WAV file first.")
            return
        self._start_region_selection("noise", "Click the beginning and end of a noise/pause-only region.")

    def select_analysis_region(self) -> None:
        if self.signal is None:
            self._warn("No file", "Load a WAV file first.")
            return
        self._start_region_selection("analysis", "Click the beginning and end of the full analysis window.")

    def select_manual_threshold(self) -> None:
        if self.signal is None:
            self._warn("No file", "Load a WAV file first.")
            return
        self._clear_selection_state()
        self._plot_signal()
        self.selection_mode = "threshold"
        self.redo_button.setEnabled(True)
        self._set_status("Click a y-value on the normalized plot to set the manual amplitude threshold.")

    def _start_region_selection(self, mode: str, message: str) -> None:
        self._clear_selection_state()
        self._plot_signal()
        self.selection_mode = mode
        self.redo_button.setEnabled(True)
        self._set_status(message)

    def _clear_selection_state(self) -> None:
        self.selection_mode = None
        self.candidate_mode = None
        self.selection_candidate = None
        self.pending_clicks = []
        if hasattr(self, "confirm_button"):
            self.confirm_button.setText("Confirm Selection")
            self.confirm_button.setEnabled(False)
        if hasattr(self, "redo_button"):
            self.redo_button.setEnabled(False)

    def _set_candidate(self, mode: str, value: tuple[int, int] | float) -> None:
        self.selection_mode = None
        self.candidate_mode = mode
        self.selection_candidate = value
        self.pending_clicks = []
        self.confirm_button.setEnabled(True)
        self.redo_button.setEnabled(True)
        if mode == "noise":
            if self.full_file_check.isChecked():
                self.confirm_button.setText("Confirm Noise + Run SPA")
            else:
                self.confirm_button.setText("Confirm Noise + Select Analysis")
        elif mode == "analysis":
            self.confirm_button.setText("Confirm Analysis + Run SPA")
        elif mode == "threshold":
            if self.full_file_check.isChecked():
                self.confirm_button.setText("Confirm Threshold + Run SPA")
            else:
                self.confirm_button.setText("Confirm Threshold + Select Analysis")

    def _on_plot_click(self, event) -> None:
        if self.signal is None or self.selection_mode is None or event.inaxes not in (self.ax_norm, self.ax_raw):
            return
        if self.selection_mode == "threshold":
            if event.inaxes != self.ax_norm:
                self._set_status("Manual threshold uses normalized amplitude. Click on the top plot.")
                return
            if event.ydata is None:
                return
            self._set_candidate("threshold", float(event.ydata))
            self._set_status(f"Proposed manual threshold: {event.ydata:.3f}. Confirm it or redo.")
            self._plot_signal()
            return

        if event.xdata is None:
            return
        self.pending_clicks.append(float(event.xdata))
        if len(self.pending_clicks) < 2:
            self._set_status("Click the second boundary.")
            return

        fs = self.signal.sample_rate
        length = len(self.signal.raw_audio)
        first, second = sorted(self.pending_clicks[:2])
        region = (
            max(0, min(length, int(round(first * fs)))),
            max(0, min(length, int(round(second * fs)))),
        )
        if region[1] <= region[0]:
            self._warn("Invalid region", "The selected region has no duration.")
        elif self.selection_mode == "noise":
            self._set_candidate("noise", region)
            self._set_status(
                f"Proposed noise region: {region[0] / fs:.3f}s to {region[1] / fs:.3f}s. Confirm it or redo."
            )
        elif self.selection_mode == "analysis":
            self._set_candidate("analysis", region)
            self._set_status(
                f"Proposed analysis region: {region[0] / fs:.3f}s to {region[1] / fs:.3f}s. Confirm it or redo."
            )
        self._plot_signal()

    def confirm_selection(self) -> None:
        if self.signal is None or self.candidate_mode is None or self.selection_candidate is None:
            return
        fs = self.signal.sample_rate
        mode = self.candidate_mode
        candidate = self.selection_candidate
        if mode == "noise" and isinstance(candidate, tuple):
            self.noise_region = candidate
            self._clear_selection_state()
            if self.full_file_check.isChecked():
                self.analysis_region = (0, len(self.signal.raw_audio))
                self._set_status("Noise region confirmed. Running SPA on the full file.")
                self.run_analysis()
            else:
                self._start_region_selection("analysis", "Noise region confirmed. Click the beginning and end of the analysis window.")
        elif mode == "analysis" and isinstance(candidate, tuple):
            self.analysis_region = candidate
            self._clear_selection_state()
            self._set_status(f"Analysis region confirmed: {candidate[0] / fs:.3f}s to {candidate[1] / fs:.3f}s. Running SPA.")
            self.run_analysis()
        elif mode == "threshold" and isinstance(candidate, float):
            self.manual_threshold = candidate
            self._clear_selection_state()
            if self.full_file_check.isChecked():
                self.analysis_region = (0, len(self.signal.raw_audio))
                self._set_status("Manual threshold confirmed. Running SPA on the full file.")
                self.run_analysis()
            else:
                self._start_region_selection(
                    "analysis",
                    "Manual threshold confirmed. Click the beginning and end of the analysis window.",
                )

    def redo_selection(self) -> None:
        if self.signal is None:
            return
        mode = self.candidate_mode or self.selection_mode
        if mode == "threshold":
            self.select_manual_threshold()
        elif mode == "noise":
            self.select_noise_region()
        elif mode == "analysis":
            self.select_analysis_region()
        else:
            self._clear_selection_state()
            self._plot_signal()

    def run_analysis(self) -> None:
        if self.signal is None:
            self._warn("No file", "Load a WAV file first.")
            return
        settings = self._settings()
        if settings.threshold_mode == "automatic" and self.noise_region is None:
            self._warn("Missing noise region", "Automatic thresholding needs a selected noise/pause-only region.")
            return
        if settings.threshold_mode == "manual" and self.manual_threshold is None:
            self._warn("Missing threshold", "Manual thresholding needs a clicked threshold value.")
            return
        if settings.use_full_file:
            analysis_region = (0, len(self.signal.raw_audio))
        else:
            analysis_region = self.analysis_region
            if analysis_region is None:
                self._warn("Missing analysis region", "Select the analysis region or enable Use full file.")
                return
        try:
            self.result = run_spa(
                self.signal,
                settings=settings,
                noise_region=self.noise_region,
                analysis_region=analysis_region,
                manual_threshold=self.manual_threshold,
            )
        except Exception as exc:
            self._warn("SPA failed", str(exc))
            return
        speech_count = int(self.result.total_matrix.iloc[-1]["Speech_events"]) if not self.result.total_matrix.empty else 0
        pause_count = int(self.result.total_matrix.iloc[-1]["Pause_events"]) if not self.result.total_matrix.empty else 0
        self._set_status(f"SPA complete: {speech_count} speech events, {pause_count} pause events.")
        self._plot_result()

    def play_full_file(self) -> None:
        if self.signal is None:
            self._warn("No file", "Load a WAV file first.")
            return
        try:
            play_audio(self.signal.raw_audio, self.signal.sample_rate)
        except Exception as exc:
            self._warn("Playback failed", str(exc))

    def play_analysis_region(self) -> None:
        if self.signal is None:
            self._warn("No file", "Load a WAV file first.")
            return
        if self.result is not None:
            start, end = self.result.analysis_region
        elif self.full_file_check.isChecked():
            start, end = 0, len(self.signal.raw_audio)
        elif self.analysis_region is not None:
            start, end = self.analysis_region
        else:
            self._warn("No analysis region", "Select an analysis region or run SPA first.")
            return
        try:
            play_audio(self.signal.raw_audio[start:end], self.signal.sample_rate)
        except Exception as exc:
            self._warn("Playback failed", str(exc))

    def save_excel(self) -> None:
        if self.result is None:
            self._warn("No result", "Run SPA before saving.")
            return
        default_name = f"{self.result.signal_path.stem}_spa.xlsx"
        path, _ = QFileDialog.getSaveFileName(self, "Save SPA Excel output", default_name, "Excel files (*.xlsx)")
        if not path:
            return
        try:
            saved_path = export_excel(self.result, path)
        except Exception as exc:
            self._warn("Save failed", str(exc))
            return
        self._set_status(f"Saved Excel output to {saved_path}.")

    def export_segments(self) -> None:
        if self.signal is None or self.result is None:
            self._warn("No result", "Run SPA before exporting segments.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Choose speech segment output folder", str(self.signal.path.parent))
        if not directory:
            return
        try:
            written = export_speech_segments(directory, self.signal, self.result.speech_events_samples)
        except Exception as exc:
            self._warn("Export failed", str(exc))
            return
        self._set_status(f"Exported {len(written)} speech segment WAV files.")


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
