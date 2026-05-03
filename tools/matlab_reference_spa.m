function matlab_reference_spa(input_csv, output_csv, max_rows)
%MATLAB_REFERENCE_SPA Noninteractive reference runner for SPA parity checks.
%
% input_csv is produced by tools/check_audio_folder.py.  The noise window is
% reused from that CSV so Python and MATLAB operate on the same samples.

if nargin < 3
    max_rows = inf;
end

T = readtable(input_csv, 'TextType', 'string');
n = min(height(T), max_rows);
[input_dir, ~, ~] = fileparts(input_csv);
project_dir = fileparts(input_dir);

file = strings(n, 1);
variant = strings(n, 1);
status = strings(n, 1);
message = strings(n, 1);
sample_rate = zeros(n, 1);
duration_seconds = zeros(n, 1);
threshold = nan(n, 1);
speech_events = nan(n, 1);
pause_events = nan(n, 1);
pct_speech = nan(n, 1);
pct_pause = nan(n, 1);
speech_duration = nan(n, 1);
pause_duration = nan(n, 1);
total_duration = nan(n, 1);

for row = 1:n
    file(row) = T.file(row);
    variant(row) = T.variant(row);
    try
        [summary, fs, duration] = run_one_reference(T(row, :), project_dir);
        status(row) = "ok";
        message(row) = "";
        sample_rate(row) = fs;
        duration_seconds(row) = duration;
        threshold(row) = summary.threshold;
        speech_events(row) = summary.speech_events;
        pause_events(row) = summary.pause_events;
        pct_speech(row) = summary.pct_speech;
        pct_pause(row) = summary.pct_pause;
        speech_duration(row) = summary.speech_duration;
        pause_duration(row) = summary.pause_duration;
        total_duration(row) = summary.total_duration;
    catch ME
        status(row) = "error";
        message(row) = string(getReport(ME, 'extended', 'hyperlinks', 'off'));
    end
end

Out = table(file, variant, status, sample_rate, duration_seconds, threshold, ...
    speech_events, pause_events, pct_speech, pct_pause, speech_duration, ...
    pause_duration, total_duration, message);
writetable(Out, output_csv);
end

function [summary, Fs, duration] = run_one_reference(row, project_dir)
filename = resolve_audio_path(char(row.file), project_dir);
variant = char(row.variant);
sd_multiplier = 3.0;
speech_threshold_ms = 25.0;
pause_threshold_ms = 100.0;

[audio, Fs] = audioread(filename);
if size(audio, 2) > 1
    audio = mean(audio, 2);
end

function filename = resolve_audio_path(filename, project_dir)
if isfile(filename)
    return
end

if ~isfolder(project_dir)
    return
end

candidate = fullfile(project_dir, filename);
if isfile(candidate)
    filename = candidate;
    return
end

candidate = fullfile(fileparts(project_dir), filename);
if isfile(candidate)
    filename = candidate;
end
end
duration = length(audio) / Fs;

sig = detrend(audio, 'constant');
sig_rect = abs(sig);
[fb, fa] = butter(5, 30 / (Fs / 2));
sig_f = filtfilt(fb, fa, sig_rect);
sig_original = sig_f;
sig_k = sig_f;
sig_k(sig_k <= 0) = 0.001;
sig_f = sig_k / max(sig_k) * 100;

noise_start = max(1, round(row.noise_start_seconds * Fs) + 1);
noise_end = max(noise_start, round(row.noise_end_seconds * Fs));
noise_end = min(length(sig_f), noise_end);
noise_section = sig_f(noise_start:noise_end);
noise_mean = mean(noise_section);
noise_std = std(noise_section);
base_threshold = noise_mean + noise_std * sd_multiplier;

if strcmpi(variant, 'adaptive')
    [sig_work, threshold_curve] = adaptive_curve(sig_f, noise_mean, noise_std, sd_multiplier);
    sig_original_work = sig_original(1:length(sig_work));
else
    sig_work = sig_f;
    sig_original_work = sig_original;
    threshold_curve = ones(length(sig_work), 1) * base_threshold;
end

[speech_events_mat, pause_events_mat] = detect_events_reference( ...
    sig_work, threshold_curve, Fs, speech_threshold_ms, pause_threshold_ms);

speech_durations = (speech_events_mat(:, 2) - speech_events_mat(:, 1)) / Fs;
pause_durations = (pause_events_mat(:, 2) - pause_events_mat(:, 1)) / Fs;
speech_duration = sum(speech_durations);
pause_duration = sum(pause_durations);
total_duration = speech_duration + pause_duration;
if total_duration <= 0
    total_duration = length(sig_work) / Fs;
end

summary.threshold = base_threshold;
summary.speech_events = size(speech_events_mat, 1);
summary.pause_events = size(pause_events_mat, 1);
summary.pct_speech = speech_duration / total_duration * 100;
summary.pct_pause = pause_duration / total_duration * 100;
summary.speech_duration = speech_duration;
summary.pause_duration = pause_duration;
summary.total_duration = total_duration;

% Keep this variable live so MATLAB performs the same filtered-envelope
% preparation as the original output path; summary checks do not need it.
if isempty(sig_original_work)
    summary.threshold = summary.threshold + 0;
end
end

function [sig_work, threshold_curve] = adaptive_curve(sig_f, noise_mean, noise_std, sd_multiplier)
number_blocks = 35;
blocklength = floor(length(sig_f) / number_blocks);
if blocklength < 1
    sig_work = sig_f;
    threshold_curve = ones(length(sig_f), 1) * (noise_mean + noise_std * sd_multiplier + std(sig_f) / 3);
    return
end

xxaxisarray = zeros(number_blocks, 1);
ll = zeros(number_blocks, 1);
for idx = 0:number_blocks - 1
    block = sig_f((idx * blocklength) + 1:(idx + 1) * blocklength);
    threshold_block = std(block) / 3 + noise_mean + noise_std * sd_multiplier;
    xxaxisarray(idx + 1) = threshold_block;
    ll(idx + 1) = (idx * blocklength) + round(blocklength / 2);
end
interp_sequence = 1:number_blocks * blocklength;
threshold_curve = interp1(ll, xxaxisarray, interp_sequence, 'spline')';
sig_work = sig_f(1:number_blocks * blocklength);
end

function [speech_events, pause_events] = detect_events_reference(sig_f, threshold_curve, Fs, speech_ms, pause_ms)
I_D_max = find(sig_f > threshold_curve);
if isempty(I_D_max)
    speech_events = zeros(0, 2);
    pause_events = zeros(0, 2);
    return
end

min_mat = sig_f;
min_mat(I_D_max) = 0;
onset_min = [];
offset_min = [];

for idx = 1:length(min_mat) - 1
    if min_mat(idx) == 0 && min_mat(idx + 1) ~= 0
        onset_min = [onset_min idx + 1]; %#ok<AGROW>
    end

    if min_mat(idx + 1) == 0 && min_mat(idx) ~= 0
        if ~isempty(onset_min)
            if onset_min(end) < idx
                offset_min = [offset_min idx]; %#ok<AGROW>
            end
        end
    end

    if idx == length(min_mat) - 1 && min_mat(idx) ~= 0
        if ~isempty(onset_min)
            onset_min(end) = [];
        end
    end
end

pause_threshold_points = (pause_ms / 1000) * Fs;
speech_threshold_points = (speech_ms / 1000) * Fs;

if ~isempty(onset_min)
    pair_count = min(length(onset_min), length(offset_min));
    onset_min = onset_min(1:pair_count);
    offset_min = offset_min(1:pair_count);
end

if length(onset_min) > 1 && length(offset_min) > 1
    speech_event_dur = onset_min(2:end) - offset_min(1:end - 1);
    h = find(speech_event_dur < speech_threshold_points);
    if ~isempty(h)
        last_small_speech_I = offset_min(h(end));
        onset_min(h + 1) = [];
        offset_min(h) = [];
        if ~isempty(offset_min) && last_small_speech_I > offset_min(end)
            onset_min(end) = [];
            offset_min(end) = [];
        end
    end
end

short_pause_I = find((offset_min - onset_min) <= pause_threshold_points);
onset_min(short_pause_I) = [];
offset_min(short_pause_I) = [];

pause_events = [onset_min(:), offset_min(:)];
onset_max = offset_min + 1;
offset_max = onset_min - 1;
onset_max = [I_D_max(1), onset_max];
offset_max = [offset_max, length(sig_f)];
speech_events = [onset_max(:), offset_max(:)];
speech_events = speech_events(speech_events(:, 2) > speech_events(:, 1), :);
end
