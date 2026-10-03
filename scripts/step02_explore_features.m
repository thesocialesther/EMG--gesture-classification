%% STEP 2: Extract and inspect RMS features
clear;
clc;
close all;

%% 1. Load the recording
filePath = ...
    fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data', 'S1_E1_A1.mat');

data = load(filePath);

emg = double(data.emg);
labels = double(data.restimulus(:));
fs = 2000;

assert(size(emg, 1) == numel(labels), ...
    'EMG samples and labels must have the same length.');

assert(all(isfinite(emg(:))), ...
    'The recording contains missing or nonfinite EMG values.');

%% 2. Set the analysis window
% 200 milliseconds contains 400 samples at 2000 Hz.
windowDuration = 0.200;
windowSamples = round(windowDuration * fs);

% Use separate, non-overlapping windows for this exploration.
numberOfWindows = floor(size(emg, 1) / windowSamples);
numberOfChannels = size(emg, 2);

rmsFeatures = zeros(numberOfWindows, numberOfChannels);
windowLabels = nan(numberOfWindows, 1);
windowTimes = zeros(numberOfWindows, 1);

%% 3. Calculate RMS in each window
for k = 1:numberOfWindows
    firstSample = (k - 1) * windowSamples + 1;
    lastSample = firstSample + windowSamples - 1;

    segment = emg(firstSample:lastSample, :);
    segmentLabels = labels(firstSample:lastSample);

    % One RMS value for each electrode channel.
    rmsFeatures(k, :) = sqrt(mean(segment.^2, 1));

    % Centre time of this window.
    windowTimes(k) = ...
        ((firstSample + lastSample) / 2 - 1) / fs;

    % Keep a label only if the whole window has the same label.
    % Windows crossing gesture boundaries are excluded below.
    if all(segmentLabels == segmentLabels(1))
        windowLabels(k) = segmentLabels(1);
    end
end

validWindows = ~isnan(windowLabels);

fprintf('Window length: %d samples (%.0f ms)\n', ...
    windowSamples, windowDuration * 1000);
fprintf('Total windows: %d\n', numberOfWindows);
fprintf('Windows with a single label: %d\n', sum(validWindows));
fprintf('Windows crossing label boundaries: %d\n', ...
    sum(~validWindows));

%% 4. Compare the RMS timeline with the gesture labels
figure('Name', 'RMS and movement labels', 'Color', 'w');

subplot(2, 1, 1);
plot(windowTimes, rmsFeatures(:, 1));
xlabel('Time (seconds)');
ylabel('RMS amplitude');
title('Channel 1 RMS — each point summarizes 200 ms');
grid on;
axis tight;

subplot(2, 1, 2);
stairs((0:numel(labels)-1)' / fs, labels);
xlabel('Time (seconds)');
ylabel('Gesture label');
title('Movement labels — 0 means rest');
grid on;
axis tight;

%% 5. Compare average RMS patterns across gestures
gestureIDs = unique(windowLabels(validWindows));
meanRMS = zeros(numel(gestureIDs), numberOfChannels);
windowCounts = zeros(numel(gestureIDs), 1);

for g = 1:numel(gestureIDs)
    selected = validWindows & windowLabels == gestureIDs(g);

    meanRMS(g, :) = mean(rmsFeatures(selected, :), 1);
    windowCounts(g) = sum(selected);
end

disp('Number of usable windows for each gesture:');
disp(table(gestureIDs, windowCounts, ...
    'VariableNames', {'GestureID', 'WindowCount'}));

figure('Name', 'Gesture RMS patterns', 'Color', 'w');
imagesc(1:numberOfChannels, 1:numel(gestureIDs), meanRMS);
colorbar;
xlabel('Electrode channel');
ylabel('Gesture ID');
title('Average RMS for each gesture and channel');
set(gca, ...
    'YTick', 1:numel(gestureIDs), ...
    'YTickLabel', arrayfun(@num2str, gestureIDs, ...
        'UniformOutput', false));

disp('Finished. These are exploratory features, not a trained model.');