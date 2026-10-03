%% STEP 6: Replay EMG through the saved classifier
clear;
clc;
close all;

%% 1. Load model and recording
projectFolder = fileparts(fileparts(mfilename('fullpath')));

saved = load(fullfile(projectFolder, ...
    'models', 'emg_selected_model.mat'));
model = saved.emgModel;

data = load(fullfile(projectFolder, ...
    'data', 'S1_E1_A1.mat'));

emg = double(data.emg);
labels = double(data.restimulus(:));
repetitions = double(data.rerepetition(:));

fs = model.fs;
windowSamples = model.windowSamples;
hopSamples = round(0.050 * fs);  % Update every 50 ms

assert(size(emg, 2) == model.numberOfChannels, ...
    'Recording channel count does not match the model.');

%% 2. Choose one movement from repetition 5
% Labels are used ONLY to locate a useful demonstration segment.
% They are not passed to the classifier.
targetGesture = 1;
targetRepetition = 5;

targetSamples = find( ...
    labels == targetGesture & repetitions == targetRepetition);

assert(~isempty(targetSamples), ...
    'Requested gesture/repetition was not found.');

% Include two seconds before and after the movement.
contextSamples = round(2 * fs);

firstSample = max(1, targetSamples(1) - contextSamples);
lastSample = min(size(emg, 1), ...
    targetSamples(end) + contextSamples);

segment = emg(firstSample:lastSample, :);
segmentLabels = labels(firstSample:lastSample);

time = (0:size(segment, 1)-1)' / fs;

% Context may contain samples with other repetition identifiers.
% This is a replay demonstration, not a new held-out accuracy score.

%% 3. Create the display
figure('Name', 'EMG replay', 'Color', 'w');

subplot(2, 1, 1);
plot(time, segment(:, 1));
xlabel('Time within replay (seconds)');
ylabel('Recorded amplitude');
title('Recorded EMG — channel 1');
grid on;
xlim([0 time(end)]);

subplot(2, 1, 2);
stairs(time, segmentLabels, 'k', 'LineWidth', 1.5);
hold on;

predictionLine = plot(NaN, NaN, 'ro-', ...
    'MarkerSize', 3, 'LineWidth', 1);

xlabel('Time within replay (seconds)');
ylabel('Gesture ID');
title('Predictions use only the preceding 200 ms');
legend({'Recorded label', 'Model prediction'}, ...
    'Location', 'best');
ylim([min(model.classIDs)-1, max(model.classIDs)+1]);
xlim([0 time(end)]);
grid on;

%% 4. Process the recording one window at a time
windowEnds = windowSamples:hopSamples:size(segment, 1);

predictionTimes = zeros(numel(windowEnds), 1);
predictions = zeros(numel(windowEnds), 1);

% Approximate display pacing; computation adds some extra delay.
% Set to 0 to process as quickly as possible.
displayPause = hopSamples / fs;

for k = 1:numel(windowEnds)
    windowEnd = windowEnds(k);
    windowStart = windowEnd - windowSamples + 1;

    % Only current and earlier samples are available to this prediction.
    window = segment(windowStart:windowEnd, :);

    % Match the training feature order exactly.
    rmsValues = sqrt(mean(window.^2, 1));
    mavValues = mean(abs(window), 1);
    wlValues = sum(abs(diff(window, 1, 1)), 1);

    features = [rmsValues, mavValues, wlValues];

    % Apply the saved transformation and normalization.
    featuresLog = log10(max(features, model.featureFloor));

    featuresScaled = ...
        (featuresLog - model.featureMean) ./ model.featureStd;

    predictions(k) = predict(model.classifier, featuresScaled);

    % Timestamp at the END of the window, when prediction is available.
    predictionTimes(k) = (windowEnd - 1) / fs;

    set(predictionLine, ...
        'XData', predictionTimes(1:k), ...
        'YData', predictions(1:k));

    drawnow;
    pause(displayPause);
end

%% 5. Save the replay output
resultsFolder = fullfile(projectFolder, 'results');

if ~isfolder(resultsFolder)
    mkdir(resultsFolder);
end

save(fullfile(resultsFolder, 'emg_replay.mat'), ...
    'predictionTimes', 'predictions', ...
    'segmentLabels', 'fs', 'firstSample', ...
    'windowSamples', 'hopSamples');

disp('Replay complete.');
disp('Black line: recorded gesture label.');
disp('Red points: predictions from EMG alone.');