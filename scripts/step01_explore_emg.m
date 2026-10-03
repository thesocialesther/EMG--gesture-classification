%% STEP 1: Load and view your EMG recording
clear;
clc;
close all;

%% 1. Set the location of your recording
dataFolder = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data');
filePath = fullfile(dataFolder, 'S1_E1_A1.mat');

% Check that the folder exists.
if ~isfolder(dataFolder)
    error('Folder not found: %s', dataFolder);
end

% Check that the recording exists.
if ~isfile(filePath)
    disp('Files found in your data folder:');
    dir(dataFolder);
    error('S1_E1_A1.mat was not found in this folder.');
end

%% 2. Load the recording
data = load(filePath);

disp('Variables inside the recording:');
disp(fieldnames(data));

% Check that the variables we need are present.
assert(isfield(data, 'emg'), ...
    'The selected recording does not contain an emg variable.');

assert(isfield(data, 'restimulus'), ...
    'The selected recording does not contain restimulus labels.');

%% 3. Extract the signals and labels
emg = double(data.emg);
labels = double(data.restimulus(:));

% Sampling rate for NinaPro DB2.
fs = 2000;

numberOfSamples = size(emg, 1);
numberOfChannels = size(emg, 2);

assert(numberOfSamples > 0, 'The EMG recording is empty.');

assert(numel(labels) == numberOfSamples, ...
    'The number of movement labels does not match the EMG samples.');

% Build a time axis in seconds.
time = (0:numberOfSamples-1)' / fs;

%% 4. Print recording information
fprintf('\nRecording loaded successfully!\n');
fprintf('File: %s\n', filePath);
fprintf('EMG channels: %d\n', numberOfChannels);
fprintf('Samples: %d\n', numberOfSamples);
fprintf('Sampling rate: %d Hz\n', fs);
fprintf('Duration: %.1f seconds\n', numberOfSamples / fs);

disp('Movement labels present:');
disp(unique(labels)');

%% 5. Plot channel 1 and movement labels
% Show up to the first 30 seconds.
plotSamples = min(30 * fs, numberOfSamples);
idx = 1:plotSamples;

figure('Name', 'EMG and movement labels', 'Color', 'w');

subplot(2, 1, 1);
plot(time(idx), emg(idx, 1));
title('EMG channel 1 — first 30 seconds');
xlabel('Time (seconds)');
ylabel('Recorded amplitude');
grid on;
axis tight;

subplot(2, 1, 2);
stairs(time(idx), labels(idx), 'LineWidth', 1);
title('Movement labels — 0 means rest');
xlabel('Time (seconds)');
ylabel('Gesture label');
grid on;
axis tight;

%% 6. Plot all EMG channels
figure('Name', 'All EMG channels', 'Color', 'w');

for channel = 1:numberOfChannels
    subplot(numberOfChannels, 1, channel);
    plot(time(idx), emg(idx, channel));
    ylabel(sprintf('Ch %d', channel));
    grid on;
    axis tight;

    if channel == 1
        title('All EMG channels — first 30 seconds');
    end

    if channel == numberOfChannels
        xlabel('Time (seconds)');
    end
end

drawnow;
disp('Finished. Two figures should now be open.');