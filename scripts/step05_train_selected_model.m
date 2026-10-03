%% STEP 5: Train, evaluate, and save the selected configuration
clear;
clc;
close all;

%% 1. Load the recording
projectFolder = fileparts(fileparts(mfilename('fullpath')));
data = load(fullfile(projectFolder, 'data', 'S1_E1_A1.mat'));

emg = double(data.emg);
labels = double(data.restimulus(:));
repetitions = double(data.rerepetition(:));

assert(size(emg, 1) == numel(labels), ...
    'EMG and labels have different lengths.');
assert(numel(labels) == numel(repetitions), ...
    'Labels and repetitions have different lengths.');
assert(all(isfinite(emg(:))), ...
    'EMG contains missing or nonfinite values.');

fs = 2000;
windowSamples = round(0.200 * fs);
numberOfChannels = size(emg, 2);
numberOfWindows = floor(size(emg, 1) / windowSamples);

%% 2. Extract RMS, MAV, and waveform length
X = zeros(numberOfWindows, 3 * numberOfChannels);
Y = zeros(numberOfWindows, 1);
R = zeros(numberOfWindows, 1);
keep = false(numberOfWindows, 1);

for k = 1:numberOfWindows
    firstSample = (k - 1) * windowSamples + 1;
    idx = firstSample:(firstSample + windowSamples - 1);

    segmentLabels = labels(idx);
    segmentTrials = repetitions(idx);

    sameGesture = all(segmentLabels == segmentLabels(1));
    sameTrial = all(segmentTrials == segmentTrials(1));

    if sameGesture && sameTrial && ...
            ismember(segmentTrials(1), 1:6)

        segment = emg(idx, :);

        rmsValues = sqrt(mean(segment.^2, 1));
        mavValues = mean(abs(segment), 1);
        wlValues = sum(abs(diff(segment, 1, 1)), 1);

        X(k, :) = [rmsValues, mavValues, wlValues];
        Y(k) = segmentLabels(1);
        R(k) = segmentTrials(1);
        keep(k) = true;
    end
end

X = X(keep, :);
Y = Y(keep);
R = R(keep);

%% 3. Separate training and test repetitions
trainRepetitions = [1 2 3 4];
testRepetitions = [5 6];

isTrain = ismember(R, trainRepetitions);
isTest = ismember(R, testRepetitions);

YTrain = Y(isTrain);
YTest = Y(isTest);
classIDs = unique(YTrain);

assert(~isempty(YTrain) && ~isempty(YTest), ...
    'Training or test data is empty.');
assert(isequal(classIDs, unique(YTest)), ...
    'Training and test gesture IDs do not match.');

%% 4. Learn normalization from training data only
featureFloor = 1e-12;

XTrainLog = log10(max(X(isTrain, :), featureFloor));
XTestLog = log10(max(X(isTest, :), featureFloor));

featureMean = mean(XTrainLog, 1);
featureStd = std(XTrainLog, 0, 1);
featureStd(featureStd < 1e-12) = 1;

XTrain = bsxfun(@rdivide, ...
    bsxfun(@minus, XTrainLog, featureMean), featureStd);

XTest = bsxfun(@rdivide, ...
    bsxfun(@minus, XTestLog, featureMean), featureStd);

%% 5. Train the selected classifier
classifier = fitcdiscr(XTrain, YTrain, ...
    'DiscrimType', 'linear', ...
    'Gamma', 0.1, ...
    'Prior', 'uniform');

%% 6. Evaluate on test repetitions
YPred = predict(classifier, XTest);

C = confusionmat(YTest, YPred, 'Order', classIDs);
recallPercent = 100 * diag(C) ./ sum(C, 2);

overallAccuracy = 100 * mean(YPred == YTest);
balancedAccuracy = mean(recallPercent);

active = YTest ~= 0;
movementAccuracy = ...
    100 * mean(YPred(active) == YTest(active));

fprintf('Training windows: %d\n', numel(YTrain));
fprintf('Test windows: %d\n\n', numel(YTest));

fprintf('Overall test accuracy: %.2f%%\n', overallAccuracy);
fprintf('Balanced test accuracy: %.2f%%\n', balancedAccuracy);
fprintf('Movement test accuracy: %.2f%%\n', movementAccuracy);

disp(table(classIDs, recallPercent, ...
    'VariableNames', {'GestureID', 'RecallPercent'}));

%% 7. Plot percentage confusion matrix
CPercent = 100 * bsxfun(@rdivide, C, sum(C, 2));

figure('Name', 'Selected model test results', 'Color', 'w');
imagesc(CPercent);
axis image;
caxis([0 100]);
colorbar;

tickLabels = arrayfun(@num2str, classIDs, ...
    'UniformOutput', false);

set(gca, ...
    'XTick', 1:numel(classIDs), 'XTickLabel', tickLabels, ...
    'YTick', 1:numel(classIDs), 'YTickLabel', tickLabels);

xlabel('Predicted gesture ID');
ylabel('Actual gesture ID');
title(sprintf('Selected model — balanced accuracy %.1f%%', ...
    balancedAccuracy));

%% 8. Package everything needed for future predictions
emgModel.classifier = classifier;
emgModel.fs = fs;
emgModel.windowSamples = windowSamples;
emgModel.numberOfChannels = numberOfChannels;
emgModel.featureOrder = {'RMS', 'MAV', 'WL'};
emgModel.featureFloor = featureFloor;
emgModel.featureMean = featureMean;
emgModel.featureStd = featureStd;
emgModel.classIDs = classIDs;
emgModel.trainRepetitions = trainRepetitions;
emgModel.testRepetitions = testRepetitions;
emgModel.preprocessing = ...
    'Recorded signals; log10 features; training-set standardization';

evaluation.overallAccuracy = overallAccuracy;
evaluation.balancedAccuracy = balancedAccuracy;
evaluation.movementAccuracy = movementAccuracy;
evaluation.confusionMatrix = C;
evaluation.recallPercent = recallPercent;

modelFolder = fullfile(projectFolder, 'models');
if ~isfolder(modelFolder)
    mkdir(modelFolder);
end

modelPath = fullfile(modelFolder, 'emg_selected_model.mat');
save(modelPath, 'emgModel', 'evaluation');

fprintf('\nSelected model saved to:\n%s\n', modelPath);