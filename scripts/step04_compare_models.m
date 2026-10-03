%% STEP 4: Compare features and classifiers using validation data
clear;
clc;
close all;

%% 1. Load data
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

%% 2. Extract three features per channel
% Column order: all RMS channels, all MAV channels, all WL channels.
features = zeros(numberOfWindows, 3 * numberOfChannels);
gesture = zeros(numberOfWindows, 1);
trial = zeros(numberOfWindows, 1);
keep = false(numberOfWindows, 1);

for k = 1:numberOfWindows
    firstSample = (k - 1) * windowSamples + 1;
    idx = firstSample:(firstSample + windowSamples - 1);

    windowLabels = labels(idx);
    windowTrials = repetitions(idx);

    sameGesture = all(windowLabels == windowLabels(1));
    sameTrial = all(windowTrials == windowTrials(1));

    % Use only development repetitions.
    allowedTrial = ismember(windowTrials(1), [1 2 3 4]);

    if sameGesture && sameTrial && allowedTrial
        segment = emg(idx, :);

        rmsValues = sqrt(mean(segment.^2, 1));
        mavValues = mean(abs(segment), 1);
        wlValues = sum(abs(diff(segment, 1, 1)), 1);

        features(k, :) = [rmsValues, mavValues, wlValues];
        gesture(k) = windowLabels(1);
        trial(k) = windowTrials(1);
        keep(k) = true;
    end
end

features = features(keep, :);
gesture = gesture(keep);
trial = trial(keep);

%% 3. Define training and validation sets
isTrain = ismember(trial, [1 2 3]);
isValidation = trial == 4;

YTrain = gesture(isTrain);
YValidation = gesture(isValidation);

assert(~isempty(YTrain) && ~isempty(YValidation), ...
    'Training or validation data is empty.');

classIDs = unique(YTrain);

assert(isequal(classIDs, unique(YValidation)), ...
    'Training and validation must contain the same gesture IDs.');

fprintf('Training windows: %d\n', numel(YTrain));
fprintf('Validation windows: %d\n', numel(YValidation));

%% 4. Define four experiments
experimentNames = {
    'RMS + diagonal LDA'
    'RMS/MAV/WL + diagonal LDA'
    'RMS + regularized LDA'
    'RMS/MAV/WL + regularized LDA'
};

featureColumns = {
    1:numberOfChannels
    1:(3 * numberOfChannels)
    1:numberOfChannels
    1:(3 * numberOfChannels)
};

overallAccuracy = zeros(4, 1);
balancedAccuracy = zeros(4, 1);
movementAccuracy = zeros(4, 1);
validationConfusions = cell(4, 1);

%% 5. Train and validate each configuration
for experiment = 1:4
    columns = featureColumns{experiment};

    XTrain = log10(max(features(isTrain, columns), 1e-12));
    XValidation = log10(max( ...
        features(isValidation, columns), 1e-12));

    % Fit normalization using training data only.
    featureMean = mean(XTrain, 1);
    featureStd = std(XTrain, 0, 1);
    featureStd(featureStd < 1e-12) = 1;

    XTrain = bsxfun(@rdivide, ...
        bsxfun(@minus, XTrain, featureMean), featureStd);

    XValidation = bsxfun(@rdivide, ...
        bsxfun(@minus, XValidation, featureMean), featureStd);

    if experiment <= 2
        model = fitcdiscr(XTrain, YTrain, ...
            'DiscrimType', 'diaglinear', ...
            'Prior', 'uniform');
    else
        % Fixed regularization value for this initial comparison.
        model = fitcdiscr(XTrain, YTrain, ...
            'DiscrimType', 'linear', ...
            'Gamma', 0.1, ...
            'Prior', 'uniform');
    end

    predictions = predict(model, XValidation);

    C = confusionmat(YValidation, predictions, ...
        'Order', classIDs);

    overallAccuracy(experiment) = ...
        100 * mean(predictions == YValidation);

    recall = diag(C) ./ sum(C, 2);
    balancedAccuracy(experiment) = 100 * mean(recall);

    active = YValidation ~= 0;
    movementAccuracy(experiment) = ...
        100 * mean(predictions(active) == YValidation(active));

    validationConfusions{experiment} = C;
end

%% 6. Display the comparison
results = table(experimentNames, overallAccuracy, ...
    balancedAccuracy, movementAccuracy, ...
    'VariableNames', {'Configuration', 'OverallPercent', ...
    'BalancedPercent', 'MovementPercent'});

disp(results);

% Select by balanced validation accuracy.
[~, bestIndex] = max(balancedAccuracy);

fprintf('\nBest balanced validation result: %s\n', ...
    experimentNames{bestIndex});

%% 7. Plot validation performance
figure('Name', 'Validation comparison', 'Color', 'w');

bar([overallAccuracy, balancedAccuracy, movementAccuracy]);
ylim([0 100]);
ylabel('Validation accuracy (%)');
title('Train: repetitions 1-3 | Validate: repetition 4');
legend({'Overall', 'Balanced', 'Movement only'}, ...
    'Location', 'best');

set(gca, 'XTick', 1:4, ...
    'XTickLabel', {'RMS-Diag', 'All-Diag', ...
                  'RMS-Reg', 'All-Reg'});
grid on;

%% 8. Save the experiment results separately from the baseline model
resultsFolder = fullfile(projectFolder, 'results');

if ~isfolder(resultsFolder)
    mkdir(resultsFolder);
end

save(fullfile(resultsFolder, 'validation_comparison.mat'), ...
    'results', 'validationConfusions', 'classIDs', 'bestIndex');

disp('Validation comparison saved. Baseline model remains unchanged.');