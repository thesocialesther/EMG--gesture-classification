%% STEP 3: Train and test an EMG gesture classifier
clear;
clc;
close all;

%% 1. Load the recording
projectFolder = fileparts(fileparts(mfilename('fullpath')));
filePath = fullfile(projectFolder, 'data', 'S1_E1_A1.mat');

data = load(filePath);

emg = double(data.emg);
labels = double(data.restimulus(:));
repetitions = double(data.rerepetition(:));

fs = 2000;
windowSamples = round(0.200 * fs);  % 200 ms

assert(size(emg, 1) == numel(labels), ...
    'EMG and labels have different lengths.');
assert(numel(labels) == numel(repetitions), ...
    'Labels and repetitions have different lengths.');
assert(all(isfinite(emg(:))), ...
    'EMG contains missing or nonfinite values.');

%% 2. Extract RMS features from non-overlapping windows
numberOfWindows = floor(size(emg, 1) / windowSamples);
numberOfChannels = size(emg, 2);

X = zeros(numberOfWindows, numberOfChannels);
Y = zeros(numberOfWindows, 1);
R = zeros(numberOfWindows, 1);
keep = false(numberOfWindows, 1);

for k = 1:numberOfWindows
    firstSample = (k - 1) * windowSamples + 1;
    lastSample = firstSample + windowSamples - 1;
    idx = firstSample:lastSample;

    segmentLabels = labels(idx);
    segmentRepetitions = repetitions(idx);

    % Keep only windows entirely inside one gesture and repetition.
    sameGesture = all(segmentLabels == segmentLabels(1));
    sameRepetition = ...
        all(segmentRepetitions == segmentRepetitions(1));
    numberedRepetition = segmentRepetitions(1) > 0;

    if sameGesture && sameRepetition && numberedRepetition
        segment = emg(idx, :);

        X(k, :) = sqrt(mean(segment.^2, 1));
        Y(k) = segmentLabels(1);
        R(k) = segmentRepetitions(1);
        keep(k) = true;
    end
end

X = X(keep, :);
Y = Y(keep);
R = R(keep);

%% 3. Split by repetition, never by random windows
trainRepetitions = [1 2 3 4];
testRepetitions = [5 6];

isTrain = ismember(R, trainRepetitions);
isTest = ismember(R, testRepetitions);

XTrain = X(isTrain, :);
YTrain = Y(isTrain);
XTest = X(isTest, :);
YTest = Y(isTest);

assert(~isempty(YTrain) && ~isempty(YTest), ...
    'Training or test data is empty.');

classIDs = unique(YTrain);

assert(isequal(classIDs, unique(YTest)), ...
    'Training and test sets do not contain the same gesture IDs.');

fprintf('Training windows: %d\n', numel(YTrain));
fprintf('Test windows: %d\n', numel(YTest));
fprintf('Number of classes, including rest: %d\n', numel(classIDs));

% Display how many examples each class has.
trainingCounts = zeros(numel(classIDs), 1);
testCounts = zeros(numel(classIDs), 1);

for k = 1:numel(classIDs)
    trainingCounts(k) = sum(YTrain == classIDs(k));
    testCounts(k) = sum(YTest == classIDs(k));
end

disp(table(classIDs, trainingCounts, testCounts, ...
    'VariableNames', {'GestureID', 'TrainingWindows', 'TestWindows'}));

%% 4. Transform and standardize features
% Log RMS compresses the amplitude range.
% Keep this same transformation when using the saved model.
featureFloor = 1e-12;

XTrainLog = log10(max(XTrain, featureFloor));
XTestLog = log10(max(XTest, featureFloor));

% Learn normalization ONLY from training data.
featureMean = mean(XTrainLog, 1);
featureStd = std(XTrainLog, 0, 1);
featureStd(featureStd < 1e-12) = 1;

XTrainScaled = bsxfun(@rdivide, ...
    bsxfun(@minus, XTrainLog, featureMean), featureStd);

XTestScaled = bsxfun(@rdivide, ...
    bsxfun(@minus, XTestLog, featureMean), featureStd);

%% 5. Train a diagonal linear discriminant classifier
% This is a simple baseline that uses a shared diagonal covariance.
% Uniform priors prevent plentiful rest windows dominating the prior.
classifier = fitcdiscr(XTrainScaled, YTrain, ...
    'DiscrimType', 'diaglinear', ...
    'Prior', 'uniform');

%% 6. Predict the held-out repetitions
YPred = predict(classifier, XTestScaled);

accuracy = 100 * mean(YPred == YTest);

% Confusion matrix: rows = actual, columns = predicted.
C = confusionmat(YTest, YPred, 'Order', classIDs);

recallPercent = 100 * diag(C) ./ sum(C, 2);
balancedAccuracy = mean(recallPercent);

fprintf('\nHELD-OUT TEST RESULTS\n');
fprintf('Overall accuracy: %.2f%%\n', accuracy);
fprintf('Balanced accuracy: %.2f%%\n', balancedAccuracy);

disp(table(classIDs, recallPercent, ...
    'VariableNames', {'GestureID', 'RecallPercent'}));

%% 7. Plot the confusion matrix
figure('Name', 'Held-out test results', 'Color', 'w');

imagesc(C);
axis image;
colorbar;

tickLabels = arrayfun(@num2str, classIDs, ...
    'UniformOutput', false);

set(gca, ...
    'XTick', 1:numel(classIDs), ...
    'XTickLabel', tickLabels, ...
    'YTick', 1:numel(classIDs), ...
    'YTickLabel', tickLabels);

xlabel('Predicted gesture ID');
ylabel('Actual gesture ID');
title(sprintf('Test repetitions 5 and 6 — accuracy %.1f%%', ...
    accuracy));

%% 8. Save the model AND its preprocessing settings
modelFolder = fullfile(projectFolder, 'models');

if ~isfolder(modelFolder)
    mkdir(modelFolder);
end

modelPath = fullfile(modelFolder, 'emg_rms_baseline.mat');

save(modelPath, ...
    'classifier', 'featureMean', 'featureStd', ...
    'featureFloor', 'fs', 'windowSamples', ...
    'numberOfChannels', 'classIDs', ...
    'trainRepetitions', 'testRepetitions', ...
    'accuracy', 'balancedAccuracy');

fprintf('\nModel saved to:\n%s\n', modelPath);
%% 9. Display a percentage confusion matrix
% Each row sums to 100%.
rowTotals = sum(C, 2);
CPercent = 100 * bsxfun(@rdivide, C, rowTotals);

figure('Name', 'Confusion matrix in percentages', 'Color', 'w');

imagesc(CPercent);
axis image;
caxis([0 100]);
colorbar;

set(gca, ...
    'XTick', 1:numel(classIDs), ...
    'XTickLabel', tickLabels, ...
    'YTick', 1:numel(classIDs), ...
    'YTickLabel', tickLabels);

xlabel('Predicted gesture ID');
ylabel('Actual gesture ID');
title('Percentage of each actual gesture classified into each category');

% Write percentages inside the cells.
for row = 1:numel(classIDs)
    for col = 1:numel(classIDs)
        value = CPercent(row, col);

        if value >= 1
            textColor = 'w';

            if value >= 50
                textColor = 'k';
            end

            text(col, row, sprintf('%.0f', value), ...
                'HorizontalAlignment', 'center', ...
                'Color', textColor, ...
                'FontSize', 8);
        end
    end
end

%% 10. Measure performance on actual movement windows
% Predictions of rest during movement still count as errors.
isMovement = YTest ~= 0;

movementAccuracy = ...
    100 * mean(YPred(isMovement) == YTest(isMovement));

fprintf('\nAccuracy on actual movement windows: %.2f%%\n', ...
    movementAccuracy);

%% 11. Identify the most common wrong prediction for each gesture
wrongCounts = C;

% Remove correct predictions from consideration.
wrongCounts(1:size(C, 1)+1:end) = 0;

[largestErrorCount, wrongColumn] = max(wrongCounts, [], 2);

mostConfusedWith = classIDs(wrongColumn);
mostConfusedWith(largestErrorCount == 0) = NaN;

confusionPercent = 100 * largestErrorCount ./ rowTotals;

disp('Most frequent mistake for each actual gesture:');

disp(table(classIDs, mostConfusedWith, confusionPercent, ...
    'VariableNames', ...
    {'ActualGesture', 'MostConfusedWith', 'PercentOfActualGesture'}));