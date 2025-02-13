% Load the Data
if ~isfile('PhysionetData.mat')
    ReadPhysionetData         
end
load PhysionetData

% Predictor
Signals(1:5)

% Labels
Labels(1:5)


summary(Labels)

% Histogram of signal lengths
L = cellfun(@length,Signals);
h = histogram(L);
xticks(0:3000:18000);
xticklabels(0:3000:18000);
title('Signal Lengths')
xlabel('Length')
ylabel('Count')

% Plotting one signal from the afib and normal classes
normal = Signals{1};
aFib = Signals{4};

subplot(2,1,1)
plot(normal)
title('Normal Rhythm')
xlim([4000,5200])
ylabel('Amplitude (mV)')
text(4330,150,'P','HorizontalAlignment','center')
text(4370,850,'QRS','HorizontalAlignment','center')

subplot(2,1,2)
plot(aFib)
title('Atrial Fibrillation')
xlim([4000,5200])
xlabel('Samples')
ylabel('Amplitude (mV)')

% preparing data so each sample is of equal length (9000 Samples)
[Signals,Labels] = segmentSignals(Signals,Labels);

% Viewing the first five elements of the Signals to verify that each sample
% is of equal length
Signals(1:5)

% The summary will show the amount of normal and afib samples 
% which can be used to determine the afib/normal ration
summary(Labels)

% Split the signals according to their class.
afibX = Signals(Labels=='A');
afibY = Labels(Labels=='A');

normalX = Signals(Labels=='N');
normalY = Labels(Labels=='N');

% Use dividerand to divide targets from each class randomly into training and testing sets.
[trainIndA,~,testIndA] = dividerand(718,0.9,0.0,0.1);
[trainIndN,~,testIndN] = dividerand(4937,0.9,0.0,0.1);

XTrainA = afibX(trainIndA);
YTrainA = afibY(trainIndA);

XTrainN = normalX(trainIndN);
YTrainN = normalY(trainIndN);

XTestA = afibX(testIndA);
YTestA = afibY(testIndA);

XTestN = normalX(testIndN);
YTestN = normalY(testIndN);

% replicate the smaller afib testing and training set to make the ration
% 1 to 1 
XTrain = [repmat(XTrainA(1:634),7,1); XTrainN(1:4438)];
YTrain = [repmat(YTrainA(1:634),7,1); YTrainN(1:4438)];

XTest = [repmat(XTestA(1:70),7,1); XTestN(1:490)];
YTest = [repmat(YTestA(1:70),7,1); YTestN(1:490);];

% test to validate that afib and normal sets are balanced
summary(YTrain)
summary(YTest)

% Define bidirectional LSTM Network Architecture
layers = [ ...
    sequenceInputLayer(1)
    bilstmLayer(200,'OutputMode','last') % more hidden units (original was 100)
    dropoutLayer(0.5) % Add dropout layer with a dropout rate of 0.5
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
    ]
% options = trainingOptions('adam', ...
%     'MaxEpochs',10, ...
%     'MiniBatchSize', 150, ...
%     'InitialLearnRate', 0.01, ...
%     'SequenceLength', 1000, ...
%     'GradientThreshold', 1, ...
%     'ExecutionEnvironment',"auto",...
%     'plots','training-progress', ...
%     'Verbose',false ...
%     );
options = trainingOptions('adam', ...
    'MaxEpochs',15, ...
    'MiniBatchSize', 150, ...
    'InitialLearnRate', 0.01, ...
    'SequenceLength', 1000, ...
    'GradientThreshold', 1, ...
    'ExecutionEnvironment',"auto",...
    'plots','training-progress', ...
    'Verbose',false, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XTest,YTest}, ... % Add validation data for testing loss
    'ValidationFrequency',30, ... % Frequency of validation, set based on the size of your dataset
    'Plots','training-progress');
%     'ValidationPatience',4, ... % Early stopping: number of times validation loss can increase before stopping training
%     'Plots','training-progress');

% Train LSTM network
net = trainNetwork(XTrain,YTrain,layers,options);

%Save the trained network
save('trainedNetwork.mat', 'net');

Visualize training accuracy and save variables
trainPred = classify(net,XTrain,'SequenceLength',1000);
trainAccuracy = sum(trainPred == YTrain)/numel(YTrain)*100;
%save('trainResults.mat', 'YTrain', 'trainPred', 'trainAccuracy');
figure
confusionchart(YTrain,trainPred,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart for LSTM');

% Visualize testing accuracy
%testPred = classify(net,XTest,'SequenceLength',1000);
testAccuracy = sum(testPred == YTest)/numel(YTest)*100;
%save('testResults.mat', 'YTest', 'testPred', 'testAccuracy');
figure
confusionchart(YTest,testPred,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart for LSTM');
