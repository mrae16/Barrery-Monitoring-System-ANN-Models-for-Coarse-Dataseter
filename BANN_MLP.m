%% MLP ANN
close all
clear all
clc

%% Options
lr_1 = 0.012;    % Learning rate for training algorithm  
loop = 'no';      % Loop over several learning rates?  'yes' or 'no'
numHiddenUnits = 10; % Number of hidden units in LSTM layer
fc1 = 3;       % Number of nodes in first hidden layer (Fully Connected)
fc2 = 2;       % Number of nodes in second hidden layer (Fully Connected)
iterations = 1000;
iterations_if_looping = 10000;   % this is to allow the smaller learning rates to converge.



% Load and preprocess your numerical data
Data = load("YourDatset.txt");  %Include your Dataset here. format in text file with columns [charging voltage, discharging voltage, Re(Ohm), Im(Ohm)]
Data = Data./max(Data);
[entries, attributes] = size(Data);
entries_breakpoint = round(entries*.70); %this is cutting out % of entries
trainData_inputs1 = Data(1:entries_breakpoint,[1,2,3,4]); %discharging voltage, Re(Ohm), Im(Ohm)
trainData_output = Data(1:entries_breakpoint, 5); %Capacitence 
testData_inputs = Data(entries_breakpoint:end, [1,2,3,4]);
testData_output = Data(entries_breakpoint:end, 5);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(loop)
switch loop 
    case 'yes'
        a = 1;       
        lr_loop(a) = 0.00012*10^(a-1);        
        lr= lr_loop;
        aa = [1:6];
        iterations = iterations_if_looping;
    case 'no'
        lr=lr_1;
        aa = 1;
end

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Initialize error vector
        errorRangeVals = zeros(6,1);
        errorVects = zeros(22,6);
        for a=aa      
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define the MLP architecture
layers = [
    featureInputLayer(4, 'Normalization', 'none', 'Name', 'input') % Input layer for 4 features
    fullyConnectedLayer(fc1, 'Name', 'fc1') % 
    sigmoidLayer('Name', 'sigmoid1') % ReLU activation layer
    fullyConnectedLayer(fc2, 'Name', 'fc2') % 
    sigmoidLayer('Name', 'sigmoid2') % ReLU activation layer
    fullyConnectedLayer(1, 'Name', 'fc3') % Output layer with a single unit for regression
    regressionLayer('Name', 'output') % Regression layer
];

switch loop 
        case 'yes' 
           lr_opt = 0.00012*10^(a-1);
        case 'no'
           lr_opt = lr;
    end

% Define training options
options = trainingOptions('adam', ...
    'MaxEpochs', 900, ...
    'MiniBatchSize', length(train_inputs1), ... %after each iteration, the all 35 predicted outputs are compared with the 35 epxerimental outputs
    'InitialLearnRate', lr_opt, ...
    'LearnRateSchedule','none', ...
    'Shuffle', 'never', ...
    'Verbose', false, ...
    'GradientDecayFactor',0.9,...
    'Plots', 'training-progress');


% Train the network

net = trainNetwork(train_inputs1, trainData_output, layers, options);

PredTestData_output = predict(net, testData_inputs);

error = PredTestData_output - testData_output;

figure('Name', 'model and error')
subplot(2,1,1)
    plot(testData_output,'r-'); hold on
        plot(PredTestData_output,'b-o')
    hold off
ylabel("Capacity %")
legend("experimental Capacity", "Predicted Capacity")

subplot(2,1,2)
    plot(error)
ylabel("error")

range_error = abs(max(error))+abs(min(error));
disp(['Range of error: ', num2str(range_error)])
disp(['Mean Absolute Error: ', num2str(mean(abs(error)))])
disp(['Standard deviation or error: ', num2str(std(error))])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch loop
    case 'yes'
% Store error value
        errorRangeVals(a) = range_error';
        errorVects(:,a) = error';

fprintf("Error of \a=%.0f", a, range_error')
fprintf("Error Vector of \a=%.0f", a, error')



%final plots
figure('Name', 'Change in error range for different learning rate')
plot(errorRangeVals)
xlabel('learning rate value (0.00012 to 120')
ylabel('Error range')
xticks("manual")

figure('Name',"Error of output for various learn rates")
    plot(errorVects(:,1)); hold on
        plot(errorVects(:,2)); hold on
            plot(errorVects(:,3)); hold on
                plot(errorVects(:,4)); hold on
                    plot(errorVects(:,5),'b-o'); hold on
                         plot(errorVects(:,6),'rx-'); hold on
hold off
legend("a=0.0005", "a=0.005","a=0.05","a=0.5","a=5","a=50")
ylabel("error")
xlabel("sample")
    case 'no'
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
MLP = PredTestData_output;
MLP = error;

%mean absolute error: 
MAE = mean(abs(PredTestData_output - testData_output));
disp(['MAE: ', num2str(MAE)])

mdl = fitlm(testData_inputs, PredTestData_output);
disp(mdl)

