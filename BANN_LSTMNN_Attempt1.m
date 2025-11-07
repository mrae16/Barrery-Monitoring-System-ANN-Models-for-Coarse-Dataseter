%LSTM NN 
close all
clear all
clc

%% Options
lr_1 = 0.012;    % Learning rate for training algorithm  
loop = 'no';      % Loop over several learning rates?   'yes' or 'no'
numHiddenUnits = 10; % Number of hidden units in LSTM layer
numFCLayer = 50;       % Number of nodes in hidden layer (Fully Connected)
iterations = 1000;
iterations_if_looping = 10000;   % this is to allow the smaller learning rates to converge.


% Load and preprocess your numerical data
Data = load("Brno_Data100%.txt");
Data = Data./max(Data);
[entries, attributes] = size(Data);
entries_breakpoint = round(entries*.70); %this is cutting out 50% of entries
trainData_inputs = Data(1:entries_breakpoint,[2,4,5,6]); %discharging voltage, Re(Ohm), Im(Ohm)
trainData_output = Data(1:entries_breakpoint, 7); %Capacitence 
testData_inputs = Data(entries_breakpoint:end, [2,4,5,6]);
testData_output = Data(entries_breakpoint:end, 7);


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

%Define network architecture 
layers = [
    sequenceInputLayer(4)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numFCLayer)
    sigmoidLayer
    fullyConnectedLayer(1)
    regressionLayer
];
switch loop 
        case 'yes' 
           lr_opt = 0.00012*10^(a-1);
        case 'no'
           lr_opt = lr;
    end

options = trainingOptions('adam', ...
    'MaxEpochs', iterations, ...
    'MiniBatchSize', length(trainData_output), ...
    'GradientThreshold', Inf, ...
    'InitialLearnRate', lr_opt,...
    'LearnRateSchedule', 'none', ...
    'GradientDecayFactor',0.9,...
    'Verbose', 0, ...
    'Plots', 'training-progress');


net = trainNetwork(trainData_inputs', trainData_output', layers, options);


PredTestData_output = predict(net, testData_inputs');

error = PredTestData_output - testData_output';



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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch loop
    case 'yes'
        % Store error value and plot after loop
        
                errorRangeVals(a) = range_error';
                errorVects(:,a) = error';
        
        fprintf("Error of \a=%.0f", a, range_error')
        fprintf("Error Vector of \a=%.0f", a, error')
      
        
        
        
        %Overall plot!
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
                            plot(errorVects(:,5)); hold on
                                 plot(errorVects(:,6),'r.-'); hold on
        hold off
        legend("a=0.00012", "a=0.0012","a=0.012","a=0.12","a=1.2","a=12")
        ylabel("error")
        xlabel("sample")

        case 'no'
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
% 
% mdl = fitlm(testData_inputs, PredTestData_output);
% disp(mdl)

% LSTM = error';
% Z = zeros(49:1);
% Z(15:49) = LSTM;
% LSTM_70 = Z

mdl = fitlm(testData_inputs, PredTestData_output);
disp(mdl)
