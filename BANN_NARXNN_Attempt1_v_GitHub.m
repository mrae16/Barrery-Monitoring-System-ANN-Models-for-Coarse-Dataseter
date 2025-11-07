%% NARX ANN 

close all
clear all
clc

%% Options
lr_1 = 0.005;    % Learning rate for training algorithm (when 'no' loop is applied)
lr_loop_series = [0.0007, 0.005, 0.01, 0.05, 0.1, 3];   %6 different learning rates for comparison (only when loop is activated 'yes')
loop = 'yes';      % Loop over several learning rates?  'yes' or 'no'
numHiddenUnits = 10; % Number of node units in hidden layer
iterations = 1000;
iterations_if_looping = 10000;   % this is to allow the smaller learning rates to converge.
feedbackDelay = 1:2; %feedback delay for NARX layer
inputDelay = 1:2; %input delay for NARX layer

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(loop)
switch loop 
    case 'yes'    
        lr_loop = lr_loop_series;    
        lr= lr_loop;
        iterations = iterations_if_looping;
        aa = [1:6];
    case 'no'
        lr=lr_1;
        aa = 1;
end

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Initialize error vector
        errorRangeVals = zeros(6,1);
        errorVects = zeros(22,6);
        for a=aa;
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%import data
Data = load("YourDatset.txt");  %Include your Dataset here. format in text file with columns [charging voltage, discharging voltage, Re(Ohm), Im(Ohm)]
Data = Data./max(Data);
[entries, attributes] = size(Data);
entries_breakpoint = round(entries*.70); %this is cutting out % of entries
Data_inputs = Data(:,[1,2,3,4]); %charging voltage, discharging voltage, Re(Ohm), Im(Ohm)
Data_output = Data(:, 7); %Capacitence 






% hidden layer size (h)
h = numHiddenUnits;
% Create a NAR neural network
net = narxnet(inputDelay, feedbackDelay, h, 'open', 'trainlm');  % Modify the architecture as needed
                              %narxnet(1:numInputDelays, 1:numFeedbackDelays, hiddenlayersize(50), feedbackMode, trainFunction)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch loop 
        case 'yes' 
           lr_opt = lr_loop_series(aa);
        case 'no'
           lr_opt = lr;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%% Configure the network
net.layers{1}.transferFcn = 'tansig';  % You can choose other activation functions
net.layers{2}.transferFcn = 'tansig';  % Output layer activation function

%setting initial weight and biases
net.IW{1,1} = rand(size(net.IW{1,1})); % replace with your desired values
net.b{1} = rand(size(net.b{1}));

% % Divide the dataset into training, validation, and testing sets
net.divideFcn = 'divideind';  % You can use other division functions
net.divideParam.trainInd = 1:entries_breakpoint; %for training 
net.divideParam.valInd = 1:entries_breakpoint;  % for validating
net.divideParam.testInd = entries_breakpoint:length(Data);  %for testing

% Prepare input time series
inputTimeSeries = tonndata(Data_inputs, false, false);
targetTimeSeries = tonndata(Data_output, false, false);

% Prepare input and target time series with delays
[Data_inputs, inputStates, layerStates, Data_output] = preparets(net, inputTimeSeries, {}, targetTimeSeries);




%List of training functions:
%   trainlm         %   
%   trainbfg        %
%   trainrp         %%%
%   trainscg        %
%   traincgb        %%%
%   traincgf        %%%
%   traincgp        %
%   trainposs       %
%   traingdx        %



% Set training parameters
net.trainParam.epochs = iterations;        % Maximum number of training epochs
net.trainParam.goal = 0;         % Performance goal
net.trainParam.max_fail = iterations/10;        % Maximum number of validation failures
net.trainParam.min_grad = 0;     % Minimum gradient for convergence
net.trainParam.lr = lr_opt(a);           % Learning rate (if using 'trainlm')
net.trainParam.mu = 0.9;           % Initial mu (if using 'trainlm')
net.trainParam.show = 10;           % Epochs between displays
net.trainParam.showCommandLine = false;   % Display training progress in command line
net.trainParam.showWindow = true; % Display training progress in a separate window

disp(['loop: ', num2str(a)])
disp(['Learn rate for current loop: ', num2str(lr_opt(a))])


% Train the NARX neural network
[net, tr] = train(net, Data_inputs, Data_output, inputStates, layerStates);

%%%%%%%%% Make predictions on the training set
predTrainData_outputs = net(Data_inputs, inputStates, layerStates);



predictedOutput = cell2mat(predTrainData_outputs);
expectedOutput = cell2mat(Data_output);
error = predictedOutput - expectedOutput;

% Plot the training, validation, and testing performance
% figure;
% plotperform(tr);


% Plot the actual vs. predicted values
figure('Name', 'model and error');
subplot(2,1,1)
plot(cell2mat(Data_output(entries_breakpoint-2:end)), 'r-');
hold on;
plot(cell2mat(predTrainData_outputs(entries_breakpoint-2:end)), 'b-o');
legend('Actual', 'Predicted');
xlabel('Sample');
ylabel('Capacitance');
title('Actual vs. Predicted Capacitance');

subplot(2,1,2)
    plot(error(entries_breakpoint-2:end))
ylabel("error")

%disp(['Mean Absolute Error: ', num2str(mean(abs(error)))])
%disp(['Standard deviation or error: ', num2str(std(error))])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % Store error value
range_error = abs(max(error))+abs(min(error));

error = error(entries_breakpoint-2:end);
errorRangeVals(a) = range_error';
errorVects(:,a) = error';

fprintf("Error of \a=%.0f", a, range_error')
fprintf("Error Vector of \a=%.0f", a, error')
                     
        end
%% final plots
figure('Name', 'Change in error range for different learning rate')
plot(errorRangeVals)
xlabel('learning rate value (0.00012 to 120')
ylabel('Error range')
xticks("manual")

figure('Name',"Error of output for various learn rates")
    plot(errorVects(:,1), lineStyle="-", Marker='d', Color='r'); hold on
        plot(errorVects(:,2), lineStyle="-", Marker='^', Color='b'); hold on
            plot(errorVects(:,3), lineStyle="-", Marker='o', Color='m'); hold on
                plot(errorVects(:,4), lineStyle="-", Marker='x', Color='g'); hold on
                    plot(errorVects(:,5), lineStyle="-", Marker='+', Color='k'); hold on
                         plot(errorVects(:,6), lineStyle="-", Marker='square', Color='cyan'); hold on
hold off
legend("a=0.0007", "a=0.005","a=0.01","a=0.05","a=0.1","a=3")
ylabel("error")
xlabel("sample")

LEV = length(errorVects);
figure('Name',"Error of output for various learn rates (Second Portion)")
    plot(errorVects(12:22,1), lineStyle="-", Marker='d', Color='r'); hold on
        plot(errorVects(12:22,2), lineStyle="-", Marker='^', Color='b'); hold on
            plot(errorVects(12:22,3), lineStyle="-", Marker='o', Color='m'); hold on
                plot(errorVects(12:22,4), lineStyle="-", Marker='x', Color='g'); hold on
                    plot(errorVects(12:22,5), lineStyle="-", Marker='+', Color='k'); hold on
                         plot(errorVects(12:22,6), lineStyle="-", Marker='s', Color='cyan'); hold on
hold off
legend("a=0.0007", "a=0.005","a=0.01","a=0.05","a=0.1","a=3") %0.0007, 0.005, 0.01, 0.05, 0.1, 3
ylabel("error")
xlabel("sample")

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% All possible adjustable training parameters:%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 'trainFcn': Specifies the training function.
% Typical values: 'trainlm' (Levenberg-Marquardt), 'trainbfg' (BFGS
% quasi-Newton), 'trainrp' (Rprop), etc. 
% 
% 'trainParam.epochs': Maximum
% number of training epochs.
% % Typical values: 100, 200, 500, etc. 
% 
% 'trainParam.goal': Performance goal.
% Typical values: 1e-6, 1e-5, etc. 
% 
% 'trainParam.max_fail': Maximum number of validation failures. 
% Typical values: 6, 10, etc. 
% 
% 'trainParam.min_grad': Minimum gradient for
% convergence. 
% Typical values: 1e-6, 1e-5, etc. 
% 
% 'trainParam.lr': Learning rate for the
% Levenberg-Marquardt algorithm. 
% Typical values: 0.01, 0.001, etc. (if using 'trainlm') 
% 
% 'trainParam.mu':
% Initial mu (Levenberg-Marquardt parameter). 
% Typical values: 0.01, 0.001, etc. 
% 
% (if using 'trainlm') 'trainParam.show':
% Epochs between displays. 
% Typical values: 10, 25, etc. 
% 
% 'trainParam.showCommandLine': Display
% training progress in command line. 
% Typical values: true or false. 
% 
% 'trainParam.showWindow': Display training
% progress in a separate window.
% Typical values: true or false.





% Possible transfer functions:
% 
% compet - Competitive transfer function.
% elliotsig - Elliot sigmoid transfer function.
% hardlim - Positive hard limit transfer function.
% hardlims - Symmetric hard limit transfer function.
% logsig - Logarithmic sigmoid transfer function.
% netinv - Inverse transfer function.
% poslin - Positive linear transfer function1.
% purelin - Linear transfer function1.
% radbas - Radial basis transfer function1.
% radbasn - Radial basis normalized transfer function1.
% satlin - Positive saturating linear transfer function1.
% satlins - Symmetric saturating linear transfer function1.
% softmax - Soft max transfer function1.
% tansig - Symmetric sigmoid transfer function1.
% tribas - Triangular basis transfer function1.


NARX = cell2mat(predTrainData_outputs((entries_breakpoint-2):end))';
errNARX = error(entries_breakpoint:end)';

mdl = fitlm((Data((entries_breakpoint:end),[1,2,3,4])),cell2mat(predTrainData_outputs((entries_breakpoint-2:end)))');
disp(mdl)

disp(['Standard deviation or error: ', num2str(std(errNARX))])


% 
% figure
% plot(NARX_4_out); hold on
% plot(NARX_10_out); hold on
% plot(NARX_20_out); hold on
% plot(NARX_40_out); hold on
% plot(NARX_100_out); hold off
% legend('H = 4', 'H = 10', 'H = 20', 'H = 40', 'H = 100')
% ylabel('SOC')
% xlabel('Sample')
% title('Change in LSTM Output for Different Hidden Layer Sizes')

