
type = 1; % 1 is AE, 2 is classifier, 



% train_x = double(train_x(1:50000,:));
% train_y = double(train_y(1:50000,:));

%train_x = dataMatrix;
train_x = trData; %vertcat(trData, valData, testData);
% train_x = cat(1, testDataResized, trainDataResized);


inputSize = size(train_x,2);

if type == 1 % AE
   outputSize  = inputSize; % in case of AE it should be equal to the number of inputs

   %if type = 1, i.e., AE then the last layer should be linear and usually a
% series of decreasing layers are used
    hiddenActivationFunctions = {'ReLu','ReLu','ReLu','linear'};%{'sigm','sigm','sigm','linear'}; 
    hiddenLayers = [2000 1000 500 50]; 
   
elseif type == 2 % classifier
    outputSize = size(train_y,2); % in case of classification it should be equal to the number of classes

    hiddenActivationFunctions = {'sigm','sigm','sigm'};%{'ReLu','ReLu','ReLu','ReLu'};%
    hiddenLayers = [500 500 1000 ]; % hidden layers sizes, does not include input or output layers

end

% parameters used for visualisation of first layer weights
visParams.noExamplesPerSubplot = 50; % number of images to show per row
visParams.noSubplots = floor(hiddenLayers(1) / visParams.noExamplesPerSubplot);
visParams.col = 50; %44;% number columns of image
visParams.row = 30; %26 number rows of image



dbnParams = dbnParamsInit(type, hiddenActivationFunctions, hiddenLayers);
dbnParams.inputActivationFunction = 'linear'; %sigm for binary inputs, linear for continuous input
dbnParams.rbmParams.epochs = 20;

% normalise data
train_x = normaliseData(dbnParams.inputActivationFunction, train_x,[]);

% train Deep Belief Network
[dbn, errorPerBatch, errorPerSample] = trainDBN(train_x, dbnParams);

% visualise weights of first layer
% visualiseHiddenLayerWeights(dbn.W{1},visParams.col,visParams.row,visParams.noSubplots);

nn = unfoldDBNtoNN(dbnParams, dbn, outputSize);



