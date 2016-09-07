function [dbn, errorPerBatch errorPerSample] = trainDBN(dataMatrix, dbnParams)
% trainDBN - Trains a DBN

% INPUTS
% dataMatrix: matrix containing the training examples, size: noExamples x
% Dimensionality
% dbnParams: structure containing the DBN params, see manual for more
% details

% OUTPUTS
% dbn: structure which contains the weights (W), the hidden biases (hidbiases) and
% the visible biases (visbiases) for each RBM layer

% errorPerBatch: 1xN cell array where N is the number of hidden layers (=
% the number of RBMs to train and stack). Each cell contains the average
% minibatch error per epoch. If number of epochs is 100 then each cell will
% be 1 x 100

% errorPerSample: same as above but contains the average error per training
% sample

activationFunctionsAllLayers = [dbnParams.inputActivationFunction, dbnParams.hiddenActivationFunctions];

hiddenLayers = dbnParams.hiddenLayers;
nHidLayers = length(hiddenLayers);

for i = 1:nHidLayers 

    noHidNeurons = hiddenLayers(i);
    [numExamples, numDims] = size(dataMatrix);

    fprintf(1,'Pretraining Layer %d with RBM: %d-%d \n',i, numDims,noHidNeurons);

    hLayer = activationFunctionsAllLayers(i + 1); % activation function of hidden layer
    vLayer = activationFunctionsAllLayers(i); % activation function of visible layer
    
    trFctnLayers = [vLayer hLayer];
    
    % train RBM
    [rbm, errorPerBatch{i}, errorPerSample{i}] = trainRBM(dataMatrix, dbnParams, noHidNeurons, trFctnLayers);
  
    % save RBM weights to corresponding DBN layer
    dbn.W{i} = rbm.W; 
    dbn.hidbiases{i} = rbm.hidbiases;
    dbn.visbiases{i} = rbm.visbiases;
    
    % compute RBMs hidden activations
    [posHidProbs, posHidStates] = RBMup(dataMatrix, rbm.W, rbm.hidbiases, hLayer);
    
    % and use them as new inputs for the following RBM
    dataMatrix = posHidProbs;

end

disp('DBN training done')