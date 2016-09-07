function [activations, states] = RBMup(data, weights, hidbiases, hL_type)
% RBMup - Computes activations and states of RBM's hidden layer

% INPUTS
% data: data matrix, noExamples x noDimensions

% weights: matrix containing the RBM weights, noVisibleUnits x
% noHiddenUNits

%hidbiases: biases of hidden layer, 1 x NoVisibleNeurons

% hL_type: activation function of hidden layer, e.g. 'sigm', 'linear',
% 'ReLu'

% OUTPUTS
% activations: activation matrix, noExamples x noNeurons (hidden neurons)

% states: states of hidden neurons, noExamples x noNeurons (hidden neurons)

[numExamples numDims] = size(data);

%  input to hidden neurons - batchSize x noHidden neurons, each row
%  contains the input to the hidden units
hidInp = data * weights; 

% create biases matrix
hidBiasesMatrx = repmat(hidbiases,numExamples,1);
  
finalHidInp = hidInp + hidBiasesMatrx;
   
% contains activations of hidden units, batchSize x noHidden neurons
activations = computeActivations(hL_type, finalHidInp);
  
% compute hidden states 
states = computeStates(hL_type, activations, finalHidInp);