function  [activations, states] = RBMdown(data, weights, visbiases, vL_type)
% RBMdown - Computes activations and states of RBM's hidden layer

% INPUTS
% data: data matrix, noExamples x noDimensions

% weights: matrix containing the RBM weights, noVisibleUnits x
% noHiddenUNits

% visbiases: biases of visible layer, 1 x NoVisibleNeurons

% vL_type: activation function of visible layer, e.g. 'sigm', 'linear',
% 'ReLu'

% OUTPUTS
% activations: activation matrix, noExamples x noNeurons (visible neurons)

% states: states of visible neurons, noExamples x noNeurons (visible neurons)


% batchSize x noDims, each row contains one example generated from the
% hidden states through backpopagating their states multiplied by the
% weights
numExamples = size(data, 1);
    
inpFromHidden = data * weights';
           
visBiasesMatrix = repmat(visbiases,numExamples,1);
         
finalVisInput = inpFromHidden + visBiasesMatrix; 

 %activations of visible units
 activations = computeActivations(vL_type, finalVisInput);
  
 % compute visible states 
 states = computeStates(vL_type, activations, finalVisInput);
