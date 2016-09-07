function [weightsClsf, biasesClsf, newActivationFunctions newLayers] = unfoldDBNToClsf(dbnParams,dbn,outputSize)
% unfoldDBNToClsf - Unfolds DBN to NN for classification purposes

% INPUTS
% dbnParams: structure containing the DBN params, see manual for more
% details

% dbn: structure which contains the weights (W), the hidden biases (hidbiases) and
% the visible biases (visbiases) for each RBM layer

% outputSize: size of output layer

% OUTPUTS
% weightsClsf: 1xN cell array, where N is the number of layers (hidden + output
% layer), each cell contains the weights of the corresponding layer

% biasesClsf: 1xN cell array, where N is the number of layers (hidden + output
% layer), each cell contains the biases of the corresponding layer

% newActivationFunctions: 1xN cell array, where N is the number of layers (hidden + output
% layer), each cell contains the activation function of the corresponding layer

% newLayers: 1xN vector, where N is the number of layers (hidden + output
% layer), each entry contains the size of the corresponding layer

% if classification then last layer is softmax
newActivationFunctions = [dbnParams.hiddenActivationFunctions 'softmax'];

newLayers = [dbnParams.hiddenLayers outputSize];
     
% initialise weights/biases of new layer
% hinton in his code initialises the last layer like this
% http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html
lastLayerW = 0.1*randn(newLayers(end - 1), outputSize);
lastLayerBiases = 0.1*randn(1, outputSize);

weightsClsf = [dbn.W lastLayerW];
biasesClsf = [dbn.hidbiases lastLayerBiases];


