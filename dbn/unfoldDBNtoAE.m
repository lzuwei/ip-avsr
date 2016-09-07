function [weightsAE, biasesAE, newActivationFunctions, newLayers] = unfoldDBNtoAE(dbnParams, dbn, outputSize)
% unfoldDBNtoAE - Unfolds DBN to an autoencoder NN

% INPUTS
% dbnParams: structure containing the DBN params, see manual for more
% details

% dbn: structure which contains the weights (W), the hidden biases (hidbiases) and
% the visible biases (visbiases) for each RBM layer

% outputSize: size of output layer

% OUTPUTS
% weightsAE: 1xN cell array, where N is the number of layers (hidden + output
% layer), each cell contains the weights of the corresponding layer

% biasesAE: 1xN cell array, where N is the number of layers (hidden + output
% layer), each cell contains the biases of the corresponding layer

% newActivationFunctions: 1xN cell array, where N is the number of layers (hidden + output
% layer), each cell contains the activation function of the corresponding layer

% newLayers: 1xN vector, where N is the number of layers (hidden + output
% layer), each entry contains the size of the corresponding layer

noLayers = length(dbnParams.hiddenLayers); 

% create encoding layers
weightsAE = dbn.W;
biasesAE = dbn.hidbiases;
inputSize = size(dbn.W{1},1);

if inputSize ~= outputSize
    error('Input size is different that output size. In an AE they should have the same size')
end
 
ind = 1;
% create decoding layers, where weights/biases are mirrored from the
% encoding layer
for i = noLayers + 1:2*noLayers
    
    index = i - ind;
    weightsAE{i} = dbn.W{index}';
    biasesAE{i} = dbn.visbiases{index};
    
    ind = ind + 2;
    
end

% create new activation functions (activFcn from encoding layer + same
% activFcn flipped for decoding layer + outputActivFcn same as inputActivFcn
newActivationFunctions = [dbnParams.hiddenActivationFunctions fliplr(dbnParams.hiddenActivationFunctions(1:end-1)) dbnParams.inputActivationFunction];
% same as above for hidden layers
newLayers = [dbnParams.hiddenLayers fliplr(dbnParams.hiddenLayers(1:end-1)) outputSize];



