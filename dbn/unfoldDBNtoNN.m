function nn = unfoldDBNtoNN(dbnParams, dbn, outputSize)
% unfoldDBNtoNN - Unfolds DBN to NN

% INPUTS
% dbnParams: structure containing the DBN params, see manual for more
% details

% dbn: structure which contains the weights (W), the hidden biases (hidbiases) and
% the visible biases (visbiases) for each RBM layer

% outputSize: size of output layer

% OUTPUTS
% nn: neural network structure, see manual for details

    
if dbnParams.type == 1 % AE
        

    disp('Unfolding DBN to AE')
    
    [weightsAE, biasesAE, newActivationFunctions, newLayers] = unfoldDBNtoAE(dbnParams, dbn, outputSize);
%   nn = paramsNNinit(newLayers, newActivationFunctions);
    nn.activationFunctions = newActivationFunctions;
    nn.layers = newLayers;
    nn.W = weightsAE;
    nn.biases = biasesAE;
        
    
elseif dbnParams.type == 2 % classification

    disp('Unfolding DBN to Classifier')
    
    [weightsClsf, biasesClsf, newActivationFunctions, newLayers] = unfoldDBNToClsf(dbnParams, dbn, outputSize);
    nn = paramsNNinit(newLayers, newActivationFunctions);   
    nn.W = weightsClsf;
    nn.biases = biasesClsf;
    
end


nn.pretraining = 1;
        


