function states = computeStates(layerType, probs, data)
% computeStates - Computes states of hidden/visible layer of an RBM

% INPUTS
% layerType: activation function of given layer, e.g. 'sigm', 'linear',
% 'ReLu'

% probs: activation matrix, noExamples x noNeurons

% data: data matrix, it's the input to the neurons, noExamples x noNeurons

% OUTPUTS
% states: states matrix, noExamples x noNeurons


[numExamples,numHid] = size(probs);

if strcmpi(layerType,'sigm')
  
      states = probs > rand(numExamples,numHid);
  
  elseif strcmpi(layerType,'linear')
      
      states = probs + randn(numExamples,numHid); 
      
  elseif strcmpi(layerType,'ReLu')
      

      sigma = 1./(1 + exp(-data));
      noise = sigma .* randn(numExamples, numHid);
      states =  max(0,data + noise); 
      
end


