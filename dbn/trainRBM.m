function [rbm, errorPerBatch, errorPerSample] = trainRBM(dataMatrix, dbnParams, numHid, layerType)
% trainRBM - Trains RBM

%INPUTS
% dataMatrix: matrix containing the training examples, size: noExamples x
% Dimensionality

% dbnParams: structure containing the DBN params, see manual for more
% details

%numHid: number of RBMs hidden neurons  

%layerType: 1x2 cell array, first cell contains activation function of
%visible layer and the second cell contains the acivation function of the
%hidden layer

% OUTPUTS
% rbm: structure which contains the weights (w), the hidden biases (hidbiases) and
% the visible biases (visbiases) of the trained RBM

% errorPerBatch: 1xM vector where M is the number of epochs. Each entry contains
% the average  minibatch error per epoch.

% errorPerSample: same as above but contains the average error per training
% sample


lrW = dbnParams.rbmParams.lrW; % Learning rate for weights 
lrVb = dbnParams.rbmParams.lrVb; % Learning rate for biases of visible units 
lrHb = dbnParams.rbmParams.lrHb; % Learning rate for biases of hidden units 


weightPenaltyL2  = dbnParams.rbmParams.weightPenaltyL2; % L2 weight decay coefficient
initialmomentum  = dbnParams.rbmParams.initMomentum; 
finalmomentum    = dbnParams.rbmParams.finalMomentum;

batchsize = dbnParams.rbmParams.batchsize;

[numExamples numDims] = size(dataMatrix);

numbatches = ceil(numExamples / batchsize);

maxepoch = dbnParams.rbmParams.epochs;

vL_type = layerType{1}; % activation function of visible layer
hL_type = layerType{2}; % activation function of hidden layer

if strcmpi(vL_type, 'linear') || strcmpi(hL_type,'linear') || strcmpi(hL_type,'ReLu') || strcmpi(vL_type, 'ReLu')
    lrW = dbnParams.rbmParams.lrW_linear; % Learning rate for weights 
    lrVb = dbnParams.rbmParams.lrVb_linear; % Learning rate for biases of visible units 
    lrHb = dbnParams.rbmParams.lrHb_linear; % Learning rate for biases of hidden units 
end

% Initializing weights and biases. Weights are initialised randomly and
% biases are set to 0 as in Hinton's code http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html
% for ReLU we use a smaller initialization weight threshold
if strcmpi(hL_type,'ReLu') || strcmpi(vL_type, 'ReLu')
    weights = 0.01*randn(numDims, numHid);
else
    weights     = 0.1*randn(numDims, numHid);
end
  hidbiases  = zeros(1,numHid);
  visbiases  = zeros(1,numDims);
  
deltaW  = zeros(numDims,numHid);
deltaVisbias = zeros(1,numDims);
deltaHidbias = zeros(1,numHid);
  
errorPerBatch = zeros(1, maxepoch);
errorPerSample = zeros(1, maxepoch);

for epoch = 1:maxepoch
    
    disp(['epoch = ',num2str(epoch)] ); 
    errSum = 0;
    randomorder = randperm(numExamples);
    
    % for the first momentumEpochThres use initialmomentum then use finalmomentum
    if epoch > dbnParams.rbmParams.momentumEpochThres,
       momentum = finalmomentum;
    else
       momentum = initialmomentum;
    end
    
    for batch = 1:numbatches,
        
        % select randomly examples for mini-batches
        if batch == numbatches       
            data = dataMatrix(randomorder(1+(batch-1)*batchsize:end), :);
        else
            data = dataMatrix(randomorder(1+(batch-1)*batchsize:batch*batchsize), :);
        end
        
        
        %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % posHidProbs: contains activations of hidden units, batchSize x noHidden neurons 
      [posHidProbs, posHidStates] = RBMup(data, weights, hidbiases, hL_type);
 
       if dbnParams.rbmParams.type == 1
           
           posprods    = data' * posHidProbs;
           % activation of hidden units over all (batch) training examples
           poshidact   = sum(posHidProbs);
  
       elseif dbnParams.rbmParams.type == 2
           posprods    = data' * posHidStates;
           % activation of hidden units over all (batch) training examples
           poshidact   = sum(posHidStates);
  
       end
       
       % activation of input units over all (batch) training examples
       posvisact = sum(data);

       %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


       %%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       % negVisProbs: batchSize x noDims, each row contains one example generated from the
       % hidden states through backpopagating their states multiplied by the
       % weights
      [negVisProbs, negVisStates] = RBMdown(posHidStates, weights, visbiases, vL_type);
         
      if dbnParams.rbmParams.type == 1
           
          [negHidProbs, negHidStates] = RBMup(negVisProbs, weights, hidbiases, hL_type);
          negprods  = negVisProbs' * negHidProbs;
          negvisact = sum(negVisProbs); 
          err = sum(sum( (data - negVisProbs).^2 ));
           
      elseif dbnParams.rbmParams.type == 2
      
          [negHidProbs, negHidStates] = RBMup(negVisStates, weights, hidbiases, hL_type);
          negprods  = negVisStates' * negHidProbs;
          negvisact = sum(negVisStates); 
          err = sum(sum( (data - negVisStates).^2 ));
      
      end
      
      neghidact = sum(negHidProbs);
      %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
      errSum = errSum + err;        

      %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
      gradEstimate = (posprods - negprods) / batchsize;
      deltaW = momentum * deltaW + lrW * (gradEstimate  - weightPenaltyL2 * weights);
    
      gradEstimateVisBias = (posvisact - negvisact) / batchsize;
      deltaVisbias = momentum * deltaVisbias + lrVb * gradEstimateVisBias;
    
      gradEstimateHidBias = (poshidact - neghidact) / batchsize;
      deltaHidbias = momentum * deltaHidbias + lrHb * gradEstimateHidBias;

      weights = weights + deltaW;
      visbiases = visbiases + deltaVisbias;
      hidbiases = hidbiases + deltaHidbias;

      nanVec = isnan(weights(:));
    
      if sum(nanVec) ~= 0
        keyboard
      end
      
      %%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    end
   
 
   
    errPerSample = err / numExamples;
    errPerBatch = err /  numbatches;
    disp(['Mean Squared Error per sample = ', num2str(errPerSample)]); 
    disp(['Mean Squared Error per Batch = ', num2str(errPerBatch)]); 
    errorPerBatch(epoch) = errPerBatch;
    errorPerSample(epoch) = errPerSample;
end

rbm.W = weights;
rbm.hidbiases = hidbiases;
rbm.visbiases = visbiases;
