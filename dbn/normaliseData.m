function [data,PS] = normaliseData(trFcn, data, PS)

% in case of linear visible layer it is recommended by Hinton in "A practical guide 
%to training RBMs" to make each dimension of the feature vector to have
%zero mean and unit standard deviation.
if strcmpi(trFcn, 'linear')
    
    if isempty(PS)
        ymean = 0;
        ystd = 1;
        [data,PS] = mapstd(data,ymean,ystd);
    else
        data = mapstd('apply',data,PS);
        
    end
%    [data,PS] = mapstd(data',ymean,ystd);
%    data = data';
   
%    each image is zero normalised and divided by the std over all pixers over
% all images
% s = std(data(:));
% 
% [dataTemp,PS] = mapstd(data,ymean,ystd);
% PS.xstd = repmat(s,size(data, 1),1);
% [data,PS] = mapstd('apply',data,PS);

   
   
% in case the activation function of the visible layer is "sigm" i.e. data
% are binary, then simply divide by the max value so the data are in the
% range [0, 1].
elseif strcmpi(trFcn, 'sigm')
    data = data/max(data(:)); %255;
end