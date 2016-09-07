function visualiseHiddenLayerWeights(weights,col,row,noImageRows)
% visualiseHiddenLayerWeights - Visualises as an image the given weights

% INPUTS
% weights: weightMatrix, noInputs x noHiddenNeurons (first hidden layer,
% since we usually visualise weights of the first hidden layer only)

% col: number of image columns 

% row: number of image rows
% The product of col and row must be equal to the number of inputs, i.e.,
% the number of rows of the weights matrix

% noImageRows: number of image rows, i.e., if 10 then there will be 10 rows
% of images where each row will contain floor(noHiddenNeurons / noImageRows)

[inpSize, N] = size(weights);

% find minimum/maximum weight value
minValue = min(weights(:));
maxValue = max(weights(:));

% no images per Row
noExPerRow = floor(N / noImageRows);

img2Disp = cell(noImageRows, noExPerRow);


for i = 1:noImageRows
    
    baseInd = (i - 1) * noExPerRow;  

    for j = 1:noExPerRow
    
        selInd =  baseInd + j;
              
        img = reshape(weights(:,selInd),row,col);

        img(:,end+1:end+3) = minValue; 
        img(end+1:end+3,:) = minValue; 
      
        img2Disp{i,j} = img;
 
    end
 
end

img2DispFinal = cell2mat(img2Disp);
imagesc(img2DispFinal,[minValue,maxValue]); colormap gray; axis equal; axis off;


drawnow;


