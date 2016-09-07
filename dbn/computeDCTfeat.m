function dctFeatures = computeDCTfeat(dataMatrix, w, h, noCoeff)



[noIm, dim] = size(dataMatrix);
imMatrix = zeros(h, w, noIm);

for i = 1:noIm
    imMatrix(:,:,i) = reshape(dataMatrix(i,:), h, w);
end

dctFeatures = DCT_Features(imMatrix,noCoeff,[]);