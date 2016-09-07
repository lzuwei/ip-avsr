function [ imMatrix ] = resizeImages( dataMatrix, oldHt, oldWt, newHt, newWt )
%RESIZEIMAGES Summary of this function goes here
%   Detailed explanation goes here
[noIm, ~] = size(dataMatrix);
imMatrix = zeros(noIm, newWt * newHt);

for i = 1:noIm
    img = reshape(dataMatrix(i,:), oldHt, oldWt);
    img = imresize(img, [newHt, newWt]);
    imMatrix(i,:) = reshape(img, 1, newHt * newWt);
end

end