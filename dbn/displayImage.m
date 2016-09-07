function [] = displayImage( image1D, h, w )
%DISPLAYIMAGE Summary of this function goes here
%   Detailed explanation goes here
image = mat2gray(reshape(image1D, h, w));
imshow(image);

end

