function features = DCT_Features(ROIs,NumberOfCoefs2Keep,visualize)
% Extract plain DCT features for given ROIs. 
% Coefs are computed over the whole ROIs (non-block approach). 
% Keep 2:NumberOfCoefs2Keep+1 zig-zag arranged coefs. 


if nargin<3 || isempty(visualize)
    visualize = 0;
end
nFrames = size(ROIs,3);
% Initialization of zigzag vectors
features = zeros(nFrames,NumberOfCoefs2Keep);

if visualize == 1
    figure;
end

for i=1:nFrames
    CurrentFrame = ROIs(:,:,i);
    DCTImage = dct2(CurrentFrame);
    DCTzigzagVector = zigzag(DCTImage);
    features(i,:) = DCTzigzagVector(2:NumberOfCoefs2Keep+1);   
    if visualize == 1
        imshow(DCTImage,[]), colormap(jet(64))
        drawnow
        pause(0.04)
    end
    clear DCTImage DCTzigzagvector;
end


end

