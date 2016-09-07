function [w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,b4,b5,b6,b7,b8] = extractNN( nn )
%EXTRACTNN Summary of this function goes here
%   Detailed explanation goes here
w1 = nn.W{1,1};
w2 = nn.W{1,2};
w3 = nn.W{1,3};
w4 = nn.W{1,4};
w5 = nn.W{1,5};
w6 = nn.W{1,6};
w7 = nn.W{1,7};
w8 = nn.W{1,8};
b1 = nn.biases{1,1};
b2 = nn.biases{1,2};
b3 = nn.biases{1,3};
b4 = nn.biases{1,4};
b5 = nn.biases{1,5};
b6 = nn.biases{1,6};
b7 = nn.biases{1,7};
b8 = nn.biases{1,8};
end

