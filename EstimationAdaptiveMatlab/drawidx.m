function [ idx ] = drawidx( numDest, Weights )
%DRAWIDX Summary of this function goes here
%   Detailed explanation goes here
idx = ones(1,numDest);
cumWeights = cumsum(Weights) / sum(Weights);
for dest = 1:numDest
    draw =rand();
    for source = 1:size(Weights,2)
        if draw <= cumWeights(source)
            idx(dest)=source;
            break;
        end
    end
end

end