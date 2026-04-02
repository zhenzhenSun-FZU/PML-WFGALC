function L = Laplacian(X)
% Calculate the Laplacian Matrix of X
%
%   X£º NxD array
    options = [];
    options.NeighborMode = 'KNN';
    options.k = 5;
    options.WeightMode = 'HeatKernel';
    options.t = 1;

    S = constructW(X, options) ;  
    L = diag(sum(S, 2))- S; 
end

