function [b, Lambda, model_outputs,LD,obj, time] = PML_WFGALC(train_data,train_p_target,param)
%PML_DLLC is a partial multi-label learning algorithm 
%    Syntax
%
%       [b, Lambda,obj] = PML_DLLC(train_data,train_p_target,param,k,ker,par);
%
%    Description
%      
%      parameters,
%           train_data     - An m * d array, the ith instance of training instance is stored in train_data(i,:)
%           train_p_target - An m * q array, if the jth class label is one of the partial labels for the ith training instance, then train_p_target(i,j) equals +1, otherwise train_p_target(i,j) equals 0
%           param          - Parameters of regularization terms
%            k             - Number of neighbors,here we set k=10
%      and returns,
%           b, Lambda      - weigthts of predictive model
%           obj            -objective function value per iteration
%           LD             - A mxq array,the label distributions of train_data
if nargin < 3
    param.lambda1 = 1;
    param.lambda2 = 1;
    param.lambda3 = 1;
    param.gamma = 1e5;
    param.k = 20;
    param.iter = 100;
end
if nargin < 2
	error('Not enough input parameters!');
end
[m,q] = size(train_p_target);
lambda1 = param.lambda1;
lambda2 = param.lambda2;
lambda3 = param.lambda3;
k = param.k;
gamma =  param.gamma;
par = 1*mean(pdist(train_data));
ker = 'rbf';
MaxIter = param.Iter;
% Initialize the label distribution
LD = train_p_target./(repmat(sum(train_p_target,2),1,q));

% similarity matrix of feature space 
S_f = build_data_manifold(train_data, train_p_target, k);
M = (eye(m,m)-S_f)'*(eye(m,m)-S_f);
M_plus = (abs(M)+M)/2;
M_minus = (abs(M)-M)/2;
%initialize local label correlation
options = [];
options.NeighborMode = 'KNN';
options.k = 0;
S_l = constructW(train_p_target', options) ; 
S_l = S_l./repmat(sum(S_l,2),1,q);
A = S_l;
L = diag(sum(A+A',2))-A-A';
L_plus = (abs(L)+L)/2;
L_minus = (abs(L)-L)/2;
Q = ones(q,q);
N = ones(m,q);
e = ones(q,1);

[b,Lambda,model_outputs,obj1] = MulRegression(train_data, LD, lambda1, par, ker);
%model_outputs(find(train_p_target==0)) = 0;
%model_outputs = model_outputs./repmat(sum(model_outputs,2),1,q);
obj_old = obj1 + lambda2*trace(LD'*M*LD) + lambda3*trace(LD*L*LD') + lambda3*(norm(A-S_l,2))^2+ gamma*(norm(LD*ones(q,1)-ones(m,1),2))^2;
obj = [];
obj = [obj; obj_old];

iter = 0;
tic;
while iter<=MaxIter
    % update label distribution
    model_outputs_plus = (abs(model_outputs)+model_outputs)/2;
    model_outputs_minus = (abs(model_outputs)-model_outputs)/2;
    
    term_1 = LD+model_outputs_minus+lambda2*M_minus*LD+lambda3*LD*L_minus+gamma*N+eps;
    term_2 = model_outputs_plus+lambda2*M_plus*LD+lambda3*LD*L_plus+gamma*LD*Q+eps;
    LD = LD.*sqrt(term_1./term_2);
    %LD = LD./repmat(sum(LD,2),1,q);
    
    %update local label correlation
    T = S_l + 1/2* EuDist2(LD'); U = 1/2*(T+T');
    A1 = U+(q+e'*U*e)*(e*e')/q^2-1/q*U*(e*e')-1/q*(e*e')*U;
    A = max(A1, 0);
    A = A-diag(diag(A));
    A = A./repmat(sum(A,2),1,q);
    L = diag(sum(A+A',2))-A-A';
    L_plus = (abs(L)+L)/2;
    L_minus = (abs(L)-L)/2;
    
    %update model weights
    [b,Lambda,model_outputs,obj1] = MulRegression(train_data, LD, lambda1, par, ker);
    %model_outputs(find(train_p_target==0)) = 0;
    %model_outputs = model_outputs./repmat(sum(model_outputs,2),1,q);
    
    iter = iter+1;
    obj_new = obj1 + lambda2*trace(LD'*M*LD) + lambda3*trace(LD*L*LD') + lambda3*(norm(A-S_l,2))^2+ gamma*(norm(LD*ones(q,1)-ones(m,1),2))^2;
    obj = [obj; obj_new];
    rate = abs(obj_old-obj_new)/obj_new;
    disp(['obj',num2str(iter),'=',num2str(obj_new),',',num2str(rate)]);
    if rate<1e-5 && iter >= 5
        break;
    end
    obj_old = obj_new;
end
time = toc;
end

