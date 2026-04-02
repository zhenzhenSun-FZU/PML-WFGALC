function [b,alpha,train_outputs,obj] = MulRegression(train_data, train_p_target, lambda, par, ker)
% this model is actually a multi-output kernel ridge regression model
[m, ~] = size(train_data);

K = kernelmatrix(ker,train_data',train_data',par); % m by m, kernel matrix

I = eye(m, m);
H = (1/(2*lambda))*K+1/2*I; % m by m
m1 = ones(m, 1);
s = (H\m1)'; % m by 1
P = train_p_target;
b = s*P/(s*m1);
alpha = H\(P-repmat(b, m, 1));

train_outputs = 1/(2*lambda)*K*alpha+repmat(b, m, 1);
err = train_outputs - train_p_target;
obj = sum(sum(err.*err)) + lambda*1/(4*lambda^2)*trace(alpha'*K*alpha);

end