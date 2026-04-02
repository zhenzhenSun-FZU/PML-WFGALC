function [test_outputs,predict_target,obj,time] = WFGALC_TrainAndPredict(train_data, train_p_target,train_real_target,test_data,param)

% PML-DLLC process
[b, Lambda, train_outputs, LD,obj, time] = PML_WFGALC(train_data,train_p_target,param);
% kernel matrix
ker  = 'rbf'; %type of kernel function ('lin', 'poly', 'rbf', 'sam')
par  = 1*mean(pdist(train_data)); %parameter of kernel function
Kt = kernelmatrix(ker,test_data',train_data',par); 
test_outputs = 1/(2*param.lambda1)*Kt*Lambda+repmat(b, size(test_data,1), 1);

test_outputs = test_outputs';
if param.tune == 1
    [tau] = TuneThreshold(train_outputs', train_real_target');
    predict_target = Predict(test_outputs,tau);
else    
    predict_target  = (test_outputs>= 0.2);    
end
predict_target  = double(predict_target);

end

