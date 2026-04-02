clc
clear
addpath(genpath(pwd));

dataset = 'mirflickr';
dst_folder = "results";
n_fold = 5;
% load data
path=strcat('./data/',dataset,'/');
datapath=strcat(path,'data.mat');
load(datapath);
targetpath=strcat(path,'target.mat');
load(targetpath);
partial_label_path=strcat(path,'partial_labels.mat');
load(partial_label_path);

% delete unlabeled samples
indx = find((sum(target,2))==0);
data(indx,:) = [];
target(indx,:) = [];
partial_labels(indx,:) = [];

%% Normalization
data  = double(data);
temp_data = data + eps;
n_sample = size(data, 1);
temp_data = temp_data./repmat(sqrt(sum(temp_data.^2,1)),size(temp_data,1),1);
if sum(sum(isnan(temp_data)))>0
   temp_data = temp_data+eps;
   temp_data = temp_data./repmat(sqrt(sum(temp_data.^2,1)),size(temp_data,1),1);
end
% preprocess
n_test = round(n_sample / n_fold);    
% create save_folder
save_folder = fullfile(dst_folder,dataset);
if exist(save_folder,'dir')==0
    mkdir(save_folder);
end
        
param.lambda1 = 1e1; param.lambda2 = 1e-3; param.lambda3 = 1e3;
param.k = 20; param.tune = 1; param.Iter = 100; param.gamma = 1e5;
% n_fold validation and evaluation
Result = zeros(15,n_fold);
for i = 1:n_fold
    fprintf('Data processing, Cross validation: %d\n', i);
    % split data
    start_idx = (i-1)*n_test + 1;
    if i == n_fold
         test_idx = start_idx : n_sample;
    else
         test_idx = start_idx:start_idx + n_test - 1;
    end
    II = 1:n_sample;
    train_idx = setdiff(II, test_idx);
    train_data = temp_data(train_idx, :);
    train_target = target(train_idx,:);
    train_p_target = partial_labels(train_idx,:);
    test_data = temp_data(test_idx, :);
    test_target = target(test_idx,:);
    [test_outputs,predict_target,obj,time] = WFGALC_TrainAndPredict(train_data, train_p_target,train_target,test_data,param);
    Result(:,i) = EvaluationAll(predict_target,test_outputs,test_target');
end
%% the average results  
Avg_Result = zeros(15,2);
Avg_Result(:,1)=mean(Result,2);
Avg_Result(:,2)=std(Result,1,2);
% save eavluation results
save_path = fullfile(save_folder,'results.mat');
save(save_path, 'Result','Avg_Result');     