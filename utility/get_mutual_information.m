function [MIxy, IGxxy, IGxyy] = get_mutual_information(X,Y)
[~,num_fea] = size(X);
num_label = size(Y,2);
Y(Y==-1)=0;
[train, target]=trans(X,Y,2);
train = train-1;
for i = 1:num_fea
    for j = 1:num_label 
       MIxy(i,j) = mi(train(:,i),target(:,j));
    end
end
for k=1:num_label
    for i=1:num_fea
        for j=1:num_fea
           IGxxy{k,1}(i,j)=cmi(train(:,i),train(:,j),target(:,k))-mi(train(:,i),train(:,j));
        end
    end
    IGxxy{k,1} = IGxxy{k,1}-diag(diag(IGxxy{k,1}));
end
for k=1:num_fea
    for i=1:num_label
        for j=1:num_label
           IGxyy{k,1}(i,j)=cmi(target(:,i),target(:,j),train(:,k))-mi(target(:,i),target(:,j));
        end
    end
    IGxyy{k,1} = IGxyy{k,1}-diag(diag(IGxyy{k,1}));
end
end

