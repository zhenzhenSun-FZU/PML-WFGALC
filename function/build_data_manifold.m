function Outputs = build_data_manifold(train_data, logicalLabel, k)
%build_data_manifold is the first phase of PML-DLLC

NumSamp = size(train_data,1);
train_data = normr(train_data);
D = EuDist2(train_data);
%D = D./repmat(sum(D,2),1,NumSamp);
options = [];
options.NeighborMode = 'KNN';
options.k = 0;
options.WeightMode = 'Cosine';
C = constructW(logicalLabel, options) ; 
WD = (1-C).*D;
WD = WD-diag(diag(WD));
[~,neighbor]=sort(WD,2);
neighbor = neighbor(:,2:k+1);
options = optimoptions('quadprog',...
'Display', 'off','Algorithm','interior-point-convex' );
Outputs = zeros(NumSamp,NumSamp);
for i = 1:NumSamp
	train_data1 = train_data(neighbor(i,:),:);
	T = repmat(train_data(i,:),k,1)-train_data1;
	TT = T*T';
	lb = sparse(k,1);
	ub = ones(k,1);
	Aeq = ub';
	beq = 1;
	s = quadprog(2*TT, [], [],[], Aeq, beq, lb, ub,[], options);
	Outputs(i,neighbor(i,:)) = s';
end
end
