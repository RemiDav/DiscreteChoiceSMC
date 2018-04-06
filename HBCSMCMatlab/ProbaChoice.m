function [ proba_choice ] = ProbaChoice( CurSubjData, obs, model , particle, param )
% CurSubjData.Xs{obs}: J x K matrix of choice set
% CurSubjData.Ys(obs): Observed outcome (Scalar)
% model: the true model to use
% (returns) proba_choice : likelihood of current observation
% it is recommended to make sure that the likelihood is not 0

X = CurSubjData.Xs{obs};
Y = CurSubjData.Ys(obs);
J = size(X,1);
K = size(X,2);


if strcmp(model,'Logit')
    %True params
    Beta = (param.attrSign .* particle.beta)';
    %utility computation
    u_x = X.^particle.r;
    v = zeros(J,1);
    for j=1:J
        v(j) = sum(Beta .* u_x(j,:)'); 
    end
    v = v - max(v); %avoid overflow
    exp_v = exp(v);
    proba_choice = exp_v(Y)/sum(exp_v);
elseif strcmp(model,'HBC-PNE')
    %True params
    beta = particle.sig;
    omega = particle.omega;
    %utility computation
    u_x = X.^particle.r;
    v = zeros(J,1);
    unnorm_u = (param.attrSign .* beta .* u_x)';
    for j=1:J
        u_y = u_x;
        u_y(j,:)=[];
        norm_coefs = sum(1 ./ (1 + omega .* (u_x(j,:) + u_y)),1);%./(J-1);
        v(j) = norm_coefs * unnorm_u(:,j); 
    end
    v = v - max(v); %avoid overflow
    exp_v = exp(v);
    proba_choice = exp_v(Y)/sum(exp_v);
elseif strcmp(model,'HierarchicalProbit')
    % [alpha sigma Omega(1,K)]
    %True params
    alpha = particle.theta(subj,1);
    sigma = particle.theta(subj,2);
    Beta = (param.attrSign)';
    omega = particle.theta(subj,3:3+K-1);
    %utility computation
    u_x = X.^alpha;
    v = zeros(J,1);
    unnorm_u = (param.attrSign .* u_x)';
    for j=1:J
        u_y = u_x;
        u_y(j,:)=[];
        norm_coefs = sum(1 ./ (sigma + omega .* (u_x(j,:) + u_y) ),1);%./(J-1);
        v(j) = norm_coefs * unnorm_u(:,j); 
    end
    v = v - max(v); %avoid overflow
    VarCov = eye(J,J);
    %Cholesky decomp + check positive def
    [CholeskyUpper,psd] = chol(VarCov);
    while psd
        VarCov = VarCov + 0.00001 * eye(size(VarCov,1));
        [CholeskyUpper,psd] = chol(VarCov);
    end
    sim = repmat(v',[size(param.NormDraw,1) 1]) + param.NormDraw(:,1:J) * CholeskyUpper;
    proba_choice = mean(sim == max(sim,[],2));
    %mixture 99.9% model and 0.1% unif
    proba_choice = 0.99 .* proba_choice + 0.01/J;
else
    error('ProbaChoice : unknown model');
end

end

