function [ proba_choice ] = ProbaChoice( X, subj , model , particle, param )
% X: J x K matrix of choice set
% model: the true model to use
% (returns) proba_choice : Jx1 vector of choice probabilities
% it is recommended to make sure that the probability of a choice is not 0

J = size(X,1);
K = size(X,2);

if strcmp(model,'Logit')
    %True params
    alpha = particle.theta(subj,1);
    Beta = (param.attrSign .* particle.theta(subj,2:end))';
    %utility computation
    u_x = X.^alpha;
    v = zeros(J,1);
    for j=1:J
        v(j) = sum(Beta .* u_x(j,:)'); 
    end
    v = v - max(v); %avoid overflow
    sum_exp_v = sum(exp(v));
    proba_choice = exp(v)./sum_exp_v;
    %mixture 99.9% model and 0.1% unif
    proba_choice = 0.99 .* proba_choice + 0.01/J;
elseif strcmp(model,'PDNNew')
    %True params
    alpha = particle.theta(subj,1);
    sigma = particle.theta(subj,2);
    Omega = particle.theta(subj,3:3+K-1);
    %utility computation
    u_x = X.^alpha;
    v = zeros(J,1);
    unnorm_u = (param.attrSign .* u_x)';
    for j=1:J
        u_y = u_x;
        u_y(j,:)=[];
        norm_coefs = sum(1 ./ (sigma + Omega .* (u_x(j,:) + u_y)),1);%./(J-1);
        v(j) = norm_coefs * unnorm_u(:,j); 
    end
    v = v - max(v); %avoid overflow
    sum_exp_v = sum(exp(v));
    proba_choice = exp(v)./sum_exp_v;
    %mixture 99.9% model and 0.1% unif
    proba_choice = 0.99 .* proba_choice + 0.01/J;
elseif strcmp(model,'RemiStand')
    %True params
    alpha = particle.theta(subj,1);
    sigma =particle.theta(subj,2);
    Beta = (param.attrSign .* particle.theta(subj,3:3+K-1))';
    %utility computation
    x_mean = mean(X,1);
    sd_x = std(X,[],1) * alpha + sigma;
    x_standardized = (X - x_mean) ./ sd_x;
    attr_signal = 1 ./ (1+exp(-x_standardized));
    v = attr_signal * Beta;
    v = v - max(v); %avoid overflow
    sum_exp_v = sum(exp(v));
    proba_choice = exp(v)./sum_exp_v;
    %mixture 99.9% model and 0.1% unif
    proba_choice = 0.99 .* proba_choice + 0.01/J;
elseif strcmp(model,'HierarchicalProbit')
    % [alpha sigma Omega(1,K)]
    %True params
    alpha = particle.theta(subj,1);
    sigma = particle.theta(subj,2);
    Beta = (param.attrSign)';
    Omega = particle.theta(subj,3:3+K-1);
    %utility computation
    u_x = X.^alpha;
    v = zeros(J,1);
    unnorm_u = (param.attrSign .* u_x)';
    for j=1:J
        u_y = u_x;
        u_y(j,:)=[];
        norm_coefs = sum(1 ./ (sigma + Omega .* (u_x(j,:) + u_y) ),1);%./(J-1);
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

