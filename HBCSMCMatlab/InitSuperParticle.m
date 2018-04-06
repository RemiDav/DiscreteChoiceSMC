function [ particle ] = InitSuperParticle( m , param)
%INITPARTICLE Returns a Theta particle drawn from prior
% Get model info
model = param.Models{m};
K = param.K;
N = param.num_subj;
particle = struct;
%model : 'logit', 'MLBA, 'PDN' or 'PDNUnitIndep'
%K : number of attributes
    if  strcmp(model,'Logit')
        particle.ha_r = nan(param.num_clust,1);
        particle.hb_r = gamrnd(2,1,param.num_clust,1);
        particle.ha_beta = nan(param.num_clust,K);
        particle.hb_beta = gamrnd(2,1,param.num_clust,K);
        % Sample ha_beta via importance resampling
        % Hyperparam: a=1, b=2, c=2
        for c = 1:param.num_clust
            draws_a = gamrnd(2,2,50,K);
            log_weights_a = draws_a/2 - log(draws_a) ...
                + 2 .* draws_a .* log(particle.hb_beta(c,:)) - 2.*log(gamma(draws_a));
            log_weights_a = log_weights_a  - max(log_weights_a,[],1 );
            weights_a = exp(log_weights_a)./sum(exp(log_weights_a),1);
            particle.ha_beta(c,:) = draws_a(mnrnd(1,weights_a')'==1)';
        end
        % Sample ha_r via importance resampling
        % Hyperparam: a=1, b=2, c=2
        for c = 1:param.num_clust
            draws_a = gamrnd(2,2,50,1);
            log_weights_a = draws_a/2 - log(draws_a) ...
                + 2 .* draws_a .* log(particle.hb_r(c,:)) - 2.*log(gamma(draws_a));
            log_weights_a = log_weights_a  - max(log_weights_a,[],1 );
            weights_a = exp(log_weights_a)./sum(exp(log_weights_a),1);
            particle.ha_r(c,:) = draws_a(mnrnd(1,weights_a')'==1)';
        end
        
        % Init subparticle container
        particle.theta = struct;
        for subj = 1:N
            particle.theta(subj).clust = nan(1,param.num_clust);
            particle.theta(subj).beta = nan(1,K);
            particle.theta(subj).r = nan;
            particle.theta(subj).log_like = 0;
            particle.theta(subj).logprior = 0;
        end
        
    elseif strcmp(model,'HBC-PNE')
        particle.ha_r = nan(param.num_clust,1);
        particle.hb_r = gamrnd(2,1,param.num_clust,1);
        particle.ha_omega = nan(param.num_clust,K);
        particle.hb_omega = gamrnd(2,1,param.num_clust,K);
        particle.ha_sig = nan(param.num_clust,K);
        particle.hb_sig = gamrnd(2,1,param.num_clust,K);
        % Sample ha_omega via importance resampling
        % Hyperparam: a=0.1, b=2, c=2
        for c = 1:param.num_clust
            draws_a = gamrnd(1,2,50,K);
            log_weights_a = draws_a/2 ...
                +(draws_a - 1) .* log(0.1) + 2 .* draws_a .* log(particle.hb_omega(c,:)) - 2.*log(gamma(draws_a));
            log_weights_a = log_weights_a  - max(log_weights_a,[],1 );
            weights_a = exp(log_weights_a)./sum(exp(log_weights_a),1);
            particle.ha_omega(c,:) = draws_a(mnrnd(1,weights_a')'==1)';
        end
        % Sample ha_sig via importance resampling
        for c = 1:param.num_clust
            draws_a = gamrnd(1,2,50,K);
            log_weights_a = draws_a/2 ...
                +(draws_a - 1) .* log(0.1) + 2 .* draws_a .* log(particle.hb_sig(c,:)) - 2.*log(gamma(draws_a));
            log_weights_a = log_weights_a  - max(log_weights_a,[],1 );
            weights_a = exp(log_weights_a)./sum(exp(log_weights_a),1);
            particle.ha_sig(c,:) = draws_a(mnrnd(1,weights_a')'==1)';
        end
        % Sample ha_r via importance resampling
        % Hyperparam: a=1, b=2, c=2
        for c = 1:param.num_clust
            draws_a = gamrnd(2,2,50,1);
            log_weights_a = draws_a/2 - log(draws_a) ...
                +(draws_a - 1) .* log(0.1) + 2 .* draws_a .* log(particle.hb_r(c,:)) - 2.*log(gamma(draws_a));
            log_weights_a = log_weights_a  - max(log_weights_a,[],1 );
            weights_a = exp(log_weights_a)./sum(exp(log_weights_a),1);
            particle.ha_r(c,:) = draws_a(mnrnd(1,weights_a')'==1)';
        end
        
        % Init subparticle container
        particle.theta = struct;
        for subj = 1:N
            particle.theta(subj).clust = nan(1,param.num_clust);
            particle.theta(subj).omega = nan(1,K);
            particle.theta(subj).sig = nan(1,K);
            particle.theta(subj).r = nan;
            particle.theta(subj).log_like = 0;
            particle.theta(subj).logprior = 0;
        end
    else
        error('InitParticle: unknown model used');
    end

end

