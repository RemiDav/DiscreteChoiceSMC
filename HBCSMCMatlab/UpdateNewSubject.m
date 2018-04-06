function [ particle ] = UpdateNewSubject( particle, model, subj ,param )
%UPDATENEWSUBJECT Summary of this function goes here

% Get model info
K = param.K;

%model : 'logit', 'MLBA, 'PDN' or 'PDNUnitIndep'
%K : number of attributes
    if  strcmp(model,'Logit')
        particle.theta(subj,:) = [betarnd(3,1,1,1) gamrnd(1,1,1,K)];
    elseif strcmp(model,'PDNNew')
        particle.theta(subj,:) = [betarnd(3,1,1,1) gamrnd(1,0.5,1,1) gamrnd(1,1,1,K)];
    elseif strcmp(model,'RemiStand')
        particle.theta(subj,:) = [betarnd(3,1,1,1) gamrnd(1,0.5,1,1) gamrnd(1,1,1,K)];
    elseif strcmp(model,'HierarchicalProbit')
        %% Particles
        % [alpha sigma Omega(1,K)]
        % alphas are indep.
        % sigma ~ Gamma(1,sigma_0), omega ~ Gamma(1,omega_0)
        % Conjugacy rules for Gamma(1,.) and Gamma(.,.)
        if subj == 1
            particle.theta(1,:) = [betarnd(3,1) ...
                gamrnd(1,gamrnd(particle.hypertheta(1),particle.hypertheta(2)),1,1) ...
                gamrnd(ones(1,K),gamrnd(particle.hypertheta(3),particle.hypertheta(4),1,K))];
        else
            %Draw from predictive conditional on previous subjects
            alpha_hyper_sigma = particle.hypertheta(1) + subj - 1; 
            alpha_hyper_beta = (particle.hypertheta(3) + subj - 1).*ones(1,K);
            beta_hyper_sigma = particle.hypertheta(2) + sum(particle.theta(1:subj-1,2),1);
            beta_hyper_beta = particle.hypertheta(4) + sum(particle.theta(1:subj-1,3:2+K),1);
            particle.theta(subj,:) = [betarnd(3,1) ...
                gbetaprrnd(1,alpha_hyper_sigma,1,beta_hyper_sigma) ...
                gbetaprrnd(ones(1,K),alpha_hyper_beta,ones(1,K),beta_hyper_beta,1,K) ];
        end
    else
        error('UpdateNewSubject: unknown model used');
    end


end

