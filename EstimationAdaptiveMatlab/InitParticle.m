function [ particle ] = InitParticle( m , param)
%INITPARTICLE Returns a Theta particle drawn from prior
% Get model info
model = param.Models{m};
K = param.K;
N = param.num_subj;
particle = struct;
%model : 'logit', 'MLBA, 'PDN' or 'PDNUnitIndep'
%K : number of attributes
    if  strcmp(model,'Logit')
        particle.theta = nan(N,K+1);
    elseif strcmp(model,'PDNNew')
        particle.theta =  nan(N,K+2);
    elseif strcmp(model,'RemiStand')
        particle.theta =  nan(N,K+2);
    elseif strcmp(model,'HierarchicalProbit')
        particle = struct;
        %% Hyper prior
        % Prior for sigma_0 ~ Gamma(1,1), omega_0k ~ Gamma(1,1)
        particle.hypertheta = [1,1,1,1];
        %% Particles
        % [alpha sigma Omega(1,K)]
        particle.theta = nan(N,K+2);
    else
        error('InitParticle: unknown model used');
    end

end

