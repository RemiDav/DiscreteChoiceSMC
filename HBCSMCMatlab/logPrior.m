function [ logprior ] = logPrior( particle, HyperParams, model , param )
%LOGPRIOR Returns the log prior by approximately integrating out the hyper parameters
    logprior = struct;
    if strcmp(model,'Logit')
        logprior.r = log(sum( ...
            gampdf(particle.r,HyperParams.ha_r(particle.clust,:,:),1./HyperParams.hb_r(particle.clust,:,:)) ...
            ));
        logprior.beta = zeros(1,param.K);
        for k = 1:param.K
            logprior.beta(k) = log(sum( ...
                gampdf(particle.beta(k),HyperParams.ha_beta(particle.clust,k,:),1./HyperParams.hb_beta(particle.clust,k,:)) ...
                )); 
        end
        logprior.total = sum(logprior.beta) + logprior.r;
    elseif strcmp(model,'HBC-PNE')
        logprior.r = log(sum( ...
            gampdf(particle.r,HyperParams.ha_r(particle.clust,:,:),1./HyperParams.hb_r(particle.clust,:,:)) ...
            ));
        logprior.omega = zeros(1,param.K);
        logprior.sig = zeros(1,param.K);
        for k = 1:param.K
            logprior.omega(k) = log(sum( ...
                gampdf(particle.omega(k),HyperParams.ha_omega(particle.clust,k,:),1./HyperParams.hb_omega(particle.clust,k,:)) ...
                ));
            logprior.sig(k) = log(sum( ...
                gampdf(particle.sig(k),HyperParams.ha_sig(particle.clust,k,:),1./HyperParams.hb_sig(particle.clust,k,:)) ...
                )); 
        end
        logprior.total = sum(logprior.omega) + sum(logprior.sig) + logprior.r;
    else
        error('logPrior : unknown model');
    end
end

