function [ VectHyperParams ] = GetHyperParams( Particles,param, model)
%GETHYPERPARAMS Returns verctorized hyper paramaeters
VectHyperParams = cell(param.G,1);
for g = 1:param.G
    VectHyperParams{g} = struct;
    if strcmp(model,'Logit')
        VectHyperParams{g}.ha_r = nan(param.num_clust,1,param.P);
        VectHyperParams{g}.hb_r = nan(param.num_clust,1,param.P);
        VectHyperParams{g}.ha_beta = nan(param.num_clust,param.K,param.P);
        VectHyperParams{g}.hb_beta = nan(param.num_clust,param.K,param.P);
        for p = 1:param.P
            VectHyperParams{g}.ha_r(:,:,p) = Particles.particle{g,p}.ha_r;
            VectHyperParams{g}.hb_r(:,:,p) = Particles.particle{g,p}.hb_r;
            VectHyperParams{g}.ha_beta(:,:,p) = Particles.particle{g,p}.ha_beta;
            VectHyperParams{g}.hb_beta(:,:,p) = Particles.particle{g,p}.hb_beta;
        end
        
    elseif strcmp(model,'HBC-PNE')
        VectHyperParams{g}.ha_r = nan(param.num_clust,1,param.P);
        VectHyperParams{g}.hb_r = nan(param.num_clust,1,param.P);
        VectHyperParams{g}.ha_omega = nan(param.num_clust,param.K,param.P);
        VectHyperParams{g}.hb_omega = nan(param.num_clust,param.K,param.P);
        VectHyperParams{g}.ha_sig = nan(param.num_clust,param.K,param.P);
        VectHyperParams{g}.hb_sig = nan(param.num_clust,param.K,param.P);
        for p = 1:param.P
            VectHyperParams{g}.ha_r(:,:,p) = Particles.particle{g,p}.ha_r;
            VectHyperParams{g}.hb_r(:,:,p) = Particles.particle{g,p}.hb_r;
            VectHyperParams{g}.ha_omega(:,:,p) = Particles.particle{g,p}.ha_omega;
            VectHyperParams{g}.hb_omega(:,:,p) = Particles.particle{g,p}.hb_omega;
            VectHyperParams{g}.ha_sig(:,:,p) = Particles.particle{g,p}.ha_sig;
            VectHyperParams{g}.hb_sig(:,:,p) = Particles.particle{g,p}.hb_sig;
        end
        
    else
        error('GetHyperParams : unknown model');
    end
end

end

