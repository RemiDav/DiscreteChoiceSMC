function [ SubParticles ] = InitSubParticles(subj,HyperParams,SupParticles,param,model)
%INITSUBPARTICLES Returns a param.G x param.S struct array of subparticles
    SubParticles = struct;
    if  strcmp(model,'Logit')
        for g=1:param.G
            for p=1:param.P
                parentp = SupParticles.particle{g,p};
                %% Draw Cluster Membership
                % Get previous subject's membership
                clust_data = [parentp.theta(:).clust];
                clust_data = reshape(clust_data,param.num_clust,numel(clust_data)/param.num_clust)';
                % Draw current subject's membership
                cluster_bin = mnrnd(1,(10+sum(clust_data(1:subj-1,:),1,'omitnan' )) / (10*param.num_clust+subj-1) )==1;
                SubParticles(g,p).clust = cluster_bin; 
                %% Draw Parameters
                SubParticles(g,p).r = gamrnd( parentp.ha_r(cluster_bin,:),1./parentp.hb_r(cluster_bin,:));
                SubParticles(g,p).beta = gamrnd( parentp.ha_beta(cluster_bin,:),1./parentp.hb_beta(cluster_bin,:));
                %% Get logPrior Proba
                SubParticles(g,p).logprior = logPrior(SubParticles(g,p),HyperParams{g},model,param);
            end
        end
    elseif  strcmp(model,'HBC-PNE')
        for g=1:param.G
            for p=1:param.P
                parentp = SupParticles.particle{g,p};
                %% Draw Cluster Membership
                % Get previous subject's membership
                clust_data = [parentp.theta(:).clust];
                clust_data = reshape(clust_data,param.num_clust,numel(clust_data)/param.num_clust)';
                % Draw current subject's membership
                cluster_bin = mnrnd(1,(10+sum(clust_data(1:subj-1,:),1,'omitnan' )) / (10*param.num_clust+subj-1) )==1;
                SubParticles(g,p).clust = cluster_bin; 
                %% Draw Parameters
                SubParticles(g,p).r = gamrnd( parentp.ha_r(cluster_bin,:),1./parentp.hb_r(cluster_bin,:));
                SubParticles(g,p).omega = gamrnd( parentp.ha_omega(cluster_bin,:),1./parentp.hb_omega(cluster_bin,:));
                SubParticles(g,p).sig = gamrnd( parentp.ha_sig(cluster_bin,:),1./parentp.hb_sig(cluster_bin,:));
                %% Get logPrior Proba
                SubParticles(g,p).logprior = logPrior(SubParticles(g,p),HyperParams{g},model,param);
            end
        end
    else
        error('InitParticle: unknown model used');
    end

end

