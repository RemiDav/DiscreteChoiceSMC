function [ SubParticles,Likelihoods,logweights ] = UpdateParticles(SubParticles,HyperParams,Likelihoods, logweights, CurSubjData, subj, obs, param)
%UPDATEPARTICLES Update particles with a resample-move(SMC) algorithm


%% Get particles info
M = numel(param.Models);
for m=1:M
    model =  param.Models{m};
    %% C Phase
    % Reweight each particle to take into account new observation
    oldweights = logweights;
    for g = 1:param.G
        for p = 1:param.P
            proba_choice = ProbaChoice( CurSubjData,obs, model , SubParticles{m}(g,p), param );
            SubParticles{m}(g,p).log_like = SubParticles{m}(g,p).log_like + log(proba_choice);
            logweights(g,p) = logweights(g,p) + log(proba_choice);
        end
    end
    % Compute relative ESS
    ress = sum(sum(exp(logweights)))^2 / (param.G*param.P*sum(sum(exp(2 .*logweights))));
    
    % Save marginal likelihood for current observation
    log_w_bar = log( sum(exp(logweights),2) ./ sum(exp(oldweights),2) );%(param.G*param.P));
    Likelihoods(m).log_marg_like_obs(subj,obs,:) = log_w_bar;
    Likelihoods(m).log_marg_like_subj(subj,:) = Likelihoods(m).log_marg_like_subj(subj,:) + log_w_bar';
    Likelihoods(m).log_marg_like_total = Likelihoods(m).log_marg_like_total + log_w_bar;
    logweights = logweights - max(logweights(:));
    fprintf('End C: Model %d ; RESS = %.5f ; (obs %d)\n',m,ress,obs );
    
    %% S Phase : Importance Resampling, within particle groups
    if ~param.Adaptive || ress < param.ress_threshold || obs == numel(CurSubjData.Ys) || obs < 5
        for g=1:param.G
            %Resample if weights are not all the same
            if length(unique(logweights(g,:))) > 1
                resample = drawidx(param.P,exp(logweights(g,:)));
                % Make a copy of the drawn particles
                NewSubP = SubParticles{m}(g,:);
                for p=1:param.P
                   NewSubP(1,p)=SubParticles{m}(g,resample(p));
                end
                % Copy back the drawn particle to the correct position
                SubParticles{m}(g,:) = NewSubP(1,:);
%                 for p=1:param.P
%                    Particles{m}.particle{g,p} = NewSubP{1,p};
%                 end
            end
        end
        logweights = zeros(param.G,param.P);
        clear NewSubP;
    
        %% M phase
        accept_count = zeros(param.G,param.P);
        for g = 1:param.G
            % Copy particles for parallel looping
            temp_Particles = SubParticles{m}(g,:);
            GroupHyperParams = HyperParams{m}{g};
            try
                parfor p = 1:param.P
                    [temp_Particles(p),accept_count(g,p)] = Mutate(CurSubjData, obs, model, temp_Particles(p),GroupHyperParams,param);
                end
            catch ME
                temp_Particles
            end
            SubParticles{m}(g,:) = temp_Particles;
        end
        fprintf('End M: Model %d ; accept. ratio = %.5f\n',m,mean(squeeze(sum(accept_count,2)) ./ param.P));
    end
end

end

