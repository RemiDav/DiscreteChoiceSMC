function [ Particles ] = UpdateParticles( Particles, SubjData, subj, obs, param)
%UPDATEPARTICLES Update particles with a resample-move(SMC) algorithm


%% Get particles info
M = numel(param.Models);
for m=1:M
    model =  param.Models{m};
    %% If new subject, draw subject specific particles values
    if obs == 1
        for g=1:param.G
            for p=1:param.P
                Particles{m}.particle{g,p} = UpdateNewSubject( Particles{m}.particle{g,p} , model, subj ,param );
            end
        end
    end
    %% C Phase
    % Reweight each particle to take into account new observation
    logweights = Particles{m}.logweights;
    ChoiceSet = SubjData{subj}.Xs{obs};
    choice = SubjData{subj}.ChoiceList(obs);
    for g = 1:param.G
        for p = 1:param.P
            proba_choice = ProbaChoice( ChoiceSet, subj, model , Particles{m}.particle{g,p}, param );
            Particles{m}.particle{g,p}.log_lik_subj(subj) = Particles{m}.particle{g,p}.log_lik_subj(subj) + log(proba_choice(choice));
            logweights(g,p) = logweights(g,p) + log(proba_choice(choice));
        end
    end
    % Compute relative ESS
    ress = sum(sum(exp(logweights)))^2 / (param.G*param.P*sum(sum(exp(2 .*logweights))));
    
    % Save marginal likelihood for current observation
    log_w_bar = log( sum(exp(logweights),2) ./ sum(exp(Particles{m}.logweights),2) );%(param.G*param.P));
    
    Particles{m}.log_marg_like(subj,:) = Particles{m}.log_marg_like(subj,:) + log_w_bar';
    Particles{m}.log_marg_like_total = Particles{m}.log_marg_like_total + log_w_bar;
    Particles{m}.logweights = logweights - max(Particles{m}.logweights(:));
    fprintf('End C: Model %d ; RESS = %.5f ; %d - %d subject avg logML = %.5f\n',m,ress,subj,obs,mean(Particles{m}.log_marg_like(subj,:)) );
    
    %% S Phase : Importance Resampling, within particle groups
    if ~param.Adaptive || ress < param.ress_threshold || obs == numel(SubjData{subj}.ChoiceList) || obs < 5
        for g=1:param.G
            %Resample if weights are not all the same
            if length(unique(logweights(g,:))) > 1
                resample = drawidx(param.P,exp(logweights(g,:)));
                % Make a copy of the drawn particles
                NewTheta = cell(1,param.P);
                for p=1:param.P
                   NewTheta{1,p}=Particles{m}.particle{g,resample(p)};
                end
                % Copy back the drawn particle to the correct position
                for p=1:param.P
                   Particles{m}.particle{g,p} = NewTheta{1,p};
                end
            end
        end
        Particles{m}.logweights = zeros(param.G,param.P);
        clear temp_Particles;
    
        %% M phase
        accept_count = zeros(param.G,param.P);
        for g = 1:param.G
            % Get the Cholesky decomposition of the Covariance matrix for each
            % subject's parameters;
            chol_cov_theta = CholCovTheta( Particles{m}.particle(g,:), param );
            % Copy particles for parallel looping
            temp_Particles = Particles{m}.particle(g,:);
            try
                parfor p = 1:param.P
                    [temp_Particles{p},accept_count(g,p)] = Mutate(SubjData, subj, obs, model, temp_Particles{p},chol_cov_theta,param);
                end
            catch ME
                temp_Particles
            end
            Particles{m}.particle(g,:) = temp_Particles;
        end
        fprintf('End M: Model %d ; accept. ratio = %.5f\n',m,mean(squeeze(sum(accept_count,2)) ./ param.P));
    end
end

end

