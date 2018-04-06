function [ particle , accept ] = Mutate(CurSubjData, obs, model, particle,HyperParams, param)
%% Number of Msteps varies accrding to observation numbers
% More Msteps are done for the first few observations
if obs < 10
    MSteps = 10 * param.Msteps;
else
    MSteps = param.Msteps;
end
%% Mutate particle
accept=0;
if  strcmp(model,'Logit')
    logLikTheta = particle.log_like;
    for m = 1 : MSteps
        %% joint resampling
        propTheta = particle;
        % Beta proposal
        step = gamrnd(100,0.01,1,param.K);
        propTheta.beta = propTheta.beta .* step;
        logQRatio = sum(- 198 .* log(step) + 100 .* (step - 1./step));
        % r proposal
        step = gamrnd(100,0.01);
        propTheta.r = propTheta.r .* step;
        logQRatio = logQRatio + sum(- 198 .* log(step) + 100 .* (step - 1./step));
        % Compute Prior Ratio
        propTheta.logprior = logPrior( propTheta, HyperParams, model, param );
        logPriorRatio = propTheta.logprior.total - particle.logprior.total;
        logLikProp = LogLikelihood( CurSubjData, obs, model , propTheta, param );
        %accept-reject
        if log(rand()) <= logPriorRatio + logLikProp - logLikTheta + logQRatio
            accept=accept + 1/MSteps;
            particle = propTheta;
            logLikTheta = logLikProp;
        end
    end
    particle.log_like = logLikTheta;

elseif  strcmp(model,'HBC-PNE')
    logLikTheta = particle.log_like;
    for m = 1 : MSteps
        %% joint resampling
        propTheta = particle;
        % Omega proposal
        step = gamrnd(100,0.01,1,param.K);
        propTheta.omega = propTheta.omega .* step;
        logQRatio = sum(- 198 .* log(step) + 100 .* (step - 1./step));
        % Sigma proposal
        step = gamrnd(100,0.01,1,param.K);
        propTheta.sig = propTheta.sig .* step;
        logQRatio = logQRatio + sum(- 198 .* log(step) + 100 .* (step - 1./step));
        % r proposal
        step = gamrnd(100,0.01);
        propTheta.r = propTheta.r .* step;
        logQRatio = logQRatio + sum(- 198 .* log(step) + 100 .* (step - 1./step));
        % Compute Prior Ratio
        propTheta.logprior = logPrior( propTheta, HyperParams, model, param );
        logPriorRatio = propTheta.logprior.total - particle.logprior.total;
        logLikProp = LogLikelihood( CurSubjData, obs, model , propTheta, param );
        %accept-reject
        if log(rand()) <= logPriorRatio + logLikProp - logLikTheta + logQRatio
            accept=accept + 1/MSteps;
            particle = propTheta;
            logLikTheta = logLikProp;
        end
    end
    particle.log_like = logLikTheta;


elseif strcmp(model,'HierarchicalProbit')
    %% If it is a new subject,update all previous subjects' parameter
    if obs == 1 && subj > 1
        for ss = 1:subj-1
            num_obs = numel(CurSubjData{ss}.ChoiceList);
            logLikTheta = LogLikelihood( CurSubjData{ss}.Xs(1:num_obs), CurSubjData{ss}.ChoiceList, ss , model , particle, param );
            %Get sufficient statistics for other subject's parameters
            other_subj_list = (1:size(particle.theta,1))~=ss & (1:size(particle.theta,1)) <= subj;
            q = sum(particle.theta(other_subj_list,:),1);
            q(2) = q(2) + particle.hypertheta(2);
            q(3:end) = q(3:end) + particle.hypertheta(4);
            a_prime = particle.hypertheta([1 3])+subj-1;
            for m = 1 : 2
                %% joint resampling
                propTheta = struct;
                propTheta.theta = particle.theta(ss,:);
                propTheta.theta = propTheta.theta + mvnrnd_chol(chol_cov_theta{ss}/8);
                if all(propTheta.theta > 0) && propTheta.theta(1) <=1
                    logPriorRatio = 2 * log (propTheta.theta(1)/particle.theta(ss,1)) ...
                        - (1+a_prime(1)) * log((q(2)+propTheta.theta(2)) / (q(2)+particle.theta(ss,2)))  ...
                        - sum( (1+a_prime(2)) * log((q(3:end)+propTheta.theta(3:end)) ./ (q(3:end)+particle.theta(ss,3:end)))  );
                    logLikProp = LogLikelihood( CurSubjData{ss}.Xs(1:num_obs), CurSubjData{ss}.ChoiceList, 1 , model , propTheta, param );
                    %accept-reject
                    if log(rand()) <= logPriorRatio + logLikProp - logLikTheta
                        particle.theta(ss,:) = propTheta.theta;
                        logLikTheta = logLikProp;
                    end
                end
            end
        end
    end
    %% Update current subject's parameters
    logLikTheta = LogLikelihood( CurSubjData{subj}.Xs(1:obs), CurSubjData{subj}.ChoiceList(1:obs), subj , model , particle, param );
    %Get sufficient statistics for other subject's parameters
    q = sum(particle.theta(1:subj-1,:),1);
    q(2) = q(2) + particle.hypertheta(2);
    q(3:end) = q(3:end) + particle.hypertheta(4);
    a_prime = particle.hypertheta([1 3])+subj-1;
    for m = 1 : MSteps
        %% joint resampling
        propTheta = struct;
        propTheta.theta = particle.theta(subj,:);
        propTheta.theta = propTheta.theta + mvnrnd_chol(chol_cov_theta{subj}/8);
        if all(propTheta.theta > 0) && propTheta.theta(1) <=1
            logPriorRatio = 2 * log (propTheta.theta(1)/particle.theta(subj,1)) ...
                - (1+a_prime(1)) * log((q(2)+propTheta.theta(2)) / (q(2)+particle.theta(subj,2)))  ...
                - sum( (1+a_prime(2)) * log((q(3:end)+propTheta.theta(3:end)) ./ (q(3:end)+particle.theta(subj,3:end)))  );
            logLikProp = LogLikelihood( CurSubjData{subj}.Xs(1:obs), CurSubjData{subj}.ChoiceList(1:obs), 1 , model , propTheta, param );
            %accept-reject
            if log(rand()) <= logPriorRatio + logLikProp - logLikTheta
                accept=accept + 1/MSteps;
                particle.theta(subj,:) = propTheta.theta;
                logLikTheta = logLikProp;
            end
        end
    end
        
else
    error('Mutate : unknown model');
end


end

