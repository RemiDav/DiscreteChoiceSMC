function [ EstimationOutput ] = EstimationAdaptiveSMC( SubjData, param, backup_file )
% External libraries
[filepath,~,~] = fileparts(mfilename('fullpath'));
addpath([filepath filesep 'ProbabilityDistributions'])

%% get data size
param.num_subj = numel(SubjData); 
M = numel(param.Models);

%% Initialize particles
Particles = cell(M,1);
for m=1:M
    Particles{m}.particle = cell(param.G,param.P);
    Particles{m}.weights = ones(param.G,param.P);
    % Initialize output values
    Particles{m}.log_marg_like = zeros(param.num_subj,param.G);
    Particles{m}.log_marg_like_total = zeros(param.G,1);
    Particles{m}.group_postmeans = cell(param.G,1);
    Particles{m}.postmeans = [];
    Particles{m}.group_postsd = cell(param.G,1);
    for g=1:param.G
        for p=1:param.P
            Particles{m}.particle{g,p} = InitParticle(m,param);
        end
    end
end

%% check if backup file provided and restore data
start_subj = 1;
if nargin > 2
   if exist(backup_file,'file')
       BackupData = load(backup_file);
       start_subj = BackupData.subj+1;
       Particles = BackupData.Particles;
       % check if subjects were added since backup
       if BackupData.param.num_subj < param.num_subj
           % Extend particles
           for m=1:M
               Particles{m}.log_marg_like(param.num_subj,1) = 0;
           end
       end
   end
end
clear BackupData
%% Check if ress threshold provided
if ~isfield(param,'ress_threshold')
    param.ress_threshold = 0.8;
end
%% Run SMC
for subj = start_subj:param.num_subj
    fprintf('Begin Subject %d\n',subj)
    num_obs = numel(SubjData{subj}.ChoiceList);
    for obs = 1:num_obs
        Particles = UpdateParticles( Particles, SubjData, subj, obs, param); 
    end
    %% Save temporary file in case of crash
    save(['backup1-' param.Tag  '.mat']);
    save(['backup2-' param.Tag  '.mat']);
end

%% Descriptive stats - Post Estimation
% save marginal likelihoods
log_marg_like = zeros(param.G,M);
for m=1:M
    log_marg_like(:,m) = Particles{m}.log_marg_like_total;
    %% Compute posterior means
    size_NK = size(Particles{m}.particle{1}.theta);
    Particles{m}.postmeans = nan(param.G,size_NK(1),size_NK(2));
    for g=1:param.G
        VectorizedTheta = nan(param.P,size_NK(2),size_NK(1));
        for p = 1:param.P
            VectorizedTheta(p,:,:) = Particles{m}.particle{g,p}.theta';
        end
        Particles{m}.group_postmeans{g} = squeeze(mean(VectorizedTheta,1))';
        Particles{m}.group_postsd{g} = squeeze(std(VectorizedTheta,[],1))';
        Particles{m}.postmeans(g,:,:) = Particles{m}.group_postmeans{g};
    end
    Particles{m}.postmeans = squeeze(mean(Particles{m}.postmeans));
end

%% Save output
% create output object
EstimationOutput = struct;
% Save Data
EstimationOutput.param = param;
EstimationOutput.Particles = Particles;
EstimationOutput.log_marg_like = log_marg_like;
%%
save(param.savefile,'EstimationOutput','param');

end

