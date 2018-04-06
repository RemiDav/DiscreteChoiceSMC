function [ EstimationOutput ] = HBCSMC( SubjData, param, backup_file )
fprintf('=== Hierarchical Bayes + Clustering - SMC ===\n');
% External libraries
[filepath,~,~] = fileparts(mfilename('fullpath'));

%% get data size
param.num_subj = numel(SubjData); 
M = numel(param.Models);

%% Initialize super particles
Particles = cell(M,1);
HyperParams = cell(M,1);
Likelihoods = struct;
for m=1:M
    Particles{m}.particle = cell(param.G,param.P);
    % Initialize output values
    Likelihoods(m).log_marg_like_obs = zeros(param.num_subj,0,param.G);
    Likelihoods(m).log_marg_like_subj = zeros(param.num_subj,param.G);
    Likelihoods(m).log_marg_like_total = zeros(param.G,1);
    for g=1:param.G
        for p=1:param.P
            Particles{m}.particle{g,p} = InitSuperParticle(m,param);
            Particles{m}.particle{g,p}.log_lik_subj = zeros(param.num_subj,1);
        end
    end
    HyperParams{m} = GetHyperParams( Particles{m}, param, param.Models{m});
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
%% Check if params provided, else set to default
if ~isfield(param,'ress_threshold')
    param.ress_threshold = 0.8;
end
if ~isfield(param,'basepath')
    param.basepath = '.';
end

%% Run HBC-SMC
for subj = start_subj:param.num_subj
    fprintf('Begin Subject %d\n',subj)
    num_obs = numel(SubjData{subj}.Ys);
    Particles{m}.log_marg_like_obs = zeros(num_obs,param.G);
    %% Initialize Sub Particles for current subject
    SubParticles = cell(M,1);
    for m=1:M
        SubParticles{m} = InitSubParticles(subj,HyperParams{m},Particles{m},param,param.Models{m});
        for g=1:param.G
            for p=1:param.P
                SubParticles{m}(g,p).log_like = 0;
            end
        end
    end
    logweights = ones(param.G,param.P);
    %% Update Sub Particles
    for obs = 1:num_obs
        [SubParticles,Likelihoods,logweights] = UpdateParticles(SubParticles,HyperParams, Likelihoods, logweights, SubjData{subj},subj, obs, param); 
    end
    %% Assimilate
    Particles = Assimilate( Particles, SubParticles, subj, param );
    %% Update Super Particles
    for m=1:M
        % Update hyper parameters
        for g=1:param.G
            for p=1:param.P
                Particles{m}.particle{g,p} = UpdateSuperParticle( Particles{m}.particle{g,p},SubjData, subj, param.Models{m}, param );
            end
        end
        HyperParams{m} = GetHyperParams( Particles{m}, param, param.Models{m});
    end
    
    %% Save temporary file in case of crash
    save([param.basepath filesep 'backup1-' param.Tag  '.mat']);
    save([param.basepath filesep 'backup2-' param.Tag  '.mat']);
end

%% Save output
% create output object
EstimationOutput = struct;
% Save Data
EstimationOutput.param = param;
EstimationOutput.Particles = Particles;
EstimationOutput.Likelihoods = Likelihoods;
%%
try
    fprintf('Saving file...\n');
    save([param.basepath filesep param.savefile],'EstimationOutput','param');
    fprintf('File saved\n');
catch
    fprintf('Waring: File not saved\n');
end

end

