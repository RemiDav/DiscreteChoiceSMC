%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimation / Model-comparison Example File %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear

%% Load Backup
% if you want to load a backup from a previous estimation (set to [] if
% not)
backup_file = 'backup1.mat';
%% Estimation Parameters
% You can create your own parameters that will be passed to the likelihood
% function and the particles initialization functions
param = struct;
param.G = 3; % Number of particles group
param.P = 256; % Number of particles per group
param.Adaptive = true; % Use the adaptive SMC (see Durham, Geweke 2014).
param.ress_threshold = 0.8;
param.Msteps = 40; % Number of mutate steps
param.Tag = 'StndVsHierPDN'; % This tag will be added to the output file
param.savefile = ['Analysis' filesep param.Tag sprintf('-%.0fx%.0f-M%.0f-',param.G,param.P,param.Msteps) datestr(datetime('now'),'yyyy-mm-dd-HH.MM') '.mat'];

% Models to use for estimations
% Each model should have a corresponding entry in the following files:
% InitParticle : Returns a draw from prior for one particle
% Mutate : The Metropolis-Hastings Mutation step
% ProbaChoice : The likelihood of one observation given a model and particle
param.Models = {'RemiStand';'HierarchicalProbit'};

% Model specific parameters
param.NormDraw = mvnrnd(zeros(4,1),eye(4),1000); % Draws for GHK
param.attrVals = cell(7,1);
param.attrNames = cell(7,1);
param.attrVals{1}= (5:25);          param.attrNames{1}="Annual fee ($)";
param.attrVals{2}= (0:0.1:0.5);    param.attrNames{2}="Transaction fee ($)";
param.attrVals{3}= (5:1:25);        param.attrNames{3}="Interest charged (%)";
param.attrVals{4}= 0:0.25:2.5;    param.attrNames{4}="Reward (%)";
param.attrVals{5}= (0:0.25:2.5);    param.attrNames{5}="ATM surcharge ($)";
param.attrVals{6}= (0:0.25:2.5);    param.attrNames{6}="Currency conversion fee (%)";
param.attrVals{7}= [0,15,30,60,90,120];    param.attrNames{7}="Warranty on purchase (days)";
param.attrSign = [-1, -1, -1, 1, -1, -1, 1];
param.K = numel(param.attrVals); 
param.attrMax = zeros(1,param.K);
for k=1:param.K
    param.attrMax(k) = max(param.attrVals{k});
end

clear k
%% Data format
% Requires a cell array (e.g. SubjData{}) of size N x 1
% Each cell must contain a structure with at least 2 fields:
% SubjData{n}.Xs : cell array of T observations.
% - SubjData{n}.Xs{t}: Matrix of J(t) options x K(t) attributes
% SubjData{n}.ChoiceList : Vector of T x 1 Choices

% Load files list
fileslist = dir(['..\CC-Exp\Analysis\LabData' filesep 'Optim-BRLAB*.mat']);
num_subj = size(fileslist,1);
Data = cell(num_subj,1);
subjList = [];
for file = 1:num_subj
    Data{file} = load([fileslist(file).folder filesep fileslist(file).name]);
    % add files to subjList if they satisfy conditions (min # answers,
    % consistency,...)
    if numel(Data{file}.ChoiceList) > 50
        if isfield(Data{file},'ConsistencyCheck')
            if mean(Data{file}.ConsistencyCheck) > 0.3
                subjList = [subjList;file];
            end
        else
            subjList = [subjList;file];
        end
    end
end
% Keep only valid subjects
SubjData = Data(subjList);

clear Data file fileslist num_subj subjList

%% Estimation
EstimationOutput = EstimationAdaptiveSMC( SubjData, param, backup_file)
