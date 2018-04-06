function [ logLik ] = LogLikelihoodML( Xs, ChoiceList, subj , model , theta, param )
%LIKELIHOOD Summary of this function goes here
%   Detailed explanation goes here
particle = struct;
particle.theta=theta;
logLik = 0;
T = numel(Xs);
proba_c = @(X) ProbaChoice( X, subj , model , particle, param );
probas = cellfun(proba_c,Xs,'UniformOutput',0);
for t = 1:T
   proba_choice = probas{t};
   logLik = logLik + log(proba_choice(ChoiceList(t)));
end
% for t = 1:T
%    proba_choice = ProbaChoice( Xs{t}, subj , model , particle, param );
%    logLik = logLik + log(proba_choice(ChoiceList(t)));
% end

end

