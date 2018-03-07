function [ logLik ] = LogLikelihood( Xs, ChoiceList, subj , model , particle, param )
%LIKELIHOOD Summary of this function goes here
%   Detailed explanation goes here
logLik = 0;
T = numel(Xs);
for t = 1:T
   proba_choice = ProbaChoice( Xs{t}, subj , model , particle, param );
   logLik = logLik + log(proba_choice(ChoiceList(t)));
end

end

