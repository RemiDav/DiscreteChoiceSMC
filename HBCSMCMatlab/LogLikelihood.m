function [ logLik ] = LogLikelihood( CurSubjData, obs, model , particle, param )
%LIKELIHOOD Summary of this function goes here
%   Detailed explanation goes here
logLik = 0;
for t = 1:obs
   proba_choice = ProbaChoice( CurSubjData, t, model , particle, param );
   logLik = logLik + log(proba_choice);
end

end

