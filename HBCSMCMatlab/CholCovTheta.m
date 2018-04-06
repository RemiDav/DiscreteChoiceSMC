function [ chol_cov_theta ] = CholCovTheta( particles, param )
%CHOLCOVTHETA Returns the upper Cholesky decomposition of the covariance
% between particles. If the Covariance is not positive definite due to
% rounding error, it adds a small value to the diagonal.
% If the covariance cannot be computed, returns NaN.

% Find the matrix of parameters
% and create a big 3D matrix of dim (P x K x N)
if isfield(particles{1,1},'theta')
    if isreal(particles{1,1}.theta)
        size_NK = size(particles{1,1}.theta);
        VectorizedTheta = nan(param.P,size_NK(2),size_NK(1));
        for p = 1:param.P
            VectorizedTheta(p,:,:) = particles{1,p}.theta';
        end
    end
else
    chol_cov_theta = nan;
    return;
end

%% Compute the Cholesky decomposition for each subject
chol_cov_theta = cell(size_NK(1),1);
for n=1:size_NK(1)
    cov_theta = cov(squeeze(VectorizedTheta(:,:,n)),'omitrows');
    if ~isnan(cov_theta(1,1))
        % Get decomposition and check positive definiteness
        [chol_cov_theta{n},p] = chol(cov_theta);
        while p
            cov_theta = cov_theta + 0.00001 * eye(size(cov_theta,1));
            [chol_cov_theta{n},p] = chol(cov_theta);
        end
    end
end

end

