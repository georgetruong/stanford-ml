function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%      

% Formatting for debugging
% format short g;
% output_precision(2);

% Calculate dimensions of X
m = size(X, 1);
num_features = size(X, 2);

% Calculate mean and standard deviation for each feature in X
for j = 1:num_features
    mu(1, j) = mean(X(:, j));       
    sigma(1, j) = std(X(:, j));     
end

% Normalize data in X
for i = 1:m
    for j = 1:num_features
        X_norm(i, j) = (X_norm(i, j) - mu(1, j)) / sigma(1, j);
    end
end

% ============================================================

end
