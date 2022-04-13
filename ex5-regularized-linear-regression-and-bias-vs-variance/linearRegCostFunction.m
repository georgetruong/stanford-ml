function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Set bias to 0 for regularization calculations
theta_tmp = theta;
theta_tmp(1) = 0;

h = X * theta;

% Calculate cost with regularization
J = (1/(2*m)) * sum((h-y) .^ 2);
J_reg = (lambda/(2*m)) * sum(theta_tmp .^ 2);
J = J + J_reg;

% Calculate gradient with regulation
grad_reg = (lambda/m) * theta_tmp';
grad = ((1/m) * sum((h-y) .* X));
grad = grad + grad_reg;

% =========================================================================

grad = grad(:);

end
