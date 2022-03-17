function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    h = X * theta;                  % Calculate h values


    % Fixed solution with only 2 theta parameters
    %
    % theta_0 = theta(1) - (alpha / m) * sum((h - y) .* X(:, 1));
    % theta_1 = theta(2) - (alpha / m) * sum((h - y) .* X(:, 2));
    % theta = [theta_0; theta_1]; 


    % Flexible solution with any number of theta parameters 

    num_params = rows(theta);       % Find # of theta parameters
    tmp = zeros(num_params, 1);     % Set a temp vector for storage

    for j = 1:num_params
        tmp(j) = theta(j) - (alpha / m) * sum((h - y) .* X(:, j));
    end

    % Simultaneously update all theta values
    theta = tmp;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
