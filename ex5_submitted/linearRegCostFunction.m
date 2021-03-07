function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);
% You need to return the following variables correctly 
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
lin = 1 / m / 2 * sum((X * theta - y).^ 2);
reg = lambda / m / 2 * sum(theta(2:n).^2);
J = lin + reg;
 
grad0 = 1 / m * X(:,1)' * (X*theta - y);
grad1 = 1 / m * X(:,2:end)' * (X*theta - y)  + lambda / m * theta(2:n);

grad(1) = grad0;
grad(2:end) = grad1;











% =========================================================================

end
