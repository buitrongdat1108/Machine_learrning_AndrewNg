function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%X�c �?nh r? r�ng c�c dimension c?a c�c matrix

% X: mx(n+1)
% y: mx1
% theta: (n+1)x1
% J:scalar
% grad: (n+1)*1

z=X*theta;
h=sigmoid(z);
J=(1/m)*sum(-y.*log(h)-((1-y).*log(1-h)));
grad=(1/m)*(X'*(h-y));
% =============================================================

end
