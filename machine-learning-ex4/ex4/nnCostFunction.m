function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% X = 5000x400
% Theta1 = 25x401
% Theta2 = 10x26

% forward prop
a1b = [ones(m, 1) X]; % 5000x401
a2 = sigmoid(a1b * Theta1'); % 5000x401 * 401x25 = 5000x25
a2b = [ones(m, 1) a2]; % 5000x26
a3 = sigmoid(a2b * Theta2'); % 5000x26 * 26x10 = 5000x10

% convert y from labels vector to binary matrix
yd = eye(num_labels);
y = yd(y,:); % 5000x10

% non-regularized cost
J = sum(sum((-y .* log(a3) - (1 - y) .* log(1 - a3)))) / m;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

% trim off bias for backprop
Theta1z = Theta1(:, 2:end); % 25x400
Theta2z = Theta2(:, 2:end); % 10x25

% back prop
d3 = a3 .- y; % 5000x10
d2 = d3 * Theta2z .* a2 .* (1 - a2); % 5000x10 * 10x25 = 5000x25

% gradient
Theta1_grad = (d2' * a1b) / m; % 25x5000 * 5000x401 = 25x401
Theta2_grad = (d3' * a2b) / m; % 10x5000 * 5000x26 = 10x26

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% disregard bias for regularization
Theta1(:,1) = 0;
Theta2(:,1) = 0;

% regularized cost
J = J + (lambda / (2 * m)) * (sum(sum(Theta1.^2)) +  sum(sum(Theta2.^2)));

% regularized gradient
Theta1_grad = Theta1_grad + (lambda / m) .* Theta1;
Theta2_grad = Theta2_grad + (lambda / m) .* Theta2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
