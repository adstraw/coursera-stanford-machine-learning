function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.01;
sigma = 0.01;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

vals = [0.01; 0.03; 0.1; 0.3; 1; 3; 10];
model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
predict = svmPredict(model, Xval);
error = mean(double(predict ~= yval));
currError = error;
bestC = 1;
bestSigma = 1;

for c =1:length(vals)
  for s = 1:length(vals)
    model = svmTrain(X, y, vals(c), @(x1, x2) gaussianKernel(x1, x2, vals(s)));
    predict = svmPredict(model, Xval);
    currError = mean(double(predict ~= yval));
    if currError < error
      error = currError;
      bestC = c;
      bestSigma = s;
    end
  end
end

C = vals(bestC);
sigma = vals(bestSigma);

% =========================================================================

end
