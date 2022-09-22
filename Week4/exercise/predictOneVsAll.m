function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1); %5000
num_labels = size(all_theta, 1); %10

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1); %5000x1

% Add ones to the X data matrix
X = [ones(m, 1) X]; %5000x401

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
% predict ch�nh l� t�nh c�i hypothesis �?y, cho theta nh�n v?i X l� ��?c
prob_mat=X*all_theta'; %5000x10
[prob,p]=max(prob_mat,[],2);
%ph�p t�nh tr�n th?c hi?n t?m gi� tr? l?n nh?t ? c�c h�ng,tr? ra 2 gi� tr? l� prob v� p, trong ��
%prob l� gi� tr? max c?a m?i h�ng trong ma tr?n prob_mat (hay c?n ��?c g?i
%l� x�c su?t h(theta)(x) cao nh?t trong 10 gi� tr? t��ng ?ng v?i kh? n�ng
%x?y ra nh?n di?n k� t? t? 1 �?n 10), c� size l� 5000x1
%p l� index c?a t?ng gi� tr? c?a vector prob trong t?ng h�ng c?a prob_mat,
%c� size l� 5000x1
% =========================================================================


end
