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
% predict chính là tính cái hypothesis ð?y, cho theta nhân v?i X là ðý?c
prob_mat=X*all_theta'; %5000x10
[prob,p]=max(prob_mat,[],2);
%phép tính trên th?c hi?n t?m giá tr? l?n nh?t ? các hàng,tr? ra 2 giá tr? là prob và p, trong ðó
%prob là giá tr? max c?a m?i hàng trong ma tr?n prob_mat (hay c?n ðý?c g?i
%là xác su?t h(theta)(x) cao nh?t trong 10 giá tr? týõng ?ng v?i kh? nãng
%x?y ra nh?n di?n kí t? t? 1 ð?n 10), có size là 5000x1
%p là index c?a t?ng giá tr? c?a vector prob trong t?ng hàng c?a prob_mat,
%có size là 5000x1
% =========================================================================


end
