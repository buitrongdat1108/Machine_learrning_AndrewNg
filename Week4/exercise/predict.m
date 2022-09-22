function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1); %5000
num_labels = size(Theta2, 1);  %10

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);  % 5000x1

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
%DIMENSIONS:
  % theta1 = 25 x 401
  % theta2 = 10 x 26
  
  % layer1 (input)  = 400 nodes + 1bias
  % layer2 (hidden) = 25 nodes + 1bias 
  % layer3 (output) = 10 nodes
  % 
  % theta dimensions = S_(j+1) x ((S_j)+1)
  % theta1 = 25 x 401
  % theta2 = 10 x 26
  
  % theta1:
  %     1st row indicates: theta corresponding to all nodes from layer1 connecting to for 1st node of layer2
  %     2nd row indicates: theta corresponding to all nodes from layer1 connecting to for 2nd node of layer2
  %     and
  %     1st Column indicates: theta corresponding to node1 from layer1 to all nodes in layer2
  %     2nd Column indicates: theta corresponding to node2 from layer1 to all nodes in layer2
  %     
  % theta2:
  %     1st row indicates: theta corresponding to all nodes from layer2 connecting to for 1st node of layer3
  %     2nd row indicates: theta corresponding to all nodes from layer2 connecting to for 2nd node of layer3
  %     and
  %     1st Column indicates: theta corresponding to node1 from layer2 to all nodes in layer3
  %     2nd Column indicates: theta corresponding to node2 from layer2 to all nodes in layer3
% ð?u tiên c? cho vào cái input trý?c
a1=[ones(m,1) X]; %5000x401
z2=a1*Theta1'; %5000x25
a2=sigmoid(z2); %5000x25
a2=[ones(size(a2,1),1) a2]; %5000x26
z3=a2*Theta2'; %5000x10
a3=sigmoid(z3);  %5000x10
[prob, p] = max(a3,[],2); %5000x1
% =========================================================================
end
