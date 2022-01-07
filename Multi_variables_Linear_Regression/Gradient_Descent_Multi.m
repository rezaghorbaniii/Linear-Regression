function [theta, J_history] = Gradient_Descent_Multi(X, y, theta, alpha, num_iters)

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    error = (X * theta) - y;
    theta = theta - (alpha * (1/m) * (X' * error));   
    J_history(iter) = Compute_Cost_Multi(X, y, theta);

end

end
