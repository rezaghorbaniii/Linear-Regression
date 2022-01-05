function [theta, J_history,theta_history] = Gradient_Descent(X, y, theta, alpha, num_iters)

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_history = zeros(num_iters, 2);

for iter = 1:num_iters
    
    theta_history(iter,1) = theta(1);
    theta_history(iter,2) = theta(2);
    J_history(iter) = Compute_Cost(X, y, theta);
    theta = theta - (alpha * (1/m) * (X' * ((X * theta) - y)));  
    
end

end
