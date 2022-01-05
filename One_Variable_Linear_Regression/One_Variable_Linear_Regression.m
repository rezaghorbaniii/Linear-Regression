% Reza Ghorbani - January 2022
% Implementation of linear regression with one variable

clc ;
clear ; 
close all ; 

data = load('data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % Number of training examples
x = X;
x = [ones(m, 1), data(:,1)]; % Add a column of ones to x

% Initialize fitting parameters
theta = [5;5];

% Gradient descent settings
iterations = 1500;
alpha = 0.01;

% Run gradient descent
[theta, J_history,theta_history] = Gradient_Descent(x, y, theta, alpha, iterations);
J = Compute_Cost(x, y, [theta(1) ; theta(2)]);

% Plot data
plot(X, y, 'bx','MarkerSize', 5);
hold on ; 
plot(x(:,2), x*theta, '-')
legend('Training data', 'Linear regression')
hold off ;

% Predict values
predict = [1, 7] *theta;

%  Plot cost values  
theta0_vals = linspace(-12, 10, 100);
theta1_vals = linspace(-5, 6, 100);

t_x = theta_history(:, 1); t_y = theta_history(:, 2);

J_vals = zeros(length(theta0_vals), length(theta1_vals));
J_vals_2 = zeros(length(t_x), length(t_y));

for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = Compute_Cost(x, y, t);
    end
end

J_vals = J_vals';
figure;
surfc(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
J_history = J_history';
plot3(t_x', t_y', J_history,'r.-','MarkerSize', 10);

% Plot Contours 
figure;
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'r*', 'MarkerSize', 10);
plot(t_x, t_y, 'b*-');
