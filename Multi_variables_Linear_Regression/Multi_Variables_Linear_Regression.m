% Reza Ghorbani - January 2022
% Implementation of linear regression with multiple variables

clc ;
clear ; 
close all ; 

data = load('data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

[X_Normalized mu sigma] = Feature_Normalize(X);

X_Normalized = [ones(m, 1) X_Normalized]; % Add intercept term to X

% Gradient descent settings
alpha = 0.1;
num_iters = 200;

theta = zeros(3, 1);
[theta, J_history] = Gradient_Descent_Multi(X_Normalized, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b');
xlabel('Number of iterations'); ylabel('Cost');

% Predict values
predict = [1, 1650, 3] * theta; 

% The closed form solution for linear regression using the Normal Equations
data = csvread('data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

[X_Normalized_2 mu sigma] = Feature_Normalize(X);

X_Normalized_2 = [ones(m, 1) X_Normalized_2]; % Add intercept term to X

% Calculate the parameters from the normal equation
theta_2 = Normal_Eqn(X_Normalized_2, y);

% Predict values
Predict_2 = [1, 1650, 3] * theta_2 ;
