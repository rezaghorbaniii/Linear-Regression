function [theta] = Normal_Eqn(X, y)

theta = zeros(size(X, 2), 1);
theta = pinv(X' * X) * X' * y;

end
