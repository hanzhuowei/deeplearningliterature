%griewank function
clear
Xpool = -4:4;
% [X1,X2] = meshgrid(Xpool, Xpool);
X = [Xpool',Xpool'];
% d = size(X,2);
% sum = 0;
% prod = 1;

sum = sum(X.^2)

y = sum - prod + 1;

figure
h= surf(X1,X2,y);
set(h,'LineStyle','none')
