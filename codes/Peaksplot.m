% clear
% [X,Y,Z] = peaks(50);
% figure
% surf(X,Y,Z)


%
% INPUT:
%
% xx = [x1, x2]
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
[X,Y] = meshgrid(-2:.02:2, -2:.02:2);
Z = X .* exp(-X.^2 - Y.^2);
% %%%%%%%%%%%%%  Eggholder function %%%%%%%%%%%%%
% x1= -512:5:512;
% x2= -512:5:512;
% [x1,x2]=meshgrid(x,x);


% term1 = -(x2+47)* sin(sqrt(abs(x2+x1/2+47)));
% term2 = -x1* sin(sqrt(abs(x1-(x2+47))));
% 
% y = term1 + term2;
% plot::Function3d(,x1=-512..512,x2=-512..512)
%%%%%%%%%%%%---six-hump camel function  %%%%%%%%%
% x1 = rand(50,1)+2;
% x2 = rand(1,50)+1;
% term1 = (4-2.1*x1.^2+(x1.^4)/3) * (x1.^2)';
% term2 = x1*x2;
% term3 = (-4+4*x2.^2)'* x2.^2;
% 
% y = term1 + term2 + term3;

figure
h= surf(X,Y,Z);
set(h,'LineStyle','none')

