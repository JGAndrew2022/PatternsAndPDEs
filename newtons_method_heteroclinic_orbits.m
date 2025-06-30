a = 0.49:-0.01:0.01;
c = zeros(length(a), 1);

pars = struct;

xi_max = 50;
dxi = 0.025;
xi = 0:dxi:xi_max;

Nxi = length(xi);   

pars.dxi = dxi;
pars.n = Nxi;

u0 = zeros(Nxi + 1, 1); 

% D Matrices
e = ones(pars.n,1);
D2 = spdiags([e -2*e e], -1:1, pars.n, pars.n) / dxi^2;
D2(1,1) = -2/dxi^2; 
D2(1,2) = 2/dxi^2;
D2(pars.n,pars.n) = -2/dxi^2; 
D2(pars.n,pars.n-1) = 1/dxi^2;
D2 = sparse(D2);

pars.D2 = D2;

D1 = spdiags([-e zeros(pars.n,1) e], -1:1, pars.n, pars.n) / (2*dxi);
D1(1,:) = 0; D1(pars.n,:) = 0;
D1 = sparse(D1);

pars.D1 = D1;

% Initial Condition
for i=1:Nxi
    u0(i) = (1/2) + (1/2)*tanh((xi(i) - (xi_max)/2)/(2*sqrt(2))) + 0.1*cos(xi(i));
end
u0(Nxi + 1) = 0;

% Numerical continuation
options = optimoptions('fsolve', 'MaxIterations', 250);

for i = 1:length(a)
    disp(['a value:', num2str(a(i))])

    residual = @(u) rhs(u, a(i), u0, pars);

    [u_new, ~, ~] = fsolve(residual, u0, options);

    c(i) = u_new(end);

    u0 = u_new;
    u0(end) = 0;
end
%% Plot tanh profile
plot(xi, u_new(1:end-1))
ylim([-0.2 1.2])
xlabel('\xi'), ylabel('u(\xi)')

%% Plot c(a)
plot(a, c)
hold on
plot(a, sqrt(2)*a - 1/sqrt(2))
title('Wavespeed c(a) as a function of a')
xlabel('a')
ylabel('c(a)')
%%
function rh = rhs(u, a, u0, pars)
    n = pars.n;
    dxi = pars.dxi;
    rh=zeros(n+1,1);
    
    rh(1) = 2*(u(2)-u(1))/(dxi^2) + u(1).*(u(1) - a).*(1 - u(1)) + (u(n+1)*(u(2) - u(1))/dxi);
    
    for i = 2:n - 1
        rh(i) = (u(i+1) + u(i-1) - 2*u(i))/(dxi^2) + u(i).*(u(i) - a).*(1 - u(i)) + (u(n+1)*(u(i) - u(i-1))/dxi);
    end
    
    rh(n) = 2*(u(n-1)-u(n))/(dxi^2) + u(n).*(u(n) - a).*(1 - u(n)) + (u(n+1)*(u(n) - u(n-1))/dxi);

    % translation constraint
    rh(n+1) = (1/dxi)*(u(3:n) - u(1:n-2))' * (u(2:n-1) - u0(2:n-1));

end
