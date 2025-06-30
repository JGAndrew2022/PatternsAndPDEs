a1 = 0.32:-0.0005:0.17;     
a2 = 0.18-0.00005:-0.00005:0.13; 
a3 = 0.13-0.0001:-0.0001:0.06;     
a4 = 0.06-0.00005:-0.00005:0.01;

a = unique([a1, a2, a3, a4], 'stable');

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
    u0_val = (1+a(1))/2 + (1-a(1))/2 * tanh(( xi(i) - 3*(xi_max)/4 )*(a(1) - 1)/(2*sqrt(2))); %+ 0.1*cos(xi(i));
    % shift everything so that a is at 0 for numerical stability reasons
    u0(i) = u0_val - a(1);
end
u0(Nxi + 1) = 0;

% Numerical continuation
options = optimoptions('fsolve', ...
    'SpecifyObjectiveGradient', true, 'MaxIterations', 250);

for i = 1:length(a)
    disp(['a value:', num2str(a(i))])

    residual = @(u) rhs(u, a(i), u0, pars);
    jacobian = @(u) jac_rhs(u, a(i), u0, pars);
    
    [u_new, ~, ~] = fsolve(@(u) deal(residual(u), jacobian(u)), u0, options);

    c(i) = u_new(end);
    u0 = u_new;
    %u0 = (1+a(i))/2 + (1-a(i))/2 * tanh(( xi - (xi_max)/2 )*(a(i) - 1)/(2*sqrt(2))); %+ 0.1*cos(xi(i));
    u0(end) = 0;
end
%% Plot tanh profile
plot(xi, u_new(1:end-1) + a(end))
hold on
yline(a(end), '-k')
ylim([-0.2 1.2])
xlabel('\xi'), ylabel('u(\xi)')

%% Plot c(a)
plot(a, c)
hold on
plot(a, a/sqrt(2) + 1/sqrt(2))
title('Wavespeed c(a) as a function of a')
xlabel('a')
ylabel('c(a)')
%%
function rh = rhs(u, a, u0, pars)
    dxi = pars.dxi;
    D1 = pars.D1;
    D2 = pars.D2;
    n = pars.n;
    rh = zeros(n+1,1);

    u_main = u(1:n);
    f_u = (u_main + a) .* u_main .* (1 - u_main - a);
    cD1u = u(n+1) * (D1 * u_main);

    rh(1:n) = D2*u_main + f_u + cD1u;

    rh(n+1) = (1/dxi)*(u(3:n) - u(1:n-2))' * (u(2:n-1) - u0(2:n-1));

end

function J = jac_rhs(u, a, u0, pars)
    n = pars.n;
    dxi = pars.dxi;
    D1 = pars.D1;
    D2 = pars.D2;

    J = spalloc(n+1, n+1, 5*n); % rough sparsity allocation

    u_main = u(1:n);
    c = u(n+1);

    df = (2*u_main).*(1 - 2*a) - 3*u_main.^2 + a - a.^2;
    J(1:n, 1:n) = D2 + spdiags(df, 0, n, n) + c * D1;

    % last column
    J(1:n, n+1) = D1 * u_main;

    % Last row
    j = (2:n-1)';
    ujm1 = u(j-1); uj = u(j); ujp1 = u(j+1); u0j = u0(j);

    row = n+1;
    J(row, j-1) = J(row, j-1) - (uj - u0j)';
    J(row, j  ) = J(row, j  ) + (ujp1 - ujm1)';
    J(row, j+1) = J(row, j+1) + (uj - u0j)';
    J(row, :) = J(row, :) / dxi;
end