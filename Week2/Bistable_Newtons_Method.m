a = 0.5:-0.01:0.01;
gamma = 1;
k = 1;
delta = 1;
ep = 0.02;

c_val = zeros(size(a));
c_analytical = k/(2*delta*sqrt(gamma)) + (a-1/2)*(sqrt(2)/delta);

xi_max = 300;
dxi = 0.1;
xi = -xi_max:dxi:xi_max;

Nxi = length(xi);   

% D Matrix
e = ones(Nxi,1);
D2 = spdiags([e -2*e e], -1:1, Nxi, Nxi) / dxi^2;
%BCs (Neumann)
D2(1,1:2) = [-2 2]/dxi^2;
D2(Nxi,Nxi-1:Nxi) = [2 -2]/dxi^2;
D2 = sparse(D2);

D1 = spdiags([-e zeros(Nxi,1) e], -1:1, Nxi, Nxi) / (2*dxi);
%BCs (Neumann)
D1(1,:) = 0;
D1(Nxi,:) = 0;
D1 = sparse(D1);

% Initial Condition
u0 = 0.5 - 0.5*tanh(-xi/(2*sqrt(2)));
u0 = u0';

v0 = zeros(size(xi));
half = round(length(xi)/2);

v0(1:half) = 1/(2*gamma) * exp(sqrt(gamma)*ep*xi(1:half));
v0(half + 1:end) = -1/(2*gamma) * exp(-sqrt(gamma)*ep*xi(half+1:end)) + 1/gamma;
v0 = v0';

options = optimoptions('fsolve', 'Display', 'off',...
    'MaxIterations', 250, 'SpecifyObjectiveGradient', true);

pars = struct('gamma', gamma, 'k', k, 'delta', delta, 'ep', ep, ...
                  'D1', D1, 'D2', D2, 'dxi', dxi, 'Nxi', Nxi);
y0 = [u0; v0; c_analytical(1)];
for i = 1:length(a)
    disp(['a value: ', num2str(a(i))]);

    residual = @(y) deal(rhs(y, a(i), pars, y0), jac_rhs(y, a(i), pars, y0));
    %residual = @(y) rhs(y, a(i), pars, y0);
    [sol, ~, ~] = fsolve(residual, y0, options);

    c_val(i) = sol(end);        
    y0 = sol;         
end
%% Plot tanh profile
plot(xi, y0(Nxi+1:end - 1))
hold on
plot(xi, v0)
plot(xi, y0(1:Nxi))
plot(xi, 0.5 - 0.5*tanh(-xi/(2*sqrt(2))))
%yline(a(end), '-k')
ylim([-0.2 1.2])
xlabel('\xi'), ylabel('u(\xi)')
legend('Final v', 'v0', 'Final u', 'u0')

%% Plot c(a)
plot(a, c_val)
hold on
%plot(a, a*(sqrt(2)/delta) - ep*(1/sqrt(2)))
plot(a, c_analytical)
title('Wavespeed c(a) as a function of a')
xlabel('a')
ylabel('c(a)')
legend('Numerical Wavespeed', 'Analytical Prediction')

%%

function rh = rhs(y, a, pars, y0)
    Nxi = pars.Nxi;
    gamma = pars.gamma;
    k = pars.k;
    delta = pars.delta;
    ep = pars.ep;
    D1 = pars.D1;
    D2 = pars.D2;
    rh = zeros(2*Nxi + 1,1);

    u0 = y0(1:Nxi);
    v0 = y0(Nxi + 1 : end - 1);

    u = y(1:Nxi);
    v = y(Nxi+1: end - 1);
    c = y(end);

    f_u = (u - a) .* u .* (1 - u);
    
    nonlinear_term = D1 * u .* (D1 * v);
    rh(1:Nxi) = delta*c*D1*u + (D2 * u) - (1/ep)*k*nonlinear_term + f_u;
    rh(Nxi+1:end - 1) = c*D1*v + (1/ep^2)*D2*v + u - gamma*v;

    rh(end) = ([D1*u; D1*v])' * ([u-u0; v-v0]);
    %rh(end) = u(floor(Nxi/2)) - 0.5;
end

function J = jac_rhs(y, a, pars, y0)
    Nxi = pars.Nxi;
    gamma = pars.gamma;
    k = pars.k;
    delta = pars.delta;
    ep = pars.ep;
    D1 = pars.D1;
    D2 = pars.D2;

    u0 = y0(1:Nxi);
    v0 = y0(Nxi + 1 : end - 1);

    u = y(1:Nxi);
    v = y(Nxi+1: end - 1);
    c = y(end);

    J = spalloc(2*Nxi+1, 2*Nxi+1, 10*Nxi);

    df = 3*u.^2 - 2*(1 + a)*u + a;

    J(1:Nxi, 1:Nxi) = delta*c*D1 + D2 + ...
        - (1/ep)*k * D1 * spdiags(D1 * v, 0, Nxi, Nxi)...
        - spdiags(df, 0, Nxi, Nxi);
    
    J(1:Nxi, Nxi+1:end-1) = -(1/ep) * k * D1 * spdiags(u, 0, Nxi, Nxi) * D1;


    J(Nxi+1:end-1, 1:Nxi) = speye(Nxi);
    
    J(Nxi+1:end-1, Nxi+1:end-1) = c*D1 + (1/ep^2)*D2 - gamma * speye(Nxi);

    % last column
    J(1:Nxi, end) = delta*D1*u;
    J(Nxi+1:end - 1, end) = D1*v;

    % Last row
    
    J(end, 1:Nxi)       = ((u - u0)' * D1 + (D1*u)');
    J(end, Nxi+1:2*Nxi) = ((v - v0)' * D1 + (D1*v)');
    J(end, end)         = 0;

    J = sparse(J);
end