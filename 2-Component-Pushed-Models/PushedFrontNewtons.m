a = 0.2;
gamma = 2;
k = 1;
delta = 1;
ep = 0.05;


c_val = zeros(size(a));

xi_max = 50;
dxi = 0.025;
xi = -xi_max:dxi:xi_max;

Nxi = length(xi);   

% D Matrix
e = ones(Nxi,1);
D2 = spdiags([e -2*e e], -1:1, Nxi, Nxi) / dxi^2;
%BCs (Dirichlet)
D2(1,1:2) = [-2 2]/dxi^2;
D2(Nxi,Nxi-1:Nxi) = [1 -3]/dxi^2;
D2 = sparse(D2);

D1 = spdiags([-e zeros(Nxi,1) e], -1:1, Nxi, Nxi) / (2*dxi);
%BCs (Dirichlet)
D1(1,:) = 0;
D1(Nxi,Nxi-1:Nxi) = [-1 -1]/(2*dxi);
D1 = sparse(D1);

%%%%% Equ. Roots %%%%
p1 = [10, -10*(1 + a(1)), 10*a(1) + 1/gamma];
r = sort(roots(p1));

u_a = r(1);
u_plus = r(2);
%%%% Fast Jump Roots %%%%%
p2 = [10, -10*(1 + a(1)), 10*a(1) + u_a/gamma];
r = sort(roots(p2));

u_a_fast = r(1);
u_plus_fast = r(2);


% Initial Condition
v0 = zeros(size(xi));
half = round(length(xi)/2);

%%%%% SOLVE FOR INITIAL B AND C %%%%%%%%%%
up = u_plus + u_a;
fun = @(x) [
    x(1) - (k*gamma*(u_a/gamma) - x(2)^2) / (x(2)*up);     % b equation
    x(2) - (2*x(1)*k) / (2*x(1)^2 - 1)             % c equation
];

x0 = [-10; 0.01];  % expect b < 0, c > 0
opts = optimoptions('fsolve');
wavespeed_result = deal(fsolve(fun, x0, opts));

b = wavespeed_result(1);
c_guess = max(wavespeed_result(2));

v0(1:half) = (1/gamma)*(u_a - u_plus)*exp(2*gamma*ep*xi(1:half)/c_guess) + u_plus/gamma;
v0(half + 1:end) = u_a/gamma;
v0 = v0' - u_a/gamma;

u0 = (u_plus + u_a)/2 + ((u_plus - u_a)/2)*tanh( (u_plus - u_a)/2 * b * xi );
u0 = u0' - u_a;  % shift down by u_a


%options = optimoptions('fsolve',...
%    'MaxIterations', 250, 'SpecifyObjectiveGradient', true);

options = optimoptions('fsolve', ...  
    'Algorithm','trust-region-dogleg', ...
    'SpecifyObjectiveGradient', true, ...         
    'MaxFunctionEvaluations', 1e4, ...
    'MaxIterations', 250, ...
    'FunctionTolerance', 1e-10, ...
    'StepTolerance', 1e-10, ...
    'Display', 'iter-detailed');            

pars = struct('gamma', gamma, 'k', k, 'delta', delta, 'ep', ep, ...
                  'D1', D1, 'D2', D2, 'dxi', dxi, 'Nxi', Nxi);
y0 = [u0; v0; c_guess];

for i = 1:length(a)

    p = [10, -10*(1 + a(i)), 10*a(i) + 1/gamma];
    r = sort(roots(p));

    u_a = r(1);
    u_plus = r(2);

    disp(['a value: ', num2str(a(i))]);

    residual = @(y) deal(rhs(y, a(i), pars, y0, u_a, u_plus), jac_rhs(y, a(i), pars, y0, u_a, u_plus));
    %residual = @(y) rhs(y, a(i), pars, y0);
    [sol, ~, ~] = fsolve(residual, y0, options);

    c_val(i) = sol(end);        
    y0 = sol;         
end
%% Plot tanh profile
figure;
%plot(xi, 0.5 - 0.5*tanh(-(1/ep)*xi/(2*sqrt(2))), 'LineWidth', 2)
plot(xi, u0, 'LineWidth', 2)
hold on
%plot(xi, y0(1:Nxi), 'LineWidth', 2)
%plot(xi, v0, 'LineWidth', 2)
plot(xi, v0, 'LineWidth', 2)


%ylim([-0.2 1.2])
xlim([min(xi), max(xi)])
xlabel('$\xi$', 'Interpreter', 'latex', 'FontSize', 20)
ylabel('Traveling Front', 'Interpreter', 'latex', 'FontSize', 20)

%legend({'Approx. Fast Solution $u(\xi)$', 'Actual Fast Solution' 'Approx. Slow Solution $v(\xi)$', 'Actual Slow Solution'}, 'Interpreter', 'latex', 'FontSize', 20, 'Location', 'best')
legend({'Front solution, $u$', 'Front solution, $v$'}, 'Interpreter', 'latex', 'FontSize', 20, 'Location', 'best')

grid on
box on
set(gca, 'FontSize', 20, 'TickLabelInterpreter', 'latex')

%% Plot c(a)
plot(a, c_val)
hold on
%plot(a, a*(sqrt(2)/delta) - ep*(1/sqrt(2)))
%plot(a, c_analytical)
title('Wavespeed c(a) as a function of a')
xlabel('a')
ylabel('c(a)')
legend('Numerical Wavespeed', 'Analytical Prediction')

%%

function rh = rhs(y, a, pars, y0, u_a, u_plus)
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

    f_u = 10*(1 - (u+u_a)) .* (u+u_a) .* ((u+u_a)-a) - (v+u_a/gamma);
    
    chi = k*v;

    nonlinear_term = (D1 * u) .* (D1 * chi) + (u + u_a) .* (D2 * chi);
    rh(1:Nxi) = c*D1*u + (D2 * u) - (1/ep)*nonlinear_term + f_u;
    rh(Nxi+1:end - 1) = c*D1*v + ep*( (u+u_a) - gamma * (v+u_a/gamma) );

    rh(end) = ([D1*u; D1*v])' * ([u-u0; v-v0]);
    %rh(end) = u(floor(Nxi/2)) - 0.5;
end

function J = jac_rhs(y, a, pars, y0, u_a, u_plus)
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

    df = -10*(3*(u+u_a).^2 - 2*(1 + a)*(u+u_a) + a);

    chi = k*v;
    dchi = k;

    J(1:Nxi, 1:Nxi) = c*D1 + D2 + ...
    - (1/ep) * ( spdiags(D1 * chi, 0, Nxi, Nxi)*D1 ...
              + spdiags(D2*chi,0,Nxi,Nxi))...
    + spdiags(df, 0, Nxi, Nxi);

    J(1:Nxi, Nxi+1:2*Nxi) = -(1/ep)*( ...
       spdiags(D1*u,0,Nxi,Nxi)*D1*dchi + ...
       spdiags((u + u_a),0,Nxi,Nxi)*D2*dchi )...
        - speye(Nxi, Nxi);


    J(Nxi+1:end-1, 1:Nxi) = ep*speye(Nxi);
    
    J(Nxi+1:end-1, Nxi+1:end-1) = c*D1 - ep*gamma * speye(Nxi);

    % last column
    J(1:Nxi, end) = D1*u;
    J(Nxi+1:end - 1, end) = D1*v;

    % Last row
    
    J(end, 1:Nxi)       = ((u - u0)' * D1 + (D1*u)');
    J(end, Nxi+1:2*Nxi) = ((v - v0)' * D1 + (D1*v)');
    J(end, end)         = 0;

    J = sparse(J);
end