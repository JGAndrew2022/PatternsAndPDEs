a = 0.1;
gamma = 1;
k = 1;
delta = 1;
ep = 0.05;
c_val = zeros(size(a));
c_analytical = k/(2*delta*sqrt(gamma)) + (a-1/2)*(sqrt(2)/delta);

a_plus = 0.3;
a_minus = 0.1;

xi_max = 15;
dxi = 0.05;
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
u0 = 0.5 - 0.5*tanh(-(1/ep)*xi/(2*sqrt(2)));
u0 = u0';

v0 = zeros(size(xi));
half = round(length(xi)/2);

v0(1:half) = 1/(2*gamma) * exp(sqrt(gamma)*xi(1:half));
v0(half + 1:end) = -1/(2*gamma) * exp(-sqrt(gamma)*xi(half+1:end)) + 1/gamma;
v0 = v0';

options = optimoptions('fsolve', 'Display', 'off',...
    'MaxIterations', 250, 'SpecifyObjectiveGradient', true);

pars = struct('gamma', gamma, 'k', k, 'delta', delta, 'ep', ep, 'D1', D1, 'D2', D2, ...
            'dxi', dxi, 'Nxi', Nxi, 'a_minus', a_minus, 'a_plus', a_plus);
y0 = [u0; v0; 0];
for i = 1:length(a)
    disp(['a value: ', num2str(a(i))]);

    residual = @(y) deal(rhs(y, a(i), pars, y0), jac_rhs(y, a(i), pars, y0));
    %residual = @(y) rhs(y, a(i), pars, y0);
    [sol_newtons, ~, ~] = fsolve(residual, y0, options);

    c_val(i) = sol_newtons(end);        
    y0 = sol_newtons;         
end
%% Plot tanh profile
figure;
%plot(xi, 0.5 - 0.5*tanh(-(1/ep)*xi/(2*sqrt(2))), 'LineWidth', 2)
plot(xi, y0(1:Nxi), 'LineWidth', 2)
hold on
%plot(xi, y0(1:Nxi), 'LineWidth', 2)
%plot(xi, v0, 'LineWidth', 2)
plot(xi, y0(Nxi+1:end - 1), 'LineWidth', 2)


ylim([-0.2 1.2])
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
    a_minus = pars.a_minus;
    a_plus = pars.a_plus;

    u0 = y0(1:Nxi);
    v0 = y0(Nxi + 1 : end - 1);

    u = y(1:Nxi);
    v = y(Nxi+1: end - 1);
    c = y(end);
    
    a_of_v = a_minus + (a_plus - a_minus)*v./(1 + v);
    f_u = (u - a_of_v) .* u .* (1 - u);

    chi = k*(8/3) * v.^2 ./ (v.^2 + 3);
    %chi = k*v;
    %chi = k*(4/3) * v.^2 ./ ( v.^2 + (3/4) );
    
    nonlinear_term = (D1 * u) .* (D1 * chi) + u.*(D2*chi);
    rh(1:Nxi) = ep*c*D1*u + ep^2*(D2 * u) - ep*nonlinear_term + f_u;
    rh(Nxi+1:end - 1) = ep*c*D1*v + D2*v + u - gamma*v;

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
    a_minus = pars.a_minus;
    a_plus = pars.a_plus;

    u0 = y0(1:Nxi);
    v0 = y0(Nxi + 1 : end - 1);

    u = y(1:Nxi);
    v = y(Nxi+1: end - 1);
    c = y(end);

    J = spalloc(2*Nxi+1, 2*Nxi+1, 10*Nxi);
    
    a_of_v = a_minus + (a_plus - a_minus)*v./(1 + v);
    dfdu = -(3*u.^2 - 2*(1 + a_of_v).*u + a_of_v);

    dfdv = -1./((1 + v).^2).*(a_plus - a_minus).*u.*(1 - u);
    
    chi = k*(8/3) * v.^2 ./ (v.^2 + 3);
    dchi = k*(8/3) * (2*v*3)./ ((3+v.^2).^2);
    %chi = k*v;
    %dchi = k;
    %chi = k*(4/3) * v.^2 ./ ( v.^2 + (3/4) );
    %dchi = k*(4/3) * (2*v*(3/4))./(( (3/4) + v.^2 ).^2);

    J(1:Nxi, 1:Nxi) = ep*c*D1 + ep^2*D2 ...
    - ep*(spdiags(D1 * chi, 0, Nxi, Nxi)*D1 ...
              + spdiags(D2*chi,0,Nxi,Nxi))...
    + spdiags(dfdu, 0, Nxi, Nxi);
    
    J(1:Nxi, Nxi+1:end-1) = -ep * ( ...
    spdiags(D1*u,0,Nxi,Nxi) * D1 * spdiags(dchi,0,Nxi,Nxi) + ...
    spdiags(u,0,Nxi,Nxi)    * D2 * spdiags(dchi,0,Nxi,Nxi) ) + ...
    spdiags(dfdv,0,Nxi,Nxi);


    J(Nxi+1:end-1, 1:Nxi) = speye(Nxi);
    
    J(Nxi+1:end-1, Nxi+1:end-1) = ep*c*D1 + D2 - gamma * speye(Nxi);

    % last column
    J(1:Nxi, end) = ep*D1*u;
    J(Nxi+1:end - 1, end) = ep*D1*v;

    % Last row
    
    J(end, 1:Nxi)       = ((u - u0)' * D1 + (D1*u)');
    J(end, Nxi+1:2*Nxi) = ((v - v0)' * D1 + (D1*v)');
    J(end, end)         = 0;

    J = sparse(J);
end