a = 0.4;
gamma = 1;
k = 3.2;
delta = 1;
ep = 0.05;

c_analytical = k/(2*delta*sqrt(gamma)) + (a-1/2)*(sqrt(2)/delta);

ell_max = 20;
ells = linspace(0, ell_max, 10);
max_real_part = zeros(size(ells));

xi_max = 100;
dxi = 0.25;
xi = -xi_max:dxi:xi_max;

Nxi = length(xi);   

% Differentiation matrices templates
e = ones(Nxi,1);
D2_small = spdiags([e -2*e e], -1:1, Nxi, Nxi) / dxi^2;
%BCs (Neumann)
D2_small(1,1:2) = [-2 2]/dxi^2;
D2_small(Nxi,Nxi-1:Nxi) = [2 -2]/dxi^2;
D2_small = sparse(D2_small);

D1_small = spdiags([-e zeros(Nxi,1) e], -1:1, Nxi, Nxi) / (2*dxi);
%BCs (Neumann)
D1_small(1,:) = 0;
D1_small(Nxi,:) = 0;
D1_small = sparse(D1_small);


I = speye(Nxi);
D1 = blkdiag(D1_small, D1_small);
D2 = blkdiag(D2_small, D2_small);


u0 = 0.5 - 0.5*tanh(-xi/(2*sqrt(2)));
u0 = u0';

v0 = zeros(size(xi));
half = round(length(xi)/2);

v0(1:half) = 1/(2*gamma) * exp(sqrt(gamma)*ep*xi(1:half));
v0(half + 1:end) = -1/(2*gamma) * exp(-sqrt(gamma)*ep*xi(half+1:end)) + 1/gamma;
v0 = v0';

%%%%%%%%%%%% Step 1: FIND TRAVELING WAVE SOLUTION W/ NEWTON'S METHOD %%%%%%%%%%%%
options = optimoptions('fsolve', 'Display', 'off',...
    'MaxIterations', 250, 'SpecifyObjectiveGradient', true);

pars = struct('gamma', gamma, 'k', k, 'delta', delta, 'ep', ep, ...
                  'D1', D1_small, 'D2', D2_small, 'dxi', dxi, 'Nxi', Nxi);
y0 = [u0; v0; c_analytical];

residual = @(y) deal(rhs(y, a, pars, y0), jac_rhs(y, a, pars, y0));
[sol, ~, ~] = fsolve(residual, y0, options);

c_val = sol(end);        
y = sol;

u = y(1:Nxi);
v = y(Nxi + 1 : end - 1);

%%%%%%%%%%%% Step 2: Build Operator that acts on perturbation and plot spectrum %%%%%%%%%%%%
% Matrix PDE Setup
D = spdiags([ones(Nxi,1); ones(Nxi,1)], 0, 2*Nxi, 2*Nxi);
U = [u; v];

df = 3*u.^2 - 2*(1+a)*u + a;

chi = 8*v.^2./(9 + 3*v.^2);

dF = [-spdiags(df, 0, Nxi, Nxi), sparse(Nxi, Nxi); I, -gamma * I];
Diag_u = spdiags(u,  0, Nxi, Nxi);

max_real = zeros(size(ells));
for j = 1:length(ells)
    ell = ells(j);
    disp(j)

    Diag_Dv = spdiags((D1_small + ep^2*1i*ell*speye(Nxi))*chi, 0, Nxi, Nxi);
    H = [Diag_Dv, Diag_u*(D1_small + ep^2*1i*ell*speye(Nxi)); sparse(Nxi, Nxi), sparse(Nxi, Nxi)];
    
    D2_u = D2_small - ep^2 * ell^2 * speye(Nxi);
    D2_v = (1/ep^2)*D2_small - ell^2 * speye(Nxi);
    D2_y = blkdiag(D2_u, D2_v);

    Operator = D2_y + c_val*D1 + dF - (k/ep)*D1*H;

    [V, Lambda] = eig(full(Operator));
    max_real(j) = max(real(diag(Lambda)));
end
%%
figure;
plot(ells, max_real, 'LineWidth', 1.5);
xlabel('$\ell$', 'Interpreter','latex'); ylabel('max Re($\lambda$)', 'Interpreter','latex');
title('$\lambda$-val with Largest Real Part vs perturbation wavenumber $\ell$', 'Interpreter','latex');
grid on;
set(findall(gcf,'-property','FontSize'),'FontSize',24); 
%%
figure;
plot(real(diag(Lambda)), imag(diag(Lambda)), 'o', 'MarkerFaceColor', 'b');
xlabel('Real part');
ylabel('Imaginary part');
title('Spectrum of Linear Operator');
grid on;
axis equal;

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

    chi = 8*v.^2./(9 + 3*v.^2);

    f_u = (u - a) .* u .* (1 - u);
    
    nonlinear_term = D1 * u .* (D1 * chi);
    rh(1:Nxi) = delta*c*D1*u + (D2 * u) - (1/ep)*k*nonlinear_term + f_u;
    rh(Nxi+1:end - 1) = c*D1*v + (1/ep^2)*D2*v + u - gamma*v;

    rh(end) = ([D1*u; D1*v])' * ([u-u0; v-v0]);
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

    chi = 8*v.^2./(9 + 3*v.^2);

    J = spalloc(2*Nxi+1, 2*Nxi+1, 10*Nxi);

    df = 3*u.^2 - 2*(1 + a)*u + a;

    J(1:Nxi, 1:Nxi) = delta*c*D1 + D2 + ...
        - (1/ep)*k * D1 * spdiags(D1 * chi, 0, Nxi, Nxi)...
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