a = 0.3;
gamma = 2;
k = 1;
delta = 1;
ep = 0.125;
d = ep*0.3;
s_0 = 0.3;

ell_max = 1;
ells = linspace(-ell_max, ell_max, 100);


c_val = 0;
c_analytical = zeros(length(a), 1);
c_num = zeros(length(a), 1);
q_star = zeros(length(a), 1);

xi_max = 100;
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

% Initial Guess
u0 = 0.5 - 0.5*tanh(-(1/ep)*xi/(2*sqrt(2)));
u0 = u0';

v0 = zeros(size(xi));
half = round(length(xi)/2);

phi_1 = (-c_val + sqrt(c_val^2 + 4*gamma))/2;
phi_2 = (-c_val - sqrt(c_val^2 + 4*gamma))/2;

v0(1:half) = phi_2/(gamma*(phi_2 - phi_1)) * exp(phi_1*xi(1:half));
v0(half + 1:end) = phi_1/(gamma*(phi_2 - phi_1)) * exp(phi_2*xi(half+1:end)) + 1/gamma;
v0 = v0';

w0 = zeros(size(xi));

w0(1:half) = 1/2 * exp(d*xi(1:half)) - s_0;
w0(half + 1:end) = -1/2 * exp(-d*xi(half+1:end)) + 1 - s_0;
w0 = w0';

options = optimoptions('fsolve', 'Display', 'off',...
    'MaxIterations', 250, 'SpecifyObjectiveGradient', true);

pars = struct('gamma', gamma, 'k', k, 'delta', delta, 'ep', ep, ...
                  'D1', D1, 'D2', D2, 'dxi', dxi, 'Nxi', Nxi, 'd', d, 's_0', s_0);
y0 = [u0; v0; w0; 0];

residual = @(y) deal(rhs(y, a, pars, y0),jac_rhs(y, a, pars, y0));
[sol, ~, ~] = fsolve(residual, y0, options);

u = sol(1:Nxi);
v = sol(Nxi + 1 : 2*Nxi);
w = sol(2*Nxi + 1 : end - 1);

c_val = sol(end);

dchi = 1;

max_real = zeros(length(ells));
for j = 1:length(ells)
    ell = ells(j);
    disp(j)

    Operator = jac_rhs(sol, a, pars, sol);
    Operator = Operator(1:end-1, 1:end-1);

    Operator = Operator - [ep^2*speye(Nxi,Nxi) -ep*spdiags(u.*dchi,0,Nxi,Nxi) sparse(Nxi, Nxi);...
        sparse(Nxi, Nxi) speye(Nxi,Nxi) sparse(Nxi, Nxi);...
        sparse(Nxi, Nxi) sparse(Nxi, Nxi) 1/(d^2)*speye(Nxi,Nxi)]*ell^2;

    max_real(j) = eigs(Operator, 1, 2);
end


%%
figure;
plot(ells, max_real, 'LineWidth', 1.5);
xlabel('$\ell$', 'Interpreter','latex'); ylabel('max Re($\lambda$)', 'Interpreter','latex');
title('$\lambda$-val with Largest Real Part vs perturbation wavenumber $\ell$', 'Interpreter','latex');
grid on;
set(findall(gcf,'-property','FontSize'),'FontSize',24);

fig = gcf;
fig.Units = 'pixels';              % work in pixels
pos = fig.Position;                % current size
fig.Position = [pos(1), pos(2), 1.8*pos(3), 1.8*pos(4)];
%%
figure;
plot(real(diag(Lambda)), imag(diag(Lambda)), 'o', 'MarkerFaceColor', 'b');
xlabel('Real part');
ylabel('Imaginary part');
title('Spectrum of Linear Operator');
grid on;
axis equal;
%% Plot Newton Results
plot(xi, u)
hold on
plot(xi, v)
plot(xi, w)
plot(xi, v0)
plot(xi, u0)
plot(xi, w0)
%yline(a(end), '-k')
%ylim([-0.2 1.2])
xlabel('\xi'), ylabel('u(\xi)')
%xlim([-1 1])
% legend('Final v', 'v0', 'Final u', 'u0')

% Note, I am using xi = x-ct instead of xi = 1/ep\tilde{x} - ct
function rh = rhs(y, a, pars, y0)
    Nxi = pars.Nxi;
    gamma = pars.gamma;
    k = pars.k;
    ep = pars.ep;
    D1 = pars.D1;
    D2 = pars.D2;
    rh = zeros(3*Nxi + 1,1);
    d = pars.d;
    s_0 = pars.s_0;

    u0 = y0(1:Nxi);
    v0 = y0(Nxi + 1 : 2*Nxi);
    w0 = y0(2*Nxi + 1 : end - 1);

    u = y(1:Nxi);
    v = y(Nxi+1 : 2*Nxi);
    w = y(2*Nxi + 1 : end - 1);
    c = y(end);

    chi = k*v;
    
    a_w = (0.5 + 0.5*tanh(w - a));

    f_u = (u - a_w) .* u .* (1 - u);
    
    nonlinear_term = (D1 * u) .* (D1 * chi) + u .* (D2 * chi);
    rh(1:Nxi) = ep*c*D1*u + ep^2*(D2 * u) - ep*nonlinear_term + f_u; % u_t
    rh(Nxi+1:2*Nxi) = c*D1*v + D2*v + u - gamma*v; % v_t
    rh(2*Nxi+1:end - 1) = c*D1*w + (1/d^2)*D2*w + ( u - w - s_0); %w_t
    rh(end) = ([D1*u; D1*v; D1*w])' * ([u-u0; v-v0; w-w0]);
    %rh(end) = u(floor(Nxi/2)) - 0.5;
end

function J = jac_rhs(y, a, pars, y0)
    Nxi = pars.Nxi;
    gamma = pars.gamma;
    k = pars.k;
    delta = pars.delta;
    ep = pars.ep;
    d = pars.d;
    D1 = pars.D1;
    D2 = pars.D2;

    u0 = y0(1:Nxi);
    v0 = y0(Nxi + 1 : 2*Nxi);
    w0 = y0(2*Nxi + 1 : end - 1);

    u = y(1:Nxi);
    v = y(Nxi + 1 : 2*Nxi);
    w = y(2*Nxi + 1 : end - 1);
    c = y(end);

    chi = k*v;
    dchi = k;

    J = spalloc(3*Nxi+1, 3*Nxi+1, 10*Nxi);
    
    a_w = 0.5 + 0.5*tanh(w - a);
    dfdu = -(3*u.^2 - 2*(1 + a_w).*u + a_w);
    
    % u equation
    %J(1:Nxi, 1:Nxi) = ep*delta*c*D1 + ep^2*D2 + ...
    %    - ep*k * D1 * spdiags(D1 * v, 0, Nxi, Nxi)...
    %    - spdiags(df, 0, Nxi, Nxi);

    J(1:Nxi, 1:Nxi) = ep*delta*c*D1 + ep^2*D2 + ...
    - ep * ( spdiags(D1 * chi, 0, Nxi, Nxi)*D1 ...
              + spdiags(D2*chi,0,Nxi,Nxi))...
    + spdiags(dfdu, 0, Nxi, Nxi);

    J(1:Nxi, Nxi+1:2*Nxi) = -ep*( ...
       spdiags(D1*u,0,Nxi,Nxi)*D1*spdiags(dchi,0,Nxi,Nxi) + ...
       spdiags(u,0,Nxi,Nxi)*D2*spdiags(dchi,0,Nxi,Nxi) );
    %J(1:Nxi, Nxi+1:2*Nxi) = -ep * k * D1 * spdiags(u, 0, Nxi, Nxi) * D1;
    
    F_w = 0.5*u.*(u-1).*(sech(w - a)).^2;
    J(1:Nxi, 2*Nxi+1:end-1) = spdiags(F_w, 0, Nxi, Nxi);

    % v equation   
    J(Nxi+1:2*Nxi, 1:Nxi) = speye(Nxi);
    J(Nxi+1:2*Nxi, Nxi+1:2*Nxi) = c*D1 + D2 - gamma * speye(Nxi);
    % 0

    % w equation:  c*D1*w + (1/d^2)*D2*w + u - w - s_0 = 0
    J(2*Nxi+1:end - 1, 1:Nxi) = speye(Nxi);
    % 0
    J(2*Nxi+1:end - 1, 2*Nxi+1:end - 1) = c*D1 + (1/d^2)*D2 - speye(Nxi);

    % last column
    J(1:Nxi, end) = ep*D1*u;
    J(Nxi+1:2*Nxi, end) = D1*v;
    J(2*Nxi+1:3*Nxi, end) = D1*w;

    % Last row
    
    J(end, 1:Nxi)       = ((u - u0)' * D1 + (D1*u)');
    J(end, Nxi+1:2*Nxi) = ((v - v0)' * D1 + (D1*v)');
    J(end, 2*Nxi+1:3*Nxi) = ((w - w0)' * D1 + (D1*w)');
    J(end, end)         = 0;

    J = sparse(J);
end