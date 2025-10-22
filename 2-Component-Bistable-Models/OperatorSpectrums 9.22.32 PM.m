% Note: in this script, xi = x - Ct, which = (1/ep)*(X - ct)
% where X - ct is the xi coordinate used in Bistable_Newtons_Method.m, so
% c_val = ep*true_c_value. This is to avoid scaling x derivatives with 1/ep
a = 0.4;
gamma = 1;
k = 2;
delta = 1;
ep = 0.05;
a_plus = 0.9;
a_minus = 0.1;

ell_max = 0.5;
ells = linspace(-ell_max, ell_max, 100);
%ells = zeros(1, 20);
max_real_part = zeros(size(ells));

xi_max = 7.5;
dxi = 0.025;
xi = -xi_max:dxi:xi_max;

Nxi = length(xi);

% Differentiation matrices templates
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

I = speye(Nxi);


% Initial Condition
u0 = 0.5 - 0.5*tanh(-(1/ep)*xi/(2*sqrt(2)));
u0 = u0';

v0 = zeros(size(xi));
half = round(length(xi)/2);

v0(1:half) = 1/(2*gamma) * exp(sqrt(gamma)*xi(1:half));
v0(half + 1:end) = -1/(2*gamma) * exp(-sqrt(gamma)*xi(half+1:end)) + 1/gamma;
v0 = v0';

%%%%%%%%%%%% Step 1: FIND TRAVELING WAVE SOLUTION W/ NEWTON'S METHOD %%%%%%%%%%%%
options = optimoptions('fsolve',...
    'MaxIterations', 250, 'display', 'iter','Jacobian','on');

pars = struct('gamma', gamma, 'k', k, 'delta', delta, 'ep', ep, 'D1', D1, 'D2', D2, ...
            'dxi', dxi, 'Nxi', Nxi, 'a_minus', a_minus, 'a_plus', a_plus);
y0 = [u0; v0; 1];

residual = @(y) deal(rhs(y, a, pars, y0),jac_rhs(y, a, pars, y0));
[sol, ~, ~] = fsolve(residual, y0, options);

u = sol(1:Nxi);
v = sol(Nxi + 1 : end - 1);
c_val = sol(end);

%dchi = k*(8/3) * (2*v*3)./ ((3+v.^2).^2);
%dchi = 2*v;
dchi = k*(4/3) * (2*v*(3/4))./(( (3/4) + v.^2 ).^2);

max_real = zeros(length(ells));
for j = 1:length(ells)
    ell = ells(j);
    disp(j)

    Operator = jac_rhs(sol, a, pars, sol);
    Operator = Operator(1:end-1, 1:end-1);

    Operator = Operator - [ep^2*speye(Nxi,Nxi) sparse(Nxi, Nxi); sparse(Nxi, Nxi) speye(Nxi,Nxi)]*ell^2 ...
    +ell^2*ep*[sparse(Nxi, Nxi) spdiags(u.*dchi,0,Nxi,Nxi); sparse(Nxi, Nxi) sparse(Nxi, Nxi)];

    max_real(j) = eigs(Operator, 1,1);
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
plot(xi, sol(Nxi+1:end - 1))
hold on
plot(xi, sol(1:Nxi))
plot(xi, v0)
plot(xi, u0)
%yline(a(end), '-k')
%ylim([-0.2 1.2])
xlabel('\xi'), ylabel('u(\xi)')
%xlim([-1 1])
legend('Final v', 'Final u', 'v0', 'u0')

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

    %chi = k*(8/3) * v.^2 ./ (v.^2 + 3);
    %chi = k*v;
    chi = k*(4/3) * v.^2 ./ ( v.^2 + (3/4) );
    
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
    
    %chi = k*(8/3) * v.^2 ./ (v.^2 + 3);
    %dchi = k*(8/3) * (2*v*3)./ ((3+v.^2).^2);
    %chi = k*v;
    %dchi = k;
    chi = k*(4/3) * v.^2 ./ ( v.^2 + (3/4) );
    dchi = k*(4/3) * (2*v*(3/4))./(( (3/4) + v.^2 ).^2);

    J(1:Nxi, 1:Nxi) = ep*c*D1 + ep^2*D2 ...
    - ep*(spdiags(D1 * chi, 0, Nxi, Nxi)*D1 ...
              + spdiags(D2*chi,0,Nxi,Nxi))...
    + spdiags(dfdu, 0, Nxi, Nxi);
    
    J(1:Nxi, Nxi+1:end-1) = -ep * ( ...
    spdiags(D1*u,0,Nxi,Nxi) * D1 * spdiags(dchi,0,Nxi,Nxi) + ...
    spdiags(u,0,Nxi,Nxi)    * D2 * spdiags(dchi,0,Nxi,Nxi) ) + ...
    spdiags(dfdv,0,Nxi,Nxi);
    

    J(Nxi+1:end-1, 1:Nxi) =  speye(Nxi);
    
    J(Nxi+1:end-1, Nxi+1:end-1) =  ep*c*D1 + D2 - gamma * speye(Nxi);

    % last column
    J(1:Nxi, end) = ep*D1*u;
    J(Nxi+1:end - 1, end) = ep*D1*v;

    % Last row
    
    J(end, 1:Nxi)       = ((u - u0)' * D1 + (D1*u)');
    J(end, Nxi+1:2*Nxi) = ((v - v0)' * D1 + (D1*v)');
    J(end, end)         = 0;

    J = sparse(J);
end