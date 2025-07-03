a = 0.5;
gamma = 2;
k = 2;
delta = 1;
ep = 0.1;

x_max = 10; 
t_max = 100;
dx = 0.01;
x = -x_max:dx:x_max;
tspan = [0 t_max];

Nx = length(x);              
Nt = length(tspan);  

% D Matrix
e = ones(Nx,1);
D2 = spdiags([e -2*e e], -1:1, Nx, Nx) / dx^2;
%BCs (Neumann)
D2(1,1:2) = [-2 2]/dx^2;
D2(Nx,Nx-1:Nx) = [2 -2]/dx^2;
D2 = sparse(D2);


D1 = spdiags([-e zeros(Nx,1) e], -1:1, Nx, Nx) / (2*dx);
%BCs (Neumann)
D1(1,:) = 0;
D1(Nx,:) = 0;
D1 = sparse(D1);

% Initial Condition
u0 = 0.5 - 0.5*tanh(-(1/ep)*x/(2*sqrt(2)));
u0 = u0';

v0 = zeros(size(x));
half = round(length(x)/2);

v0(1:half) = 1/(2*gamma) * exp(sqrt(gamma)*x(1:half));
v0(half + 1:end) = -1/(2*gamma) * exp(-sqrt(gamma)*x(half+1:end)) + 1/gamma;
v0 = v0';

y0 = [u0; v0];

rhs = @(t, y) rhs_pde(t, y, D1, D2, a, gamma, delta, ep, k);

opts = odeset('Jacobian', @(t, y) jacobian(y, D1, D2, a, gamma, delta, ep, k), ...
              'RelTol', 1e-6, 'AbsTol', 1e-8, ...
              'InitialStep', 1e-4, 'MaxStep', 0.1, 'Stats', 'on');

[t, Y] = ode15s(rhs, tspan, y0, opts);

u_sol = Y(:, 1:Nx);
v_sol = Y(:, Nx+1:end);
%%
plot(x, u_sol(1,:)); hold on;
plot(x, u_sol(round(end/4),:));
plot(x, u_sol(end,:));
%plot(x, v_sol(end,:));
%plot(x, v_sol(1,:));

ylim([-0.2 1.2]);
legend('t = 0', 't = 25', 't = 100')
xlim([-x_max, x_max])


function dydt = rhs_pde(~, y, D1, D2, a, gamma, delta, ep, k)
    Nx = length(y)/2;
    u = y(1:Nx);
    v = y(Nx+1:end);

    f_u = u .* (u - a) .* (1 - u);
                      
    nonlinear_term = D1 * u .* (D1 * v); 

    udot = (1/delta)*(ep^2 * (D2 * u) - ep * k * nonlinear_term + f_u);
    vdot = D2 * v + u - gamma * v;

    dydt = [udot; vdot];
end

function J = jacobian(y, D1, D2, a, gamma, delta, ep, k)
    Nx = length(y)/2;
    u = y(1:Nx);
    v = y(Nx+1:end);
    
    q = D1 * v;
    fprime = 3*u.^2 - 2*(1 + a)*u + a;
    
    A = (1/delta) * ( ep^2 * D2 ...
                    - ep * k * D1 * spdiags(q, 0, Nx, Nx) ...
                    + spdiags(fprime, 0, Nx, Nx) );
    
    B = -(ep * k) * D1 * spdiags(u, 0, Nx, Nx) * D1;

    C = speye(Nx);
    
    D = D2 - gamma * speye(Nx);
    
    J = [A, B;
         C, D];
    J = sparse(J);
end


