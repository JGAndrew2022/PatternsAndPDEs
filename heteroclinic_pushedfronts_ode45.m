a = 0.1;

x_max = 125; 
t_max = 125;
dx = 0.1;
dt = 0.001; 
x = -x_max:dx:x_max;
t = 0:dt:t_max;

Nx = length(x);              
Nt = length(t);  

% D Matrix
e = ones(Nx,1);
D2 = spdiags([e -2*e e], -1:1, Nx, Nx) / dx^2;
D2(1,1) = -2/dx^2; D2(1,2) = 2/dx^2;
D2 = sparse(D2);

u0 = zeros(Nx, 1); 

pert = zeros(size(x));
pert(1:round(length(x)/3)) = 0.1*cos( x( 1:round(length(x)/3) ) );

% Initial Condition
for i=1:Nx
    u0_val = (1+a)/2 + (1-a)/2 * tanh(( x(i) )*(a - 1)/(2*sqrt(2))) + pert(i);
    % shift everything so that a is at 0 for numerical stability reasons
    u0(i) = u0_val - a;
end

% Analytical jacobian
jacfun = @(t, u) jacobian(u, D2, a);
opts = odeset('Jacobian', jacfun, 'JPattern', sparse(ones(Nx)));  % Optional: use sparsity pattern

[t,u_final] = ode15s(@(t,u) rhs(u, D2, a), t, u0, opts);
%%
plot(x, u_final(1, :) + a)
hold on 
plot(x, u_final(10000, :) + a)
plot(x, u_final(40000, :) + a)
yline(a, '--k', 'a', 'LabelHorizontalAlignment', 'left');
hold off
ylim([-0.2 1.2]);
legend('t = 0', 't = 10', 't = 100')
xlim([-125, 125])

function udot = rhs(u, D2, a)
    udot = D2 * u + (u + a) .* u .* (1 - u - a);
end

function J = jacobian(u, D2, a)
    df = (2*u + a) - 3*u.^2;
    J = D2 + spdiags(df, 0, length(u), length(u));
    J = sparse(J);
end

