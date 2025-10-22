a = 0.3;
gamma = 2;
k = 1;
delta = 1;
ep = 0.125;
d = ep*0.3;
s_0 = 0.3;

xi_max = 100;
dxi = 0.1;
xi = -xi_max:dxi:xi_max;

y_max = 350;
dy = 5;
y = -y_max:dy:y_max;

Nxi = length(xi); 
Ny = length(y);

dt = 1;
K = 50;  

tend = 500;

%% differentiation matrices
e0xi = ones(Nxi,1);
e0y = ones(Ny,1);

% identity matrices
ex = sparse(1:Nxi,[1:Nxi],e0xi,Nxi,Nxi); % Nxi identity
ey = sparse(1:Ny,[1:Ny],e0y,Ny,Ny); % Ny identity

% d_x
Dx = sparse(1:Nxi-1,[2:Nxi-1 Nxi],ones(Nxi-1,1)/2,Nxi,Nxi);
%Dx(1,Nxi) = -1/2; % Periodic boundary conditions
Dx = (Dx - Dx')/dxi;
Dx(1,2) = 0; Dx(Nxi,Nxi-1) = 0; % Neumann boundary conditions

% d_y
Dy = sparse(1:Ny-1,[2:Ny-1 Ny],ones(Ny-1,1)/2,Ny,Ny);
Dy(1,Ny) = -1/2; % Periodic boundary conditions
Dy = (Dy - Dy')/dy;
%Dy(1,2) = 0; Dy(Ny,Ny-1) = 0; % Neumann boundary conditions

% d_xx
D2x = sparse(1:Nxi-1,[2:Nxi-1 Nxi],ones(Nxi-1,1),Nxi,Nxi) - sparse(1:Nxi,[1:Nxi],e0xi,Nxi,Nxi);
D2x = (D2x + D2x');
%D2x(1,Nxi) = 1; D2x(Nxi,1) = 1; % Periodic boundary conditions
D2x(1,2)=2; D2x(Nxi,Nxi-1)=2; % Neumann boundary conditions
D2x = D2x/dxi^2;

% d_yy
D2y = sparse(1:Ny-1,[2:Ny-1 Ny],ones(Ny-1,1),Ny,Ny) - sparse(1:Ny,[1:Ny],e0y,Ny,Ny);
D2y = (D2y + D2y');
%D2y(1,2)=2; D2y(Ny,Ny-1)=2; % Neumann boundary conditions
D2y(1,Ny) = 1; D2y(Ny,1) = 1; % Periodic boundary conditions
D2y = D2y/dy^2;

% create differentiation matrices
Dx = sparse(kron(ey,Dx));
Dy = sparse(kron(Dy,ex));
D2x = sparse(kron(ey,D2x));
D2y = sparse(kron(D2y,ex));

%% Initial Conditions
u_line = sol_newtons(1:Nxi); % run newtonsThreePar.m first to get ICs
v_line = sol_newtons(Nxi+1:2*Nxi);
w_line = sol_newtons(2*Nxi+1:end-1);

u0 = repmat(u_line(:), Ny, 1);
v0 = repmat(v_line(:), Ny, 1);   
w0 = repmat(w_line(:), Ny, 1); 
c = sol_newtons(end);
pars = struct('gamma', gamma, 'k', k, 'delta', delta, 'ep', ep, ...
                  'Dx', Dx, 'Dy', Dy, 'D2y', D2y, 'D2x', D2x, 'dxi', dxi,...
                   'Nxi', Nxi, 'dy', dy, 'Ny', Ny, 'c', c, 'a', a, 'd', d, 's_0', s_0);

% ADD TRANSVERSAL PERTURBATIONS (center around ell = 0.05)
Ly = y(end)-y(1);

m = 10:50;              
phi = 2*pi*rand(1, numel(m));
Y = (y(:)-y(1))/Ly;     
S = sin(2*pi*(Y*m) + phi);    
siny_sum = sum(S,2)/sqrt(numel(m));
amp = 0.05;

Upert = amp * (u_line(:)) * (siny_sum.');
Vpert = amp * (v_line(:)) * (siny_sum.');
Wpert = amp * (w_line(:)) * (siny_sum.');

u_init = reshape(u0, [Nxi, Ny]) + Upert;
v_init = reshape(v0, [Nxi, Ny]) + Vpert;
w_init = reshape(w0, [Nxi, Ny]) + Wpert;

% put back in Nxi*Ny x 1 vector form
sol = [u_init(:); v_init(:); w_init(:)];
%% Solve
solution= [];
times = [];
jac_rhs_s = @(t,y)jac_rhs(t,y,pars);

%options=odeset('RelTol',1e-8,'AbsTol',1e-8,'Jacobian',jac_rhs_s);

for j=10:K-1
    disp(j)
    time = 0:dt:tend;
    sol = ode15s(@(t,y)rhs(t,y,pars), time, sol, options);
    times = [times j*tend];
    sol = sol.y(:,end);
    solution = [solution sol];
end
%% View 1-D solution at final time 
N = Nxi * Ny;
final_state = solution(:, end);


mid_y_index = ceil(Ny / 2);

u_final = final_state(1:N); 
u_row = u_final((mid_y_index-1)*Nxi + 1 : mid_y_index*Nxi);

figure;
plot(xi, u_row, 'LineWidth', 2);
xlabel('x');
ylabel('u(x, t_{final})');
title(sprintf('Final u profile at y = %.3f', y(mid_y_index)));
grid on;
%% plot solution

for i = 1:length(times)
    figure(1)
    imagesc(xi, y, reshape(solution(1:Nxi*Ny, i), [Nxi, Ny])')
    set(gca, 'YDir', ['normal' ...
        '']);
    colormap(flipud(summer));
    shading flat
    drawnow
    pause;
end 

%{
for i = 1:length(times)
    figure(1)
    imagesc(xi, y, reshape(solution(1:Nxi*Ny, i), [Nxi, Ny])')
    set(gca, 'YDir', 'normal');
    colormap(flipud(summer));
    shading flat
    axis tight
    exportgraphics(gca, sprintf('frame_%03d.pdf', i), 'ContentType', 'vector');
end
%}
%% RHS and Jacobian Functions
function rh = rhs(t, solution, pars)
    Nxi = pars.Nxi;
    Ny = pars.Ny;
    gamma = pars.gamma;
    k = pars.k;
    ep = pars.ep;
    Dx = pars.Dx;
    Dy = pars.Dy;
    D2x = pars.D2x;
    D2y = pars.D2y;
    c = pars.c;
    d = pars.d;
    a = pars.a;
    s_0 = pars.s_0;

    u = solution(1:Nxi*Ny);
    v = solution(Nxi*Ny+1:2*Nxi*Ny);
    w = solution(2*Nxi*Ny+1:3*Nxi*Ny);

    chi = k*v;
    
    a_w = (0.5 + 0.5*tanh(w - a));

    f_u = (u - a_w) .* u .* (1 - u);
    
    nonlinear_term = ((Dx + Dy) * u) .* ((Dx + Dy) * chi) + u .* ((D2x + D2y) * chi);
    rh1 = ep*c*Dx*u + ep^2*((D2x + D2y) * u) - ep*nonlinear_term + f_u;
    rh2 = c*(Dx)*v + (D2x + D2y)*v + u - gamma*v;
    rh3 = c*Dx*w + (1/d^2)*(D2x + D2y)*w + ( u - w - s_0); %w_t
    
    rh = [rh1; rh2; rh3];
end

function J = jac_rhs(t, solution, pars)
    Nxi = pars.Nxi;
    Ny = pars.Ny;
    gamma = pars.gamma;
    k = pars.k;
    ep = pars.ep;
    Dx = pars.Dx;
    Dy = pars.Dy;
    D2x = pars.D2x;
    D2y = pars.D2y;
    c = pars.c;
    d = pars.d;
    a = pars.a;
    s_0 = pars.s_0;

    u = solution(1:Nxi*Ny);
    v = solution(Nxi*Ny+1:2*Nxi*Ny);
    w = solution(2*Nxi*Ny+1:3*Nxi*Ny);

    chi = k*v;
    dchi = k;
    
    a_w = (0.5 + 0.5*tanh(w - a));
    df = -(3*u.^2 - 2*(1 + a_w).*u + a_w);
    
    
    J11 = ep*c*Dx + ep^2*(D2x + D2y) ...
    - ep * ( ...
        spdiags(Dx*chi,0,Nxi*Ny,Nxi*Ny) * Dx + ...
        spdiags(Dy*chi,0,Nxi*Ny,Nxi*Ny) * Dy + ...
        spdiags((D2x + D2y)*chi,0,Nxi*Ny,Nxi*Ny) ) ...
    + spdiags(df,0,Nxi*Ny,Nxi*Ny);
    
    J12 = -ep*( ...
       spdiags( (Dx*u) ,0,Nxi*Ny,Nxi*Ny) * Dx * spdiags(dchi, 0, Nxi*Ny,Nxi*Ny) + ...
       spdiags( (Dy*u) ,0,Nxi*Ny,Nxi*Ny) * Dy * spdiags(dchi, 0, Nxi*Ny,Nxi*Ny) +...
       spdiags(u,0,Nxi*Ny,Nxi*Ny) * (D2x + D2y) * spdiags(dchi,0,Nxi*Ny,Nxi*Ny));

    F_w = 0.5*u.*(u-1).*(sech(w - a)).^2;
    J13 = spdiags(F_w, 0, Nxi*Ny, Nxi*Ny);

    J21 = speye(Nxi*Ny);
    
    J22 = c*(Dx) + (D2x + D2y) - gamma * speye(Nxi*Ny);

    J23 = sparse(Nxi*Ny, Nxi*Ny);
    
    J31 = speye(Nxi*Ny);
    J32 = sparse(Nxi*Ny, Nxi*Ny);
    J33 = c*Dx + (1/d^2)*(D2x + D2y) - speye(Nxi*Ny);
    
    J = [J11, J12, J13; J21, J22, J23; J31, J32, J33];

    J = sparse(J);
end