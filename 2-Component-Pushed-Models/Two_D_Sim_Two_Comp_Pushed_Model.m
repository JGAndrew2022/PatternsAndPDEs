% Params
a = 0.4;
gamma = 1;
k = 10;
delta = 1;
ep = 0.05;
tend = 100;

xi_max = 100;
dxi = 0.5;
xi = -xi_max:dxi:xi_max;

y_max = 200;
dy = 1;
y = -y_max:dy:y_max;

Nxi = length(xi); 
Ny = length(y);

dt = 0.1;
K = 50;

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
u_line = sol_newtons(1:Nxi); % get solutions from Bistable_Newtons_Method.m
v_line = sol_newtons(Nxi+1:end-1);

u0 = repmat(u_line(:), Ny, 1);     % [Nxi*Ny x 1]
v0 = repmat(v_line(:), Ny, 1);     % [Nxi*Ny x 1]
c = sol_newtons(end);

%sol = [u0;v0]+ 0.05*(rand(2*Nxi*Ny,1));

pars = struct('gamma', gamma, 'k', k, 'delta', delta, 'ep', ep, ...
                  'Dx', Dx, 'Dy', Dy, 'D2y', D2y, 'D2x', D2x,...
                  'dxi', dxi, 'Nxi', Nxi, 'dy', dy, 'Ny', Ny, 'c', c, 'a', a);
% ADD TRANSVERSAL PERTURBATIONS (for small modes since we want \ell << 1)
Ly = y(end)-y(1);

m = 0:40;              
phi = 2*pi*m;
Y = (y(:)-y(1))/Ly;     
S = sin(2*pi*(Y*m) + phi);    
siny_sum = sum(S,2)/sqrt(numel(m));
amp = 0.05;

Upert = amp * (u_line(:)) * (siny_sum.');
Vpert = amp * (v_line(:)) * (siny_sum.');

u_init = reshape(u0, [Nxi, Ny]) + Upert;
v_init = reshape(v0, [Nxi, Ny]) + Vpert;

sol = [u_init(:); v_init(:)];
%% Solve
solution= [];
times = [];
jac_rhs_s = @(t,y)jac_rhs(t,y,pars);

options=odeset('RelTol',1e-8,'AbsTol',1e-8,'Jacobian',jac_rhs_s);

for j=0:K-1
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
%{
for i = 1:length(times)
    figure(1)
    imagesc(xi, y, reshape(solution(1:Nxi*Ny, i), [Nxi, Ny])')
    set(gca, 'YDir', 'normal');
    colormap(flipud(summer));
    shading flat
    drawnow
    pause;
end 
%}

for i = 1:length(times)
    figure(1)
    imagesc(xi, y, reshape(solution(1:Nxi*Ny, i), [Nxi, Ny])')
    set(gca, 'YDir', 'normal');
    colormap(flipud(summer));
    shading flat
    axis tight
    exportgraphics(gca, sprintf('frame_%03d.pdf', i), 'ContentType', 'vector');
end


%% RHS and Jacobian Functions

function rh = rhs(t, solution, pars)
    Nxi = pars.Nxi;
    Ny = pars.Ny;
    gamma = pars.gamma;
    k = pars.k;
    delta = pars.delta;
    ep = pars.ep;
    Dx = pars.Dx;
    Dy = pars.Dy;
    D2x = pars.D2x;
    D2y = pars.D2y;
    c = pars.c;
    a = pars.a;

    u = solution(1:Nxi*Ny);
    v = solution(Nxi*Ny+1:2*Nxi*Ny);

    f_u = (u - a) .* u .* (1 - u);
    
    chi = k*(8*sqrt(3)/9) * v.^2 ./ (v.^2 + 16);

    nonlinear_term = ((Dx + Dy) * u) .* ((Dx + Dy) * chi) + u .* ((D2x + D2y) * chi);
    rh1 = c*Dx*u + ((D2x + D2y) * u) - (1/ep)*nonlinear_term + f_u;
    rh2 = c*(Dx)*v + (1/ep^2)*(D2x + D2y)*v + u - gamma*v;

    rh = [rh1; rh2];
end

function J = jac_rhs(t, solution, pars)
    Nxi = pars.Nxi;
    Ny = pars.Ny;
    gamma = pars.gamma;
    k = pars.k;
    delta = pars.delta;
    ep = pars.ep;
    Dx = pars.Dx;
    Dy = pars.Dy;
    D2x = pars.D2x;
    D2y = pars.D2y;
    c = pars.c;
    a = pars.a;

    u = solution(1:Nxi*Ny);
    v = solution(Nxi*Ny+1:2*Nxi*Ny);

    chi = k*(8*sqrt(3)/9) * v.^2 ./ (v.^2 + 16);
    dchi = k*(8*sqrt(3)/9) * (2*v*16)./ ((16+v.^2).^2);

    df = -(3*u.^2 - 2*(1 + a)*u + a);

    J11 = c*Dx + (D2x + D2y) + ...
    - (1/ep) * ( spdiags((Dx + Dy) * chi, 0, Nxi*Ny, Nxi*Ny)*(Dx + Dy) ...
              + spdiags((D2x + D2y)*chi,0,Nxi*Ny,Nxi*Ny))...
    + spdiags(df, 0, Nxi*Ny, Nxi*Ny);
    
    J12 = -(1/ep)*( ...
       spdiags((Dx + Dy)*u,0,Nxi*Ny,Nxi*Ny) * (Dx + Dy) * spdiags(dchi,0,Nxi*Ny,Nxi*Ny) + ...
       spdiags(u,0,Nxi*Ny,Nxi*Ny)*(D2x + D2y)*spdiags(dchi,0,Nxi*Ny,Nxi*Ny) );


    J21 = speye(Nxi*Ny);
    
    J22 = c*(Dx) + (1/ep^2)*(D2x + D2y) - gamma * speye(Nxi*Ny);

    J=[J11,J12;J21,J22];
    J = sparse(J);
end