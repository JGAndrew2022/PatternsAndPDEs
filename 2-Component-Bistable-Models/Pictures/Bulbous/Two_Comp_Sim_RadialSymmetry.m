% Params
a = 0.1;
gamma = 1;
k = 1;
delta = 1;
ep = 0.05;
tend = 50;

xi_max = 7.5;
dxi = 0.05;
xi = -xi_max:dxi:xi_max;

y_max = 7.5;
dy = 0.05;
y = -y_max:dy:y_max;

Nxi = length(xi); 
Ny = length(y);

dt = 1;
K = 40;

a_plus = 0.3;
a_minus = 0.1;

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
%Dy(1,Ny) = -1/2; % Periodic boundary conditions
Dy = (Dy - Dy')/dy;
Dy(1,2) = 0; Dy(Ny,Ny-1) = 0; % Neumann boundary conditions

% d_xx
D2x = sparse(1:Nxi-1,[2:Nxi-1 Nxi],ones(Nxi-1,1),Nxi,Nxi) - sparse(1:Nxi,[1:Nxi],e0xi,Nxi,Nxi);
D2x = (D2x + D2x');
%D2x(1,Nxi) = 1; D2x(Nxi,1) = 1; % Periodic boundary conditions
D2x(1,2)=2; D2x(Nxi,Nxi-1)=2; % Neumann boundary conditions
D2x = D2x/dxi^2;

% d_yy
D2y = sparse(1:Ny-1,[2:Ny-1 Ny],ones(Ny-1,1),Ny,Ny) - sparse(1:Ny,[1:Ny],e0y,Ny,Ny);
D2y = (D2y + D2y');
D2y(1,2)=2; D2y(Ny,Ny-1)=2; % Neumann boundary conditions
%D2y(1,Ny) = 1; D2y(Ny,1) = 1; % Periodic boundary conditions
D2y = D2y/dy^2;

% create differentiation matrices
Dx = sparse(kron(ey,Dx));
Dy = sparse(kron(Dy,ex));
D2x = sparse(kron(ey,D2x));
D2y = sparse(kron(D2y,ex));


%% Initial Conditions (radially symmetric)
% get 1-D front from Bistable_Newtons_Method.m
u_line = sol_newtons(1:Nxi);          % length Nxi profile in x
v_line = sol_newtons(Nxi+1:end-1);    % length Nxi profile in x
c_front = sol_newtons(end);

% Usually set c=0 so the comoving x-advection doesn't break radial symmetry.
% If you want a drifting circle in +x, keep c_front; otherwise set c=0:
c = 0;

% Build (X,Y) grid
[X, Y] = meshgrid(xi, y);      % X, Y are Ny-by-Nxi
x0 = 0; y0 = 0;                 % center of the circle
R0 = 4.0;                       % radius where interface sits

% Interpolants of the 1-D profiles (extrapolate flat at ends)
Fu = griddedInterpolant(xi(:), u_line(:), 'linear', 'nearest');
Fv = griddedInterpolant(xi(:), v_line(:), 'linear', 'nearest');

% Radial coordinate and "signed distance" to the circle
R = sqrt((X - x0).^2 + (Y - y0).^2);
S = R0 - R;

% Evaluate front profiles on S so the jump is on r = R0
U0 = Fu(S);   % Ny-by-Nxi
V0 = Fv(S);   % Ny-by-Nxi

% (Optional) smooth the very center a touch to avoid tiny stiffness:
% U0(R < (R0 - 3*dxi)) = U0(R < (R0 - 3*dxi));
% V0(R < (R0 - 3*dxi)) = V0(R < (R0 - 3*dxi));

% Vectorize to match your kron(ey,Dx)/reshape usage:
u0 = reshape(U0.', Nxi*Ny, 1);  % note transpose before reshape
v0 = reshape(V0.', Nxi*Ny, 1);

% small perturbation if you want to break perfect symmetry:
rng(0);
sol = [u0; v0] + 0.02*randn(2*Nxi*Ny,1);

% pack params (use c chosen above)
pars = struct('gamma', gamma, 'k', k, 'delta', delta, 'ep', ep, ...
              'Dx', Dx, 'Dy', Dy, 'D2y', D2y, 'D2x', D2x, ...
              'dxi', dxi, 'Nxi', Nxi, 'dy', dy, 'Ny', Ny, 'c', c, 'a', a, ...
              'a_minus', a_minus, 'a_plus', a_plus);
% ADD TRANSVERSAL PERTURBATIONS (for small modes since we want \ell << 1)
Ly = y(end)-y(1);

sol = [u0;v0]+ 0.05*(rand(2*Nxi*Ny,1));

solution= [];
times = [];
jac_rhs_s = @(t,y)jac_rhs(t,y,pars);

options=odeset('RelTol',1e-8,'AbsTol',1e-8,'Jacobian',jac_rhs_s);
%% Solve
for j=0:K-1
    disp(j)
    time = 0:dt:tend;
    sol = ode15s(@(t,y)rhs(t,y,pars), time, sol, options);
    times = [times j*tend];
    sol = sol.y(:,end);
    solution = [solution sol];


    % This saves a plot
    % Used settings from:
    % https://www.nature.com/articles/s41467-020-19160-7 for
    % multi-sequential data
    % Install at https://www.mathworks.com/matlabcentral/fileexchange/45208-colorbrewer-attractive-and-distinctive-colormaps?status=SUCCESS
    % then run in MATLAB terminal
    % addpath('<pathToFolder>/BrewerMap')
    fig = figure('Visible','off');
    imagesc(xi, y, reshape(sol(1:Nxi*Ny), [Nxi, Ny])')
    set(gca, 'YDir', 'normal');

    cmap = colormap(brewermap([],"YlGnBu"));       
    
    colormap(cmap);
    
    shading flat;
    axis tight;
    exportgraphics(gca, sprintf('frame_%02d_bulbus.pdf', j), 'ContentType', 'vector');
    close(fig);
    
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
title(sprintf('Final u profile at y = 0'));
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
%}

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
    a_minus = pars.a_minus;
    a_plus = pars.a_plus;

    u = solution(1:Nxi*Ny);
    v = solution(Nxi*Ny+1:2*Nxi*Ny);

    a_of_v = a_minus + (a_plus - a_minus)*v./(1 + v);
    f_u = (u - a_of_v) .* u .* (1 - u);

    chi = k*(8/3) * v.^2 ./ (v.^2 + 3);
    %chi = k*v;
    %chi = k*(4/3) * v.^2 ./ ( v.^2 + (3/4) );

    nonlinear_term = ep*(Dx*u).*(Dx*chi) + ep*(Dy*u).*(Dy*chi) + ep*u.*( (D2x + D2y) * chi );
    rh1 = ep*c*Dx*u + ep^2*((D2x + D2y) * u) - nonlinear_term + f_u;
    rh2 = ep*c*(Dx)*v + (D2x + D2y)*v + u - gamma*v;

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
    a_minus = pars.a_minus;
    a_plus = pars.a_plus;

    u = solution(1:Nxi*Ny);
    v = solution(Nxi*Ny+1:2*Nxi*Ny);

    chi = k*(8/3) * v.^2 ./ (v.^2 + 3);
    dchi = k*(8/3) * (2*v*3)./ ((3+v.^2).^2);
    %chi = k*v;
    %dchi = k;
    %chi = k*(4/3) * v.^2 ./ ( v.^2 + (3/4) );
    %dchi = k*(4/3) * (2*v*(3/4))./(( (3/4) + v.^2 ).^2);

    a_of_v = a_minus + (a_plus - a_minus)*v./(1 + v);
    dfdu = -(3*u.^2 - 2*(1 + a_of_v).*u + a_of_v);
    dfdv = -1./((1 + v).^2).*(a_plus - a_minus).*u.*(1 - u);

    J11 = ep*c*Dx + ep^2*( D2x +  D2y) ...
    - ep*( ...
        spdiags(Dx*chi, 0,Nxi*Ny,Nxi*Ny) * Dx + ...
        spdiags(Dy*chi, 0,Nxi*Ny,Nxi*Ny) * Dy + ...
        spdiags( (D2x + D2y) * chi, 0,Nxi*Ny,Nxi*Ny) ) ...
    + spdiags(dfdu, 0,Nxi*Ny,Nxi*Ny);
    
    J12 = -ep*( ...
       spdiags( Dx*u ,0,Nxi*Ny,Nxi*Ny) * Dx * spdiags(dchi,0,Nxi*Ny,Nxi*Ny) + ...
       spdiags( Dy*u ,0,Nxi*Ny,Nxi*Ny) * Dy * spdiags(dchi,0,Nxi*Ny,Nxi*Ny) + ...
       spdiags(u,0,Nxi*Ny,Nxi*Ny) * (D2x + D2y) * spdiags(dchi,0,Nxi*Ny,Nxi*Ny) ) + ...
       spdiags(dfdv,0,Nxi*Ny,Nxi*Ny);


    J21 = speye(Nxi*Ny);
    
    J22 = ep*c*(Dx) + ( D2x + D2y ) - gamma * speye(Nxi*Ny);

    J=[J11,J12;J21,J22];
    J = sparse(J);
end