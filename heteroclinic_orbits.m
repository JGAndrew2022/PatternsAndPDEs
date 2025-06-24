a = 0.25;

x_max = 100;    
dx = x_max/1e3;
dt = 0.1;    

Nx = round(x_max / dx);  
T_max = 10;              
Nt = round(T_max / dt);  

u0 = zeros(Nx, 1); 

% Initial Condition
for i=-Nx:Nx - 1
    u0(i+Nx+1) = (1/2) + (1/2)*tanh(i*dx/(2*sqrt(2))) + 0.2*cos(i*dx);
end

tspan = [0 T_max];

[t,u] = ode45(@(t,u) rhs(t, u, dx, a, Nx), tspan, u0);
%%
plot(u(1, :))
ylim([0 1.2]);
xlim([-100, x_max])

function udot=rhs(t,u, dx, a, n)

    udot=zeros(n,1);
    
    udot(1) = 2*(u(2)-u(1))/(dx^2) + u(1).*(u(1) - a).*(1 - u(1));
    
    for i = 2:2*n - 1
        udot(i) = (u(i+1) + u(i-1) - 2*u(i))/(dx^2) + u(i).*(u(i) - a).*(1 - u(i));
    end
    
    udot(2*n) = 2*(u(2*n-1)-u(2*n))/(dx^2) + u(2*n).*(u(2*n) - a).*(1 - u(2*n));

end

