a = 0.25;

x_max = 100; 
t_max = 100;
dx = 0.1;
dt = 0.001; 
x = -100:dx:100;
t = 0:dt:t_max;

Nx = length(x);              
Nt = length(t);  

u0 = zeros(Nx, 1); 

% Initial Condition
for i=1:Nx
    u0(i) = (1/2) + (1/2)*tanh((x(i))/(2*sqrt(2))) + 0.1*cos(x(i));
end

[t,u_final] = ode45(@(t,u) rhs(t, u, dx, a, Nx), t, u0);
%%
plot(x, u_final(1000, :))
hold on 
plot(x, u_final(1, :))
plot(x, u_final(100000, :))
hold off
ylim([-0.2 1.2]);
%xlim([-100, 2000])

function udot=rhs(t,u, dx, a, n)

    udot=zeros(n,1);
    
    udot(1) = 2*(u(2)-u(1))/(dx^2) + u(1).*(u(1) - a).*(1 - u(1));
    
    for i = 2:n - 1
        udot(i) = (u(i+1) + u(i-1) - 2*u(i))/(dx^2) + u(i).*(u(i) - a).*(1 - u(i));
    end
    
    udot(n) = 2*(u(n-1)-u(n))/(dx^2) + u(n).*(u(n) - a).*(1 - u(n));

end

