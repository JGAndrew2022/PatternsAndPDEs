
% Parameters
a = 0.001;

dx = 0.1;
x = -100:dx:100;
dt = 0.001;
t = 0:dt:100; 
h = 1/(length(x)-1) ; %step size
u = zeros(length(t),length(x));

%Initial condition
for i=1:length(x)
    u(1,i)= (1/2) + (1/2)*tanh(x(i)/(2*sqrt(2))) + 0.1*cos(x(i));   
end 

n = length(x);

for j = 1:length(t)-1
    % left Neummann BC
    u(j+1,1) = u(j,1) + dt * (2*(u(j, 2)-u(j, 1))/(dx^2) + u(j, 1).*(u(j, 1) - a).*(1 - u(j, 1)));

    for i=2:n-1
        u(j+1,i) = u(j,i) + dt *( (u(j, i+1) + u(j, i-1) - 2*u(j, i))/(dx^2) + u(j, i).*(u(j, i) - a).*(1 - u(j, i)) );
    end 
    
    % right Neumann BC
    u(j+1, n) = u(j,n) + dt * (2*(u(j, n-1) - u(j, n))/(dx^2) + u(j, n).*(u(j, n) - a).*(1 - u(j, n)));

end
%%
figure(1)
plot(x, u(100000,:))
hold on
plot(x, u(1,:))
ylim([-0.2 1.2])