a = 0.01:0.01:0.33;
c = zeros(length(a), 1);

xi_max = 100;
dxi = 0.1;
xi = 0:dxi:xi_max;

Nxi = length(xi);   

u0 = zeros(Nxi + 1, 1); 

% Initial Condition
for i=1:Nxi
    u0(i) = (1/2) + (1/2)*tanh((xi(i) - (xi_max)/2)/(2*sqrt(2))) + 0.1*cos(xi(i));
end
u0(Nxi + 1) = 0;

% Numerical continuation
for i=1:length(a)
    F = @(u) rhs(dxi, u, a(i), Nxi, u0);
    u_new = fsolve(F, u0);

    c(i) = u_new(end);
    u0 = u_new;
    u0(end) = 0;
end
%% Plot tanh profile
plot(xi, u(1:end-1))
ylim([-0.2 1.2])
xlabel('\xi'), ylabel('u(\xi)')
%hold on 
%plot(xi, u(1, :))
%plot(xi, u(100000, :))
%hold off
%xlim([-100, 2000])

%% Plot c(a)
plot(a, c)
hold on
plot(a, sqrt(2)*a - 1/sqrt(2))
title('Wavepeed c(a) as a function of a')
xlabel('a')
ylabel('c(a)')
%%
function rh = rhs(dxi, u, a, n, u0)

    rh=zeros(n+1,1);
    
    rh(1) = 2*(u(2)-u(1))/(dxi^2) + u(1).*(u(1) - a).*(1 - u(1)) + (u(n+1)*(u(2) - u(1))/dxi);
    
    for i = 2:n - 1
        rh(i) = (u(i+1) + u(i-1) - 2*u(i))/(dxi^2) + u(i).*(u(i) - a).*(1 - u(i)) + (u(n+1)*(u(i) - u(i-1))/dxi);
    end
    
    rh(n) = 2*(u(n-1)-u(n))/(dxi^2) + u(n).*(u(n) - a).*(1 - u(n)) + (u(n+1)*(u(n) - u(n-1))/dxi);

    % translation constraint
    rh(n+1) = diff(u(1:n))' * (u(1:n-1) - u0(1:n-1));

end

