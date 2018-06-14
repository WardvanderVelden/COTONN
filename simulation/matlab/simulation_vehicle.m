%% clear all
clc
clear all


%% load nn
cd nn
vehicle
cd ..


%% controller parameters

% state space
tmp = size(s_ll);
s_dim = tmp(2);
s_ipd = calculateGridHelper(s_eta, s_ll, s_ur);

% input space
tmp = size(u_ll);
u_dim = tmp(2);
u_ipd = calculateGridHelper(u_eta, u_ll, u_ur);


%% nn parameters
tmp = size(W);
layers = tmp(2);

tmp = size(W{1});
inputs = tmp(1);

tmp = size(W{layers});
outputs = tmp(2);


%% simulation parameters
tau = 0.3;
%s = [0.6 0.6 0]; line_color = 1;
%s = [1.5 0.6 0]; line_color = 3;
s = [6.3 9 0]; line_color = 5;

s_list = s;
u_list = [];

loop = 250;

% target set
lb=[9 0];
ub=lb+0.5;


%% simulate system
while(loop>0)
    s = s_list(end,:);
    
    % check if state is (still) within the controller boundaries
    inside = true;
    for i=1:s_dim
       if(s(i) < s_ll(i) || s(i) > s_ur(i)) 
          inside = false;
       end
    end
    
    if(inside == false)
       disp("State is out of controller bounds")
       break
    end
    
    % stop when goal is reached
    if (lb(1) <= s(end,1) & s(end,1) <= ub(1) && lb(2) <= s(end,2) & s(end,2) <= ub(2))
        break;
    end 
        
    % get state binary
    %s_bin = x2bin(s, s_ipd, s_eta, s_ll, inputs);
    s_vec = x2vec(s, s_ipd, s_eta, s_ll, s_ur, inputs);

    % get input for given state
    %u_bin = neuralNetwork(s_bin, W, b);
    u_vec = neuralNetwork(s_vec, W, b);

    % bin input to input id
    %u = bin2x(u_bin, u_ipd, u_eta, u_ll, outputs);
    u = vec2x(u_vec, u_ipd, u_eta, u_ll, u_ur, outputs);

    % numerically integrate one tau
    u_list = [u_list; u];
    [t s] = ode45(@ode_vehicle, [0 tau], s_list(end,:), odeset('abstol',1e-12,'reltol',1e-12), u);
    s_list = [s_list; s(end,:)];
    
    loop = loop - 1;
end

%% plot system
colors = get(groot, 'DefaultAxesColorOrder');

% plot trajectory
hold on
plot(s_list(:,1),s_list(:,2),'k.-','color',colors(line_color,:))
plot(s_list(1,1),s_list(1,2),'.','color',zeros(1,3),'markersize',20)
hold on

box on
axis([-.5 10.5 -.5 10.5])


