% Assignment 3 - CFRM - Hardik Chugh - 1005587866
clear all;
clc
format long;

Nout  = 100000; % number of out-of-sample scenarios
Nin   = 5000;   % number of in-sample scenarios
Ns    = 5;      % number of idiosyncratic scenarios for each systemic

C = 8;          % number of credit states

% Filename to save out-of-sample scenarios
filename_save_out  = 'scen_out';

% Read and parse instrument data
instr_data = dlmread('instrum_data.csv', ',');
instr_id   = instr_data(:,1);           % ID
driver     = instr_data(:,2);           % credit driver
beta       = instr_data(:,3);           % beta (sensitivity to credit driver)
recov_rate = instr_data(:,4);           % expected recovery rate
value      = instr_data(:,5);           % value
prob       = instr_data(:,6:6+C-1);     % credit-state migration probabilities (default to A)
exposure   = instr_data(:,6+C:6+2*C-1); % credit-state migration exposures (default to A)
retn       = instr_data(:,6+2*C);       % market returns

K = size(instr_data, 1); % number of  counterparties

% Read matrix of correlations for credit drivers
rho = dlmread('credit_driver_corr.csv', '\t');
sqrt_rho = (chol(rho))'; % Cholesky decomp of rho (for generating correlated Normal random numbers)

disp('======= Credit Risk Model with Credit-State Migrations =======')
disp('============== Monte Carlo Scenario Generation ===============')
disp(' ')
disp(' ')
disp([' Number of out-of-sample Monte Carlo scenarios = ' int2str(Nout)])
disp([' Number of in-sample Monte Carlo scenarios = ' int2str(Nin)])
disp([' Number of counterparties = ' int2str(K)])
disp(' ')

% Find credit-state for each counterparty
% 8 = AAA, 7 = AA, 6 = A, 5 = BBB, 4 = BB, 3 = B, 2 = CCC, 1 = default
[Ltemp, CS] = max(prob, [], 2);
clear Ltemp

% Account for default recoveries
exposure(:, 1) = (1-recov_rate) .* exposure(:, 1);

% Compute credit-state boundaries
CS_Bdry = norminv( cumsum(prob(:,1:C-1), 2) );
CS_Bdry1=zeros(K,C+1);
for i=1:K
    for j=1:C-1
        CS_Bdry1(i,j+1) = CS_Bdry(i,j);
    end
end
CS_Bdry1(:,C+1)=999;
CS_Bdry1(:,1)=-999;
CS_Bdry=CS_Bdry1;

if(~exist('scenarios_out.mat','file'))

    % Calculated out-of-sample losses (100000 x 100)  
    losses_out=zeros(K,1,Nout);
    Losses_Out=zeros(K,Nout);
    for s = 1:Nout
         z= normrnd(0,1);            %generating a normal random number for idiosyncratic factor
        z_2= normrnd(0,1,50,1);     %y=cz  for generating correlated random number
        y= sqrt_rho*z_2;
        w=zeros(K,1);
        loss=zeros(K,1);
        for i = 1:K
            simga1 = sqrt(1-beta(i)^2);
            w(i)= beta(i)* y(driver(i)) + simga1 * z;     %calculating weight
        
         for j=1:C
                if (CS_Bdry(i,j)<=w(i)) && (w(i)<CS_Bdry(i,j+1))

                    loss(i)=exposure(i,j);
                end
            end
        end
        
     losses_out(:,:,s)=loss;
     Losses_Out(:,s)=losses_out(:,:,s);       % Calculated out-of-sample losses (100000 x 100)
    end
        
    clear Losses_out

    save('scenarios_out', 'Losses_Out')
else
    load('scenarios_out', 'Losses_Out')
end

% Compute portfolio weights
portf_v = sum(value);     % portfolio value
w0{1} = value / portf_v;  % asset weights (portfolio 1)
w0{2} = ones(K, 1) / K;   % asset weights (portfolio 2)
x0{1} = (portf_v ./ value) .* w0{1};  % asset units (portfolio 1)
x0{2} = (portf_v ./ value) .* w0{2};  % asset units (portfolio 2)

% Compute portfolio losses in outsampling senarioes
port1_out=Losses_Out'*w0{1};
port2_out=Losses_Out'*w0{2};
port_out=[sort(port1_out) sort(port2_out)];

% Normal approximation computed from out-of-sample scenarios
mu_out=mean(port_out);
sigma_out= std(port_out);

% Quantile levels (99%, 99.9%)
alphas = [0.99 0.999];

% Compute VaR and CVaR (non-Normal and Normal) for 100000 scenarios
VaRout=zeros(2);
CVaRout=zeros(2);
VaRinN=zeros(2);
CVaRinN=zeros(2);

% Compute VaR and CVaR (non-Normal and Normal) for 100000 scenarios
for(portN = 1:2)
    for(q=1:length(alphas))
        alf = alphas(q);
      VaRout(portN,q)  = port_out(ceil(Nout*alf),portN);
      VaRinN(portN,q)  = (1/(Nout*(1-alf))) * ( (ceil(Nout*alf)-Nout*alf) * VaRout(portN,q) + sum(port_out(ceil(Nout*alf)+1:Nout,portN)));
      CVaRout(portN,q) = mu_out(:,portN)+norminv(alf)+sigma_out(:,portN);
      CVaRinN(portN,q) = mu_out(:,portN)+(normpdf(norminv(alf))/(1-alf))*sigma_out(:,portN);  
 end
end


% Perform 100 trials
N_trials = 100;

for(tr=1:N_trials)

    % Monte Carlo approximation 1
    losses_in1=zeros(K,1,Nin);
    Losses_inMC1 =zeros(K,Nin);
    for s = 1:ceil(Nin/Ns) % systemic scenarios
        z_21= normrnd(0,1,50,1);     %y=cz  for generating correlated random number
        y1= sqrt_rho*z_21;
        for si = 1:Ns % idiosyncratic scenarios for each systemic
                    z1= normrnd(0,1);            %generating a normal random number for idiosyncratic factor
                    w1=zeros(K,1);
                    loss1=zeros(K,1);
                    for i=1:K
                      sigma1=sqrt(1-beta(i)^2);
                      w1(i)=beta(i)*y1(driver(i))+sigma1*z1;     %calculating weight
            
                      for j=1:C
                         if (CS_Bdry(i,j)<=w1(i)) && (w1(i)<CS_Bdry(i,j+1))
                            loss1(i)=exposure(i,j);
                         end
                      end
                    end
        end
       losses_in1(:,:,s)=loss1;    
       Losses_inMC1(:,s)=losses_in1(:,:,s);       % Calculated losses for MC1 approximation (5000 x 100)
    end
    clear losses_in1
    clear loss1
   
    % Monte Carlo approximation 2
    losses_in2=zeros(K,1,Nin);
    Losses_inMC2=zeros(K,Nin);
    for s = 1:Nin % systemic scenarios (1 idiosyncratic scenario for each systemic)
        z2= normrnd(0,1);            %generating a normal random number for idiosyncratic factor
        z_22= normrnd(0,1,50,1);     %y=cz  for generating correlated random number
        y2= sqrt_rho*z_22;
        w2=zeros(K,1);
        loss2=zeros(K,1);
        for i=1:K
            sigma1=sqrt(1-beta(i)^2);
            w2(i)=beta(i)*y2(driver(i))+sigma1*z2;     %calculating weight
            
            for j=1:C
                if (CS_Bdry(i,j)<=w2(i)) && (w2(i)<CS_Bdry(i,j+1))
           
                    loss2(i)=exposure(i,j);
                end
            end
        end
     losses_in2(:,:,s)=loss2;
     Losses_inMC2(:,s)=losses_in2(:,:,s);    % Calculated losses for MC2 approximation (5000 x 100)

    end
    clear losses_in2
    clear loss2
    
    % Compute portfolio losses in MC1 senario
    port1_MC1=Losses_inMC1'*w0{1};
    port2_MC1=Losses_inMC1'*w0{2};
    port_MC1=[sort(port1_MC1) sort(port2_MC1)];
    
    % Compute portfolio losses in MC1 senario
    port1_MC2=Losses_inMC2'*w0{1};
    port2_MC2=Losses_inMC2'*w0{2};
    port_MC2=[sort(port1_MC2) sort(port2_MC2)];
    
    % Compute portfolio mean loss mu_p_MC1 and portfolio standard deviation of losses sigma_p_MC1
      mu_p_MC1=mean(port_MC1);
      sigma_p_MC1= std(port_MC1);
      
    % Compute portfolio mean loss mu_p_MC2 and portfolio standard deviation of losses sigma_p_MC2
      mu_p_MC2=mean(port_MC2);
      sigma_p_MC2= std(port_MC2);
      
    % Compute VaR and CVaR
    VaRinMC1=zeros(2);
    VaRinMC2=zeros(2);
    VaRinN1=zeros(2);
    VaRinN2=zeros(2);
    CVaRinMC1=zeros(2);
    CVaRinMC2=zeros(2);
    CVaRinN1=zeros(2);
    CVaRinN2=zeros(2);
    
    for(portN = 1:2)
        for(q=1:length(alphas))
            alf = alphas(q);
                    
           % Compute VaR and CVaR for the current trial             
           VaRinMC1(portN,q) = port_MC1(ceil(Nin*alf),portN);
            VaRinMC2(portN,q) = port_MC2(ceil(Nin*alf),portN);
            VaRinN1(portN,q) = mu_p_MC1(:,portN)+norminv(alf)+sigma_p_MC1(:,portN);
            VaRinN2(portN,q) = mu_p_MC2(:,portN)+norminv(alf)+sigma_p_MC2(:,portN);
            CVaRinMC1(portN,q) = (1/(Nin*(1-alf))) * ( (ceil(Nin*alf)-Nin*alf) * VaRinMC1(portN,q) + sum(port_MC1(ceil(Nin*alf)+1:Nin,portN)));
            CVaRinMC2(portN,q) = (1/(Nin*(1-alf))) * ( (ceil(Nin*alf)-Nin*alf) * VaRinMC2(portN,q) + sum(port_MC2(ceil(Nin*alf)+1:Nin,portN)));
            CVaRinN1(portN,q) = mu_p_MC1(:,portN)+(normpdf(norminv(alf))/(1-alf))*sigma_p_MC1(:,portN);
            CVaRinN2(portN,q) = mu_p_MC2(:,portN)+(normpdf(norminv(alf))/(1-alf))*sigma_p_MC2(:,portN);
        end
    end


% Display portfolio VaR and CVaR

for(portN = 1:2)
fprintf('\nPortfolio %d:\n\n', portN)    
 for(q=1:length(alphas))
    alf = alphas(q);
    fprintf('Out-of-sample: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f\n', 100*alf, VaRout(portN,q), 100*alf, CVaRout(portN,q))
    fprintf('In-sample MC1: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f\n', 100*alf, mean(VaRinMC1(portN,q)), 100*alf, mean(CVaRinMC1(portN,q)))
    fprintf('In-sample MC2: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f\n', 100*alf, mean(VaRinMC2(portN,q)), 100*alf, mean(CVaRinMC2(portN,q)))
    fprintf(' In-sample No: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f\n', 100*alf, VaRinN(portN,q), 100*alf, CVaRinN(portN,q))
    fprintf(' In-sample N1: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f\n', 100*alf, mean(VaRinN1(portN,q)), 100*alf, mean(CVaRinN1(portN,q)))
    fprintf(' In-sample N2: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f\n\n', 100*alf, mean(VaRinN2(portN,q)), 100*alf, mean(CVaRinN2(portN,q)))
 end
end
VaRinMC1_100(:,:,tr)= VaRinMC1;
VaRinMC2_100(:,:,tr)= VaRinMC2;
VaRinN1_100(:,:,tr)=VaRinN1;
VaRinN2_100(:,:,tr)=VaRinN2;
CVaRinMC1_100(:,:,tr)=CVaRinMC1;
CVaRinMC2_100(:,:,tr)=CVaRinMC2;
CVaRinN1_100(:,:,tr)=CVaRinN1;
CVaRinN2_100(:,:,tr)=CVaRinN2;
end
%mean of 100 trails for each MC senarios
mu_VaRinMC1=mean(VaRinMC1_100,3);
mu_VaRinMC2=mean(VaRinMC2_100,3);
mu_CVaRinMC1=mean(CVaRinMC1_100,3);
mu_CVaRinMC2=mean(CVaRinMC2_100,3);
          %%%%%%%%%%%%%%
st_VaRinMC1=std(VaRinMC1_100,0,3);
st_VaRinMC2=std(VaRinMC1_100,0,3);
st_CVaRinMC1=std(CVaRinMC1_100,0,3);
st_CVaRinMC2=std(CVaRinMC2_100,0,3);


disp('======= Mean and Sigma of 100 trails for each Monte carlo Approximation scenario =======')
disp(' ')
for(portN = 1:2)
 for(q=1:length(alphas))
    alf = alphas(q);
    fprintf('MC1 Scenario in portfolio %4.0f:mean of VaR %4.1f%% = $%6.2f, Sigma of VaR %4.1f%% = $%6.2f\n',portN, 100*alf, mu_VaRinMC1(portN,q), 100*alf, st_VaRinMC1(portN,q));
    fprintf('MC1 Scenario in portfolio %4.0f:mean of CVaR %4.1f%% = $%6.2f, Sigma of CVaR %4.1f%% = $%6.2f\n',portN, 100*alf, mu_CVaRinMC1(portN,q), 100*alf, mu_CVaRinMC1(portN,q))
    fprintf('MC2 Scenario in portfolio %4.0f:mean of VaR %4.1f%% = $%6.2f, Sigma of VaR %4.1f%% = $%6.2f\n',portN, 100*alf, mu_VaRinMC2(portN,q), 100*alf, st_VaRinMC2(portN,q))
    fprintf('MC2 Scenario in portfolio %4.0f:mean of CVaR %4.1f%% = $%6.2f, Sigma of CVaR %4.1f%% = $%6.2f\n',portN, 100*alf, mu_CVaRinMC2(portN,q), 100*alf, st_CVaRinMC2(portN,q))
 end
end

% Plot results
%%%%%%%%%%%%%%%%%%%%%%%
figure(1);
hold on
xlabel('Loss in Portfolio Value') % x-axis label
ylabel('Frequency') % y-axis label
[frequencyCounts, binLocations] = hist(port1_out,1000); bar(binLocations, frequencyCounts); xlim([-2*10^6 8*10^6]);ylim([0 200]);
title('Out-of-Sample Loss distribution for portfolio with one unit invested in each 100 Bonds')
line([VaRout(1,1) VaRout(1,1)], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-');
line([CVaRout(1,1) CVaRout(1,1)], [0 max(frequencyCounts)/2], 'Color', 'b', 'LineWidth', 1, 'LineStyle', '-');
line([VaRinN(1,1) VaRinN(1,1)], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
line([CVaRinN(1,1) CVaRinN(1,1)], [0 max(frequencyCounts)/2], 'Color', 'b', 'LineWidth', 1, 'LineStyle', '--');
legend ('',['VaR(%99)=' num2str(VaRout(1,1))],['CVar(%99)=' num2str(CVaRout(1,1))],['Va_rNormal(%99)=' num2str(VaRinN(1,1))],['CVa_rNormal(%99)=' num2str(VaRinN(1,1))]);
hold on
%%%%%%%%%%%%%%%%%%%%%%%
figure(2);
hold on
xlabel('Loss in Portfolio Value') % x-axis label
ylabel('Frequency') % y-axis label
[frequencyCounts, binLocations] = hist(port1_out,1000); bar(binLocations, frequencyCounts); xlim([-2*10^6 8*10^6]);ylim([0 200]);
title('Out-of-Sample Loss distribution for portfolio with one unit invested in each 100 Bonds')
line([VaRout(1,2) VaRout(1,2)], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-');
line([CVaRout(1,2) CVaRout(1,2)], [0 max(frequencyCounts)/2], 'Color', 'b', 'LineWidth', 1, 'LineStyle', '-');
line([VaRinN(1,2) VaRinN(1,2)], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
line([CVaRinN(1,2) CVaRinN(1,2)], [0 max(frequencyCounts)/2], 'Color', 'b', 'LineWidth', 1, 'LineStyle', '--');
legend ('',['VaR(%99.9)=' num2str(VaRout(1,2))],['CVar(%99.9)=' num2str(CVaRout(1,2))],['Va_rNormal(%99.9)=' num2str(VaRinN(1,2))],['CVa_rNormal(%99.9)=' num2str(VaRinN(1,2))]);
hold on
%%%%%%%%%%%%%%%%%%%%%%%
figure(3);
hold on
xlabel('Loss in Portfolio Value') % x-axis label
ylabel('Frequency') % y-axis label
[frequencyCounts, binLocations] = hist(port2_out,1000); bar(binLocations, frequencyCounts); xlim([-2*10^6 8*10^6]);ylim([0 200]);
title('Out-of-Sample Loss distribution for portfolio with equal value invested in each 100 Bonds')
line([VaRout(2,1) VaRout(2,1)], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-');
line([CVaRout(2,1) CVaRout(2,1)], [0 max(frequencyCounts)/2], 'Color', 'b', 'LineWidth', 1, 'LineStyle', '-');
line([VaRinN(2,1) VaRinN(2,1)], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
line([CVaRinN(2,1) CVaRinN(2,1)], [0 max(frequencyCounts)/2], 'Color', 'b', 'LineWidth', 1, 'LineStyle', '--');
legend ('',['VaR(%99)=' num2str(VaRout(2,1))],['CVar(%99)=' num2str(CVaRout(2,1))],['Va_rNormal(%99)=' num2str(VaRinN(2,1))],['CVa_rNormal(%99)=' num2str(VaRinN(2,1))]);
hold on
%%%%%%%%%%%%%%%%%%%%%%% 
figure(4);
hold on
xlabel('Loss in Portfolio Value') % x-axis label
ylabel('Frequency') % y-axis label
[frequencyCounts, binLocations] = hist(port2_out,1000); bar(binLocations, frequencyCounts); xlim([-2*10^6 8*10^6]);ylim([0 200]);
title('Out-of-Sample Loss distribution for portfolio with equal value invested in each 100 Bonds')
line([VaRout(2,2) VaRout(2,2)], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-');
line([CVaRout(2,2) CVaRout(2,2)], [0 max(frequencyCounts)/2], 'Color', 'b', 'LineWidth', 1, 'LineStyle', '-');
line([VaRinN(2,2) VaRinN(2,2)], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
line([CVaRinN(2,2) CVaRinN(2,2)], [0 max(frequencyCounts)/2], 'Color', 'b', 'LineWidth', 1, 'LineStyle', '--');
legend ('',['VaR(%99)=' num2str(VaRout(2,2))],['CVar(%99.9)=' num2str(CVaRout(2,2))],['Va_rNormal(%99.9)=' num2str(VaRinN(2,2))],['CVa_rNormal(%99.9)=' num2str(VaRinN(2,2))]);
hold on
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(5);
hold on
xlabel('Loss in Portfolio Value') % x-axis label
ylabel('Frequency') % y-axis label
[frequencyCounts, binLocations] = hist(port_MC1,1000); bar(binLocations, frequencyCounts); xlim([-2*10^6 8*10^6]);ylim([0 40]);
title('In-Sample MC1: Loss distribution for portfolio with one unit invested in each 100 Bonds')
line([VaRinMC1(1,1) VaRinMC1(1,1)], [0 1000], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-');
line([CVaRinMC1(1,1) CVaRinMC1(1,1)], [0 1000], 'Color', 'b', 'LineWidth', 1, 'LineStyle', '-');
line([VaRinN1(1,1) VaRinN1(1,1)], [0 1000], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
line([CVaRinN1(1,1) CVaRinN1(1,1)], [0 1000], 'Color', 'b', 'LineWidth', 1, 'LineStyle', '--');
legend ('','',['VaR(%99)=' num2str(VaRinMC1(1,1))],['CVar(%99)=' num2str(CVaRinMC1(1,1))],['Va_rNormal(%99)=' num2str(VaRinN1(1,1))],['CVa_rNormal(%99)=' num2str(CVaRinN1(1,1))]);
hold on

%%%%%%%%%%%%%%%%%%%%%%% 
figure(6);
hold on
xlabel('Loss in Portfolio Value') % x-axis label
ylabel('Frequency') % y-axis label
[frequencyCounts, binLocations] = hist(port_MC1,1000); bar(binLocations, frequencyCounts); xlim([-2*10^6 8*10^6]);ylim([0 40]);
title('In-Sample MC1: Loss distribution for portfolio with one unit invested in each 100 Bonds')
line([VaRinMC1(1,2) VaRinMC1(1,2)], [0 1000], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-');
line([CVaRinMC1(1,2) CVaRinMC1(1,2)], [0 1000], 'Color', 'b', 'LineWidth', 1, 'LineStyle', '-');
line([VaRinN1(1,2) VaRinN1(1,2)], [0 1000], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
line([CVaRinN1(1,2) CVaRinN1(1,2)], [0 1000], 'Color', 'b', 'LineWidth', 1, 'LineStyle', '--');
legend ('','',['VaR(%99.9)=' num2str(VaRinMC1(1,2))],['CVar(%99.9)=' num2str(CVaRinMC1(1,2))],['Va_rNormal(%99.9)=' num2str(VaRinN1(1,2))],['CVa_rNormal(%99.9)=' num2str(CVaRinN1(1,2))]);
hold on
%%%%%%%%%%%%%%%%%%%%%%% 
figure(7);
hold on
xlabel('Loss in Portfolio Value') % x-axis label
ylabel('Frequency') % y-axis label
[frequencyCounts, binLocations] = hist(port_MC1,1000); bar(binLocations, frequencyCounts); xlim([-2*10^6 8*10^6]);ylim([0 40]);
title('In-Sample MC1: Loss distribution for portfolio with equal value invested in each 100 Bonds')
line([VaRinMC1(2,1) VaRinMC1(2,1)], [0 1000], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-');
line([CVaRinMC1(2,1) CVaRinMC1(2,1)], [0 1000], 'Color', 'b', 'LineWidth', 1, 'LineStyle', '-');
line([VaRinN1(2,1) VaRinN1(2,1)], [0 1000], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
line([CVaRinN1(2,1) CVaRinN1(2,1)], [0 1000], 'Color', 'b', 'LineWidth', 1, 'LineStyle', '--');
legend ('','',['VaR(%99)=' num2str(VaRinMC1(2,1))],['CVar(%99)=' num2str(CVaRinMC1(2,1))],['Va_rNormal(%99)=' num2str(VaRinN1(2,1))],['CVa-rNormal(%99)=' num2str(CVaRinN1(2,1))]);
hold on
%%%%%%%%%%%%%%%%%%%%%%% 
figure(8);
hold on
xlabel('Loss in Portfolio Value') % x-axis label
ylabel('Frequency') % y-axis label
[frequencyCounts, binLocations] = hist(port_MC1,1000); bar(binLocations, frequencyCounts); xlim([-2*10^6 8*10^6]);ylim([0 40]);
title('In-Sample MC1: Loss distribution for portfolio with equal value invested in each 100 Bonds')
line([VaRinMC1(2,2) VaRinMC1(2,2)], [0 1000], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-');
line([CVaRinMC1(2,2) CVaRinMC1(2,2)], [0 1000], 'Color', 'b', 'LineWidth', 1, 'LineStyle', '-');
line([VaRinN1(2,2) VaRinN1(2,2)], [0 1000], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
line([CVaRinN1(2,2) CVaRinN1(2,2)], [0 1000], 'Color', 'b', 'LineWidth', 1, 'LineStyle', '--');
legend ('','',['VaR(%99.9)=' num2str(VaRinMC1(2,2))],['CVar(%99.9)=' num2str(CVaRinMC1(2,2))],['Va-rNormal(%99.9)=' num2str(VaRinN1(2,2))],['CVa-rNormal(%99.9)=' num2str(CVaRinN1(2,2))]);
hold on
%%%%%%%%%%%%%%%%%%%%%%% 
figure(9);
hold on
xlabel('Loss in Portfolio Value') % x-axis label
ylabel('Frequency') % y-axis label
[frequencyCounts, binLocations] = hist(port_MC2,1000); bar(binLocations, frequencyCounts); xlim([-2*10^6 8*10^6]);ylim([0 40]);
title('In-Sample MC2: Loss distribution for portfolio with one unit invested in each 100 Bonds')
line([VaRinMC2(1,1) VaRinMC2(1,1)], [0 1000], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-');
line([CVaRinMC2(1,1) CVaRinMC2(1,1)], [0 1000], 'Color', 'b', 'LineWidth', 1, 'LineStyle', '-');
line([VaRinN2(1,1) VaRinN2(1,1)], [0 1000], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
line([CVaRinN2(1,1) CVaRinN2(1,1)], [0 1000], 'Color', 'b', 'LineWidth', 1, 'LineStyle', '--');
legend ('','',['VaR(%99)=' num2str(VaRinMC2(1,1))],['CVar(%99)=' num2str(CVaRinMC2(1,1))],['Va_rNormal(%99)=' num2str(VaRinN2(1,1))],['CVa_rNormal(%99)=' num2str(CVaRinN2(1,1))]);
hold on
%%%%%%%%%%%%%%%%%%%%%%% 
figure(10);
hold on
xlabel('Loss in Portfolio Value') % x-axis label
ylabel('Frequency') % y-axis label
[frequencyCounts, binLocations] = hist(port_MC2,1000); bar(binLocations, frequencyCounts); xlim([-2*10^6 8*10^6]);ylim([0 40]);
title('In-Sample MC2: Loss distribution for portfolio with one unit invested in each 100 Bonds')
line([VaRinMC2(1,2) VaRinMC2(1,2)], [0 1000], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-');
line([CVaRinMC2(1,2) CVaRinMC2(1,2)], [0 1000], 'Color', 'b', 'LineWidth', 1, 'LineStyle', '-');
line([VaRinN2(1,2) VaRinN2(1,2)], [0 1000], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
line([CVaRinN2(1,2) CVaRinN2(1,2)], [0 1000], 'Color', 'b', 'LineWidth', 1, 'LineStyle', '--');
legend ('','',['VaR(%99.9)=' num2str(VaRinMC2(1,2))],['CVar(%99.9)=' num2str(CVaRinMC2(1,2))],['Va_rNormal(%99.9)=' num2str(VaRinN2(1,2))],['CVa_rNormal(%99.9)=' num2str(CVaRinN2(1,2))]);
hold on
%%%%%%%%%%%%%%%%%%%%%%% 
figure(11);
hold on
xlabel('Loss in Portfolio Value') % x-axis label
ylabel('Frequency') % y-axis label
[frequencyCounts, binLocations] = hist(port_MC2,1000); bar(binLocations, frequencyCounts); xlim([-2*10^6 8*10^6]);ylim([0 40]);
title('In-Sample MC2: Loss distribution for portfolio with equal value invested in each 100 Bonds')
line([VaRinMC2(2,1) VaRinMC2(2,1)], [0 1000], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-');
line([CVaRinMC2(2,1) CVaRinMC2(2,1)], [0 1000], 'Color', 'b', 'LineWidth', 1, 'LineStyle', '-');
line([VaRinN2(2,1) VaRinN2(2,1)], [0 1000], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
line([CVaRinN2(2,1) CVaRinN2(2,1)], [0 1000], 'Color', 'b', 'LineWidth', 1, 'LineStyle', '--');
legend ('','',['VaR(%99)=' num2str(VaRinMC2(2,1))],['CVar(%99)=' num2str(CVaRinMC2(2,1))],['Va_rNormal(%99)=' num2str(VaRinN2(2,1))],['CVa-rNormal(%99)=' num2str(CVaRinN2(2,1))]);
hold on
%%%%%%%%%%%%%%%%%%%%%%% 
figure(12);
hold on
xlabel('Loss in Portfolio Value') % x-axis label
ylabel('Frequency') % y-axis label
[frequencyCounts, binLocations] = hist(port_MC2,1000); bar(binLocations, frequencyCounts); xlim([-2*10^6 8*10^6]);ylim([0 40]);
title('In-Sample MC2: Loss distribution for portfolio with equal value invested in each 100 Bonds')
line([VaRinMC2(2,2) VaRinMC2(2,2)], [0 1000], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-');
line([CVaRinMC2(2,2) CVaRinMC2(2,2)], [0 1000], 'Color', 'b', 'LineWidth', 1, 'LineStyle', '-');
line([VaRinN2(2,2) VaRinN2(2,2)], [0 1000], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
line([CVaRinN2(2,2) CVaRinN2(2,2)], [0 1000], 'Color', 'b', 'LineWidth', 1, 'LineStyle', '--');
legend ('','',['VaR(%99.9)=' num2str(VaRinMC2(2,2))],['CVar(%99.9)=' num2str(CVaRinMC2(2,2))],['Va-rNormal(%99.9)=' num2str(VaRinN2(2,2))],['CVa-rNormal(%99.9)=' num2str(CVaRinN2(2,2))]);
hold on
%%%%%%%%%%%%%%%%%%%%%%% 