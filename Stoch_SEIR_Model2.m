
beta = 2.34;
delta1 = 0.5;
gamma1 = 1/14;
theta = 1/4;
kappa = 1/5;
mu1 = 0.002;
mu2 = 0.015;
p_s = 0.8;
gamma3 = 1/14;
N = 100000;
I0 = 10;
E0 = 0;
A0 = 0;
Q0 = 0;
MaxTime = 150;
 
[t2,S2,E2,I2,A2,Q2,R2] = Flu_SEIAQR_stoch(beta,delta1,gamma1,theta,kappa,mu1,mu2,p_s,gamma3,I0,E0,A0,Q0,N,MaxTime);

figure(2)
%plot(t',S, 'DisplayName','S Susceptible');
hold on
plot(t2',E2, 'DisplayName','E Exposed');
plot(t2',I2, 'DisplayName','I_S Infected Symptomatic');
plot(t2',A2, 'DisplayName','I_A Infected Asymptomatic');
plot(t2',Q2, 'DisplayName','Q Quarantined');
%plot(t2',R2, 'DisplayName','R Removed/Recovered');
legend


function [t,S,E,I,A,Q,R] = Flu_SEIAQR_stoch(beta,delta1,gamma1,theta,kappa,mu1,mu2,p_s,gamma3,I0,E0,A0,Q0,N,MaxTime)

S0 = N-I0-E0-A0-Q0;
S=S0; I=I0; E=E0; A=A0; Q=Q0; R=N-S-E-I-A-Q;
% The main iteration
[t, pop] = Stoch_Iteration([0 MaxTime],[S0 E0 I0 A0 Q0 R],[beta delta1 gamma1 theta kappa mu1 p_s gamma3 mu2 N]);
T=t;
S=pop(:,1); E=pop(:,2); I=pop(:,3); A=pop(:,4); Q=pop(:,5);
R=pop(:,6);
numRows = size(T);
end



% Do the iterations using the full evnt driven stochastic methodology
% relatively general version of Gillespie's Direct Algorithm
function [T,P]=Stoch_Iteration(Time,Initial,Parameters)
S=Initial(1); E=Initial(2); I=Initial(3); A=Initial(4); Q=Initial(5);
R=Initial(6);
T=0; P(1,:)=[S E I A Q R];
old=[S E I A Q R];
loop=1;
while (T(loop)<Time(2))
 [step,new]=Iterate(old,Parameters);
 loop=loop+1;
 T(loop)=T(loop-1)+step;
 P(loop,:)=new; 
 old=new;
%  loop=loop+1;
%  T(loop)=T(loop-1);
%  P(loop,:)=new; old=new;
 if loop>=length(T)
    T(loop*2)=0;
    P(loop*2,:)=0;
 end
end
T=T(1:loop); P=P(1:loop,:);
end
% Do the actual iteration step
function[step, new_value]=Iterate(old, Parameters)
beta=Parameters(1); delta1=Parameters(2); gamma1=Parameters(3);
theta=Parameters(4); kappa=Parameters(5); mu1=Parameters(6);
p_S=Parameters(7); gamma3=Parameters(8); mu2=Parameters(9); N=Parameters(10);
S=old(1); E=old(2); I=old(3); A=old(4); Q=old(5); R=old(6);
Rate(1) = beta*S*(I+delta1*A)/N; Change(1,:)=[-1 +1 0 0 0 0];
Rate(2) = p_S*kappa*E; Change(2,:)=[0 -1 +1 0 0 0];
Rate(3) = (1-p_S)*kappa*E; Change(3,:)=[0 -1 0 +1 0 0];
Rate(4) = gamma1*I; Change(4,:)=[0 0 -1 0 0 +1];
Rate(5) = theta*I; Change(5,:)=[0 0 -1 0 +1 0];
Rate(6) = mu1*I; Change(6,:)=[0 0 -1 0 0 0];
Rate(7) = gamma1*A; Change(7,:)=[0 0 0 -1 0 +1];
Rate(8) = gamma3*Q; Change(8,:)=[0 0 0 0 -1 +1];
Rate(9) = mu2*Q; Change(9,:)=[0 0 0 0 -1 +1];

R1=rand(1,1);
R2=rand(1,1);
if(sum(Rate) > 0)
 step = -log(R2)/(sum(Rate));
else
 return
end
% find which event to do
m=min(find(cumsum(Rate)>=R1*sum(Rate)));
new_value=old+Change(m,:);
end
