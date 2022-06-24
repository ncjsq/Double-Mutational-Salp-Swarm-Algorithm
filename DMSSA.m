%__________________________________________________________________________________%
% Double Mutational Salp Swarm Algorithm(DMSSA) source codes demo version 1.0      %
%            By "Chao Lin" Dated: 24-06-2022                                       %
%__________________________________________________________________________________%
function [FoodPosition,Convergence_curve]=DMSSA(N,MaxFEs,lb,ub,dim,fobj)
tic
lb=ones(1,dim).*lb;
ub=ones(1,dim).*ub;

Convergence_curve = [];
FEs=0;
SalpPositions=initialization(N,dim,ub,lb);
SalpFitness=ones(N,1)*inf;
FoodPosition=zeros(1,dim);
FoodFitness=inf;

%calculate the fitness of initial salps
for i=1:size(SalpPositions,1)
    SalpFitness(i)=fobj(SalpPositions(i,:));
    FEs=FEs+1;
end


[FoodFitness,min_salp_fitness] = min(SalpFitness);
FoodPosition = SalpPositions(min_salp_fitness(1),:);


t=2; % start from the second iteration since the first iteration was dedicated to calculating the fitness of salps


CRm=0.5;SCR=[];
Fm=0.5;SF=[];
%Main loop
while FEs<MaxFEs+2
    SalpPre=SalpPositions;
    SalpFitPre=SalpFitness;
    CR=CRm+0.1*randn();
    CR=min(1,max(0,CR));
    pa=CR;
    new_nest=empty_nests(SalpPositions,lb,ub,pa) ;
    SalpPositions(1:N/2,:)=new_nest(1:N/2,:);
    A=randperm(N);
    A(A<=N/2)=[];
    r1=A(1);
    r2=A(2);
    F = Fm + 0.1 * tan(pi * (rand() - 0.5));
    F = min(1, F);
    while F<=0
        F = Fm + 0.1 * tan(pi * (rand() - 0.5));
        F = min(1, F);
    end
    for i=N/2+1:N
        for j=1:dim
            if rand()<CR
                SalpPositions(i,j)=SalpPositions(i,j)+(FoodPosition(j)-SalpPre(r1,j))*F+(SalpPositions(r1,j)-SalpPre(r2,j))*F;
            else
                SalpPositions(i,j)=(SalpPositions(i,j)+SalpPositions(i-1,j))/2;
            end
        end
    end
    for i=1:size(SalpPositions,1)
        SalpPositions(i,:)=max(SalpPositions(i,:),lb);
        SalpPositions(i,:)=min(SalpPositions(i,:),ub); 
        SalpFitness(i)=fobj(SalpPositions(i,:));
        FEs=FEs+1;
        if SalpFitness(i)<FoodFitness
            FoodPosition=SalpPositions(i,:);
            FoodFitness=SalpFitness(i);
        end
        if SalpFitness(i)>SalpFitPre(i)
            SalpPositions(i,:)=SalpPre(i,:);
            SalpFitness(i)=SalpFitPre(i);
        else
            if i>N/2
                SCR=[SCR;CR];
                SF=[SF;F];
            end
        end
    end
    if size(SCR,1)>N/2
        rndpos=randperm(size(SCR,1));
        rndpos=rndpos(1:N/2);
        SCR=SCR(rndpos,:);
    end
    if size(SF,1)>N/2
        rndpos=randperm(size(SF,1));
        rndpos=rndpos(1:N/2);
        SF=SF(rndpos,:);
    end
    c=1/10;
    CRm=(1-c)*CRm+c*mean(SCR);
    Fm = (1 - c) * Fm + c * sum(SF .^ 2) / sum(SF);
    Convergence_curve(t-1)=FoodFitness;
    t = t + 1;
end
toc
end


function new_nest=empty_nests(nest,Lb,Ub,pa)
% A fraction of worse nests are discovered with a probability pa
n=size(nest,1);
% Discovered or not -- a status vector
K=rand(size(nest))>pa;

% In the real world, if a cuckoo's egg is very similar to a host's eggs, then 
% this cuckoo's egg is less likely to be discovered, thus the fitness should 
% be related to the difference in solutions.  Therefore, it is a good idea 
% to do a random walk in a biased way with some random step sizes.  
%%% New solution by biased/selective random walks
stepsize=rand*(nest(randperm(n),:)-nest(randperm(n),:));
new_nest=nest+stepsize.*K;
for j=1:size(new_nest,1)
    s=new_nest(j,:);
  new_nest(j,:)=simplebounds(s,Lb,Ub);  
end
end
function s=simplebounds(s,Lb,Ub)
  % Apply the lower bound
  ns_tmp=s;
  I=ns_tmp<Lb;
  ns_tmp(I)=Lb(I);
  
  % Apply the upper bounds 
  J=ns_tmp>Ub;
  ns_tmp(J)=Ub(J);
  % Update this new move 
  s=ns_tmp;
end