clear
clc


load  hw4_nnet_train.dat;
load  hw4_nnet_test.dat;
tx = hw4_nnet_train(:,1:2);
ty = hw4_nnet_train(:,3);
tx = tx';
ty = ty';
vx = hw4_nnet_test(:,1:2);
vy = hw4_nnet_test(:,3);
vx = vx';
vy = vy';

lr = [0.001 0.01 0.1 1 10];

itr = 500;
dnum = size(vy);

[I N] = size(tx);
[O,N] = size(ty);

M = [1,6,11,16,21];
m = 3;
m14 = [8 3];

for k=1:5
net = newff(tx,ty,m,{'tansig' 'tansig'},'traingdm','learngd','sse');
%net = newff(tx,ty,m,{'tansig' 'tansig' 'tansig'},'traingdm');
net.trainParam.lr = lr(k);
net.trainParam.epochs = 1000;
net.trainParam.goal = 1e-20;
net.trainParam.min_grad = 1e-20;
net.divideFcn = '';
net.trainParam.max_fail = 50;


cof = 0.1;
IW = cof*randi([-1000,1000],m,I)/10^3;
LW = cof*randi([-1000,1000],O,m)/10^3;
% IW = cof*randi([-1000,1000],8,I)/10^3;
% IW2 = cof*randi([-1000,1000],3,8)/10^3;
% LW = cof*randi([-1000,1000],O,3)/10^3;
net.iw{1,1} = IW;
net.lw{2,1} = LW;
%net.lw{2,1} = IW2;
%net.lw{3,2} = LW;

err_t = 0;

for j = 1:itr

%train
net_r = train(net,tx,ty);
BPoutput=sim(net_r,vx); 

%Validation
    err = 0;
    for i=1:dnum(2)
        if BPoutput(i)*vy(i) < 0  
           err = err+1;    
        end
    end
    err_t = err_t + err/dnum(2);
end

err_t = err_t/itr;
fprintf(1,'Learning rate = %d\n',lr(k));
fprintf(1,'Avg err = %d\n',err_t);



end
    



