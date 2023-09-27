close all;
clc;
clear all;

%%%%%%%%%%%%%%%%%% 数据生成 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 100;                % 样本量大小
center1 = [1,1];        % 第一类数据中心
center2 = [3,4];        % 第二类数据中心
X = zeros(2*n,2);       % 2n * 2的数据矩阵，每一行表示一个数据点，第一列表示x轴坐标，第二列表示y轴坐标
Y = zeros(2*n,1);       % 类别标签
X(1:n,:) = ones(n,1)*center1 + randn(n,2);           %生成数据：中心点+高斯噪声
X(n+1:2*n,:) = ones(n,1)*center2 + randn(n,2);       %矩阵X的前n行表示类别1中数据，后n行表示类别2中数据
Y(1:n) = 1; 
Y(n+1:2*n) = -1;        % 第一类数据标签为1，第二类为-1 

figure(1)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(1:n,1),X(1:n,2),'ro','LineWidth',1,'MarkerSize',10);            % 画第一类数据点
hold on;
plot(X(n+1:2*n,1),X(n+1:2*n,2),'b*','LineWidth',1,'MarkerSize',10);    % 画第二类数据点
hold on;
xlabel('x axis');
ylabel('y axis');
legend('class 1','class 2');

%%%%%%%%%%%%%%%%%  感知机模型   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%  学生实现,求出感知机模型的参数(w,b)     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w = zeros(2,1);
b = zeros(1);               % 感知机模型 y = x*w + b


%%%%%%% 使用梯度下降法训练模型；即最小化f(w,b) = 1/2 * || X*w + ones(2*n,1)*b - Y ||^2
% 计算损失函数的梯度
%定义精确度
E=0.05;
%定义学习率
s=0.1;
%是否收敛至E
is_sl=0;
%迭代次数计数器
times=0;
while(is_sl==0)
    if(times==100)
        break;
    end
% 对w1求梯度
temp_1=0;
for i=1:200
    x_t=[X(i,1),X(i,2)];
    temp_1=temp_1+(x_t*w + b - Y(i))*X(i,1);
end
t_w_1=1/(2*n)*temp_1;
% 对w2求梯度
temp_2=0;
for i=1:200
    x_t=[X(i,1),X(i,2)];
    temp_2=temp_2+(x_t*w + b - Y(i))*X(i,2);
end
t_w_2=1/(2*n)*temp_2;
t_w=[t_w_1,t_w_2]';%w的梯度向量
%对b求梯度
temp_b=0;
for i=1:200
    x_t=[X(i,1),X(i,2)];
    temp_b=temp_b+(x_t*w + b - Y(i));
end
t_b=1/(2*n)*temp_b;
if(abs(t_w(1))<E&&abs(t_w(2))<E&&abs(t_b)<E)
    is_sl=1;
else
    w(1)=w(1)-t_w_1*s;
    w(2)=w(2)-t_w_2*s;
    b=b-t_b*s;
end
times=times+1;
end
%%%%%%%%%%%%%%%  分类器可视图  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% 即画出 x*w + b =0 的图像 %%%%%%%%%%%%%%%%%%%%%%%%%%%%

x1 = -2 : 0.00001 : 7;
y1 = ( -b * ones(1,length(x1)) - w(1) * x1 )/w(2);         % 分类界面,length()表示向量长度
                                                           %x1为分类界面横轴，y1为纵轴
figure(3)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(1:n,1),X(1:n,2),'ro','LineWidth',1,'MarkerSize',10);            % 画第一类数据点
hold on;
plot(X(n+1:2*n,1),X(n+1:2*n,2),'b*','LineWidth',1,'MarkerSize',10);    % 画第二类数据点
hold on;
plot( x1,y1,'k','LineWidth',1,'MarkerSize',10);                         % 画分类界面
xlabel('x axis');
ylabel('y axis');
legend('class 1','class 2','classification surface');

%%%%%%%%%%%%%%%%%%  测试  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% 生成测试数据:与训练数据同分布 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m = 10;                % 测试样本量大小
Xt = zeros(2*m,2);       % 2n * 2的数据矩阵，每一行表示一个数据点，第一列表示x轴坐标，第二列表示y轴坐标
Yt = zeros(2*m,1);       % 类别标签
Xt(1:m,:) = ones(m,1)*center1 + randn(m,2);
Xt(m+1:2*m,:) = ones(m,1)*center2 + randn(m,2);       %矩阵X的前n行表示类别1中数据，后n行表示类别2中数据
Yt(1:m) = 1; 
Yt(m+1:2*m) = -1;        % 第一类数据标签为1，第二类为-1 

figure(4)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(1:n,1),X(1:n,2),'ro','LineWidth',1,'MarkerSize',10);              % 画第一类数据点
hold on;
plot(X(n+1:2*n,1),X(n+1:2*n,2),'b*','LineWidth',1,'MarkerSize',10);      % 画第二类数据点
hold on;
plot(Xt(1:m,1),Xt(1:m,2),'go','LineWidth',1,'MarkerSize',10);            % 画第一类数据点
hold on;
plot(Xt(m+1:2*m,1),Xt(m+1:2*m,2),'g*','LineWidth',1,'MarkerSize',10);    % 画第二类数据点
hold on;
plot( x1,y1,'k','LineWidth',1,'MarkerSize',10);                           % 画分类界面
xlabel('x axis');
xlabel('x axis');
ylabel('y axis');
legend('class 1: train','class 2: train','class 1: test','class 2: test','classification surface');

%%%%%%%%%%%%%%%%%  学生实现     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%  给出模型的预测输出，并与测试数据的真实输出比较，计算错误率     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sum_of_fault=0;%标记分类错误的个数；
for i=1:10
    x_tt=[Xt(i,1),Xt(i,2)];
    if(x_tt*w+b<0)
        sum_of_fault=sum_of_fault+1;
    end
end
for i=11:20
    x_tt=[Xt(i,1),Xt(i,2)];
    if(x_tt*w+b>=0)
        sum_of_fault=sum_of_fault+1;
    end
end
sum_of_right=20-sum_of_fault;
rate_of_right=sum_of_right/20;
str=['分类正确率:'];
disp(str);
disp(rate_of_right);