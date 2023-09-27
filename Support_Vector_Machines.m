%%%%%%%%%%%%%%%%%%% SVM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;clear all;clc;
%%%%%%%%%%%%%%%%%%% 数据生成 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 100;                % 样本量大小
center1 = [1,1];        % 第一类数据中心
center2 = [6,6];        % 第二类数据中心
%线性可分数据：center2 = [6,6]；线性不可分数据，改为center2 = [3,3]
X = zeros(2*n,2);       % 2n * 2的数据矩阵，每一行表示一个数据点，第一列表示x轴坐标，第二列表示y轴坐标
Y = zeros(2*n,1);       % 类别标签
X(1:n,:) = ones(n,1)*center1 + randn(n,2);
X(n+1:2*n,:) = ones(n,1)*center2 + randn(n,2);       %矩阵X的前n行表示类别1中数据，后n行表示类别2中数据
Y(1:n) = 1; 
Y(n+1:2*n) = -1;        % 第一类数据标签为1，第二类为-1 

figure(1)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(1:n,1),X(1:n,2),'go','LineWidth',1,'MarkerSize',10);            % 画第一类数据点
hold on;
plot(X(n+1:2*n,1),X(n+1:2*n,2),'b*','LineWidth',1,'MarkerSize',10);    % 画第二类数据点
hold on;
xlabel('x axis');
ylabel('y axis');
legend('class 1','class 2');

%%%%%%%%%%%%%%%%%%  SVM模型   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%  学生实现,求出SVM的参数(w,b)     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w = zeros(2,1);
b = zeros(1);               % SVM: y = x*w + b
alpha = zeros(2*n,1);       % 对偶问题变量

%%%%%%%% %%%%%%%% 使用线性增广拉格朗日法训练模型
C = 20000;%定义软间隔系数C
lambda = 0;%增广拉格朗日算法中的增广项系数1
beta = 0.01;%增广拉格朗日算法中的增广项系数2
eta = 0.0001;%增广拉格朗日算法中的更新步长

%求增广拉格朗日函数
X_h = zeros(2*n,2);%求X^
for i=1:2*n
    X_h(i,1) = X(i,1)*Y(i);
    X_h(i,2) = X(i,2)*Y(i);
end
L1 = 0.5*alpha'*(X_h*X_h')*alpha-ones(1,2*n)*alpha+lambda*Y'*alpha+0.5*beta*(Y'*alpha)^2;%定义增广拉格朗日函数
alpha_h = zeros(10000,1);
A=zeros(2*n,1);
XX=zeros(10000,1);
k=1;
while(k<=10000)%在大约进行了1w次迭代后算法收敛
    A(k)=(Y'*alpha)^2;
    XX(k)=k;
    alpha_h = alpha - eta*(X_h*X_h'*alpha-1+lambda*Y+beta*Y'*alpha*Y);%alpha_h
    for i=1:2*n
        if(alpha_h(i)<0)
            alpha(i)=0;
        elseif(alpha_h(i)>=0 && alpha(i)<=C)
                alpha(i)=alpha_h(i);
        elseif(alpha_h(i)>C)
                alpha(i)=C;
        end
    end
    lambda = lambda+beta*Y'*alpha;
    k=k+1;
end
figure;
plot(XX,A,'-o');
%由alpha得出w的值
for i=1:2*n
    w(1)=w(1)+alpha(i)*Y(i)*X(i,1);
    w(2)=w(1)+alpha(i)*Y(i)*X(i,2);
end
disp(w);

min_of_wx = w(1)*X(1,1)+w(2)*X(1,2);
for i=2:n
    temp1=w(1)*X(i,1)+w(2)*X(i,2);
    if(temp1<min_of_wx)
        min_of_wx=temp1;
    end
end

max_of_wx = w(1)*X(101,1)+w(2)*X(101,2);
for i=101:2*n
    temp2=w(1)*X(i,1)+w(2)*X(i,2);
    if(temp2>max_of_wx)
        max_of_wx=temp2;
    end
end
b=-(max_of_wx+min_of_wx)/2;
%%%%%%%%%%%%%%%%  分类器可视图  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% 即画出 x*w + b =0 的图像 %%%%%%%%%%%%%%%%%%%%%%%%%%%%

x1 = -2 : 0.00001 : 7;
y1 = ( -b * ones(1,length(x1)) - w(1) * x1 )/w(2);         % 分类界面
                                                           % x1为分类界面横轴，y1为纵轴
y2 = ( ones(1,length(x1)) - b * ones(1,length(x1)) - w(1) * x1 )/w(2);
y3 = ( -ones(1,length(x1)) - b * ones(1,length(x1)) - w(1) * x1 )/w(2);  %画出间隔边界

figure(4)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(1:n,1),X(1:n,2),'go','LineWidth',1,'MarkerSize',10);            % 画第一类数据点
hold on;
plot(X(n+1:2*n,1),X(n+1:2*n,2),'b*','LineWidth',1,'MarkerSize',10);    % 画第二类数据点
hold on;
plot( x1,y1,'k','LineWidth',1,'MarkerSize',10);                         % 画分类界面
hold on;
plot( x1,y2,'k-.','LineWidth',1,'MarkerSize',10);                         % 画分间隔边界
hold on;
plot( x1,y3,'k-.','LineWidth',1,'MarkerSize',10);                         % 画分间隔边界
hold on;
plot(X(alpha>0,1),X(alpha>0,2),'rs','LineWidth',1,'MarkerSize',10);    % 画支持向量
hold on;
plot(X(alpha<C&alpha>0,1),X(alpha<C&alpha>0,2),'rs','MarkerFaceColor','r','LineWidth',1,'MarkerSize',10);    % 画间隔边界上的支持向量
hold on;
xlabel('x axis');
ylabel('y axis');
set(gca,'Fontsize',10)
legend('class 1','class 2','classification surface','boundary','boundary','support vectors','support vectors on boundary');
