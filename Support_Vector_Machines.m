%%%%%%%%%%%%%%%%%%% SVM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;clear all;clc;
%%%%%%%%%%%%%%%%%%% �������� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 100;                % ��������С
center1 = [1,1];        % ��һ����������
center2 = [6,6];        % �ڶ�����������
%���Կɷ����ݣ�center2 = [6,6]�����Բ��ɷ����ݣ���Ϊcenter2 = [3,3]
X = zeros(2*n,2);       % 2n * 2�����ݾ���ÿһ�б�ʾһ�����ݵ㣬��һ�б�ʾx�����꣬�ڶ��б�ʾy������
Y = zeros(2*n,1);       % ����ǩ
X(1:n,:) = ones(n,1)*center1 + randn(n,2);
X(n+1:2*n,:) = ones(n,1)*center2 + randn(n,2);       %����X��ǰn�б�ʾ���1�����ݣ���n�б�ʾ���2������
Y(1:n) = 1; 
Y(n+1:2*n) = -1;        % ��һ�����ݱ�ǩΪ1���ڶ���Ϊ-1 

figure(1)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(1:n,1),X(1:n,2),'go','LineWidth',1,'MarkerSize',10);            % ����һ�����ݵ�
hold on;
plot(X(n+1:2*n,1),X(n+1:2*n,2),'b*','LineWidth',1,'MarkerSize',10);    % ���ڶ������ݵ�
hold on;
xlabel('x axis');
ylabel('y axis');
legend('class 1','class 2');

%%%%%%%%%%%%%%%%%%  SVMģ��   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%  ѧ��ʵ��,���SVM�Ĳ���(w,b)     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w = zeros(2,1);
b = zeros(1);               % SVM: y = x*w + b
alpha = zeros(2*n,1);       % ��ż�������

%%%%%%%% %%%%%%%% ʹ�����������������շ�ѵ��ģ��
C = 20000;%��������ϵ��C
lambda = 0;%�������������㷨�е�������ϵ��1
beta = 0.01;%�������������㷨�е�������ϵ��2
eta = 0.0001;%�������������㷨�еĸ��²���

%�������������պ���
X_h = zeros(2*n,2);%��X^
for i=1:2*n
    X_h(i,1) = X(i,1)*Y(i);
    X_h(i,2) = X(i,2)*Y(i);
end
L1 = 0.5*alpha'*(X_h*X_h')*alpha-ones(1,2*n)*alpha+lambda*Y'*alpha+0.5*beta*(Y'*alpha)^2;%���������������պ���
alpha_h = zeros(10000,1);
A=zeros(2*n,1);
XX=zeros(10000,1);
k=1;
while(k<=10000)%�ڴ�Լ������1w�ε������㷨����
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
%��alpha�ó�w��ֵ
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
%%%%%%%%%%%%%%%%  ����������ͼ  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% ������ x*w + b =0 ��ͼ�� %%%%%%%%%%%%%%%%%%%%%%%%%%%%

x1 = -2 : 0.00001 : 7;
y1 = ( -b * ones(1,length(x1)) - w(1) * x1 )/w(2);         % �������
                                                           % x1Ϊ���������ᣬy1Ϊ����
y2 = ( ones(1,length(x1)) - b * ones(1,length(x1)) - w(1) * x1 )/w(2);
y3 = ( -ones(1,length(x1)) - b * ones(1,length(x1)) - w(1) * x1 )/w(2);  %��������߽�

figure(4)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(1:n,1),X(1:n,2),'go','LineWidth',1,'MarkerSize',10);            % ����һ�����ݵ�
hold on;
plot(X(n+1:2*n,1),X(n+1:2*n,2),'b*','LineWidth',1,'MarkerSize',10);    % ���ڶ������ݵ�
hold on;
plot( x1,y1,'k','LineWidth',1,'MarkerSize',10);                         % ���������
hold on;
plot( x1,y2,'k-.','LineWidth',1,'MarkerSize',10);                         % ���ּ���߽�
hold on;
plot( x1,y3,'k-.','LineWidth',1,'MarkerSize',10);                         % ���ּ���߽�
hold on;
plot(X(alpha>0,1),X(alpha>0,2),'rs','LineWidth',1,'MarkerSize',10);    % ��֧������
hold on;
plot(X(alpha<C&alpha>0,1),X(alpha<C&alpha>0,2),'rs','MarkerFaceColor','r','LineWidth',1,'MarkerSize',10);    % ������߽��ϵ�֧������
hold on;
xlabel('x axis');
ylabel('y axis');
set(gca,'Fontsize',10)
legend('class 1','class 2','classification surface','boundary','boundary','support vectors','support vectors on boundary');
