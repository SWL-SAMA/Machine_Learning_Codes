close all;
clc;
clear all;

%%%%%%%%%%%%%%%%%% �������� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 100;                % ��������С
center1 = [1,1];        % ��һ����������
center2 = [3,4];        % �ڶ�����������
X = zeros(2*n,2);       % 2n * 2�����ݾ���ÿһ�б�ʾһ�����ݵ㣬��һ�б�ʾx�����꣬�ڶ��б�ʾy������
Y = zeros(2*n,1);       % ����ǩ
X(1:n,:) = ones(n,1)*center1 + randn(n,2);           %�������ݣ����ĵ�+��˹����
X(n+1:2*n,:) = ones(n,1)*center2 + randn(n,2);       %����X��ǰn�б�ʾ���1�����ݣ���n�б�ʾ���2������
Y(1:n) = 1; 
Y(n+1:2*n) = -1;        % ��һ�����ݱ�ǩΪ1���ڶ���Ϊ-1 

figure(1)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(1:n,1),X(1:n,2),'ro','LineWidth',1,'MarkerSize',10);            % ����һ�����ݵ�
hold on;
plot(X(n+1:2*n,1),X(n+1:2*n,2),'b*','LineWidth',1,'MarkerSize',10);    % ���ڶ������ݵ�
hold on;
xlabel('x axis');
ylabel('y axis');
legend('class 1','class 2');

%%%%%%%%%%%%%%%%%  ��֪��ģ��   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%  ѧ��ʵ��,�����֪��ģ�͵Ĳ���(w,b)     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w = zeros(2,1);
b = zeros(1);               % ��֪��ģ�� y = x*w + b


%%%%%%% ʹ���ݶ��½���ѵ��ģ�ͣ�����С��f(w,b) = 1/2 * || X*w + ones(2*n,1)*b - Y ||^2
% ������ʧ�������ݶ�
%���徫ȷ��
E=0.05;
%����ѧϰ��
s=0.1;
%�Ƿ�������E
is_sl=0;
%��������������
times=0;
while(is_sl==0)
    if(times==100)
        break;
    end
% ��w1���ݶ�
temp_1=0;
for i=1:200
    x_t=[X(i,1),X(i,2)];
    temp_1=temp_1+(x_t*w + b - Y(i))*X(i,1);
end
t_w_1=1/(2*n)*temp_1;
% ��w2���ݶ�
temp_2=0;
for i=1:200
    x_t=[X(i,1),X(i,2)];
    temp_2=temp_2+(x_t*w + b - Y(i))*X(i,2);
end
t_w_2=1/(2*n)*temp_2;
t_w=[t_w_1,t_w_2]';%w���ݶ�����
%��b���ݶ�
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
%%%%%%%%%%%%%%%  ����������ͼ  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% ������ x*w + b =0 ��ͼ�� %%%%%%%%%%%%%%%%%%%%%%%%%%%%

x1 = -2 : 0.00001 : 7;
y1 = ( -b * ones(1,length(x1)) - w(1) * x1 )/w(2);         % �������,length()��ʾ��������
                                                           %x1Ϊ���������ᣬy1Ϊ����
figure(3)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(1:n,1),X(1:n,2),'ro','LineWidth',1,'MarkerSize',10);            % ����һ�����ݵ�
hold on;
plot(X(n+1:2*n,1),X(n+1:2*n,2),'b*','LineWidth',1,'MarkerSize',10);    % ���ڶ������ݵ�
hold on;
plot( x1,y1,'k','LineWidth',1,'MarkerSize',10);                         % ���������
xlabel('x axis');
ylabel('y axis');
legend('class 1','class 2','classification surface');

%%%%%%%%%%%%%%%%%%  ����  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% ���ɲ�������:��ѵ������ͬ�ֲ� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m = 10;                % ������������С
Xt = zeros(2*m,2);       % 2n * 2�����ݾ���ÿһ�б�ʾһ�����ݵ㣬��һ�б�ʾx�����꣬�ڶ��б�ʾy������
Yt = zeros(2*m,1);       % ����ǩ
Xt(1:m,:) = ones(m,1)*center1 + randn(m,2);
Xt(m+1:2*m,:) = ones(m,1)*center2 + randn(m,2);       %����X��ǰn�б�ʾ���1�����ݣ���n�б�ʾ���2������
Yt(1:m) = 1; 
Yt(m+1:2*m) = -1;        % ��һ�����ݱ�ǩΪ1���ڶ���Ϊ-1 

figure(4)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(1:n,1),X(1:n,2),'ro','LineWidth',1,'MarkerSize',10);              % ����һ�����ݵ�
hold on;
plot(X(n+1:2*n,1),X(n+1:2*n,2),'b*','LineWidth',1,'MarkerSize',10);      % ���ڶ������ݵ�
hold on;
plot(Xt(1:m,1),Xt(1:m,2),'go','LineWidth',1,'MarkerSize',10);            % ����һ�����ݵ�
hold on;
plot(Xt(m+1:2*m,1),Xt(m+1:2*m,2),'g*','LineWidth',1,'MarkerSize',10);    % ���ڶ������ݵ�
hold on;
plot( x1,y1,'k','LineWidth',1,'MarkerSize',10);                           % ���������
xlabel('x axis');
xlabel('x axis');
ylabel('y axis');
legend('class 1: train','class 2: train','class 1: test','class 2: test','classification surface');

%%%%%%%%%%%%%%%%%  ѧ��ʵ��     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%  ����ģ�͵�Ԥ�����������������ݵ���ʵ����Ƚϣ����������     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sum_of_fault=0;%��Ƿ������ĸ�����
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
str=['������ȷ��:'];
disp(str);
disp(rate_of_right);