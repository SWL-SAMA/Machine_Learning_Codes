clear all;
clc;
close all;
%%%%%%%%%%%%%%%%%%% �������� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 2000;                % ��������С
X = rand(n,2)*10;        % n * 2�����ݾ���ÿһ�б�ʾһ�����ݵ㣬��һ�б�ʾx�����꣬�ڶ��б�ʾy������
Y = zeros(n,1);          % ����ǩ

for i=1:n
   if 0<X(i,1) && X(i,1)<3 && 0<X(i,2) && X(i,2)<3              % ����x��y������ȷ������      
       Y(i) = 1;
   end
   if 0<X(i,1) && X(i,1)<3 && 3.5<X(i,2) && X(i,2)<6.5
       Y(i) = 2;
   end
   if 0<X(i,1) && X(i,1)<3 && 7<X(i,2) && X(i,2)<10
       Y(i) = 3;
   end
   if 3.5<X(i,1) && X(i,1)<6.5 && 0<X(i,2) && X(i,2)<3
       Y(i) = 4;
   end
   if 3.5<X(i,1) && X(i,1)<6.5 && 3.5<X(i,2) && X(i,2)<6.5
       Y(i) = 5;
   end
   if 3.5<X(i,1) && X(i,1)<6.5 && 7<X(i,2) && X(i,2)<10
       Y(i) = 6;
   end
   if 7<X(i,1) && X(i,1)<10 && 0<X(i,2) && X(i,2)<3
       Y(i) = 7;
   end
   if 7<X(i,1) && X(i,1)<10 && 3.5<X(i,2) && X(i,2)<6.5
       Y(i) = 8;
   end
   if 7<X(i,1) && X(i,1)<10 && 7<X(i,2) && X(i,2)<10
       Y(i) = 9;
   end
end

X = X(Y>0,:);                                                    % ע��X����[0,10]*[0,10]��Χ�ھ������ɵģ�������ֻ�����һ����X�����֮��İ�ɫ����еĵ�û�б꣬�����Ҫ����Щ��ȥ��
Y = Y(Y>0,:);                                                    % X(Y>0,:)��ʾֻȡX�ж�Ӧ��Y����0���У�������Ϊ��ɫ����еĵ��Y��Ϊ0
nn = length(Y);                                                  % ȥ������ɫ���ʣ�µĵ�

n = 2000;
X(nn+1:n,:) = rand(n-nn,2)*10;                                   % ����n-nn��������
Y(nn+1:n) = ceil( rand(n-nn,1)*9 );                              % ������ı�ǩ���ѡȡ��rand(n-nn,1)*9��ʾ����[0,9]�ľ��ȷֲ���ceil��ʾ��ȡ�����ʽ��Ϊ1,2,...,9

figure(1)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(Y==1,1),X(Y==1,2),'ro','LineWidth',1,'MarkerSize',10);            % ����һ�����ݵ�    X(Y==1,1)��ʾ���Ϊ1��Y==1���ĵ�ĵ�һά�����꣬X(Y==1,2)��ʾ���Ϊ1�ĵ�ĵڶ�ά������
hold on;
plot(X(Y==2,1),X(Y==2,2),'ko','LineWidth',1,'MarkerSize',10);            % ���ڶ������ݵ�
hold on;
plot(X(Y==3,1),X(Y==3,2),'bo','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(X(Y==4,1),X(Y==4,2),'g*','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(X(Y==5,1),X(Y==5,2),'m*','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(X(Y==6,1),X(Y==6,2),'c*','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(X(Y==7,1),X(Y==7,2),'b+','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(X(Y==8,1),X(Y==8,2),'r+','LineWidth',1,'MarkerSize',10);            % ���ڰ������ݵ�
hold on;
plot(X(Y==9,1),X(Y==9,2),'k+','LineWidth',1,'MarkerSize',10);            % ���ھ������ݵ�
hold on;
xlabel('x axis');
ylabel('y axis');

%%%%%%%%%%%%%%%%%%%  ���ɲ�������  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% ���ɲ�������:��ѵ������ͬ�ֲ� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m = 100;                % ������������С
Xt = rand(m,2)*10;       
Yt = zeros(m,1);
for i=1:m
   if 0<Xt(i,1) && Xt(i,1)<3 && 0<Xt(i,2) && Xt(i,2)<3
       Yt(i) = 1;
   end
   if 0<Xt(i,1) && Xt(i,1)<3 && 3.5<Xt(i,2) && Xt(i,2)<6.5
       Yt(i) = 2;
   end
   if 0<Xt(i,1) && Xt(i,1)<3 && 7<Xt(i,2) && Xt(i,2)<10
       Yt(i) = 3;
   end
   if 3.5<Xt(i,1) && Xt(i,1)<6.5 && 0<Xt(i,2) && Xt(i,2)<3
       Yt(i) = 4;
   end
   if 3.5<Xt(i,1) && Xt(i,1)<6.5 && 3.5<Xt(i,2) && Xt(i,2)<6.5
       Yt(i) = 5;
   end
   if 3.5<Xt(i,1) && Xt(i,1)<6.5 && 7<Xt(i,2) && Xt(i,2)<10
       Yt(i) = 6;
   end
   if 7<Xt(i,1) && Xt(i,1)<10 && 0<Xt(i,2) && Xt(i,2)<3
       Yt(i) = 7;
   end
   if 7<Xt(i,1) && Xt(i,1)<10 && 3.5<Xt(i,2) && Xt(i,2)<6.5
       Yt(i) = 8;
   end
   if 7<Xt(i,1) && Xt(i,1)<10 && 7<Xt(i,2) && Xt(i,2)<10
       Yt(i) = 9;
   end
end
Xt = Xt(Yt>0,:);
Yt = Yt(Yt>0,:);
m = length(Yt);
Ym = zeros(m,1);                                                         % ��¼ģ��������

figure(2)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(Y==1,1),X(Y==1,2),'ro','LineWidth',1,'MarkerSize',10);            % ����һ�����ݵ�
hold on;
plot(X(Y==2,1),X(Y==2,2),'ko','LineWidth',1,'MarkerSize',10);            % ���ڶ������ݵ�
hold on;
plot(X(Y==3,1),X(Y==3,2),'bo','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(X(Y==4,1),X(Y==4,2),'g*','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(X(Y==5,1),X(Y==5,2),'b*','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(X(Y==6,1),X(Y==6,2),'c*','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(X(Y==7,1),X(Y==7,2),'b+','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(X(Y==8,1),X(Y==8,2),'r+','LineWidth',1,'MarkerSize',10);            % ���ڰ������ݵ�
hold on;
plot(X(Y==9,1),X(Y==9,2),'k+','LineWidth',1,'MarkerSize',10);            % ���ھ������ݵ�
hold on;
plot(Xt(:,1),Xt(:,2),'ms','MarkerFaceColor','m','LineWidth',1,'MarkerSize',10);            % ���������ݵ�   Xt(:,2)��ʾXt�ĵڶ��У�����������
hold on;
xlabel('x axis');
ylabel('y axis');

%%%%%%%%%%%%%%%%%%  ��Ҷ˹�㷨��ѧ��ʵ��     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%  ����ģ�͵�Ԥ�����������������ݵ���ʵ����Ƚϣ����������     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%�����Խ�������Ym��
s_y=zeros(9,1);
P_y=zeros(9,1);%����ÿһ���ĸ���,��Ӧ��ʽ�е�P(y)
for i=1:9
    for j=1:n
        if(Y(j)==i)
            s_y(i)=s_y(i)+1;
        end
    end
end
for i=1:9
    P_y(i)=s_y(i)/n;
end
%����ÿ��������Ӧ��������������(һ�������������ֱ���ѵ������xy����)
p_x1_y=zeros(3,9);%������������
p_x2_y=zeros(3,9);%����������ֱ��Ӧ3��������ȡֵ��9�ֿ��ܵķ���
for i=1:3
    for j=1:9
        for k=1:n
            if (i==1 && X(k,1)<=3 && X(k,1)>=0 && Y(k)==j)%�����k�����x����Ϊ��һ�ಢ�����ķ���Ϊjʱ�����һ
                p_x1_y(i,j)=p_x1_y(i,j)+1;
            end
            if (i==2 && X(k,1)>=3.5 && X(k,1)<=6.5 && Y(k)==j)%�����k�����x����Ϊ�ڶ��ಢ�����ķ���Ϊjʱ�����һ
                p_x1_y(i,j)=p_x1_y(i,j)+1;
            end
            if (i==3 && X(k,1)<=10 && X(k,1)>=7 && Y(k)==j)%�����k�����x����Ϊ�����ಢ�����ķ���Ϊjʱ�����һ
                p_x1_y(i,j)=p_x1_y(i,j)+1;
            end
        end
    end
end
for i=1:3
    for j=1:9
        for k=1:n
            if (i==1 && X(k,2)<=3 && X(k,2)>=0 && Y(k)==j)%�����k�����x����Ϊ��һ�ಢ�����ķ���Ϊjʱ�����һ
                p_x2_y(i,j)=p_x2_y(i,j)+1;
            end
            if (i==2 && X(k,2)>=3.5 && X(k,2)<=6.5 && Y(k)==j)%�����k�����x����Ϊ�ڶ��ಢ�����ķ���Ϊjʱ�����һ
                p_x2_y(i,j)=p_x2_y(i,j)+1;
            end
            if (i==3 && X(k,2)<=10 && X(k,2)>=7 && Y(k)==j)%�����k�����x����Ϊ�����ಢ�����ķ���Ϊjʱ�����һ
                p_x2_y(i,j)=p_x2_y(i,j)+1;
            end
        end
    end
end
for i=1:9
    p_x1_y(:,i)=p_x1_y(:,i)/s_y(i);
    p_x2_y(:,i)=p_x2_y(:,i)/s_y(i);
end
%����ÿ��ʵ���������ϸ��ʷֲ�
P=ones(m,9);%������Ե�x�ֱ�Ϊ��i��ĸ���
for i=1:m
    for j=1:9
        if(Xt(i,1)<=3 && Xt(i,1)>=0)
            P(i,j)=p_x1_y(1,j)*P_y(j)*P(i,j);
        end
        if(Xt(i,1)<=6.5 && Xt(i,1)>=3.5)
            P(i,j)=p_x1_y(2,j)*P_y(j)*P(i,j);
        end
        if(Xt(i,1)<=10 && Xt(i,1)>=7)
            P(i,j)=p_x1_y(3,j)*P_y(j)*P(i,j);
        end
    end
end
for i=1:m
    for j=1:9
        if(Xt(i,2)<=3 && Xt(i,2)>=0)
            P(i,j)=p_x2_y(1,j)*P(i,j);
        end
        if(Xt(i,2)<=6.5 && Xt(i,2)>=3.5)
            P(i,j)=p_x2_y(2,j)*P(i,j);
        end
        if(Xt(i,2)<=10 && Xt(i,2)>=7)
            P(i,j)=p_x2_y(3,j)*P(i,j);
        end
    end
end
P_of_max=zeros(m,1);
for i=1:m
    [P_of_max(i),Ym(i)]=max(P(i,:));%���������ϸ����ܶ������ֵ�����ֵ��Ӧ�ط���������������
end
%���������ȷ�ʲ���ӡ
rate_of_right=0;
for i=1:m
    if(Yt(i)==Ym(i))
        rate_of_right=rate_of_right+1;
    end
end
rate_of_right=rate_of_right/m;
disp("������ȷ��Ϊ��");
disp(rate_of_right);
%%%%%%%%%%%%%%%%%%  ������ӻ�     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(3)
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(Y==1,1),X(Y==1,2),'ro','LineWidth',1,'MarkerSize',10);            % ����һ�����ݵ�
hold on;
plot(X(Y==2,1),X(Y==2,2),'ko','LineWidth',1,'MarkerSize',10);            % ���ڶ������ݵ�
hold on;
plot(X(Y==3,1),X(Y==3,2),'bo','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(X(Y==4,1),X(Y==4,2),'g*','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(X(Y==5,1),X(Y==5,2),'b*','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(X(Y==6,1),X(Y==6,2),'c*','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(X(Y==7,1),X(Y==7,2),'b+','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(X(Y==8,1),X(Y==8,2),'r+','LineWidth',1,'MarkerSize',10);            % ���ڰ������ݵ�
hold on;
plot(X(Y==9,1),X(Y==9,2),'k+','LineWidth',1,'MarkerSize',10);            % ���ھ������ݵ�
hold on;
plot(Xt(Ym==1,1),Xt(Ym==1,2),'ro','MarkerFaceColor','r','LineWidth',1,'MarkerSize',10);            % ����һ�����ݵ�
hold on;
plot(Xt(Ym==2,1),Xt(Ym==2,2),'ko','MarkerFaceColor','k','LineWidth',1,'MarkerSize',10);            % ���ڶ������ݵ�
hold on;
plot(Xt(Ym==3,1),Xt(Ym==3,2),'bo','MarkerFaceColor','b','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(Xt(Ym==4,1),Xt(Ym==4,2),'go','MarkerFaceColor','g','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(Xt(Ym==5,1),Xt(Ym==5,2),'bo','MarkerFaceColor','b','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(Xt(Ym==6,1),Xt(Ym==6,2),'co','MarkerFaceColor','c','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(Xt(Ym==7,1),Xt(Ym==7,2),'bo','MarkerFaceColor','b','LineWidth',1,'MarkerSize',10);            % �����������ݵ�
hold on;
plot(Xt(Ym==8,1),Xt(Ym==8,2),'ro','MarkerFaceColor','r','LineWidth',1,'MarkerSize',10);            % ���ڰ������ݵ�
hold on;
plot(Xt(Ym==9,1),Xt(Ym==9,2),'ko','MarkerFaceColor','k','LineWidth',1,'MarkerSize',10);            % ���ھ������ݵ�
hold on;
xlabel('x axis');
ylabel('y axis');