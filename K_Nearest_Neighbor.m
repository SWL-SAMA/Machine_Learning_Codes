clear all;close all;clc;
%%%%%%%%%%%%%%%%%%% �������� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 2000;                % ��������С
X = rand(n,2)*10;        % 2n * 2�����ݾ���ÿһ�б�ʾһ�����ݵ㣬��һ�б�ʾx�����꣬�ڶ��б�ʾy������
Y = zeros(n,1);          % ����ǩ

for i=1:n
   if (X(i,1)^2+X(i,2)^2<100&&-4*X(i,1)+X(i,2)<0)              % ����x��y������ȷ������      
       Y(i) = 1;
   else
       Y(i)=2;
   end
end
% X = X(Y>0,:);                                                    % ע��X����[0,10]*[0,10]��Χ�ھ������ɵģ�������ֻ�����һ����X�����֮��İ�ɫ����еĵ�û�б꣬�����Ҫ����Щ��ȥ��
% Y = Y(Y>0,:);                                                    % X(Y>0,:)��ʾֻȡX�ж�Ӧ��Y����0���У�������Ϊ��ɫ����еĵ��Y��Ϊ0
% n = length(Y);                                                   % ȥ������ɫ���ʣ�µĵ�

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

%%%%%%%%%%%%%%%%%%%  ����  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% ���ɲ�������:��ѵ������ͬ�ֲ� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m = 100;                % ������������С
Xt = rand(m,2)*10;       
Yt = zeros(m,1);
for i=1:m
    if (Xt(i,1)^2+Xt(i,2)^2<100&&-4*Xt(i,1)+Xt(i,2)<0)              % ����x��y������ȷ������      
       Yt(i) = 1;
   else
       Yt(i)=2;
   end
end
Xt = Xt(Yt>0,:);
Yt = Yt(Yt>0,:);
m = length(Yt);

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
%%%%%%%%%%%%%%%%%%  K-�����㷨��ѧ��ʵ��     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%  ����ģ�͵�Ԥ�����������������ݵ���ʵ����Ƚϣ����������     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Ym = zeros(m,1);                    % ������������Ԥ���������Ym��
%�����������������ľ���
Dis=zeros(m,n);%�������ľ���
for i=1:m %m����������ݳ���
    for j=1:n %n����ѵ�����ݳ���
        Dis(i,j)=pdist([[Xt(i,1),Xt(i,2)];[X(j,1),X(j,2)]],'euclidean');%����ŷʽ����
    end
end

%����KNN�㷨�е�kֵ
k=input("input k:");
%�Ծ�������������,index�����¼����ǰ������λ��
Dis_sorted=zeros(m,n);
index=zeros(m,n);
for i=1:m
    [Dis_sorted(i,:),index(i,:)]=sort(Dis(i,:));
end
Y_index=zeros(m,k);
for i=1:m%������Ե������ǰk��ѵ��������洢��Y_index��
    for j=1:k  
        Y_index(i,j)=Y(index(i,j));
    end
end
%ʵ��KNN����
for i=1:m
    temp=Y_index(i,:);
    Ym(i)=mode(temp);
end
sum_of_right=0;
for i=1:m
    if(Ym(i)==Yt(i))
        sum_of_right=sum_of_right+1;
    end
end
%�����ȷ��
rate_of_right=sum_of_right/m;
disp("KNN�㷨������ȷ�ʣ�");
disp(rate_of_right);
%%%%%%%%%%%%%%%%%%  ����Ԥ����     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
hold on;