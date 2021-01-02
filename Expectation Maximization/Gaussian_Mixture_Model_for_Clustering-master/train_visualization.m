clc;
clear all;
close all;

%%visulaizing the training of algorithm for the iris dataset reduced to 2
%%dimensions using PCA,

data=dlmread('iris.data');%loading the iris dataset
n=length(data);
[coeff,score] = pca(data(:,1:end-1)); %computing PCA component to make 2D data

PCA_data=zeros(n,2); % x,y,label 
X=score(:,1);
Y=score(:,2);
real_label=data(:,end);

PCA_data(:,1)=X;
PCA_data(:,2)=Y;

x_lim=[min(X),max(X)];
y_lim=[min(Y),max(Y)];
figure;
scatter(X(real_label==1),Y(real_label==1),'r+');
hold on;
scatter(X(real_label==2),Y(real_label==2),'g+');
hold on;
scatter(X(real_label==3),Y(real_label==3),'b+');
xlabel("X");
ylabel("Y");
legend(["class 1","class 2","class 3"])
title("PCA component of iris dataset");
hold on;


%% Setting the hyper-parameters
choose_norm=2; % Normalization methods, 0: no normalization, 1: z-score, 2: max-min
init=1; % Initialization methods, 1: random, 2: K-means, 3: fuzzt c-means, 4: K-means clustering, accelerated by matlab matrix operations.


%% Load data
K=length(unique(real_label)); % number of cluster,for iris K=3
[N, ~]=size(data);
label_old=zeros(N,1);
s_1=0; 
%% Initialization & Normalization
%PCA_data = normlization(PCA_data, choose_norm);
label_old(:)=init_methods(PCA_data, K, init);

%% Fit the data with GMM_EM function
t0=cputime;
max_iter=1000;
[label_new,fitness,iter_GMM,sigma_hist,mu_hist]=GMM_EM(PCA_data, K, label_old,max_iter);


%some intermediate iterations of all the training iterations.
mid_iterations= [int64(1:iter_GMM/3:iter_GMM),iter_GMM];
figure;
cnt=0;
for iter=mid_iterations
    cnt=cnt+1;
    subplot(2,2,cnt);
    scatter(X(real_label==1),Y(real_label==1),'r+');
    hold on;
    scatter(X(real_label==2),Y(real_label==2),'g+');
    hold on;
    scatter(X(real_label==3),Y(real_label==3),'b+');
    xlabel("X");
    ylabel("Y");
    title(['iteration:  ',num2str(iter)])
    hold on;
    for k=1:K
        mu=reshape(mu_hist(iter,k,:),1,2);
        cov=reshape(sigma_hist(iter,:,:,k),2,2);
        x=linspace(x_lim(1), x_lim(2),100);
        y=linspace(y_lim(1), y_lim(2),100);
        [X_grid,Y_grid]=meshgrid(x,y);

        z=mvnpdf( reshape([X_grid,Y_grid],10000,2),mu,cov);
        %surf(X_grid,Y,reshape(z,100,100));
        contour(X_grid,Y_grid,reshape(z,100,100),3);
    end
end

%%visualizing 3d probability distribution of 2d gaussian distribution
 figure;
 for k=1:K
    mu=reshape(mu_hist(iter_GMM,k,:),1,2);
    cov=reshape(sigma_hist(iter_GMM,:,:,k),2,2);
    x=linspace(x_lim(1)-1, x_lim(2)+1,100);
    y=linspace(y_lim(1)-1, y_lim(2)+1,100);
    [X_grid,Y_grid]=meshgrid(x,y);

    z=mvnpdf( reshape([X_grid,Y_grid],10000,2),mu,cov);
    
    surf(X_grid,Y_grid,reshape(z,100,100));
    hold on;
    
 end
  title("Gassuian mixture model(GMM) for the iris data")
  xlabel("X");
  ylabel("Y");
  zlabel("P(x,y| {\theta})");



