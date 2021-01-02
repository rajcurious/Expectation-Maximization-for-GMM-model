
% Demo Iris.data
% Written by Raj Patel. (18bce190@nirmauni.ac.in)
clc;
clear all;
close all;


%% Setting the hyper-parameters
choose_norm=2; % Normalization methods, 0: no normalization, 1: z-score, 2: max-min
init=4; % Initialization methods, 1: random, 2: K-means, 3: fuzzt c-means, 4: K-means clustering, accelerated by matlab matrix operations.


%% Load data
addpath(genpath('.'));
data_load=dlmread('iris.data');
data=data_load(:, 1:end-1);
real_label=data_load(:, end);
K=length(unique(real_label)); % number of cluster
[N, ~]=size(data);
label_old=zeros(N,1);
s_1=0; 

%% Initialization & Normalization
data = normlization(data, choose_norm);
label_old(:)=init_methods(data, K, init);

%% Repeat the experiment repeat_num times
t0=cputime;
max_iter=1000;
[label_new,fitness,iter_GMM,sigma_hist,mu_hist]=GMM_EM(data, K, label_old,max_iter);

figure;
plot(1:iter_GMM,fitness(1:iter_GMM));
xlabel("iterations");
ylabel("LogLikelihood of model");
title("LogLikelihood vs iterations  ");
%% calculating accuracy
accuracy=label_map(real_label,label_new);
fprintf('Accuary in percentatge: %.8f\n',accuracy*100);
run_time=cputime-t0;
fprintf('Runtime in seconds: %.4fs\n',run_time);


%% Visulaizing the output
figure;
subplot(1,3,1);
hist(label_old);
title("old label");
xlabel("class");
ylabel("samples");

subplot(1,3,2);
hist(label_new);
title("new label");
xlabel("class");
ylabel("samples");

subplot(1,3,3);
hist(real_label);
title("real label(Ground truth)");
xlabel("class");
ylabel("samples");



