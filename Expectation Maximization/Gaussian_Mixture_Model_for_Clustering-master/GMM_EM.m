function [label_new,fitness,iter_GMM,sigma_hist,mu_hist,NegativeLogLikelihood]=GMM_EM(X,K,label_old,max_iter)
% Input:
% K: number of cluster
% X: dataset, N*D where 
% label_old: initializing label. N*1
% max_iter:max of number of iterations to train the model
% Output: 
% label_new: results of cluster. N*1
% para_mu: the mean of clusters.  K*D
% para_mu: the coavariance matrix of the clusters. D*D*K
% NegativeLogLikelihood:The NegativeLogLikelihood estimate of GMM model 1
% fitness:The fitness of the model achived after training for max_iter  1*N
% sigma_hist:The history of the sigma in each iteration. max_iter*D*D*K
% mu_hist:The history of the mean in each iteration. max_iter*K*D
% This GMM_EM estimates the data X using K Gaussian
% distribution using Expectation Maximization iterations.
% Written by Raj Patel. (18BCE190@nirmauni.ac.in)
format long 
%% initializing parameters
esp=1e-6;  % stopping criterion for iteration
if nargin==3
    max_iter=1000;    % maximum number of iterations 
end
beta=1e-4;  % a regularization coefficient of covariance matrix
fitness=zeros(max_iter,1);
[X_num, X_dim]=size(X);
para_sigma=zeros(X_dim, X_dim, K); % the covariance matrix
sigma_hist=zeros(max_iter,X_dim,X_dim,K); %covariance matrix for each iteration
para_sigma_inv=zeros(X_dim, X_dim, K); % sigma^(-1)
para_mu=zeros(K, X_dim); % the mean
mu_hist=zeros(max_iter,K,X_dim);  %mean history for each iteration
para_pi=zeros(1, K); % the mixing proportion
log_N_pdf=zeros(X_num, K);  % log pdf

   
%% initializing the mixing proportion, the mean and the covariance matrix
for k=1:K
    X_k=X(label_old==k, :); 
    para_pi(k)=size(X_k, 1)/X_num;  
    para_mu(k, :)=mean(X_k);   
    sample_cov=cov(X_k)+beta*eye(X_dim);
    
    para_sigma_inv(:, :, k)=inv(sample_cov);  %sigma^(-1)
end
%% Expectation maximization (EM) algorithm
for t=1:max_iter
    %% E-step
    
    for k=1:K
        % pdf of each cluster 
        X_mu=X-repmat(para_mu(k,:), X_num, 1);  % X-mu. X_num*X_dim
        exp_up=sum((X_mu*para_sigma_inv(:, :, k)).*X_mu, 2);  % (X-mu)'*sigma^(-1)*(X-mu)
        log_N_pdf(:,k)=log(para_pi(k))-0.5*X_dim*log(2*pi)+0.5*log(abs(det(para_sigma_inv(:, :, k))))-0.5*exp_up; % N*1
    end
    
    T = logsumexp(log_N_pdf,2);
    responsivity = exp(bsxfun(@minus,log_N_pdf,T)); % posterior probability
    responsivity(isnan(responsivity)==1) = 1;
    
    %% M-step
    R_k=sum(responsivity, 1);  % 1*K
    
    % update mu
    para_mu=(responsivity'*X)./repmat(R_k', 1, X_dim);
    
    % update sigma
    for k=1:K
        X_mu=X-repmat(para_mu(k, :), X_num, 1); % N*D
        temp_X_mu_r=X_mu.*repmat(sqrt(responsivity(:, k)), 1, X_dim); % N*D
        para_sigma(:, :, k)=(temp_X_mu_r'*temp_X_mu_r)/R_k(k);
        para_sigma(:, :, k)=para_sigma(:, :, k)+beta*eye(X_dim);
        para_sigma_inv(:, :, k)=inv(para_sigma(:, :, k));  % sigma^(-1)
    end
    
    % update pi
    para_pi=R_k/sum(R_k);
    
    %update mu_hist
    mu_hist(t,:,:)=para_mu;
    
    %update_sima_hist
    sigma_hist(t,:,:,:)=para_sigma;
    
    %% Negative logLikelihood function
    %  fitness(t)=-sum(sum(log_N_pdf));
    fitness(t)=sum(sum(responsivity.*log_N_pdf));
    
    %% stopping criterion for iteration
    if t>1 
        if abs(fitness(t)-fitness(t-1))<esp
            break;
        end
    end
end
iter_GMM=t;  % iterations
NegativeLogLikelihood=fitness(iter_GMM);
%% clustering
[~, label_new]=max(responsivity, [], 2);
end