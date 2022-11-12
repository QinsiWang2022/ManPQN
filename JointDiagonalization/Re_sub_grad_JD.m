function [X_Rsub,F_Sub,sparsity,time_Rsub,i,succ_flag] = Re_sub_grad_JD(A,option)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Riemannian subgradient;
% min sum_{l=1}^N norm(diag(X'*A_l*X),'fro')^2+ mu*norm(X,1) 
% s.t. X'*X=Ir. X \in R^{p*r}
tic;
r=option.r;
n=option.n;
mu = option.mu;
N = option.N;
maxiter =option.maxiter + 1;
tol = option.tol;
X = option.phi_init;

AX = zeros(n,r,N);
for l=1:N
    AX(:,:,l) = A(:,:,l)*X;
end

h=@(X) mu*sum(sum(abs(X)));

f_re_sub = zeros(maxiter,1); 
succ_flag = 0;
f_re_sub(1) = func_f(AX,X,N) + h(X);
for i = 2:maxiter
    gx = - nabla_f(AX,X,N) - mu*sign(X) ; %negative Euclidean gradient
    xgx = X'*gx;
    pgx = gx - 0.5*X*(xgx+xgx');   %negative  Riemannian gradient using Euclidean metric
    %pgx = gx;
    %eta = 0.6*0.99^i; 
    eta = 1/i^(3/4);  
    %eta = 1/i;  
    X = X + eta * pgx;    % Riemannian step
    %[q,~] = qr(q);    % retraction
    [U, SIGMA, S] = svd(X'*X);   SIGMA =diag(SIGMA);    X = X*(U*diag(sqrt(1./SIGMA))*S');
    for l=1:N
        AX(:,:,l) = A(:,:,l)*X;
    end
    f_re_sub(i) = func_f(AX,X,N) + h(X);
    if  f_re_sub(i) < option.F_manpg + option.tol
        succ_flag = 1;
        break;
    end
   
end
X((abs(X)<=1e-5))=0;
X_Rsub = X;
time_Rsub = toc;
sparsity= sum(sum(X_Rsub==0))/(n*r);
F_Sub = f_re_sub(i-1);

%    fprintf('Rsub:Iter ***  Fval *** CPU  **** sparsity \n');
%     
%     print_format = ' %i     %1.5e    %1.2f     %1.2f    \n';
%     fprintf(1,print_format, i-1, F_Sub, time_Rsub, sparsity);
end

function f_value = func_f(AX,X,N)
    f_value = 0;
    for l=1:N
        f_value = f_value - norm(diag(X'*AX(:,:,l)),2)^2;
    end
end

function nabla_value = nabla_f(AX,X,N)
    nabla_value = 0;
    for l=1:N
        nabla_value = nabla_value - 4*AX(:,:,l)*diag(diag(X'*AX(:,:,l)));
    end
end
