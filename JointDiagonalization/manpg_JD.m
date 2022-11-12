function [X_manpg, F_manpg,sparsity,time_manpg,iter,flag_succ,num_linesearch,mean_ssn] =manpg_JD(A,option)
% min sum_{l=1}^N norm(diag(X'*A_l*X),'fro')^2+ mu*norm(X,1)
% s.t. X'*X=Ir.
%%
%parameters
tic;
r = option.r;%number of col
n = option.n;%dim
mu = option.mu;
N = option.N;
maxiter =option.maxiter+1;
tol = option.tol;
%inner_tol =  option.inner_tol;
inner_iter = option.inner_iter;
h=@(X) sum(mu.*sum(abs(X)));
prox_fun = @(b,lambda,r) proximal_l1(b,lambda,r);
inner_flag = 0;
%setduplicat_pduplicat(r);
Dn = sparse(DuplicationM(r));
pDn = (Dn'*Dn)\Dn';
t_min=1e-4; % minimum stepsize
%%
%initial point
X = option.phi_init;
L = option.L;
AX = zeros(n,r,N);
for l=1:N
    AX(:,:,l) = A(:,:,l)*X;
end

F(1) = func_f(AX,X,N)+h(X);
num_inner = zeros(maxiter,1);
opt_sub = num_inner;
num_linesearch = 0;
alpha = 1;
t0 = 1/L; t =t0;
linesearch_flag = 0;
num_inexact = 0;
for iter = 2:maxiter
    %   fprintf('manpg__________________iter: %3d, \n', iter);
    ngx = -nabla_f(AX,X,N); % negative gradient 2AX
    %xgx = X'*ngx;
    %pgx = ngx - 0.5*X*(xgx+xgx');
    pgx= ngx; % grad or projected gradient both okay
    %% subproblem
    if alpha < t_min || num_inexact > 10
        inner_tol = max(5e-16, min(1e-14,1e-5*tol*t^2)); % subproblem inexact;
    else
        inner_tol = max(1e-13, min(1e-11,1e-3*tol*t^2));
    end
    
    if iter == 2
        [ PY,num_inner(iter),Lam, opt_sub(iter),in_flag]=Semi_newton_matrix(n,r,X,t,X+2*t*pgx,mu*t,inner_tol,prox_fun,inner_iter,zeros(r),Dn,pDn);
        
    else
        [ PY,num_inner(iter),Lam,  opt_sub(iter) ,in_flag]=Semi_newton_matrix(n,r,X,t,X+2*t*pgx,mu*t,inner_tol,prox_fun,inner_iter,Lam,Dn,pDn);
        
    end
    
    %     if iter ==2
    %         [Y,~, dual_y,dual_z, ~,  ~, ~, ~, subiter, ls, cg_iter] =SSNAL_mat(eye(n),eye(n),X-t*pgx, X, eye(n), eye(n),X, X,zeros(r),zeros(n,r), eye(r), mu*t, 50, 1e-4) ;
    %     else
    %         [Y,~, dual_y,dual_z, ~,  ~, ~, ~, subiter, ls, cg_iter] = SSNAL_mat(eye(n),eye(n),X-t*pgx, X, eye(n),eye(n),X, X, dual_y, dual_z, eye(r), mu*t, 50, 1e-3) ;
    %     end
    if in_flag == 1   % subprolem total iteration.
        inner_flag = 1 + inner_flag;
    end
    alpha=1;
    D = PY-X; %descent direction D
    
    [U, SIGMA, S] = svd(PY'*PY);
    SIGMA =diag(SIGMA);
    Z = PY*(U*diag(sqrt(1./SIGMA))*S');
    % [Z,R]=qr(PY,0);       Z = Z*diag(sign(diag(R))); %old version need consider the sign
    
    AZ = zeros(n,r,N);
    for l=1:N
        AZ(:,:,l) = A(:,:,l)*Z;
    end
    
    F_trial = func_f(AZ,Z,N)+h(Z);
    normDsquared = norm(D,'fro')^2;
    
    if  normDsquared/t^2 < tol
        % if  abs(F(iter)-F(iter-1))/(abs(F(iter))+1)<tol
        break;
    end
    %% linesearch
    while F_trial>= F(iter-1)-0.5/t*alpha*normDsquared
        alpha = 0.5*alpha;
        linesearch_flag = 1;
        num_linesearch = num_linesearch+1;
        if alpha<t_min
            num_inexact = num_inexact + 1;
            break;
        end
%         if alpha< 1e-16
%             break;
%         end
        PY =X + alpha*D;
        %  [U,~,V]=svd(PY,0);  Z=U*V';
        %  [Z,R]=qr(PY,0);   Z = Z*diag(sign(diag(R)));  %old version need consider the sign
        [U, SIGMA, S] = svd(PY'*PY);   SIGMA =diag(SIGMA);
        Z = PY*(U*diag(sqrt(1./SIGMA))*S');
        for l=1:N
            AZ(:,:,l) = A(:,:,l)*Z;
        end
        F_trial =  func_f(AZ,Z,N) + h(Z);
    end
    X = Z;
    AX = AZ;
    F(iter) = F_trial;
    %     if option.adap ==1
    %         if linesearch_flag == 0
    %             t = t*1.01;
    %         else
    %             t = max(t0,t/1.01);
    %         end
    %     end
    linesearch_flag = 0;
end
X((abs(X)<=1e-5))=0;
X_manpg=X;
time_manpg = toc;
mean_ssn = sum(num_inner)/(iter-1);
% plot(F);

if iter == maxiter && sqrt(normDsquared)/t > 1e-1
    flag_succ = 0;
%     sparsity  = 0;
%     F_manpg = 0;
%     time_manpg = 0;
else
    flag_succ = 1;
end
sparsity= sum(sum(X_manpg==0))/(n*r);
F_manpg =  F(iter);
    
%     fprintf('ManPG:Iter ***  Fval *** CPU  **** sparsity ***inner_inexact&averge_No. ** opt_norm ** total_linsea \n');
%     print_format = ' %i     %1.5e    %1.2f     %1.2f         %4i   %2.2f                %1.3e        %d \n';
%     fprintf(1,print_format, iter-1,min(F), time_manpg,sparsity, inner_flag, sum(num_inner)/(iter-1) ,sqrt(normDsquared)/t,num_linesearch);
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