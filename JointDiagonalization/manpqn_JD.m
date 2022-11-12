function [X_nls, F_nls,F,sparsity,time_nls,iter,flag_succ,num_linesearch,mean_ssn] = manpqn_JD(A,option,M,bb_mod)
% min sum_{l=1}^N norm(diag(X'*A_l*X),'fro')^2+ mu*norm(X,1)
% s.t. X'*X=Ir.
% parameters:
%   A: parameter matrix for Joint Diagonalization problem
%   option.phi_init: initial iterative point
%   option.maxiter: maximal ietration number
%   option.tol: tolerance for stopping criterion
%   option.r, option.n, option.mu
%   option.inner_iter: maximal iteration number for SSN methods in the
%   inner loop
%   M£ºparameter for nonmonotone line search
%   bb_mod: the way computing BB stepsize strategy
%%
%parameters
tic;
% sigma = 0.1;  % parameter for line-search
r = option.r;  % number of col
n = option.n;  % dim
mu = option.mu;
N = option.N;
maxiter =option.maxiter+1;
tol = option.tol;
%inner_tol = option.inner_tol;
inner_iter = option.inner_iter;
h = @(X) sum(mu.*sum(abs(X)));  % norm(X,1)
prox_fun = @(b,lambda,r,h) proximal_l1_diag(b,lambda,r,h);
% prox_fun = @(b,lambda,r) proximal_l1(b,lambda,r);
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
hk = ones(n, 1);
para_step = 1.03;
linesearch_flag = 0;
num_inexact = 0;
Cval = F(1);
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
    
    if iter == 3
        gk_1 = grad_f(AXk_1,Xk_1,N);
        gk = grad_f(AX,X,N);
    elseif iter > 3
        gk_1 = gk;
        gk = grad_f(AX,X,N);
        sk_1 = X-Xk_1;
        yk_1 = gk-gk_1;
        rho_k = abs(sum(sum(sk_1.*yk_1)));
        if bb_mod == 2
            if mod(iter,2) == 1
                t_bb = (norm(sk_1, 'fro')^2)/rho_k;
            else
                t_bb = rho_k/(norm(yk_1, 'fro')^2);
            end
        elseif bb_mod == 1
            t_bb = (norm(sk_1, 'fro')^2) / rho_k;
        else 
            t_bb = rho_k / (norm(yk_1, 'fro')^2);
        end
        t = max(t0, t_bb);
    end
    if linesearch_flag == 0
        t = t*para_step;
    else
        t = max(t0,t/para_step);
%         ls_flag_sum = ls_flag_sum + 1;
    end
    linesearch_flag = 0;
    
    if iter>2 && abs(F(iter-1)-F(iter-2))<1e-6
        hk = ones(size(hk));
    else
        if iter == 4
            h0 =  rho_k / (norm(yk_1, 'fro'));
            rho_k = 1/rho_k;
            Hk = h0 * eye(n) + (rho_k-1/rho_k) * (yk_1*yk_1');
            hk = diag(Hk);
        elseif iter > 4
            rho_k = 1/rho_k;
            vk = yk_1'*Hk;
            Hk = Hk - (vk'*vk)/abs(trace(yk_1'*Hk*yk_1)) + rho_k*(sk_1*sk_1');
            hk = diag(Hk);
        end

        hk = abs(hk);
        hk = hk/sqrt(norm(hk,2));
        hk(hk<1e-4)=1e-4;
    end
    invDiag = hk;
    
    if iter == 2
        [ PY, num_inner(iter), Lam, opt_sub(iter), ...
            in_flag]=Semi_newton_matrix_prox_diag(n,r,X,invDiag,t,...
            X+2*t*invDiag.*pgx,mu*t,inner_tol,prox_fun,inner_iter,zeros(r),Dn,pDn);
    else
        [ PY, num_inner(iter), Lam, opt_sub(iter), ...
            in_flag]=Semi_newton_matrix_prox_diag(n,r,X,invDiag,t,...
            X+2*t*invDiag.*pgx,mu*t,inner_tol,prox_fun,inner_iter,Lam,Dn,pDn);
    end

    if in_flag == 1   % subprolem total iteration.
        inner_flag = 1 + inner_flag;
    end
    alpha=1;
    
    if nnz(isnan(PY))~=0||nnz(isinf(PY))~=0
        invDiag = ones(size(hk));
       	[ PY, num_inner(iter), Lam, opt_sub(iter), ...
            in_flag]=Semi_newton_matrix_prox_diag(n,r,X,invDiag,t,...
            X+2*t*invDiag.*pgx,mu*t,inner_tol,prox_fun,inner_iter,Lam,Dn,pDn);
    end
    
    D = PY-X; %descent direction D
    [U, SIGMA, S] = svd(PY'*PY);
    SIGMA =diag(SIGMA);
    Z = PY*(U*diag(sqrt(1./SIGMA))*S');
        
    AZ = zeros(n,r,N);
    for l=1:N
        AZ(:,:,l) = A(:,:,l)*Z;
    end

    F_trial = func_f(AZ,Z,N)+h(Z);
    normDsquared = norm(D,'fro')^2;
    
    %% nonmonotone-linesearch
    if iter <= M+1
        Cval = max(F(1:iter-1));
    else
        Cval = max(F(iter-M:iter-1));
    end
    while F_trial>= Cval-0.5/t*alpha*normDsquared  %*sigma
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
        PY = X + alpha*D;
        [U, SIGMA, S] = svd(PY'*PY);
        SIGMA =diag(SIGMA);
        Z = PY*(U*diag(sqrt(1./SIGMA))*S');
        
        for l=1:N
            AZ(:,:,l) = A(:,:,l)*Z;
        end
        F_trial = func_f(AZ,Z,N) + h(Z);
    end
    Xk_1 = X;
    AXk_1 = AX;
    X = Z;
    AX = AZ;
    F(iter) = F_trial;
    if F_trial < option.F_manpg
        break;
    end
end

X((abs(X)<=1e-5))=0;
X_nls=X;
time_nls = toc;
mean_ssn = sum(num_inner)/(iter-1);

if iter == maxiter && sqrt(normDsquared)/t > 1e-1
    flag_succ = 0;
else
    flag_succ = 1;
end
sparsity= sum(sum(X_nls==0))/(n*r);
F_nls =  F(iter);
    
%     fprintf('NLS:Iter ***  Fval *** CPU  **** sparsity ***inner_inexact&averge_No. ** opt_norm ** total_linsea \n');
%     print_format = ' %i     %1.5e    %1.2f     %1.2f         %4i   %2.2f                %1.3e        %d \n';
%     fprintf(1,print_format, iter-1,min(F), time_nls,sparsity, inner_flag, sum(num_inner)/(iter-1) ,sqrt(normDsquared)/t,num_linesearch);

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

function grad_value = grad_f(AX,X,N)
    nf = nabla_f(AX,X,N);
    nfx = X'*nf;
    grad_value = nf-0.5*X*(nfx+nfx');
end