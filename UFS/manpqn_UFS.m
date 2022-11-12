function [X_nls, F_nls,sparsity,time_nls,iter,flag_succ,num_linesearch,mean_ssn] = manpqn_UFS(H,option,d_l,V,M,bb_mod)
% solving min -Tr(X'*H*X)+ mu*norm(X,1) s.t. X'*X=Ir.
% parameters:
%   H: parameter matrix for UFS problem
%   option.phi_init: initial iterative point
%   option.maxiter: maximal ietration number
%   option.tol: tolerance for stopping criterion
%   option.r, option.n, option.mu
%   option.inner_iter: maximal iteration number for SSN methods in the
%   inner loop
%   M：parameter for nonmonotone line search
%   bb_mod: the way computing BB stepsize strategy
%%
%parameters
tic;
r = option.r;  % number of col
n = option.n;  % dim
mu = option.mu;
maxiter =option.maxiter+1;
tol = option.tol;
%inner_tol = option.inner_tol;
inner_iter = option.inner_iter;
h = @(X) mu*sum(vecnorm(X,2,2));  % norm(X,1)
prox_fun = @(b,lambda,h) proximal_l21_diag(b,lambda,h);
inner_flag = 0;
%setduplicat_pduplicat(r);
Dn = sparse(DuplicationM(r));
pDn = (Dn'*Dn)\Dn';
t_min=1e-4; % minimum stepsize

%%
%initial point
X = option.phi_init;
L = 8/d_l^2.*(sin(pi/4))^2 + V;
%dx = option.L/n;
%LAM_A =  (cos(2*pi*[0:n-1]'/n)-1)/dx^2;
%HX = real(fft(bsxfun(@times,ifft( X ),LAM_A)));
HX = H*X;

F(1) = -sum(sum(X.*(HX)))+h(X);
num_inner = zeros(maxiter,1);
opt_sub = num_inner;
num_linesearch = 0;
alpha = 1;
t0 = 1/L; t =t0;
hk = ones(n, 1);
para_step = 1.01;
linesearch_flag = 0;
num_inexact = 0;

Cval = F(1);
for iter = 2:maxiter
    %   fprintf('manpg__________________iter: %3d, \n', iter);
    ngx = HX; % negative gradient 2AX
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
        HXk_1 = H*Xk_1;
        gk_1 = -2*HXk_1 + 2*(Xk_1*(Xk_1'*HXk_1));
        gk = -2*HX + 2*(X*(X'*HX));
    elseif iter > 3
        gk_1 = gk;
        gk = -2*HX + 2*(X*(X'*HX));
        sk_1 = X-Xk_1;
        yk_1 = gk-gk_1;
        rho_k = abs(sum(sum(sk_1.*yk_1)));
        if bb_mod == 2
            % 分奇偶步交替生成BB步长
            if mod(iter,2) == 1
                t_bb = norm(sk_1, 'fro')/rho_k;
            else
                t_bb = rho_k/norm(yk_1, 'fro');
            end
        elseif bb_mod == 1
            % 仅使用一种形式的BB步长
            t_bb = (norm(sk_1, 'fro')) / rho_k;
        else 
            t_bb = rho_k / (norm(yk_1, 'fro'));
        end
        % 对BB步长t取下界t0进行截断，避免步长太小而使得迭代停滞不前
        t = max(t0, t_bb);
    end
    if linesearch_flag == 0
        t = t*para_step;
    else
        t = max(t0,t/para_step);
%         ls_flag_sum = ls_flag_sum + 1;
    end
    linesearch_flag = 0;
    
% DFP
    if iter == 4
        h0 =  rho_k / (norm(yk_1, 'fro')^2);
        rho_k = 1/rho_k;
        vk = eye(n)-(rho_k*yk_1)*sk_1';
        Hk = h0*(vk'*vk) + rho_k*(sk_1*sk_1');
        hk = diag(Hk);
    elseif iter > 4
        rho_k = 1/rho_k;
        vk = eye(n)-(rho_k*yk_1)*sk_1';
        Hk = vk'*Hk*vk + rho_k*(sk_1*sk_1');
        hk = diag(Hk);
    end

    hk = abs(hk);
    hk(hk<1e-4)=1e-4;
    invDiag = hk;
    
    if iter == 2
        [ PY,num_inner(iter),Lam, opt_sub(iter),...
            in_flag]=Semi_newton_matrix_l21_prox_diag(n,r,X,invDiag,t,...
            X+2*t*invDiag.*pgx,mu*t,inner_tol,prox_fun,inner_iter,zeros(r),Dn,pDn);
    else
        [ PY,num_inner(iter),Lam, opt_sub(iter),...
            in_flag]=Semi_newton_matrix_l21_prox_diag(n,r,X,invDiag,t,...
            X+2*t*invDiag.*pgx,mu*t,inner_tol,prox_fun,inner_iter,Lam,Dn,pDn);
    end
    
    if in_flag == 1   % subprolem total iteration.
        inner_flag = 1 + inner_flag;
    end
    alpha=1;
    D = PY-X; %descent direction D

    [U, SIGMA, S] = svd(PY'*PY);
    SIGMA =diag(SIGMA);
    Z = PY*(U*diag(sqrt(1./SIGMA))*S');

    HZ = H*Z;

    F_trial = -sum(sum(Z.*(HZ)))+h(Z);
    normDsquared = norm(D,'fro')^2;
    
    if  normDsquared/t^2 < tol
        % if  abs(F(iter)-F(iter-1))/(abs(F(iter))+1)<tol
        break;
    end
    
    %% nonmonotone-linesearch
    if iter <= M+1
        Cval = max(F(1:iter-1));
    else
        Cval = max(F(iter-M:iter-1));
    end
    while F_trial>= Cval-0.5/t*alpha*normDsquared
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
        
        HZ= H*Z;
        %HZ = real(fft(bsxfun(@times,ifft( Z ),LAM_A)));
        F_trial =  -sum(sum(Z.*(HZ))) + h(Z);
    end
    Xk_1 = X;
    X = Z;
    HX = HZ;
    F(iter) = F_trial;
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
F_nls =  F(iter-1);
    
%     fprintf('NLS:Iter ***  Fval *** CPU  **** sparsity ***inner_inexact&averge_No. ** opt_norm ** total_linsea \n');
%     print_format = ' %i     %1.5e    %1.2f     %1.2f         %4i   %2.2f                %1.3e        %d \n';
%     fprintf(1,print_format, iter-1,min(F), time_nls,sparsity, inner_flag, sum(num_inner)/(iter-1) ,sqrt(normDsquared)/t,num_linesearch);

end
