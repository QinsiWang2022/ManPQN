function [X_nls, F_nls,F,sparsity,time_nls,iter,flag_succ,num_linesearch,mean_ssn] = manpqn_orth_sparse(B,option,M,bb_mod)
%min -Tr(X'*B*X)+ mu*norm(X,1) s.t. X'*X=Ir. X \in R^{p*r}
% mu can be a vector with weighted parameter
%parameters:
%   B: parameter matrix for sparse PCA problem
%   option.phi_init: initial iterative point
%   option.maxiter: maximal ietration number
%   option.tol: tolerance for stopping criterion
%   option.r, option.n, option.mu
%   option.inner_iter: maximal iteration number for SSN methods in the
%   inner loop
%   M：parameter for nonmonotone line search
%   bb_mod: the way computing BB stepsize strategy
%   retrac_mod: the way of choosing retraction

tic;
% profile on
r = option.r;%number of col
n = option.n;%dim
mu = option.mu;

maxiter = option.maxiter;
F = zeros(maxiter, 1);

tol = option.tol;
h = @(X) sum(mu.*sum(abs(X)));
inner_iter = option.inner_iter;
% prox_fun = @(b,lambda,r) proximal_l1(b,lambda,r);
inner_flag = 0;
%setduplicat_pduplicat(r);
Dn = sparse(DuplicationM(r));
pDn = (Dn'*Dn)\Dn';
type = option.type; % type=0
t_min = 1e-4; % minimum stepsize
%%
%initial point
X = option.phi_init;
if type == 1
    L = 2*abs(eigs(full(B),1));
    %  L=2*abs(eigs(B,1));
else
    L = 2*(svds(full(B),1))^2;
end

% if mu > L/2
%     fprintf('Too large penalty parameter mu, trivial solution\n');
%     X_manpg =
%     return;
% end

% % set of preconditioner
if type == 1
    AX = B*X;
else
    AX = B'*(B*X);
end

t = 1/L; t0=1/L;
hk = ones(n, 1);

prox_fun_diag = @(b,lambda,r,h) proximal_l1_diag(b,lambda,r,h);
prox_fun = @(b,lambda,r) proximal_l1(b,lambda,r);

F(1) = -sum(sum(X.*(AX)))+h(X);
num_inner = zeros(maxiter,1);
opt_sub = num_inner;
num_linesearch = 0;
num_inexact = 0;
alpha =1;
% t = 1/L; t0=1/L;
para_step = 1.01;
%inner_tol  = 0.1*tol^2*t^2;
linesearch_flag = 0;
ls_flag_sum = 0;
pg_flag = 0;
for iter = 2:maxiter
    %   fprintf('manpg__________________iter: %3d, \n', iter);
    ngx = 2*AX; % negative gradient       pgx=gx-X*xgx;  %projected gradient
    neg_pgx = ngx; % grad or projected gradient both okay
    %% subproblem
%     if alpha < t_min || num_inexact > 10
%         inner_tol = max(5e-16, min(1e-14,1e-5*tol*t^2)); % subproblem inexact;
%     else
%         inner_tol = max(1e-13, min(1e-11,1e-3*tol*t^2));
%     end

    if alpha < t_min || num_inexact > 10
        inner_tol = max(1e-15, min(1e-13,1e-5*tol*t^2)); % subproblem inexact;
    else
        inner_tol = max(1e-13, min(1e-11,1e-3*tol*t^2));
    end
    
    if linesearch_flag == 0
        t = t*para_step;
    else
        t = max(t0,t/para_step);
        ls_flag_sum = ls_flag_sum + 1;
    end
    linesearch_flag = 0;


    if pg_flag == 0
        if iter <= 3
            gk = -2*AX + 2*(X*(X'*AX));
        elseif iter > 3
            gk_1 = gk;
            gk = -2*AX + 2*(X*(X'*AX));
            sk_1 = X-Xk_1;
            yk_1 = gk-gk_1;
            rho_k = abs(sum(sum(sk_1.*yk_1)));
            if bb_mod == 2
                % 分奇偶步交替生成BB步长
                if mod(iter,2) == 1
                    t_bb = trace(sk_1'*sk_1)/abs(trace(sk_1'*yk_1));
                else
                    t_bb = trace(yk_1'*sk_1)/abs(trace(yk_1'*yk_1));
                end
            elseif bb_mod == 1
                % 仅使用一种形式的BB步长
                t_bb = trace(sk_1'*sk_1)/abs(trace(sk_1'*yk_1));
            else 
                t_bb = trace(yk_1'*sk_1)/abs(trace(yk_1'*yk_1));
            end
            % 对BB步长t取下界t0进行截断，避免步长太小而使得迭代停滞不前
            t = max(t0, t_bb);
        end
        if iter == 4
            h0 = rho_k/(norm(yk_1,'fro'));
            rho_k = 1/rho_k;
            Hk = h0 * eye(n) + (rho_k-1/rho_k) * (yk_1*yk_1');
            hk = diag(Hk);
        elseif iter > 4
            if n*r <= 600
                Hk = hk_update_mat(yk_1,sk_1,Hk,rho_k);
                hk = diag(Hk);
            else
                hk = hk_update_diag(yk_1,sk_1,hk,rho_k);
            end
        end
        hk = abs(hk);
        hk = hk/max(hk);
        hk(hk<1e-4)=1e-4;
        invDiag = hk;
        
        if iter == 2
             [ PY,num_inner(iter),Lam, opt_sub(iter),...
                 in_flag] = Semi_newton_matrix_prox_diag(n,r,X,invDiag,t,...
                 X + t*invDiag.*neg_pgx,...
                 mu*t,inner_tol,prox_fun_diag,inner_iter,zeros(r),Dn,pDn);
        else
             [ PY,num_inner(iter),Lam, opt_sub(iter),...
                 in_flag] = Semi_newton_matrix_prox_diag(n,r,X,invDiag,t,...
                 X + t*invDiag.*neg_pgx,...
                 mu*t,inner_tol,prox_fun_diag,inner_iter,Lam,Dn,pDn);
        end
    else
        if iter == 2
            [ PY,num_inner(iter),Lam, opt_sub(iter),...
                in_flag] = Semi_newton_matrix(n,r,X,t,X + t*neg_pgx,...
                mu*t,inner_tol,prox_fun,inner_iter,zeros(r),Dn,pDn);
        elseif iter == 3
            gk = -2*AX + 2*(X*(X'*AX));
        elseif iter > 3
            gk_1 = gk;
            gk = -2*AX + 2*(X*(X'*AX));
            sk_1 = X-Xk_1;
            yk_1 = gk-gk_1;
            if bb_mod == 2
                % 分奇偶步交替生成BB步长
                if mod(iter,2) == 1
                    t_bb = trace(sk_1'*sk_1)/abs(trace(sk_1'*yk_1));
                else
                    t_bb = trace(yk_1'*sk_1)/abs(trace(yk_1'*yk_1));
                end
            elseif bb_mod == 1
                % 仅使用一种形式的BB步长
                t_bb = trace(sk_1'*sk_1)/abs(trace(sk_1'*yk_1));
            else 
                t_bb = trace(yk_1'*sk_1)/abs(trace(yk_1'*yk_1));
            end
            % 对BB步长t取下界t0进行截断，避免步长太小而使得迭代停滞不前
            t = max(t0, t_bb);
            [ PY,num_inner(iter),Lam, opt_sub(iter),...
                in_flag] = Semi_newton_matrix(n,r,X,t,X + t*neg_pgx,...
                mu*t,inner_tol,prox_fun,inner_iter,Lam,Dn,pDn);
        end
    end

    if in_flag == 1   % subprolem not exact.
        inner_flag = 1 + inner_flag;
    end
    alpha = 1;
    if nnz(isnan(PY))~=0||nnz(isinf(PY))~=0
       	[ PY, num_inner(iter), Lam, opt_sub(iter), ...
            in_flag]=Semi_newton_matrix(n,r,X,t,X+t*neg_pgx,mu*t,...
            inner_tol,prox_fun,inner_iter,Lam,Dn,pDn);
    end
    D = PY-X; %descent direction D
    
%     [U, ~, S] = svd(PY,'econ');      
%     Z = U*S';

    [U, SIGMA, S] = svd(PY'*PY);   
    SIGMA =diag(SIGMA);    
    Z = PY*(U*diag(sqrt(1./SIGMA))*S');
    % [Z,R]=qr(PY,0);       Z = Z*diag(sign(diag(R))); %old version need consider the sign
    
    if type == 1
        AZ = B*Z;
    else
        AZ = B'*(B*Z);
    end
    %   AZ = real(ifft( LAM_manpg.*fft(Z) ));
    
    f_trial = -sum(sum(Z.*(AZ)));
    F_trial = f_trial+h(Z);   normDsquared=norm(D,'fro')^2;
    
    %% nonmonotone-linesearch
    if iter <= M+1
        Cval = max(F(1:iter-1));
    else
        Cval = max(F(iter-M:iter-1));
    end
    while F_trial >= Cval-0.5/t*alpha*normDsquared
        alpha = 0.5*alpha;
        linesearch_flag = 1;
        num_linesearch = num_linesearch+1;
        if alpha<t_min
            num_inexact = num_inexact + 1;
            break;
        end
        PY = X+alpha*D;
        %  [U,~,V]=svd(PY,0);  Z=U*V';
        %  [Z,R]=qr(PY,0);   Z = Z*diag(sign(diag(R)));  %old version need consider the sign
%         [U, ~, S] = svd(PY,'econ');      
%         Z = U*S';
        [U, SIGMA, S] = svd(PY'*PY);   
        SIGMA =diag(SIGMA);   
        Z = PY*(U*diag(sqrt(1./SIGMA))*S');
        if type ==1
            AZ= B*Z;
        else
            AZ = B'*(B*Z);
        end
        %  flag_linesearch(iter) = 1+flag_linesearch(iter); % linesearch flag
        f_trial = -sum(sum(Z.*(AZ)));     F_trial = f_trial+ h(Z);
    end
    Xk_1 = X;
    X = Z; AX = AZ;
    F(iter) = F_trial;
    if  pg_flag==0 && normDsquared/(t^2) < tol
        pg_flag = 1;
    end
    if F_trial < option.F_manpg+1e-7 || normDsquared/(t^2) < tol
        break;
    end
    if iter == maxiter
        flag_maxiter =1;
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
F = F(1:iter);
% profile viewer
end

function ak = quad_itp(a1,a2,p1,p2,dp1)
a1_a2 = a1-a2;
ak = a1-(1/2)*((a1_a2*dp1)/(dp1-((p1-p2)/a1_a2)));
end

function mat_k = hk_update_mat(y,s,H,rho)
rho = 1/rho;
v = y'*H;
mat_k = H - (v'*v)/abs(trace(v*y)) + rho*(s*s');
end

function diag_k = hk_update_diag(y,s,h,rho)
rho = 1/rho;
v = y.*h;
diag_k = h - (sum(y.^2,2))/abs(sum(sum(v.*y))) + rho*(sum(s.^2,2));
end