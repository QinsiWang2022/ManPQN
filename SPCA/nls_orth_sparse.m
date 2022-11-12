function [X_nls,F_nls,F,sparsity,time_nls,iter,flag_succ,num_linesearch,mean_ssn] = nls_orth_sparse(B,option,M,bb_mod)
%min -Tr(X'*B*X)+ mu*norm(X,1) s.t. X'*X=Ir. X \in R^{p*r}
% mu can be a vector with weighted parameter
%parameters
tic;
r = option.r;%number of col
n = option.n;%dim
mu = option.mu;
maxiter = option.maxiter;
tol = option.tol;
h = @(X) sum(mu.*sum(abs(X)));
inner_iter = option.inner_iter;
prox_fun = @(b,lambda,r) proximal_l1(b,lambda,r);
inner_flag = 0;
%setduplicat_pduplicat(r);
Dn = sparse(DuplicationM(r));
pDn = (Dn'*Dn)\Dn';
type = option.type;
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

if type == 1
    AX = B*X;
else
    AX = B'*(B*X);
end
F(1) = -sum(sum(X.*(AX)))+h(X);
num_inner = zeros(maxiter,1);
opt_sub = num_inner;
num_linesearch = 0;
num_inexact = 0;
alpha =1;
t = 1/L; t0=1/L;
%inner_tol  = 0.1*tol^2*t^2;
linesearch_flag = 0;
% ls_flag_sum = 0;
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
    
    if iter == 3
        gk = -2*AX + 2*(X*(X'*AX));
    elseif iter > 3
        gk_1 = gk;
        gk = -2*AX + 2*(X*(X'*AX));
        sk_1 = X-Xk_1;
        yk_1 = gk-gk_1;
        if bb_mod == 2
            if mod(iter,2) == 1
                t_bb = trace(sk_1'*sk_1)/abs(trace(sk_1'*yk_1));
            else
                t_bb = trace(yk_1'*sk_1)/abs(trace(yk_1'*yk_1));
            end
        elseif bb_mod == 1
            t_bb = trace(sk_1'*sk_1)/abs(trace(sk_1'*yk_1));
        else 
            t_bb = trace(yk_1'*sk_1)/abs(trace(yk_1'*yk_1));
        end
        t = max(t0, t_bb);
    end
    
%     display(t)
    
    if iter == 2
         [ PY,num_inner(iter),Lam, opt_sub(iter),in_flag] = Semi_newton_matrix(n,r,X,t,X + t*neg_pgx,mu*t,inner_tol,prox_fun,inner_iter,zeros(r),Dn,pDn);
        %      [ PY,num2(iter),r_norm(iter)]=fista(X,pgx,mu,t);
    else
         [ PY,num_inner(iter),Lam, opt_sub(iter),in_flag] = Semi_newton_matrix(n,r,X,t,X + t*neg_pgx,mu*t,inner_tol,prox_fun,inner_iter,Lam,Dn,pDn);
        %     [ PY,num2(iter),r_norm(iter)]=fista(X,pgx,mu,t);
    end

    if in_flag == 1   % subprolem not exact.
        inner_flag = 1 + inner_flag;
    end
    alpha = 1;
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
    if  normDsquared/t^2 < tol  
        % if  abs(F(iter)-F(iter-1))/(abs(F(iter))+1)<tol
        break;
    end
%     if iter == maxiter
%         flag_maxiter =1;
%     end
    
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
    
%     fprintf('ManPG:Iter ***  Fval *** CPU  **** sparsity ***inner_inexact&averge_No. ** opt_norm ** total_linsea \n');
%     print_format = ' %i     %1.5e    %1.2f     %1.2f         %4i   %2.2f                %1.3e        %d \n';
%     fprintf(1,print_format, iter-1,min(F), time_manpg,sparsity, inner_flag, sum(num_inner)/(iter-1) ,sqrt(normDsquared)/t,num_linesearch);
end