%function compare_spca

clc
clear;
close all;
addpath ../misc
addpath ../SSN_subproblem

% profile on

n_set = [1000];  % [100;200;500;800;1000;1500]; %dimension
r_set = [4]; % [1;2;4;6;8;10];   % rank

mu_set = [1.0];  % [0.55;0.6;0.65;0.7;0.75;0.8];
% M_set = [5;10;20;30];
index = 1;
M = 5;
m_lbfgs = 3;
exp_time = 1;
for id_n = 1:length(n_set)        % n  dimension
    
    n = n_set(id_n);
    fid =1;
    
    for id_r = 1:length(r_set) % r  number of column
        r = r_set(id_r);
        for id_mu = 1:length(mu_set)         % mu  sparse parameter
            mu = mu_set(id_mu);
            
            succ_no_manpg = 0;  succ_no_manpg_BB = 0; succ_no_nls = 0; 
            succ_no_pn = 0;  succ_no_sub = 0;  
            diff_no_sub = 0;  fail_no_sub = 0;
            
            fprintf(fid,'==============================================================================================\n');
            fprintf(fid,'- n -- r -- mu --------\n');
            fprintf(fid,'%4d %3d  %3.2f \n',n,r,mu);
%           fprintf(fid,'----------------------------------------------------------------------------------\n');

            for test_random = 1:50  %times average.
                rng('shuffle');
                %rng(70);
                m = 50;
                B = randn(m,n);
                type = 0; % random data matrix
                if (type == 1) %covariance matrix
                    scale = max(diag(B)); % Sigma=A/scale;
                elseif (type == 0) %data matrix
                    B = B - repmat(mean(B,1),m,1);
                    %                     scale = [];
                    %                     for id = 1:n
                    %                         scale = [scale norm(B(:,id))];
                    %                     end
                    %                     scale = max(scale);
                    %                     B = B/scale;
                    B = normc(B);
                    %  Sigma=A'*A;
                end
                %B(abs(B)<0.1) = 0;
%                 mu = mu_set(id_mu);
                
                rng('shuffle');
                %rng(177);
                [phi_init,~] = svd(randn(n,r),0);  % random intialization
                %[phi_init,~] = eigs(H,r);    % singular value initialization
                option_Rsub.F_manpg = -1e10;
                option_Rsub.phi_init = phi_init; option_Rsub.maxiter = 5e2;  option_Rsub.tol = 5e-3;
                option_Rsub.r = r;    option_Rsub.n= n;  option_Rsub.mu=mu;  option_Rsub.type = type;
                
                [phi_init, F_Rsub(test_random),sparsity_Rsub(test_random),time_Rsub(test_random),...
                    maxit_att_Rsub(test_random),succ_flag_sub]= Re_sub_grad_spca(B,option_Rsub);
                
                
                %%  manpg parameter
                option_manpg.adap = 0;    option_manpg.type =type;
                option_manpg.phi_init = phi_init; option_manpg.maxiter = 30000;  option_manpg.tol =1e-8*n*r;
                option_manpg.r = r;    option_manpg.n = n;  option_manpg.mu = mu;
                option_manpg.inner_iter = 100;

                %% ManPG
                [X_manpg, F_manpg(test_random),F_manpg_list, sparsity_manpg(test_random),time_manpg(test_random),...
                    maxit_att_manpg(test_random),succ_flag_manpg, lins(test_random),...
                    in_av(test_random)]= manpg_orth_sparse(B,option_manpg);
                if succ_flag_manpg == 1
                    succ_no_manpg = succ_no_manpg + 1;
                end
                
                %% ManPG-Ada
                option_manpg.F_manpg = F_manpg(test_random);
                [X_manpg_BB, F_manpg_BB(test_random),F_manpg_BB_list,sparsity_manpg_BB(test_random),time_manpg_BB(test_random),...
                    maxit_att_manpg_BB(test_random),succ_flag_manpg_BB,lins_adap(test_random),...
                    in_av_adap(test_random),ls_flag_adap(test_random)]= manpg_orth_sparse_adap(B,option_manpg);
                if succ_flag_manpg_BB == 1
                    succ_no_manpg_BB = succ_no_manpg_BB + 1;
                end
                
                %%  ManPG-NLS algorithm with alternating BB stepsize
                [X_nls, F_nls(test_random),F_nls_list,sparsity_nls(test_random),time_nls(test_random),...
                    maxit_att_nls(test_random),succ_flag_nls,lins_nls(test_random),...
                    in_av_nls(test_random)]= nls_orth_sparse(B,option_manpg,M,1);
                if succ_flag_nls == 1
                    succ_no_nls = succ_no_nls + 1;
                end

                %%  ManPQN algorithm with proximal newton method (approximated by diagonal matrix)
                [X_pn, F_pn(test_random),F_pn_list,sparsity_pn(test_random),time_pn(test_random),...
                    maxit_att_pn(test_random),succ_flag_pn,lins_pn(test_random),...
                    in_av_pn(test_random)]= manpqn_orth_sparse(B,option_manpg,M,1);
                if succ_flag_pn == 1
                    succ_no_pn = succ_no_pn + 1;
                end
                
                %% Riemannian subgradient parameter
                option_Rsub.F_manpg = F_manpg(test_random);
                option_Rsub.phi_init = phi_init; option_Rsub.maxiter = 1e1;  option_Rsub.tol = 5e-3;
                option_Rsub.r = r;    option_Rsub.n= n;  option_Rsub.mu=mu;  option_Rsub.type = type;
                
                [X_Rsub, F_Rsub(test_random),sparsity_Rsub(test_random),time_Rsub(test_random),...
                    maxit_att_Rsub(test_random),succ_flag_sub]= Re_sub_grad_spca(B,option_Rsub);
                %phi_init = X_Rsub;
                if succ_flag_sub == 1
                    succ_no_sub = succ_no_sub + 1;
                end
                
                if succ_flag_sub == 0
                    fail_no_sub = fail_no_sub + 1;
                end
                if succ_flag_sub == 2
                    diff_no_sub = diff_no_sub + 1;
                end
                if succ_flag_manpg == 1
                    F_best(test_random) =  F_manpg(test_random);
                end
                if succ_flag_manpg_BB == 1
                    F_best(test_random) =  min( F_best(test_random), F_manpg_BB(test_random));
                end
                if succ_flag_nls == 1
                    F_best(test_random) =  min(F_best(test_random), F_nls(test_random));
                end
                if succ_flag_pn == 1
                    F_best(test_random) =  min(F_best(test_random), F_pn(test_random));
                end

            end
            Result(index,1) = sum(lins)/succ_no_manpg;  Result(index,2) = sum(in_av)/succ_no_manpg;
            Result(index,3) = sum(lins_adap)/succ_no_manpg;  Result(index,4) = sum(in_av_adap)/succ_no_manpg;
            Result(index,5) = sum(lins_nls)/succ_no_nls;  Result(index,6) = sum(in_av_nls)/succ_no_nls;
            Result(index,7) = sum(lins_pn)/succ_no_pn;  Result(index,8) = sum(in_av_pn)/succ_no_pn;
            index = index +1;

            iter.manpg(id_n, id_r, id_mu) =  sum(maxit_att_manpg)/succ_no_manpg;
            iter.manpg_BB(id_n, id_r, id_mu) =  sum(maxit_att_manpg_BB)/succ_no_manpg_BB;
            iter.nls(id_n, id_r, id_mu) =  sum(maxit_att_nls)/succ_no_nls;  
            iter.pn(id_n, id_r, id_mu) =  sum(maxit_att_pn)/succ_no_pn; 
            iter.Rsub(id_n, id_r, id_mu) =  sum(maxit_att_Rsub)/succ_no_sub;

            time.manpg(id_n, id_r, id_mu) =  sum(time_manpg)/succ_no_manpg;
            time.manpg_BB(id_n, id_r, id_mu) =  sum(time_manpg_BB)/succ_no_manpg_BB;
            time.nls(id_n, id_r, id_mu) =  sum(time_nls)/succ_no_nls;  
            time.pn(id_n, id_r, id_mu) =  sum(time_pn)/succ_no_pn;
            time.Rsub(id_n, id_r, id_mu) =  sum(time_Rsub)/succ_no_sub;

            Fval.manpg(id_n, id_r, id_mu) =  sum(F_manpg)/succ_no_manpg;
            Fval.manpg_BB(id_n, id_r, id_mu) =  sum(F_manpg_BB)/succ_no_manpg_BB;
            Fval.nls(id_n, id_r, id_mu) =  sum(F_nls)/succ_no_nls;   
            Fval.pn(id_n, id_r, id_mu) =  sum(F_pn)/succ_no_pn;  
            Fval.Rsub(id_n, id_r, id_mu) =  sum(F_Rsub)/succ_no_sub;
            Fval.best(id_n, id_r, id_mu) = sum(F_best)/succ_no_manpg;

            Sp.manpg(id_n, id_r, id_mu) =  sum(sparsity_manpg)/succ_no_manpg;
            Sp.manpg_BB(id_n, id_r, id_mu) =  sum(sparsity_manpg_BB)/succ_no_manpg_BB;
            Sp.nls(id_n, id_r, id_mu) =  sum(sparsity_nls)/succ_no_nls; 
            Sp.pn(id_n, id_r, id_mu) =  sum(sparsity_pn)/succ_no_pn; 
            Sp.Rsub(id_n, id_r, id_mu) =  sum(sparsity_Rsub)/succ_no_sub;

            fprintf(fid,' Alg ****        Iter *****  Fval ******* sparsity ***** cpu *** line-search *** SSN *****\n');

            print_format =  'ManPG       &   %.2f  & %1.5e  &    %1.2f  &   %3.4f   &   %4.2f   &   %.2f  \\\\ \n';
            fprintf(fid, print_format, iter.manpg(id_n, id_r, id_mu), ...
                Fval.manpg(id_n, id_r, id_mu), Sp.manpg(id_n, id_r, id_mu),...
                time.manpg(id_n, id_r, id_mu), Result(index-1,1), Result(index-1,2));
            print_format =  'ManPG-adap  &   %.2f  & %1.5e  &    %1.2f  &   %3.4f   &   %4.2f   &   %.2f  \\\\ \n';
            fprintf(fid, print_format, iter.manpg_BB(id_n, id_r, id_mu), ...
                Fval.manpg_BB(id_n, id_r, id_mu), Sp.manpg_BB(id_n, id_r, id_mu),...
                time.manpg_BB(id_n, id_r, id_mu), Result(index-1,3), Result(index-1,4));
            print_format =  'NLS-ManPG   &   %.2f  & %1.5e  &    %1.2f  &   %3.4f   &   %4.2f   &   %.2f  \\\\ \n';
            fprintf(fid, print_format, iter.nls(id_n, id_r, id_mu), ...
                Fval.nls(id_n, id_r, id_mu), Sp.nls(id_n, id_r, id_mu), ...
                time.nls(id_n, id_r, id_mu), Result(index-1,5), Result(index-1,6));
            print_format =  'ManPQN      &   %.2f  & %1.5e  &    %1.2f  &   %3.4f   &   %4.2f   &   %.2f  \\\\ \n';
            fprintf(fid, print_format, iter.pn(id_n, id_r, id_mu), ...
                Fval.pn(id_n, id_r, id_mu), Sp.pn(id_n, id_r, id_mu), ...
                time.pn(id_n, id_r, id_mu), Result(index-1,7), Result(index-1,8));

        end
        
    end
end

% profile viewer



