clc
clear;
close all;
addpath ../misc
addpath ../SSN_subproblem


n_set = [128];  % [128;256;512;1024]
r_set = [4];  % [1;2;4;6;8]
mu_set = [1.5];  % [1.0;1.5;2.0;2.5;3.0]
% mu: sparsity parameter
Result = zeros(length(n_set)*length(r_set)*length(mu_set), 11);
index = 1;
for id_n = 1:length(n_set)
    n = n_set(id_n);
    fid = 1;
    for id_r = 1:length(r_set)
        r = r_set(id_r);
        for id_mu = 1:length(mu_set)
            mu = mu_set(id_mu);
            succ_no_manpg = 0;  succ_no_manpg_BB = 0;  succ_no_nls = 0;  succ_no_sub = 0;
            succ_no_pn = 0;
            diff_no_sub = 0;  fail_no_sub = 0;
            fprintf(fid, 'n *** r *** mu *** \n');
            print_format =  '%d  %d    %1.2f \n';
            fprintf(fid,print_format, n, r, mu);
       
            for test_random = 1:50
%                 fprintf(fid, 'test\n');
%                 test_random
                L = 50; dx = L/n;  V = 0;
                H = -Sch_matrix(0,L,n);
                rng('shuffle');  
                [phi_init,~] = svd(randn(n,r),0);  
                
                %% Riemannian subgradient method
                option_Rsub.F_manpg = -1e10;
                option_Rsub.phi_init = phi_init; option_Rsub.maxiter = n*r;  option_Rsub.tol = 5e-3;
                option_Rsub.r = r;    option_Rsub.n= n;  option_Rsub.mu=mu;  option_Rsub.type = 1;
                
                [phi_init, F_Rsub(test_random),sparsity_Rsub(test_random),time_Rsub(test_random),...
                    maxit_att_Rsub(test_random),succ_flag_sub]= Re_sub_grad_UFS(H,option_Rsub);

                %%  ManPG parameter update
                %option_manpg.adap = 0;
                option_manpg.phi_init = phi_init; option_manpg.maxiter = 30000;  option_manpg.tol =1e-8*n*r;
                option_manpg.r = r;    option_manpg.n = n;  option_manpg.mu = mu;
                option_manpg.L = L;
                %option_manpg.inner_tol =1e-11;
                option_manpg.inner_iter = 100;
                
                %%  ManPG algorithm
                [X_manpg, F_manpg(test_random),sparsity_manpg(test_random),time_manpg(test_random),...
                    maxit_att_manpg(test_random),succ_flag_manpg,lins(test_random),...
                    in_av(test_random)]= manpg_UFS(H,option_manpg,dx,V);
                if succ_flag_manpg == 1
                    succ_no_manpg = succ_no_manpg + 1;
                end

                %%  ManPG-Adap algorithm
                option_manpg.F_manpg = F_manpg(test_random);
                [X_manpg_BB, F_manpg_BB(test_random),sparsity_manpg_BB(test_random),time_manpg_BB(test_random),...
                    maxit_att_manpg_BB(test_random),succ_flag_manpg_BB,lins_adap(test_random),...
                    in_av_adap(test_random)]= manpg_UFS_adap(H,option_manpg,dx,V);
                if succ_flag_manpg_BB == 1
                    succ_no_manpg_BB = succ_no_manpg_BB + 1;
                end
                
                %%  ManPG-NLS algorithm with alternating BB stepsize
                M = 10;
                [X_nls, F_nls(test_random),sparsity_nls(test_random),time_nls(test_random),...
                    maxit_att_nls(test_random),succ_flag_nls,lins_nls(test_random),...
                    in_av_nls(test_random)]= manpg_nls_UFS(H,option_manpg,dx,V,M,1);
                % 
                if succ_flag_nls == 1
                    succ_no_nls = succ_no_nls + 1;
                end
                
                %%  ManPG-NLS algorithm with proximal newton method (approximated by diagonal matrix)
                M=10;
                [X_pn, F_pn(test_random),sparsity_pn(test_random),time_pn(test_random),...
                    maxit_att_pn(test_random),succ_flag_pn,lins_pn(test_random),...
                    in_av_pn(test_random)]= manpqn_UFS(H,option_manpg,dx,V,M,1);
                if succ_flag_pn == 1
                    succ_no_pn = succ_no_pn + 1;
                end

%                 %% Riemannian subgradient parameter update
%                 option_Rsub.F_manpg = F_manpg(test_random);
%                 option_Rsub.phi_init = phi_init; option_Rsub.maxiter = 1e1;  option_Rsub.tol = 1e-3;
%                 option_Rsub.r = r;    option_Rsub.n= n;  option_Rsub.mu=mu;  option_Rsub.type = 1;
%                 [X_Rsub, F_Rsub(test_random),sparsity_Rsub(test_random),time_Rsub(test_random),...
%                     maxit_att_Rsub(test_random),succ_flag_sub]= Re_sub_grad_UFS(H,option_Rsub);
%                 %phi_init = X_Rsub;
%                 if succ_flag_sub == 1
%                     succ_no_sub = succ_no_sub + 1;
%                 end
                
                F_best(test_random) =  F_manpg(test_random);
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
                    F_best(test_random) =  min(F_best(test_random), F_manpg_BB(test_random));
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

            time.manpg(id_n, id_r, id_mu) =  sum(time_manpg)/succ_no_manpg;
            time.manpg_BB(id_n, id_r, id_mu) =  sum(time_manpg_BB)/succ_no_manpg_BB;
            time.nls(id_n, id_r, id_mu) =  sum(time_nls)/succ_no_nls;   
            time.pn(id_n, id_r, id_mu) =  sum(time_pn)/succ_no_pn;

            Fval.manpg(id_n, id_r, id_mu) =  sum(F_manpg)/succ_no_manpg;
            Fval.manpg_BB(id_n, id_r, id_mu) =  sum(F_manpg_BB)/succ_no_manpg_BB;
            Fval.nls(id_n, id_r, id_mu) =  sum(F_nls)/succ_no_nls;   
            Fval.pn(id_n, id_r, id_mu) =  sum(F_pn)/succ_no_pn;  
            Fval.best(id_n, id_r, id_mu) = sum(F_best)/succ_no_manpg;

            Sp.manpg(id_n, id_r, id_mu) =  sum(sparsity_manpg)/succ_no_manpg;
            Sp.manpg_BB(id_n, id_r, id_mu) =  sum(sparsity_manpg_BB)/succ_no_manpg_BB;
            Sp.nls(id_n, id_r, id_mu) =  sum(sparsity_nls)/succ_no_nls;   
            Sp.pn(id_n, id_r, id_mu) =  sum(sparsity_pn)/succ_no_pn; 
            fprintf(fid,'==============================================================================================\n');

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



