# ManPQN
documentation by:
Qinsi Wang
qinsiwang20@fudan.edu.cn
Fudan University
2022

----------------------------------

This package contains the code used in the paper "Proximal Quasi-Newton Method for Composite Optimization over the Stiefel Manifold" (see ManPQN.pdf).

Our algorithm ManPQN is tested and compared with three proximal gradient related algorithms ManPG, ManPG-Ada and NLS-ManPG on four experiments, including compressed modes problem, sparse PCA (principal component analysis) problem, unsupervised feature selection and joint diagonalization problem.

This code has been tested to run in MATLAB R2018b.

If you use this code in an academic paper, please cite our paper:
Qinsi Wang and Wei Hong Yang. Proximal Quasi-Newton Method for Composite Optimization over the Stiefel Manifold.  

If you have any questions, please let me know by <qinsiwang20@fudan.edu.cn>. I would appreciate any suggestions regarding the algorithms.

----------------------------------

- CM/demo_CMS: compares 4 algorithms ManPG, ManPG-Ada, NLS-ManPGC and ManPQN for solving compressed modes problem with orthogonal constraint;

- SPCA/demo_compare_SPCA: compares 4 algorithms for solving sparse PCA problem with orthogonal constraint and randomly generated coefficient matrix;

- SPCA/demo_compare_SPCA_UF_matrices: compares 4 algorithms for solving sparse PCA problem with orthogonal constraint and coefficient matrix chosen from "UF Sparse Matrix Collection";

- UFS/demo_UFS: compares 4 algorithms for solving unsupervised feature selection problem with orthogonal constraint;

- JointDiagonalization/demo_JD: compares 4 algorithms for solving joint diagonalization problem with orthogonal constraint;

- SSN_subproblem: conjugate gradient method and semismooth Newton method;

- misc: proximal mapping, duplication matrices.

