function [ x_prox, delta ,Inact_set] = proximal_l21_diag(b, lambda, ...
    invdiaglist)
%proximal mapping of L_21 norm
[n,r] = size(b);  delta = cell(n,1);
% diag_list should be a column vector
if ~iscolumn(invdiaglist)
    invdiaglist=invdiaglist';
end
hlist = lambda.*invdiaglist;
nr = vecnorm(b,2,2); % compute the l2 norm of each row of matrix b
nr = max(nr, hlist);
a =  nr - hlist;
if r < 15
    Act_set = double( a > 0);
else
    Act_set = ( a > 0);
end
% return indexes of nonzeros elements of \max{\norm{b_i}_2-lambda, 0}

x_prox = Act_set.*(1- hlist./nr).*b;
for i = 1:n
    delta{i} = (eye(r) - hlist(i)./nr(i).*(eye(r) - b(i,:)'*b(i,:)./nr(i)))*Act_set(i);
end
% diag = 1 - lambda*ones(size(b))./nr + lambda.*b.^2./(nr.^3);
% diag = Act_set.*diag;
if nargout==3
    Inact_set= (a <= 0);
end

end

