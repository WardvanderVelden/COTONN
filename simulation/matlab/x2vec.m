function vec = x2vec(x, ipd, eta, ll, ur, inputs)
    tmp = size(ipd);
    dim = tmp(2);
    
    vec = zeros(1, dim);

    % state-space to vec
    for i = 1:dim
        vec(i) = (x(i) - ll(i))/(ur(i) - ll(i));
    end
end