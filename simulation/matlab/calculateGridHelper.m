function ipd = calculateGridHelper(eta, ll, ur)
    tmp = size(eta);
    dim = tmp(2);

    ngp = zeros(1, dim);
    ipd = ones(1, dim);

    for i = 1:dim
        ngp(i) = (ur(i)-ll(i))/eta(i) + 1;
    
        if (i ~= 1)
            ipd(i) = ipd(i-1)*ngp(i-1); 
        end
    end
end

