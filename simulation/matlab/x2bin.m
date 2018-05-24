function bin = x2bin(x, ipd, eta, ll, inputs)
    tmp = size(ipd);
    dim = tmp(2);

    % space to id
    id = 0;
    for i = 1:dim
       d_id = x(i) - ll(i);
       id = id + floor((d_id+eta(i)/2.0)/eta(i))*ipd(i);
    end
    
    % id to binary
    raw_bin = dec2bin(id, inputs);
    bin = zeros(1, inputs);

    for i=1:inputs
        bin(i) = str2num(raw_bin(i));
    end 
end