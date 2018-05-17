function [x, id] = x2bin(bin, ipd, eta, ll, outputs)
    tmp = size(ipd);
    dim = tmp(2);
    
    % binary to id
    raw_bin = "";
    for i=1:outputs
        raw_bin = raw_bin + num2str(bin(i));
    end
    id = bin2dec(raw_bin);

    % id to space
    i = dim;
    x = zeros(1, dim);
    while (i>1)
        num = floor(id/ipd(i));
        id = mod(id, ipd(i));
        x(i) = ll(i)+num*eta(i);
        
        i = i - 1;
    end
    num = id;
    x(1) = ll(i)+num*eta(1);
end