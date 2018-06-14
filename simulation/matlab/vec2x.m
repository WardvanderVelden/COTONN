function x = vec2x(vec, ipd, eta, ll, ur, outputs)
    tmp = size(ipd);
    dim = tmp(2);
    
    x = [];
    for i = 1:dim
        x(i) = vec(i)*(ur(i) - ll(i)) + ll(i);
    end
    
    u = [];
    for i = 1:dim
        u(i) = [ll(i):eta(i):ur(i)];
    end 
    
    for i=1:dim
        for j=1:length(u(i))
            if abs(x(i) - u(j))<eta(i) 
                x(i) = u(j) ;
       end
    end
end





