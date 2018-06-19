function x = vec2x(vec, u_dim, eta, ll, ur, outputs)
    dim = u_dim;
    
    x = [];
    for i = 1:dim
        x(i) = vec(i)*(ur(i) - ll(i)) + ll(i);
    end
    
    for i = 1:dim
        u{i} = [ll(i):eta(i):ur(i)]';
    end 
    u = cell2mat(u);
    
    for i=1:dim
        for j=1:length(u(i))
            if abs(x(i) - u(i,j))<eta(i) 
                x(i) = u(i,j);
            end 
        end 
    end
end





