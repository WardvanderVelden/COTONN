eta = [0.01 0.01];
lower_left = [0.5 5];
upper_right = [1.5 6];
dim = 2;

% // number of grid points:     (m_last[i]-m_first[i])/m_eta[i] + 1
% // ids per dimension:         ipd[0]=1; ipd[i]=ipd[i-1]*no_grid_points[i-1]
ngp = zeros(1, dim);
ipd = ones(1, dim);

for i = 1:dim
    ngp(i) = (upper_right(i)-lower_left(i))/eta(i) + 1;
    
    if (i ~= 1)
        ipd(i) = ipd(i-1)*ngp(i-1); 
    end
end

% const grid_point_t& x; // state space vector
% abs_type id = 0; // id
% double d_id; // helper variable
% double eta_h; // eta per dimension
% 
% // for every dimension
% for(int k=0; k<m_dim; k++) {
% 
%   d_id = x[k]-m_first[k]; // relative difference between point and the corner of the state space
% 
%   //id += static_cast<abs_type>((d_id+eta_h )/m_eta[k])*m_NN[k];
%   id += round((d_id+m_eta[k]/2.0)/m_eta[k])*ipd[k]
% }
% return id;

x = [0.5 5.0];
id = 0;
for i = 1:dim
   d_id = x(i) - lower_left(i);
   id = id + floor((d_id+eta(i)/2.0)/eta(i))*ipd(i);
end

id






