function g = sigmoidFunction(x)

  g = zeros(size(x));

  % Instructions: z can be a matrix, vector or scalar
  g = 1.0 ./ ( 1.0 + exp(-x));
   
end