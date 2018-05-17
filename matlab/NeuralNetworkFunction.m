% function that describes the neural network
function input = NeuralNetwork(state_ID, number_of_layers,w1,w2, w3,b1,b2,b3)
w{1} = w1
w{2} = w2
w{3} = w3

b{1} = b1
b{2} = b2
b{3} = b3

output_layers{1} = state_ID

for i=1:number_of_layers
    input_layers{i} = output_layers{i}*w{i}+b{i};

    output_layers{i+1} = sigmoidFunction(input_layers{i});
end

input = output_layers{number_of_layers}
   
end



