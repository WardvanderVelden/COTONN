% function that describes the neural network
function input = neuralNetwork(state, W, b)
    % get number of layers
    shape = size(W);
    layers = shape(2);

    input = state;
    for i=1:layers
        input = sigmoid(input*W{i}+b{i}');
    end
    
    input = round(input);
end



