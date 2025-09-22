clear;
clc;
close all;

%% Initialization

% Input vector
x1 = 0.1:1/22:1;

% Desired output
d = (1 + 0.6 * sin(2 * pi * x1 / 0.7) + 0.3 * sin(2 * pi * x1)) / 2;

% # of training samples
num_samples = length(x1);

% Extra stuff
learning_rate = 0.02;
iterations = 100000;


% Using negative numbers to break symmetry
% Hidden layer weights
wh11 = rand() - 0.5;
wh12 = rand() - 0.5;
wh13 = rand() - 0.5;
wh14 = rand() - 0.5;

% Hidden layer biases
bh1 = rand() - 0.5;
bh2 = rand() - 0.5;
bh3 = rand() - 0.5;
bh4 = rand() - 0.5;

% Output layer weights
wo11 = rand() - 0.5;
wo12 = rand() - 0.5;
wo13 = rand() - 0.5;
wo14 = rand() - 0.5;

% Output layer bias
bo1 = rand() - 0.5;

%% 2. Network training

%This just adds a progress bar for qol
fprintf('Training started...\n');
h_waitbar = waitbar(0, 'Starting Training...', 'Name', 'MLP Training Progress');

% Outer loop for iterations
for iteration = 1:iterations
    % Inner loop for each x input sample
    for k = 1:num_samples
        % Select the current input and desired output
        current_x = x1(k);
        current_d = d(k);

        % Feedforward
        % Input and activation for each hidden neuron
        % Activation function is hyperbolic tangent tanh(x)
        net_h1 = wh11 * current_x + bh1;
        yh1 = (exp(net_h1) - exp(-net_h1)) / (exp(net_h1) + exp(-net_h1));

        net_h2 = wh12 * current_x + bh2;
        yh2 = (exp(net_h2) - exp(-net_h2)) / (exp(net_h2) + exp(-net_h2));

        net_h3 = wh13 * current_x + bh3;
        yh3 = (exp(net_h3) - exp(-net_h3)) / (exp(net_h3) + exp(-net_h3));

        net_h4 = wh14 * current_x + bh4;
        yh4 = (exp(net_h4) - exp(-net_h4)) / (exp(net_h4) + exp(-net_h4));
        
        % Net input and final output for the output neuron
        % Activation function is linear
        net_o1 = wo11*yh1 + wo12*yh2 + wo13*yh3 + wo14*yh4 + bo1;
        yo1 = net_o1;

        % Backpropogation
    
        error = current_d - yo1;

        % Delta for the output layer neuron
        % Derivative of the linear activation function is 1
        delta_o1 = error * 1;

        % Deltas for the hidden layer neurons
        % Derivative of tanh(x) is 1 - tanh(x)^2
        delta_h1 = (delta_o1 * wo11) * (1 - yh1^2);
        delta_h2 = (delta_o1 * wo12) * (1 - yh2^2);
        delta_h3 = (delta_o1 * wo13) * (1 - yh3^2);
        delta_h4 = (delta_o1 * wo14) * (1 - yh4^2);

        % Update everything

        % Update output layer weights and bias
        wo11 = wo11 + learning_rate * delta_o1 * yh1;
        wo12 = wo12 + learning_rate * delta_o1 * yh2;
        wo13 = wo13 + learning_rate * delta_o1 * yh3;
        wo14 = wo14 + learning_rate * delta_o1 * yh4;
        bo1  = bo1  + learning_rate * delta_o1;

        % Update hidden layer weights and biases
        wh11 = wh11 + learning_rate * delta_h1 * current_x;
        wh12 = wh12 + learning_rate * delta_h2 * current_x;
        wh13 = wh13 + learning_rate * delta_h3 * current_x;
        wh14 = wh14 + learning_rate * delta_h4 * current_x;
        
        bh1 = bh1 + learning_rate * delta_h1;
        bh2 = bh2 + learning_rate * delta_h2;
        bh3 = bh3 + learning_rate * delta_h3;
        bh4 = bh4 + learning_rate * delta_h4;
    end
    if mod(iteration, 100) == 0 % Update every 100 iterations
        progress = iteration / iterations;
        waitbar(progress, h_waitbar, sprintf('Iteration: %d / %d (%.1f%%)', iteration, iterations, progress*100));
    end
end
delete(h_waitbar);
fprintf('Training finished!\n\n');

%% 3. Final result

yo_final = zeros(1, num_samples);
for k = 1:num_samples
    % Hidden Layer
    net_h1 = wh11 * x1(k) + bh1;
    yh1 = (exp(net_h1) - exp(-net_h1)) / (exp(net_h1) + exp(-net_h1));
    net_h2 = wh12 * x1(k) + bh2;
    yh2 = (exp(net_h2) - exp(-net_h2)) / (exp(net_h2) + exp(-net_h2));
    net_h3 = wh13 * x1(k) + bh3;
    yh3 = (exp(net_h3) - exp(-net_h3)) / (exp(net_h3) + exp(-net_h3));
    net_h4 = wh14 * x1(k) + bh4;
    yh4 = (exp(net_h4) - exp(-net_h4)) / (exp(net_h4) + exp(-net_h4));
    
    % Output Layer
    net_o1 = wo11*yh1 + wo12*yh2 + wo13*yh3 + wo14*yh4 + bo1;
    yo_final(k) = net_o1;
end

% Display Final Coefficients
disp('--- Final Trained Coefficients ---');

weightsHidden = [wh11, wh12, wh13, wh14];
disp('Hidden Layer Weights (input -> hidden):');
disp(weightsHidden);

biasesHidden = [bh1, bh2, bh3, bh4];
disp('Hidden Layer Biases:');
disp(biasesHidden);

weightsOutput = [wo11, wo12, wo13, wo14];
disp('Output Layer Weights (hidden -> output):');
disp(weightsOutput);

biasOutput = bo1;
disp('Output Layer Bias:');
disp(biasOutput);

% Plot the comparison graph
figure;
plot(x1, d, 'b-o', 'LineWidth', 2, 'MarkerFaceColor', 'b');
hold on;
plot(x1, yo_final, 'r--x', 'LineWidth', 1.5);
title('MLP Function Approximation');
xlabel('Input (x1)');
ylabel('Output');
legend('Desired Response', 'MLP Approximation');
grid on;
hold off;