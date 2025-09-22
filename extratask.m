% --- MLP Surface Approximator (2 Inputs) ---
% This program trains a multilayer perceptron to approximate a 3D surface.
%
% Network Structure:
% - Input Layer: 2 neurons (x1, x2)
% - Hidden Layer: 12 neurons (hyperbolic tangent activation)
% - Output Layer: 1 neuron (linear activation)

clear;
clc;
close all;

%% 1. Problem Definition and Network Initialization

% Define the input range and create a grid
step = 0.4;
[x1_grid, x2_grid] = meshgrid(-8:step:8, -8:step:8);

% Calculate the desired response 'd' (Sombrero function)
r = sqrt(x1_grid.^2 + x2_grid.^2) + eps; % Add eps to avoid division by zero
d_grid = sin(r) ./ r;

% Reshape the grid data into column vectors for training
x1_vec = reshape(x1_grid, [], 1);
x2_vec = reshape(x2_grid, [], 1);
d_vec = reshape(d_grid, [], 1);

num_samples = length(x1_vec);

% --- Hyperparameters ---
learning_rate = 0.01;
epochs = 10000 ; % Note: Full training can take thousands

% --- Initialize Weights and Biases for a 2-Input, 12-Neuron Hidden Layer ---
% Weights from Input 1 (x1) to Hidden Neurons
wh11=rand-0.5; wh12=rand-0.5; wh13=rand-0.5; wh14=rand-0.5;
wh15=rand-0.5; wh16=rand-0.5; wh17=rand-0.5; wh18=rand-0.5;
wh19=rand-0.5; wh110=rand-0.5; wh111=rand-0.5; wh112=rand-0.5;

% Weights from Input 2 (x2) to Hidden Neurons
wh21=rand-0.5; wh22=rand-0.5; wh23=rand-0.5; wh24=rand-0.5;
wh25=rand-0.5; wh26=rand-0.5; wh27=rand-0.5; wh28=rand-0.5;
wh29=rand-0.5; wh210=rand-0.5; wh211=rand-0.5; wh212=rand-0.5;

% Biases for Hidden Neurons
bh1=rand-0.5; bh2=rand-0.5; bh3=rand-0.5; bh4=rand-0.5;
bh5=rand-0.5; bh6=rand-0.5; bh7=rand-0.5; bh8=rand-0.5;
bh9=rand-0.5; bh10=rand-0.5; bh11=rand-0.5; bh12=rand-0.5;

% Weights from Hidden Layer to Output Neuron
wo1=rand-0.5; wo2=rand-0.5; wo3=rand-0.5; wo4=rand-0.5;
wo5=rand-0.5; wo6=rand-0.5; wo7=rand-0.5; wo8=rand-0.5;
wo9=rand-0.5; wo10=rand-0.5; wo11=rand-0.5; wo12=rand-0.5;

% Bias for Output Neuron
bo1 = rand() - 0.5;

%% 2. Training the Network using Backpropagation

fprintf('Training started...\n');
h_waitbar = waitbar(0, 'Starting Training...', 'Name', 'MLP Training Progress');

for epoch = 1:epochs
    for k = 1:num_samples
        current_x1 = x1_vec(k);
        current_x2 = x2_vec(k);
        current_d = d_vec(k);

        % --- a) Feedforward Pass ---
        % Net input now includes contributions from BOTH inputs
        net_h1=wh11*current_x1+wh21*current_x2+bh1; yh1=tanh(net_h1);
        net_h2=wh12*current_x1+wh22*current_x2+bh2; yh2=tanh(net_h2);
        net_h3=wh13*current_x1+wh23*current_x2+bh3; yh3=tanh(net_h3);
        net_h4=wh14*current_x1+wh24*current_x2+bh4; yh4=tanh(net_h4);
        net_h5=wh15*current_x1+wh25*current_x2+bh5; yh5=tanh(net_h5);
        net_h6=wh16*current_x1+wh26*current_x2+bh6; yh6=tanh(net_h6);
        net_h7=wh17*current_x1+wh27*current_x2+bh7; yh7=tanh(net_h7);
        net_h8=wh18*current_x1+wh28*current_x2+bh8; yh8=tanh(net_h8);
        net_h9=wh19*current_x1+wh29*current_x2+bh9; yh9=tanh(net_h9);
        net_h10=wh110*current_x1+wh210*current_x2+bh10; yh10=tanh(net_h10);
        net_h11=wh111*current_x1+wh211*current_x2+bh11; yh11=tanh(net_h11);
        net_h12=wh112*current_x1+wh212*current_x2+bh12; yh12=tanh(net_h12);
        
        % Output neuron sums activations from all 12 hidden neurons
        net_o1 = wo1*yh1+wo2*yh2+wo3*yh3+wo4*yh4+wo5*yh5+wo6*yh6+wo7*yh7+wo8*yh8+wo9*yh9+wo10*yh10+wo11*yh11+wo12*yh12+bo1;
        yo1 = net_o1;

        % --- b) Backpropagation of Error ---
        error = current_d - yo1;
        delta_o1 = error;
        
        delta_h1=(delta_o1*wo1)*(1-yh1^2); delta_h2=(delta_o1*wo2)*(1-yh2^2);
        delta_h3=(delta_o1*wo3)*(1-yh3^2); delta_h4=(delta_o1*wo4)*(1-yh4^2);
        delta_h5=(delta_o1*wo5)*(1-yh5^2); delta_h6=(delta_o1*wo6)*(1-yh6^2);
        delta_h7=(delta_o1*wo7)*(1-yh7^2); delta_h8=(delta_o1*wo8)*(1-yh8^2);
        delta_h9=(delta_o1*wo9)*(1-yh9^2); delta_h10=(delta_o1*wo10)*(1-yh10^2);
        delta_h11=(delta_o1*wo11)*(1-yh11^2); delta_h12=(delta_o1*wo12)*(1-yh12^2);

        % --- c) Update Weights and Biases ---
        % Output weights and bias
        wo1=wo1+learning_rate*delta_o1*yh1; wo2=wo2+learning_rate*delta_o1*yh2;
        wo3=wo3+learning_rate*delta_o1*yh3; wo4=wo4+learning_rate*delta_o1*yh4;
        wo5=wo5+learning_rate*delta_o1*yh5; wo6=wo6+learning_rate*delta_o1*yh6;
        wo7=wo7+learning_rate*delta_o1*yh7; wo8=wo8+learning_rate*delta_o1*yh8;
        wo9=wo9+learning_rate*delta_o1*yh9; wo10=wo10+learning_rate*delta_o1*yh10;
        wo11=wo11+learning_rate*delta_o1*yh11; wo12=wo12+learning_rate*delta_o1*yh12;
        bo1 = bo1 + learning_rate * delta_o1;

        % Hidden layer weights (each weight update depends on its corresponding input)
        wh11=wh11+learning_rate*delta_h1*current_x1; wh21=wh21+learning_rate*delta_h1*current_x2;
        wh12=wh12+learning_rate*delta_h2*current_x1; wh22=wh22+learning_rate*delta_h2*current_x2;
        wh13=wh13+learning_rate*delta_h3*current_x1; wh23=wh23+learning_rate*delta_h3*current_x2;
        wh14=wh14+learning_rate*delta_h4*current_x1; wh24=wh24+learning_rate*delta_h4*current_x2;
        wh15=wh15+learning_rate*delta_h5*current_x1; wh25=wh25+learning_rate*delta_h5*current_x2;
        wh16=wh16+learning_rate*delta_h6*current_x1; wh26=wh26+learning_rate*delta_h6*current_x2;
        wh17=wh17+learning_rate*delta_h7*current_x1; wh27=wh27+learning_rate*delta_h7*current_x2;
        wh18=wh18+learning_rate*delta_h8*current_x1; wh28=wh28+learning_rate*delta_h8*current_x2;
        wh19=wh19+learning_rate*delta_h9*current_x1; wh29=wh29+learning_rate*delta_h9*current_x2;
        wh110=wh110+learning_rate*delta_h10*current_x1; wh210=wh210+learning_rate*delta_h10*current_x2;
        wh111=wh111+learning_rate*delta_h11*current_x1; wh211=wh211+learning_rate*delta_h11*current_x2;
        wh112=wh112+learning_rate*delta_h12*current_x1; wh212=wh212+learning_rate*delta_h12*current_x2;
        
        % Hidden layer biases
        bh1=bh1+learning_rate*delta_h1; bh2=bh2+learning_rate*delta_h2; bh3=bh3+learning_rate*delta_h3;
        bh4=bh4+learning_rate*delta_h4; bh5=bh5+learning_rate*delta_h5; bh6=bh6+learning_rate*delta_h6;
        bh7=bh7+learning_rate*delta_h7; bh8=bh8+learning_rate*delta_h8; bh9=bh9+learning_rate*delta_h9;
        bh10=bh10+learning_rate*delta_h10; bh11=bh11+learning_rate*delta_h11; bh12=bh12+learning_rate*delta_h12;
    end
    
    if mod(epoch, 5) == 0
        progress = epoch / epochs;
        waitbar(progress, h_waitbar, sprintf('Epoch: %d / %d', epoch, epochs));
    end
end
delete(h_waitbar);
fprintf('Training finished!\n\n');

%% 3. Final Results and Visualization

% Create an empty grid for the final output
yo_grid = zeros(size(x1_grid));

% Perform a final feedforward pass over the entire grid
for row = 1:size(x1_grid, 1)
    for col = 1:size(x1_grid, 2)
        current_x1 = x1_grid(row, col);
        current_x2 = x2_grid(row, col);
        
        net_h1=wh11*current_x1+wh21*current_x2+bh1; yh1=tanh(net_h1);
        net_h2=wh12*current_x1+wh22*current_x2+bh2; yh2=tanh(net_h2);
        net_h3=wh13*current_x1+wh23*current_x2+bh3; yh3=tanh(net_h3);
        net_h4=wh14*current_x1+wh24*current_x2+bh4; yh4=tanh(net_h4);
        net_h5=wh15*current_x1+wh25*current_x2+bh5; yh5=tanh(net_h5);
        net_h6=wh16*current_x1+wh26*current_x2+bh6; yh6=tanh(net_h6);
        net_h7=wh17*current_x1+wh27*current_x2+bh7; yh7=tanh(net_h7);
        net_h8=wh18*current_x1+wh28*current_x2+bh8; yh8=tanh(net_h8);
        net_h9=wh19*current_x1+wh29*current_x2+bh9; yh9=tanh(net_h9);
        net_h10=wh110*current_x1+wh210*current_x2+bh10; yh10=tanh(net_h10);
        net_h11=wh111*current_x1+wh211*current_x2+bh11; yh11=tanh(net_h11);
        net_h12=wh112*current_x1+wh212*current_x2+bh12; yh12=tanh(net_h12);
        
        net_o1 = wo1*yh1+wo2*yh2+wo3*yh3+wo4*yh4+wo5*yh5+wo6*yh6+wo7*yh7+wo8*yh8+wo9*yh9+wo10*yh10+wo11*yh11+wo12*yh12+bo1;
        yo_grid(row, col) = net_o1;
    end
end

% --- Plot the comparison surfaces ---
figure;
sgtitle('MLP Surface Approximation');

% Plot the original function
subplot(1, 2, 1);
surf(x1_grid, x2_grid, d_grid);
title('Original Function ("Sombrero")');
xlabel('x1');
ylabel('x2');
zlabel('d');

% Plot the MLP's approximation
subplot(1, 2, 2);
surf(x1_grid, x2_grid, yo_grid);
title('MLP Approximation');
xlabel('x1');
ylabel('x2');
zlabel('yo');