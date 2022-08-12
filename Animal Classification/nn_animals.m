% NN Model Training
%% clear
% Initialization
clear all; close all;
%% load dataset
load('data.mat');

%% Normalization
%X1 = mapminmax(X, 0, 1);

%% PCA
%X2=X';
%[COEFF,SCORE,latent] = pca(X2);
%total_sum=0;

%for i= 1:length(latent)
%total_sum=total_sum+latent(i);
%end
%sum_k=0;i=1;
%while ((sum_k/total_sum)<0.9)
%    sum_k = sum_k + latent(i);
%    i=i+1;
%end
%P=COEFF(:,1:i-1);
%X3=X2*P;
%X=X3';

%% NN Model Training
% create a neural network
net = patternnet([100,100,100]);
% divided into training, validation and testing simulate
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 0.2;

rand_indices = randperm(size(X, 2));
trainData = X(:, rand_indices(1:2400));
trainLabels = y(:, rand_indices(1:2400));
testData = X(:, rand_indices(2401:end));
testLabels = y(:, rand_indices(2401:end));

% train a neural network
net = train(net, trainData, trainLabels);

net.performFcn='mse';

% show the network
view(net);

preds = net(testData);
est = vec2ind(preds) - 1;
tar = vec2ind(testLabels) - 1;

% find percentage of correct classifications
accuracy = 100*length(find(est==tar))/length(tar);
fprintf('Accuracy rate is %.2f\n', accuracy);

% confusion matrix
plotconfusion(testLabels, preds)
save ('nn_model.mat','net');