% Load and set up data
DATA = load("data.mat");
LABEL = load("label.mat");

Data = DATA.('imageTrain');
labelData = LABEL.('labelTrain');

testData = DATA.('imageTest');
labelTest = LABEL.('labelTest');

covariance = eye(size(testData,1), size(testData,2));
probability = ones(10,1) / 10;
% Stack the column vectors
FEATURE_SIZE = size(Data, 1) * size(Data, 2);
x_train_stacked = reshape(Data, FEATURE_SIZE, length(labelData));
x_test_stacked = reshape(testData, FEATURE_SIZE, length(labelTest));

% Class initialization
Classifier = BayesClassifier();
Classifier = create(Classifier, x_train_stacked, labelData, covariance, FEATURE_SIZE);

% Question 1
Classifier = populateSample(Classifier);

%Images of Sample Mean
for index = 0:9
   subplot(3, 10, index + 1)
   imshow(uint8(Classifier.sample_mean(:,:,index + 1)));
   title("Sample Mean of " + index);
   subplot(3, 10, index + 11)
   image(Classifier.sample_mean(:,:,index + 1));
   title("Sample Mean of " + index);
end

% Question 2
aResult = predict(Classifier, x_test_stacked, labelTest, probability);

% Error Rates, 2a)
error = getError(Classifier, aResult, labelTest);
plot([0:9], error(:, 3) * 100);
xlim([0, 9]);
ytickformat('percentage');
title("Error Rates for different Labels");
xlabel("Labels");
ylabel("Percentages");

% Error Rates
% P(E | 0) = 10.0000000000000%
% P(E | 1) = 24.6913580246914%
% P(E | 2) = 16.2790697674419%
% P(E | 3) = 32.0000000000000%
% P(E | 4) = 24.0000000000000%
% P(E | 5) = 42.8571428571429%
% P(E | 6) = 11.1111111111111%
% P(E | 7) = 14.2857142857143%
% P(E | 8) = 16.6666666666667%
% P(E | 9) = 32.2033898305085%

% Total Error Rate, 2b)
totalSum = sum(error(:,1)/sum(error(:,2)));
% P(Error) = 24.20%

% Optional Credit
aCov = getCov(Classifier);

% It is because the covariance matrix is not positive definite or semi
% definite. The reason why they're not any of them is because they are not
% symmetric. This means that the mahalobian distance formula will not work
% as it only works with positive definite matrices in x'Mx.
for anImage = 0:length(unique(labelTest)) - 1
    aCova = cov(aCov(:, :, anImage + 1));
    subplot(3, 10, anImage + 1)
    imshow(uint8(aCova));
    title("Imshow of " + anImage);
    subplot(3, 10, anImage + 11)
    image(uint8(aCova));
    title("Image of " + anImage);
end