DATA = load("data.mat");
LABEL = load("label.mat");

Data = DATA.('imageTrain');
labelData = LABEL.('labelTrain');

testData = DATA.('imageTest');
labelTest = LABEL.('labelTest');

% Stack the column vectors
FEATURE_SIZE = size(testData, 1) * size(testData, 2);
x_train_stacked = reshape(Data, FEATURE_SIZE, length(labelData));
x_test_stacked = reshape(testData, FEATURE_SIZE, length(labelTest));

% Initialize class, Question 1
Classifier = NNClassifier();
Classifier = create(Classifier, x_train_stacked, labelData); % instantiate members
y_pred = predict(Classifier, x_test_stacked, labelTest); % predict test data
error = getError(Classifier, y_pred, labelTest); % Get error for each label, total count of numbers, and error rate for each


% Question 1 error rate
error(:,3) = error(:,3) * 100;
plot([0:9], error(:, 3));
xlim([0, 9]);
ytickformat('percentage');
title("Error Rates for different Labels");
xlabel("Labels");
ylabel("Percentages");

% Error for Class 0: 8.69565217391304%
% Error for Class 1: 5.63380281690141%
% Error for Class 2: 2.04081632653061%
% Error for Class 3: 7.31707317073171%
% Error for Class 4: 11.1111111111111%
% Error for Class 5: 7.84313725490196%
% Error for Class 6: 8.88888888888889%
% Error for Class 7: 12.7272727272727% 
% Error for Class 8: 16.6666666666667%
% Error for Class 9: 15.3846153846154%



% Question 2 Total Error
totalError = sum(error(:,1)) / sum(error(:,2)); % Total count / total error
% Total Error = 9.40%



% Question 3
images = getImages(Classifier, x_test_stacked, labelTest);
mismatchTrain = images(:,1);
mismatchTest = images(:,2);

x_data_reshaped = reshape(x_train_stacked, 28, 28, 5000);
x_test_reshaped = reshape(x_test_stacked, 28, 28, 500);

imageArray = zeros(10,2);
for index = 1:10
    subplot(4, 5, index)
    image(x_data_reshaped(:,:,mismatchTrain(index)));
    title("Train " + index)
    subplot(4, 5, index + 10)
    image(x_test_reshaped(:,:,mismatchTest(index)));
    title("Test " + index)
end

% The NN Classification failed as the digits looked similar and would have
% similar distance from one another. The most obvious is 9 and 4 but the
% least obvious is 2 and 1.