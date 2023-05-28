% Load and set up data
DATA = load("data.mat");
LABEL = load("label.mat");

Data = DATA.('imageTrain')/255;
Data = normalize(Data, 'range', [0, 1]);
labelData = LABEL.('labelTrain');

testData = DATA.('imageTest');
testData = normalize(testData, 'range', [0, 1]);
labelTest = LABEL.('labelTest');

probability = ones(10,1)/10;

% Question 1
% Stack the column vectors
FEATURE_SIZE = size(Data, 1) * size(Data, 2);
x_train_stacked = reshape(Data, FEATURE_SIZE, length(labelData));
x_test_stacked = reshape(testData, FEATURE_SIZE, length(labelTest));

% Class initialization
Classifier = BayesClassifier();
[eigenvector, eigenvalue, class_mean] = PCA(Classifier, x_train_stacked, x_test_stacked, 784);
[eigenvector1, eigenvalue1, class_mean1] = PCA_5(Classifier, x_train_stacked, x_test_stacked, labelData, 784);
for index = 1:10
                subplot(3, 10, index+10);
                imshow(reshape(eigenvector(:, index), 28, 28), []);
                title("Principle Component " + index);
                subplot(3, 10, index + 20);
                imshow(reshape(eigenvector1(:, index), 28, 28), []);
                title("Principle Component " + index);
                subplot(3, 2, 1)
                plot(eigenvalue);
                title("Principle Values of Data");
                subplot(3, 2, 2);
                plot(eigenvalue1);
                title("Principle Values of Class 5");
end
% Principle Values of Data
% 5.32575817665228 
% 4.15765444035059
% 3.38257986030941
% 2.99124656995504
% 2.50883076046970
% 2.39768258064119
% 1.85644563443779
% 1.59480208618075
% 1.49162171425832
% 1.25547873832038

% Principle Values of Class 5
% 3.08076398066810
% 0.720342248582902
% 0.346577955141297
% 0.313452189340450
% 0.182100974457166
% 0.164234518241075
% 0.127208857259258
% 0.115259923077638
% 0.110726993208023
% 0.0970769551073043

% Question 2a)
% I would say 72, because that's when the eigenvalue drops below 0.1.
% Realistically though, I'd say it depends on how many dimensions you want
% by testing out which has the lowest errors. Since we get higher errors
% the higher dimensions we go, it's more logical to have lower dimensions.
% If this is the case, having 12 would be best because after 12 dimensions,
% it dips below 1.

x_train_stacked = eigenvector.' * (x_train_stacked - class_mean);
x_test_stacked = eigenvector.' * (x_test_stacked - class_mean);
Classifier = create(Classifier, x_train_stacked, labelData, 784);
Classifier = populateSample(Classifier);
Classifier = getCov(Classifier);
aResult = predict(Classifier, x_test_stacked, labelTest, probability);

error = getError(Classifier, aResult, labelTest);
%% Could not figure out how to make it work.