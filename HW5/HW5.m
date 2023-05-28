%% Initialization of data
DATA = load("data.mat");
LABEL = load("label.mat");
Data = double(DATA.('imageTrain'));
labelTrain = LABEL.('labelTrain');

%% Reshaping of features and data
testData = double(DATA.('imageTest'));
labelTest = LABEL.('labelTest');
FEATURE_SIZE = 784;
Data = reshape(Data, FEATURE_SIZE, []);
testData = reshape(testData, FEATURE_SIZE, []);

%% Create Classifier
KMeans1 = KMeansClassifier();
KMeans1 = create(KMeans1, Data);

%% Randomize 10 class means
KMeans1 = random(KMeans1);

%% For n = 1
KMeans1 = predictRand(KMeans1);
KMeans1 = learnMean(KMeans1);
%% For n >1
while(KMeans1.changed_assignments > 10)
    KMeans1 = predictRand(KMeans1);
    KMeans1 = learnMean(KMeans1);
end

%% Reshape into images and print them out
%means1 = reshape(KMeans1.class_means, 28, 28, 10);
%for index = 0:9
 %  subplot(3, 10, index + 1)
 %  imshow(uint8(means1(:,:,index+1)));
 %  title("Sample Mean of " + index);
 %  subplot(3, 10, index + 11)
 %  image(means1(:,:,index+1));
 %  title("Sample Mean of " + index);
%end

%% Answer to Question 1
% The problem with running this K-Means Algorithm with random
% initialization images is that it can cause empty clusters to occur. 
% Some clusters could be on the decision boundary line and could have no
% class attached to them, and thus there could be no iterative step for
% them. These results in random colorized pixels and indicate no value.

% To tackle it, you simply do supervised learning with labels, but realistically in
% unsupervised learning, it's hard to self learn. We can create classes,
% but they won't be unique, and it might not be k = 10 classes and could be
% k =6 or k = 4 due to the randomly selected image not being the best
% represenative of it's class.


%% Question 2
% Create classes and select 10 random images within training
KMeans2 = KMeansClassifier();
KMeans2 = create(KMeans2, Data);
KMeans2 = select(KMeans2, labelTrain);

% Reshape into image.
originals = reshape(KMeans2.class_means, 28, 28, 10);

% Print the original images
%for index = 0:9
%    subplot(3, 10, index + 1)
%    imshow(uint8(originals(:,:,index+1)));
%end

% For n = 1
KMeans2 = predictRand(KMeans2);
KMeans2 = learnMean(KMeans2);

% For n > 1
while(KMeans2.changed_assignments > 10)
    KMeans2 = predictRand(KMeans2);
    KMeans2 = learnMean(KMeans2);
end
% Reshape into image
means2 = reshape(KMeans2.class_means, 28, 28, 10);
% Print the learned class means
%for index = 0:9
%   subplot(3, 10, index + 11)
%   imshow(uint8(means2(:,:,index+1)));
%   title("Sample Mean of " + index);
%   subplot(3, 10, index + 21)
%   image(means2(:,:,index+1));
%   title("Sample Mean of " + index);
%end

%% Question 3
covariance = eye(784);
probability = ones(10,1) / 10;

% Manually assign class labels
% Question 3 classLabels = [9 4 6 8 2 7 1 -1 0 3].';
% Question 4 classLabels = [7 0 1 8 6 2 9 -1 3 -1].';

% Change non-unique or empty clusters into zeros
means2(:, :, 8) = zeros(28, 28, 1);
%means2(:, :, 3) = zeros(28, 28, 1);
means2(:, :, 10) = zeros(28, 28, 1);

% Class initialization
Classifier = BayesClassifier();
Classifier = create(Classifier, covariance, FEATURE_SIZE);
% Take the class labels and learned means as 'supervised' labels
Classifier = meanify(Classifier, means2, classLabels);
% Use BDR to classify
predictions = predict(Classifier, testData, labelTest, probability);

% Error Rates
error = getError(Classifier, predictions, labelTest);
% Assign an error rate of 50% for non-unique or empty cluster class mean.
error(5,3) = .5; 
error(6,3) = .5; 
plot([0:9], error(:, 3) * 100);
xlim([0, 9]);
ytickformat('percentage');
title("Error Rates for different Labels");
xlabel("Labels");
ylabel("Percentages");

% Error Rates

% P(E | 0) = 13.3333333333333%
% P(E | 1) = 42.6086956521739%
% P(E | 2) = 7.14285714285714%
% P(E | 3) = 61.7977528089888%
% P(E | 4) = 64.1509433962264%
% P(E | 5) = 50.0000000000000% No Label
% P(E | 6) = 26.3157894736842%
% P(E | 7) = 70.0000000000000%
% P(E | 8) = 54.2857142857143%
% P(E | 9) = 100%

% Total Error Rate, 3
totalSum = sum(error(:,3))/10;
% P(Error) = 48.96%


%% The reason why the error rates for 4, 7, 9 were high is due to their close resemblance to each other.
%% If we look at 2, we can see how strongly bright it is and close to the class mean it is and how there are no other class means close to '2'.
%% Another example is 0, because although we have another class mean, the "best" representing class mean yielded a lower error rate.
%% As such, to have a lower error rate, we want to have images that are as close to the class means of the labels possible.

%% Question 4
% Reused Q3.

% Error Rates
% P(E | 0) = 13.3333333333333%
% P(E | 1) = 33.3333333333333%
% P(E | 2) = 9.67741935483871%
% P(E | 3) = 60.2150537634409%
% P(E | 4) = 50.0000000000000% No Label
% P(E | 5) = 50.0000000000000% No Label
% P(E | 6) = 26.6666666666667%
% P(E | 7) = 0%
% P(E | 8) = 65.4545454545455%
% P(E | 9) = 70.1298701298701%

% Total Error Rate, 4
% P(Error) = 37.88%

%% Answer
% Initialization is super important because the more close the images are
% to the class mean, the more accurate our results will be.
% For example, the class 3 had the lowest because although it was similar
% to 8, the highlighting features of 3 were much brighter than the 8
% features.
% 
% The highest errors were 4 and 9 because of how similar they were. 
% 8 was also high due to it's close resemblance to 3.

% The closer the initializing images are to the class mean, the better our
% unsupervised results will become.
