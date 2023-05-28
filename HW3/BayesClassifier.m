classdef BayesClassifier
    properties
        x_train = [] % 784 x 5000 vector
        y_train = [] % 5000 x 1 vector
        cov = [] % covariance matrix
        sample_mean = []
        feature_size;
        prob_class = [];
    end
    
    methods
        
        function obj = create(obj, x_train, y_train, feature)
            obj.x_train = x_train;
            obj.y_train = y_train;;
            obj.feature_size = feature; 
        end
        
        function obj = populateSample(obj)
            theLabelVector = unique(obj.y_train); % Get the unique labels or classes
            theDigitCount = zeros(length(theLabelVector), 1); % Create zeros for the length of labels
            theAvgVector = zeros(obj.feature_size, length(theLabelVector)); % Create the container for which the summed values will go
            theDigit = 0;
            % Populate the sum of all the features of samples with respect
            % to each sample digit
            for anIndex = 1:length(obj.y_train) % 1 -> 5000
                theDigit = obj.y_train(anIndex) + 1; % Labels are from 0 to 9, we shift to 1 to 10.
                theAvgVector(:, theDigit) =  theAvgVector(:, theDigit) + obj.x_train(:, anIndex);
                theDigitCount(theDigit) = theDigitCount(theDigit) + 1;
            end
            
            for averageIndex = 1:length(theLabelVector)
                theAvgVector(:, averageIndex) = theAvgVector(:, averageIndex) / theDigitCount(theDigit); % Average each class by count;
            end
    
            obj.sample_mean = reshape(theAvgVector, 28, 28, length(theLabelVector));  % Reshape and store into internal data member.  
        end
      
        function prediction = predict(obj, x_test, y_test, prob_class)
            sample_means = reshape(obj.sample_mean, length(x_test), length(unique(y_test)));
            labelVector = unique(y_test);
            theResult = zeros(length(y_test), 1);
            mahaDistVector = zeros(500,1);
            cov = eye(784);
            
            for testIndex = 1:length(y_test)
                for classIndex = 1:length(labelVector)
                    class_mean = sample_means(:, classIndex);
                    normalization = (-1/2) * 784 * log (2 * pi)  * det(obj.cov) + log (prob_class(classIndex));
                    mahaDist = (-1/2) * (x_test(:,testIndex) - class_mean(:)).' * cov * (x_test(:, testIndex) - class_mean(:)) + normalization;
                    
                    if (classIndex == 1)
                        mahaDistVector(testIndex) = mahaDist;
                        theResult(testIndex) = classIndex - 1;
                    else
                        maxValue = mahaDistVector(testIndex);
                        if(maxValue < mahaDist)
                            mahaDistVector(testIndex) = mahaDist;
                            theResult(testIndex) = classIndex - 1;
                        end
                    end
                end
            end
            
           prediction = theResult;
        end
        
        function error = getError(obj, y_pred, labelTest)
            labelVector = unique(labelTest);
            theResult = zeros(size(labelVector));
            for indexPrediction = 1:length(y_pred)
                   if(y_pred(indexPrediction) ~= labelTest(indexPrediction))
                        errorIndex = find(labelVector == y_pred(indexPrediction));
                      theResult(errorIndex) = theResult(errorIndex) + 1;
                    end
            end
                
                for indexCount = 1:length(labelVector)
                    count(indexCount) = sum(y_pred == labelVector(indexCount));
               end
                error(:,1) = theResult;
                error(:,2) = count;
            
                for errorRateIndex = 1:length(labelVector)
                    rate(errorRateIndex) = theResult(errorRateIndex)/count(errorRateIndex);
                end
            
                error(:,3) = rate;
        end
        
        function obj = getCov(obj)
                 theLabelVector = unique(obj.y_train);
                 for aIndex = 1:length(theLabelVector)
                     theIndices = find(obj.y_train == aIndex - 1);
                     for theIndex = 1:length(theIndices)
                         theCovar(theIndex, :, aIndex) = obj.x_train(:, theIndices(theIndex));
                     end
                 end
                 obj.cov = theCovar;

        end
        
  
    end
end