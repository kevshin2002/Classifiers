classdef KMeansClassifier
    properties
        x_train = []; % 784 x 5000 vector
        class_means = [];
        class_labels = [];
        class_difference = zeros(784,10);
        changed_assignments = 11;
    end
    
    methods
        function obj = create(obj, x_train)
            obj.x_train = x_train;
        end
        function obj = random(obj)
            for classes = 1:10
                obj.class_means(:, classes) = rand(784, 1) * 255;
            end
            obj.class_labels = zeros(5000, 1);
        end
        
        function obj = select(obj, trainLabels)
          % Question 3  indices = [373 4241 3986 716 1560 1236 1376 1078 294 925].';
          % Question 4  indices = [724 4850 784 3512 3476 2825 3311 3682 2341 1132].';
            obj.class_means = obj.x_train(:, indices);
            obj.class_labels = zeros(5000, 1);
        end
        
        
        function obj = predictRand(obj)
       obj.changed_assignments = 0;  
        distVector = zeros(10, 1);
            for trainLength = 1:length(obj.x_train)
                for classes = 1:10
                theDifference = norm(obj.class_means(:, classes) - obj.x_train(:, trainLength))^2;        
                distVector(classes) = theDifference;
                end
                [value, theIndex] = min(distVector);
                if(obj.class_labels(trainLength, 1) ~= theIndex -1)
                    obj.class_labels(trainLength, 1) = theIndex - 1;
                    obj.changed_assignments = obj.changed_assignments + 1;
                end
            end
        end
        
            function obj = learnMean(obj)
                for classes = 1:10
                    indices = find(obj.class_labels == (classes - 1));
                    aMean = mean(obj.x_train(:, [indices]), 2);
                    if(~(isnan(aMean)))
                       obj.class_difference(:, classes) = abs(obj.class_means(:, classes) - aMean);
                        obj.class_means(:, classes) = aMean;
                    else
                        obj.class_difference(classes) = 0;
                        obj.class_means(classes) = obj.class_means(classes);
                    end
                end
            end
        
        
        
        function obj = predictSelect(obj)
            indices = randsample(5000, 10);
            for classes = 1:10
                obj.class_means(:,classes) = obj.x_train(:, indices(classes));
            end
            for trainLength = 1:length(obj.x_train)
                for classes = 1:10
                theDifference = norm(obj.class_means(:, classes) - obj.x_train(:, trainLength));
                if(theDifference < theMinimum)
                    theMinimum = theDifference;
                    theClass = classes;
                end
                end
                obj.class_labels(trainLength) = theClass - 1;
            end
        end
        
        function error = getError(obj, labelTest)
            labelVector = unique(labelTest);
            theResult = zeros(size(labelVector));
            for indexPrediction = 1:length(obj.class_labels)
                   if(obj.class_labels(indexPrediction) ~= labelTest(indexPrediction))
                        errorIndex = find(labelVector == obj.class_labels(indexPrediction));
                      theResult(errorIndex) = theResult(errorIndex) + 1;
                    end
            end
                
                for indexCount = 1:length(labelVector)
                    count(indexCount) = sum(obj.class_labels == labelVector(indexCount));
               end
                error(:,1) = theResult;
                error(:,2) = count;
            
                for errorRateIndex = 1:length(labelVector)
                    rate(errorRateIndex) = theResult(errorRateIndex)/count(errorRateIndex);
                end
            
                error(:,3) = rate;
        end
        
    end
end