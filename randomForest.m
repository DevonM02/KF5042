%Clearing the workspace
clear;
%Loading the dataset
filename = "amazon_reviews.csv";
dataReviews = readtable(filename,'TextType','string');
reviews = dataReviews.reviewText;
sentiment = dataReviews.overall;
%Data cleaning and pre-processing
cleanTextData = lower(reviews);
documents = tokenizedDocument(cleanTextData);
documents = erasePunctuation(documents);
processedReviews = removeWords(documents,stopWords);
%Feature extraction
b = bagOfWords(processedReviews);
t = tfidf(b);
d = full(t);
%Splitting the data
cvp = cvpartition(sentiment,'HoldOut',0.2);
Xtrain = d(training(cvp),:);
Ytrain = sentiment(training(cvp));
Xtest = d(test(cvp),:);
Ytest = sentiment(test(cvp));
%Training the model
random_forest_model = TreeBagger(150, Xtrain,Ytrain);
%Predicting labels of test data
predictedSent =str2double(predict(random_forest_model,Xtest));
%Evaluation metrics
confusion = confusionmat(Ytest,predictedSent);
accuracy = sum(diag(confusion)) /sum(confusion(:));
disp(accuracy*100);
precision = confusion(1,1) / sum(confusion(:,1));
disp(precision*100);
recall = confusion(1,1) / sum(confusion(1,:));
disp(recall*100);
f1_score = 2 * (precision * recall) /(precision + recall);