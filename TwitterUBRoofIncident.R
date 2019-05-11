#To extract Twitter data
install.packages("rtweet")
#For Write_csv
install.packages("readr")
#to convert List to Data Frame
install.packages("twitteR")
install.packages("dplyr")
install.packages("tidytext")
install.packages("tm")
install.packages("stringr")
install.packages("gradDescent")
install.packages("sgd")

library(rtweet)
library(readr)
library(twitteR)
library(dplyr)
library(tidytext)
library(tm)
library(stringr)
library(gradDescent)

consumerKey = "ddRk8ThASj3L5A6AEO28rn0lm"
consumerSecretKey = "8mj2v80QfQgrarZT2YcFqzLDahFltdhVvfreN8zR6TP3euIcne"
accessToken = "851278434-cInNpKzDyjFzvzwcs9Fp8WuUvnGgfo3bfms8dwMq"
accessTokenSecret = "zULiS1MXyPtCVzpHX1yDKLwbW4llErPWBV5y0oSdl4dpU"
appName = "ML_ExtractTweets"

create_token(app = appName, consumer_key = consumerKey, consumer_secret = consumerSecretKey, access_token = accessToken, 
	access_secret = accessTokenSecret)

#Extract all tweets from the mentioned Twitter handles.
extractedDataList <- get_timeline(c("UBFacilities", "WGRZ", "news4buffalo", "TheBuffaloNews", "937WBLK", "gina_ann95", 
	"UBSpectrum", "kenzxg", "meowburger", "PrevalWGRZ", "jessjones220", "The_realcash35", "kelsomoreau"), n = 32922)
write_as_csv(extractedDataList, "/Users/varshilaredkar/Documents/UB_Sem2/Machine Learning/RoofData.csv", prepend_ids = TRUE, 
	na = "NA", fileEncoding = "UTF-8")

#Filter the above tweets to be in the date range from Feb 22 to Feb 27 as the Roof incident occurred on Feb 24
#Also, Filter for english tweets and eliminate retweets
extractedDataListsubset <- extractedDataList %>% dplyr::filter(created_at > "2019-02-22" & created_at < "2019-02-27") %>% 
	dplyr::filter(is_retweet == FALSE) %>% dplyr::filter(lang == "en")

write_as_csv(extractedDataListsubset, "/Users/varshilaredkar/Documents/UB_Sem2/Machine Learning/RoofDataSubset.csv", 
	prepend_ids = TRUE, na = "NA", fileEncoding = "UTF-8")

#Selected 2 columns: Text & Retweet_count. Manually annotated positive examples
roofData <- extractedDataListsubset %>% select(text, retweet_count)
write_as_csv(roofData, "/Users/varshilaredkar/Documents/UB_Sem2/Machine Learning/RoofData.csv", prepend_ids = TRUE, 
	na = "NA", fileEncoding = "UTF-8")

# remove punctuation, convert to lowercase, add id for each tweet!
roofDataClean <- roofData %>% dplyr::select(text) %>% unnest_tokens(word, text)
write_as_csv(roofDataClean, "/Users/varshilaredkar/Documents/UB_Sem2/Machine Learning/RoofDataCleaned.csv", prepend_ids = TRUE, 
	na = "NA", fileEncoding = "UTF-8")

# load list of stop words - from the tidytext package
data("stop_words")
# view first 6 words
head(stop_words)

nrow(roofDataClean)

# remove stop words from your list of words
# retain unique words
#remove any numbers
roofDataClean_words <- roofDataClean %>% anti_join(stop_words) %>% distinct() %>% filter(!grepl("[0-9]+", word))

write_as_csv(roofDataClean_words, "/Users/varshilaredkar/Documents/UB_Sem2/Machine Learning/RoofDataCleaned_words.csv", 
	prepend_ids = TRUE, na = "NA", fileEncoding = "UTF-8")

# there should be fewer words now
nrow(roofDataClean_words)

newdata <- iconv(roofDataClean_words$word, "ASCII", "UTF-8", sub = "")
roof_Corpus <- Corpus(VectorSource(newdata))

roofDataClean_words <- read_csv("/Users/varshilaredkar/Documents/UB_Sem2/Machine Learning/RoofDataCleaned_words.csv")

#Stemming words using Porter Stemmer method and converting it to lowercase
#Keep unique words
roof_stemmed <- stemDocument(roofDataClean_words$word, language = "english")
roof_stemmed <- unique(roof_stemmed)
head(roof_stemmed)


#BUILD DATA SET
#Get the CSV containing Tweet Text, Retweet Count, Target
roofData <- read_csv("/Users/varshilaredkar/Documents/UB_Sem2/Machine Learning/RoofData.csv")
#Stem Twitter Text
roofDataStemmed <- stemDocument(roofData$text, language = "english")
#Retain text which is only in UTF-8
Encoding(roofDataStemmed) <- "UTF-8"
roofDataStemmed <- iconv(roofDataStemmed, "UTF-8", sub = "")
#Convert Twitter text to lowercase
roofDataStemmed <- tolower(roofDataStemmed)

#Create final data frame with above with following columns
roofDataFrame <- data.frame(TwitterText = roofData$text, TwitterTextStemmed = roofDataStemmed, RetweetCount = roofData$retweet_count, 
	Target = roofData$target)
write_as_csv(roofDataFrame, "/Users/varshilaredkar/Documents/UB_Sem2/Machine Learning/RoofDataFrame.csv", prepend_ids = TRUE, 
	na = "NA", fileEncoding = "UTF-8")

#Create data frame of features
roof_features <- matrix(data = "", nrow = nrow(roofDataFrame), ncol = length(roof_stemmed))
colnames(roof_features) <- roof_stemmed
roof_features <- as.data.frame(roof_features)
head(roof_features)

#Append features(unique text words) Skeleton(Data Set)
dataSet <- cbind(roofDataFrame, roof_features)
write_as_csv(dataSet, "/Users/varshilaredkar/Documents/UB_Sem2/Machine Learning/DataSet.csv", prepend_ids = TRUE, 
	na = "NA", fileEncoding = "UTF-8")


#Making a Document Term Matrix
readFile <- read_csv("/Users/varshilaredkar/Documents/UB_Sem2/Machine Learning/DataSet.csv", col_names = TRUE)
for (j in 1:nrow(readFile)) {
	for (k in 5:ncol(readFile)) {
		readFile[j, k] <- str_count(readFile[j, "TwitterTextStemmed"], names(readFile[k]))
	}
}

write_as_csv(readFile, "/Users/varshilaredkar/Documents/UB_Sem2/Machine Learning/DataSetFreq.csv", prepend_ids = TRUE, 
	fileEncoding = "UTF-8")
readFile1 <- read_csv("/Users/varshilaredkar/Documents/UB_Sem2/Machine Learning/DataSetFreq.csv", col_names = TRUE)
head(readFile1)

#Split data set into train and test manually because numeric values were getting converted to logical (as a 50-50)
#Read train and test set
trainSet <- read_csv("/Users/varshilaredkar/Documents/UB_Sem2/Machine Learning/TrainDataTwitter.csv", col_names = TRUE)
testSet <- read_csv("/Users/varshilaredkar/Documents/UB_Sem2/Machine Learning/TestDataTwitter.csv", col_names = TRUE)
head(trainSet)
head(testSet)

#Selecting on numeric columns
trainSet1 <- trainSet[, 3:ncol(trainSet)]
#Moving Target variable column to the extreme right because SAGD() function requires it in that format
testSet1 <- testSet1[c(setdiff(names(testSet1), "Target"), "Target")]
head(trainSet1)

#Selecting on numeric columns
testSet1 <- testSet[, 3:ncol(testSet)]
#Moving Target variable column to the extreme right because predictions() function requires it in that format
testSet1 <- testSet1[c(setdiff(names(testSet1), "Target"), "Target")]
testSet1[, "Target"]
#Removing actual target value before predicting
testSet2 <- testSet1[, 1:ncol(testSet1) - 1]
head(testSet2)

#SAGD model
SAGDmodel <- SAGD(trainSet1, alpha = 0.1, maxIter = 10, seed = NULL)
print(SAGDmodel)

#Predictions
SAGDpredictions <- prediction(SAGDmodel, testSet2)
print(SAGDpredictions)
SAGDpredictions[, "V1"]
sum(SAGDpredictions[,"V1"] > 0)
as.list(testSet1[, "Target"])