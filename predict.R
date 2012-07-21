library(randomForest);

rmse <- function (x, y) mean((x - y)^2)^0.5;

train <- read.csv('data/train.csv');
test <- read.csv('data/test.csv');
words <- read.csv('data/words.csv');
users <- read.csv('data/users.csv')

words$X <- NULL;
rows <- is.na(words$Good.Lyrics)
words$Good.Lyrics[rows] <- words$Good.lyrics[rows];
words$Good.lyrics <- NULL;
words$OWN_ARTIST_MUSIC <- as.character(words$OWN_ARTIST_MUSIC);
words$OWN_ARTIST_MUSIC[substr(words$OWN_ARTIST_MUSIC, 2, 2) == 'o'] <- 'DK';
words$OWN_ARTIST_MUSIC <- as.factor(words$OWN_ARTIST_MUSIC);
words$HEARD_OF[words$HEARD_OF == ''] <- 'Never heard of';
words$HEARD_OF[words$HEARD_OF == 'Ever heard music by'] <- 'Heard of and listened to music EVER';
words$HEARD_OF[words$HEARD_OF == 'Ever heard of'] <- 'Heard of';
words$HEARD_OF[words$HEARD_OF == 'Listened to recently'] <- 'Heard of and listened to music RECENTLY';
words$HEARD_OF <- droplevels(words$HEARD_OF);

words <- na.roughfix(words);

fixtimes <- function (x) {
  x <- as.character(x)
  x[x == 'Less than an hour'] <- '.5';
  x[x == 'More than 16 hours'] <- '18';
  x <- as.numeric(substr(x, 1, 2));
  return(x)
}
users$LIST_OWN <- fixtimes(users$LIST_OWN);
users$LIST_BACK <- fixtimes(users$LIST_BACK);

domerge <- function (data) {
  data$RowID <- 1:nrow(data);
  merged <- merge(data, words, all.x=T);
  merged <- merge(merged, users, by.x='User', by.y='RESPID', all.x=T)
  merged <- na.roughfix(merged);
  merged <- merged[order(merged$RowID),];
  merged$RowID <- NULL;
  return(merged);
}

trainfeats <- domerge(train)[,-4];
testfeats <- domerge(test);

rf <- randomForest(trainfeats, train$Rating, do.trace=T, sampsize=50000, ntree=500);
cat('Estimated RMSE: ', rmse(train$Rating, rf$predicted), '\n');

pred <- predict(rf, testfeats);
cv <- rf$predicted;

argv <- commandArgs(T);
if (length(argv) == 0) {
  filename <- 'predictions/scratch.csv';
} else {
  filename <- argv[1];
}

write.csv(pred, filename, row.names=F, quote=F);
write.csv(cv, paste(filename, '.cross', sep=''), row.names=F, quote=F)