library(nnet);
clamp <- function (x, bot, top) pmax(pmin(x, top), bot)
rmse <- function (x, y) mean((x - y)^2)^0.5;

train <- read.csv('data/train.csv');

ratings <- train$Rating;

preds <- c('lm', 'svd', 'rf', 'demo');

trains <- sapply(preds, function (name) read.csv(paste('predictions/', name, '.csv.cross', sep=''))$x)/100;
tests <- sapply(preds, function (name) read.csv(paste('predictions/', name, '.csv', sep=''))$x)/100;

mix <- nnet(trains, ratings, size=3, decay=0.1, maxit=500, linout=T, reltol=0, abstol=0);
print(summary(mix));
cat('RMSE:', rmse(predict(mix), ratings), '\n');

blendname <- do.call(paste, as.list(c(preds, sep='-')));
write.csv(clamp(predict(mix, tests), 0, 100), paste('blends/', blendname, '.csv', sep=''), row.names=F, quote=F)