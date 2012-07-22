library(randomForest);
library(plyr);
nuser <- 50928;
ntrack <- 184;

rmse <- function (x, y) mean((x - y)^2)^0.5;

cleaned <- tryCatch({load('data/clean.Rdata'); cleaned},
  error = function (e) {source('clean.R'); save(cleaned, file='data/clean.Rdata'); return(cleaned);});
  
trainfeats <- cleaned[[1]];
testfeats <- cleaned[[2]];
ratings <- cleaned[[3]];

rfpred <- function (trainfeats, testfeats, ratings) {
  rf <- randomForest(trainfeats, ratings, do.trace=T, sampsize=1000, ntree=100);
  pred <- predict(rf, testfeats);
  cv <- rf$predicted;
  return(list(pred=pred, cv=cv));
}

svdpred <- function (trainfeats, testfeats, ratings, n=64) {
  library(irlba);
  mu <- mean(ratings);
  spmat <- sparseMatrix(i=trainfeats$User + 1, j=trainfeats$Track + 1, x=ratings - mu);
  sv <- irlba(spmat, nu=n, nv=n);
  rec <- sv$u %*% (sv$d * t(sv$v));
  pred <- sapply(1:nrow(testfeats), function (i) rec[testfeats$User[i] + 1, testfeats$Track[i] + 1]) + mu;
  return(list(pred=pred));
}

svmpred <- function (trainfeats, testfeats, ratings) {
  library(e1071);
  n <- nrow(trainfeats);
  big <- rbind(trainfeats, testfeats);
  model <- model.matrix(~., data=big);
  sv <- svm(model[1:n,], ratings);
  pred <- predict(svm, model[-(1:n)]);
  return(list(pred=pred));
}

rfbyartistpred <- function (train, test, ratings) {
  train$Rating <- ratings;
  rfs <- dlply(train, .(Artist), function (x) {
    rating <- x$Rating;
    x$Rating <- NULL;
    return(randomForest(x, rating, ntree=100, do.trace=F));
  }, .progress='text');
  pred <- numeric(nrow(test));
  for (i in 0:max(test$Artist)) {
    ids <- test$Artist == i;
    if (sum(ids) > 0) pred[ids] <- predict(rfs[[as.character(i)]], test[ids,]);
  }
  return(list(pred=pred));
}
cross.val <- function (predictor, folds, trainfeats, testfeats, ratings, ...) {
  cat('Running primary predictor\n');
  pred <- predictor(trainfeats, testfeats, ratings);
  items <- sample(folds, nrow(trainfeats), T);
  cv <- numeric(nrow(trainfeats));
  for (i in 1:folds) {
    samp <- which(items == i);
    cat('Running CV fold', i, '\n');
    cpred <- predictor(trainfeats[-samp,], trainfeats[samp,], ratings[-samp]);
    cat('Estimated RMSE for fold:', rmse(ratings[samp], cpred$pred), '\n');
    cv[samp] <- cpred$pred;
  }
  pred$cv <- cv;
  return(pred);
}

remove.global <- function (predictor) {
  function (train, test, ratings) {
    n <- nrow(train);
    mu <- mean(ratings);
    baseline <- rep(mu, n);

    usum <- numeric(nuser);
    ucount <- numeric(nuser);
    for (i in 1:n) {
      u <- train$User[i] + 1;
      usum[u] <- usum[u] + ratings[i] - baseline[i];
      ucount[u] <- ucount[u] + 1;
    }
    umean <- usum / (ucount + 5);
    baseline <- baseline + umean[train$User + 1];

    tsum <- numeric(ntrack);
    tcount <- numeric(ntrack);
    for (i in 1:n) {
        t <- train$Track[i] + 1;
        tsum[t] <- tsum[t] + ratings[i] - baseline[i];
        tcount[t] <- tcount[t] + 1;
      }
    tmean <- tsum / (tcount + 25);
    baseline <- baseline + tmean[train$Track + 1];
    
    pred <- predictor(train, test, ratings - baseline);
    trk <- test$Track + 1;
    usr <- test$User + 1;
    base <- umean[usr] + tmean[trk] + mu;
    
    pred$pred <- pred$pred + base;
    return(pred);
  }
} 
source('funk.R');

s <- sample(nrow(test), 10000)
pred <- cross.val(remove.global(rfbyartistpred), 5, trainfeats, testfeats, ratings);
cat('Estimated RMSE: ', rmse(ratings, pred$cv), '\n');

argv <- commandArgs(T);
if (length(argv) == 0) {
  filename <- 'predictions/scratch.csv';
} else {
  filename <- argv[1];
}

write.csv(pred$pred, filename, row.names=F, quote=F);
write.csv(pred$cv, paste(filename, '.cross', sep=''), row.names=F, quote=F)