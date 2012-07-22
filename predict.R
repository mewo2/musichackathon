library(randomForest);
library(plyr);

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
  function (trainfeats, testfeats, ratings) {
    trainfeats$Rating <- ratings;
    usr <- daply(trainfeats, .(User), function (x) mean(x$Rating));
    trainfeats$Rating <- NULL;
    offset <- usr[as.character(trainfeats$User)];
    offset[is.na(offset)] <- mean(usr);
    ratings <- ratings - offset;
    pred <- predictor(trainfeats, testfeats, ratings);
    offset <- usr[as.character(testfeats$User)];
    offset[is.na(offset)] <- mean(usr);
    pred$pred <- pred$pred + offset;
    return(pred);
  }
} 
source('funk.R');

pred <- cross.val(funkpred, 10, trainfeats, testfeats, ratings);
cat('Estimated RMSE: ', rmse(ratings, pred$cv), '\n');

argv <- commandArgs(T);
if (length(argv) == 0) {
  filename <- 'predictions/scratch.csv';
} else {
  filename <- argv[1];
}

write.csv(pred$pred, filename, row.names=F, quote=F);
write.csv(pred$cv, paste(filename, '.cross', sep=''), row.names=F, quote=F)