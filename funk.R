funksvd <- function (users, tracks, ratings, nfeats=16) {
  n <- length(users);
  nuser <- max(users) + 1;
  ntrack <- max(tracks) + 1;
  ufeats <- double(nuser * nfeats);
  tfeats <- double(ntrack * nfeats);
  dyn.load('funk.so');
  sv <- .C('funksvd', as.integer(n), as.integer(users), as.integer(tracks), as.double(ratings), 
            as.integer(nfeats), as.integer(nuser), as.integer(ntrack), 
            ufeats=ufeats, tfeats=tfeats);
  ufeats <- matrix(sv$ufeats, nuser, nfeats);
  tfeats <- matrix(sv$tfeats, ntrack, nfeats);
  return(list(ufeats=ufeats, tfeats=tfeats));
  # baseline <- rep(mean(ratings), n);
  # cached <- baseline;
  # sqerr <- 0;
  # sqerr.old <- 0;
  # for (feat in 1:nfeats) {
  #   cat('Calculating feature', feat, '\n');
  #   loops <- 0;
  #   while (loops < min.loops | ((sqerr.old/n)^.5 - (sqerr/n)^.5 > min.improve)) {
  #     loops <- loops + 1;
  #     cat('Iterating... (error =', (sqerr/n)^.5, 'improvement =', (sqerr.old/n)^.5 - (sqerr/n)^.5, ')\n');
  #     sqerr.old <- sqerr;
  #     sqerr <- 0;
  #     prediction <- numeric(n);
  #     for (i in 1:n) {
  #       u <- users[i] + 1;
  #       t <- tracks[i] + 1;
  #       prediction <- cached[i] + t(ufeats[u,feat]) %*% tfeats[t,feat];
  #       err <- (ratings[i] - prediction);
  #       sq <- err^2;
  #       uv <- ufeats[u,feat];
  #       ufeats[u,feat] <- ufeats[u,feat] * (1 - k * lrate) + lrate * err * tfeats[t,feat];
  #       tfeats[t,feat] <- tfeats[t,feat] * (1 - k * lrate) + lrate * err * uv;
  #       sqerr <- sqerr + sq;
  #     }
  #   }
  #   for (i in 1:n) {
  #     u <- users[i] + 1;
  #     t <- tracks[i] + 1;
  #     cached[i] <- cached[i] + t(ufeats[u,feat]) %*% tfeats[t,feat];
  #   }
  # }
  # return(list(baseline=baseline,ufeats=ufeats, tfeats=tfeats));
}

funkpred <- function(train, test, ratings) {
  n <- nrow(train);
  ntrack <- max(train$Track) + 1;
  nuser <- max(train$User) + 1;
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



  sv <- funksvd(train$User, train$Track, ratings - baseline, 64);
  pred <- sapply(1:nrow(test), 
    function (i) {
      trk <- test$Track[i] + 1;
      usr <- test$User[i] + 1;
      base <- umean[usr] + tmean[trk] + mu;
      s <- t(sv$ufeats[usr,]) %*% sv$tfeats[trk,];
      return(base + s);
    });
  pred.train <- sapply(1:nrow(train), 
    function (i) {
      trk <- train$Track[i] + 1;
      usr <- train$User[i] + 1;
      base <- umean[usr] + tmean[trk] + mu;
      s <- t(sv$ufeats[usr,]) %*% sv$tfeats[trk,];
      return(base + s);
    });
  cat("Sanity check RMSE:", rmse(pred.train, ratings), '\n');
  return(list(pred=pred));
}