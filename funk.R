funksvd <- function (users, tracks, ratings, nfeats=16) {
  n <- length(users);
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

funkpred <- function(train, test, ratings, nfeats=32) {
  sv <- funksvd(train$User, train$Track, ratings, nfeats);
  pred <- sapply(1:nrow(test), 
    function (i) {
      trk <- test$Track[i] + 1;
      usr <- test$User[i] + 1;
      s <- t(sv$ufeats[usr,]) %*% sv$tfeats[trk,];
      return(s);
    });
  return(list(pred=pred));
}