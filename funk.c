#include <math.h>
#include <stdlib.h>
#include <stdio.h>
void funksvd(int* pn, int* users, int* tracks, double* ratings, int* pnfeats, int* pnuser, int* pntrack, double* ufeats, double* tfeats) {
  int n = *pn;
  int nfeats = *pnfeats;
  int nuser = *pnuser;
  int ntrack = *pntrack;
  int min_loops = 30;
  double min_improve = 1e-4;
  double k = 4;
  double lrate = 1e-3;
  
  int i = 0;
  int j = 0;
  printf("Initialising user feats (%d x %d)\n", nuser, nfeats);
  for (i = 0; i < nuser * nfeats; i++) {
    //printf("Item %d\n", i);
    ufeats[i] = 1;
  }
  
  printf("Initialising track feats\n");
  for (i = 0; i < ntrack * nfeats; i++) {
    tfeats[i] = 1;
  }
  printf("Allocating cache\n");
  
  double* cached = calloc(n, sizeof(double));
  
  double sqerr = 0;
  double sqerr_old = 0;
  for (int feat = 0; feat < nfeats; feat++) {
    printf("Calculating feature %d (error = %f)\n", feat, sqrt(sqerr/n));
    int loops = 0;
    
    
    while ((loops < min_loops) || (sqrt(sqerr_old/n) - sqrt(sqerr/n) > min_improve)) {
      loops += 1;
      //printf("Iterating... (error = %f improvement = %f )\n", sqrt(sqerr/n), sqrt(sqerr_old/n) - sqrt(sqerr/n));
      sqerr_old = sqerr;
      sqerr = 0;
      for (i = 0; i < n; i++) {
        double prediction;
        int u = users[i];
        int t = tracks[i];
        prediction = cached[i] + ufeats[u + feat * nuser] * tfeats[t + feat * ntrack];
        double err = ratings[i] - prediction;
        double uv = ufeats[u + feat * nuser];
        ufeats[u + feat * nuser] += lrate * (err * tfeats[t + feat * ntrack] - ufeats[u + feat * nuser] * k);
        tfeats[t + feat * ntrack] += lrate * (err * uv - tfeats[t + feat * ntrack] * k);
        sqerr += err * err;
      }
    }
    for (i = 0; i < n; i++) {
      int u = users[i];
      int t = tracks[i];
      cached[i] += ufeats[u + feat * nuser] * tfeats[t + feat * ntrack];
    }
  }
  free(cached);
}