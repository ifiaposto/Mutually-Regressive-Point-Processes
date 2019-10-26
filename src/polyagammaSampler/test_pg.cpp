
// #include "PolyaGamma.h"
#include "PolyaGamma.h"
#include <stdio.h>
#include <time.h>

const int N = 100;
const int nthreads = 4;
//g++ -g -std=c++11 -Wall PolyaGamma.cpp RNG.cpp GRNG.cpp test_pg.cpp -lgsl -lcblas -llapack
int main(void) {
  
  int M = 500;

  double samp[N];
  int    df[N];
  double psi[N];

  for (int i = 0; i < N; i++) {
    df [i] = 1;
    psi[i] = 2.0;
  }

  //------------------------------------------------------------------------------

  PolyaGamma pg;
  RNG r(time(NULL));
  time_t start, end;
  
  start = time(NULL);

  
  for(int i = 0; i < N; i++){
	  samp[i] = pg.draw(df[i], psi[i], r);
	  std::cout<<"sample "<<i<<" "<<samp[i]<<std::endl;
  }

  end = time(NULL);
  double diff = (double)(end - start);
  printf("Time: %f sec. (serial) for %i.\n", diff, N);

  // Write it.
  FILE *file = fopen("ser.txt","w");
  for (int i = 0; i < N; i++) {
    fprintf(file,"%g\n", samp[i]);
  }
  fclose(file);

  return 0;
}

