#include <cstdio>
#include <climits>
#include "KahanSummer.hpp"

using Kahan::KahanSummer;

const int kill_limit = 1000;
int main(){
  auto sum = KahanSummer<float>(0);
  for(int i = 1, kill = kill_limit; i < INT_MAX; ++i){
    auto temp = sum.get();
    sum.add(1.0 / i);
    if(sum.get() == temp){
      if(kill > 0){
        --kill;continue;
      }else{
        printf("convergence at %d, %f\n", i, sum.get());
        return 0;
      }
    }else{
      kill = kill_limit;
    }
  }
  printf("can not convergence\n");
}
