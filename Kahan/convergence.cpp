// author bravomikekilo bravomikekilo@buaa.edu.cn
// under GPLv3
// for LICENSE see LICENSE at root.
#include <cstdio>
#include <climits>

int main(){
  float sum = 0;
  for(int i = 1; i < INT_MAX; ++i){
    float temp = sum + 1.0 / i;
    if(sum == temp){
      printf("convergence at %d, %f\n", i, sum);
      return 0;
    }
    sum = temp;
  }
  printf("can not convergence\n");
}
