#include <cstdio>
#include <random>
#include <functional>

int main(){
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(9, 11);
  auto dice = std::bind(distribution, generator);
  const float base = 1E9;
  float sum_big = base;
  float sum_small = 0;
  for(int i = 0; i < 1000000; ++i){
    auto temp = dice();
    sum_big += temp;
    sum_small += temp;
  }
  printf("result without Kahan is %f \n", sum_big);
  printf("result with Kahan is %f \n", sum_small + base);
  printf("mathematical expect of result is %f \n", 10 * 1000000 + base);
}
