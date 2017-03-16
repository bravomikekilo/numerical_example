// author bravomikekilo bravomikekilo@buaa.edu.cn
// under GPLv3
// for LICENSE see LICENSE at root.
#include <cstdio>
#include <random>
#include <functional>
#include "KahanSummer.hpp"
#include <gmp.h>

using Kahan::KahanSummer;
const int times = 100000000; // 1 billion

int main(){
  std::random_device rd;
  auto engine = std::default_random_engine(rd());
  auto dist = std::uniform_real_distribution<float>(9, 11);
  auto dice = std::bind(dist, engine);
  float sum = times * 10.0;
  auto summer = KahanSummer<float>(times * 10.0);
  for(int i= 0; i < times; ++i){
    auto temp = dice();
    sum += temp;
    summer.add(temp);
  }
  printf("expect value of sum is %f\n", 2.0 * 10.0 * times);
  printf("value of naive sum is %f\n", sum);
  printf("value of kahan sum is %f\n", summer.get());
  printf("value of kahan res is %f\n", summer.get_res());
}
