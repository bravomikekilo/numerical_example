#ifndef KAHAN_SUMMER_HPP_H
#define KAHAN_SUMMER_HPP_H

namespace Kahan{

  template <typename T>
  class KahanSummer{
  public:
    KahanSummer(const T& initial = 0):sum(initial), res(0) {}

    void add(const T& in){
      auto term = in + res;
      auto temp = sum + term;
      auto fake = temp - sum;
      res = term - fake;
      sum = temp;
    }

    T get(){return sum;}

    T get_res(){return res;}

    void clear(const T& target=0){sum = target; res = 0;};

  private:
    T sum;
    T res;
  };

} // namespace Kahan
#endif
