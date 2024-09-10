#include <cstddef>
#include <cstdint>
#include <iostream>

#define container_of(ptr, type, member)                                        \
  ({                                                                           \
    const typeof(((const type *)0)->member) *__mptr = (ptr);                   \
    (const type *)((const char *)__mptr - offsetof(type, member));             \
  })

template <class T> struct RcBox {
  size_t rc;
  void *next;
  void *vtable;
  T data;
};
struct List {
  int64_t data;
  RcBox<List> *next;
};

extern "C" void opaque(List *ref) {
  for (int i = 0; i < 100; ++i) {
    std::cout << ref->data << std::endl;
    auto *c = container_of(ref, RcBox<List>, data);
    if (c->rc == 0)
      std::abort();
    ref = &ref->next->data;
  }
};

extern "C" void test();

int main() { test(); }
