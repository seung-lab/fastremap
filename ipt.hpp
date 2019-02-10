/* ipt.hpp - In-Place Transposition
 *
 * When transitioning between different media,
 * e.g. CPU to GPU, CPU to Network, CPU to disk,
 * it's often necessary to physically transpose
 * multi-dimensional arrays to reformat as C or
 * Fortran order. Tranposing matrices is also 
 * a common action in linear algebra, but often
 * you can get away with just changing the strides.
 *
 * An out-of-place transposition is easy to write,
 * often faster, but will spike peak memory consumption.
 *
 * This library grants the user the option of performing
 * an in-place transposition which trades CPU time for
 * peak memory usage.
 *
 * Author: William Silversmith
 * Date: Feb. 2019
 */

#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstdint>
#include <stdio.h>
#include <iostream>
#include <vector>

#include "libdivide.h"

#ifndef IN_PLACE_TRANSPOSE_H
#define IN_PLACE_TRANSPOSE_H

// ipt = in-place transpose
namespace ipt {

template <typename T>
void square_ipt_2d(T* arr, const int sx, const int sy) {
  T tmp = 0;

  int k = 0;
  int kprime = 0;
  for (int y = 0; y < sy; y++) {
    k = sx * y;
    kprime = y;
    for (int x = y; x < sx; x++) {
      k += x;
      kprime += sy * x;

      tmp = arr[kprime];
      arr[kprime] = arr[k];
      arr[k] = tmp;
    }
  }
}

/* A permutation, P(k), is a mapping of
  * one arrangement of numbers to another.
  * For an m x n array, the permuatation
  * mapping from C to Fortran order is:
  *
  * P(k) := mk mod mn - 1
  * iP(k) := nk mod mn - 1 (the inverse)
  *
  * Where does this come from? Assume we are
  * going from C to Fortran order (it doesn't
  * matter either way). The indicies are defined
  * as:
  * 
  * k = C(x,y) = x + sx * y
  *     F(x,y) = y + sy * x
  *
  * The permutation P(k) is the transformation:
  * 
  * P(C(x,y)) = F(x,y)
  *
  * 1. P(x + sx * y) = y + sx * x
  * 2. sy (x + sx y) = sy x + sx sy y 
  * 3. Let q = (sx sy - 1)
  * 4. sy x + sx sy y % q
  * 5. ((sy x % q) + (sx sy y % q)) % q by distributive identity
  * 6. sy x is identical b/c q is always bigger
  * 7. sx sy y reduces to y 
  * 8 q is always bigger than sy x + y so it disappears
  * 
  * ==> P(k) = y + sy * x = F(x,y)
  * ==> P(k) = sy * k % (sx sy - 1)
  * 
  * Note that P(0) and P(q) are always 0 and q respectively.
  *
  * Now we need a way to implement this insight.
  * How can we move the data around without using too
  * much extra space? A simple algorithm is 
  * "follow-the-cycles". Each time you try moving a
  * k to P(k), it displaces the resident tile. Eventually,
  * this forms a cycle. When you reach the end of a cycle,
  * you can stop processing and move to unvisited parts of
  * the array. This requires storing a packed bit representation
  * of where we've visited to make sure we get everything.
  * This means we need to store between 2.0x and 1.016x
  * memory in the size of the original array depending on its
  * data type (2.0x would be a transpose of another bit packed 
  * array and 1.016x would be 64-bit data types).
  *
  * There are fancier algorithms that use divide-and-conquer,
  * and SIMD tricks, and near zero extra memory, but 
  * this is a good place to start. Fwiw, the bit vector
  * has an O(nm) time complexity (really 2nm) while the 
  * sans-bit vector algorithms are O(nm log nm).
  */
template <typename T>
void rect_ipt_2d(T* arr, const int sx, const int sy) {
  const int sxy = sx * sy;

  std::vector<bool> visited;
  visited.resize(sxy);

  visited[0] = true;
  visited[sxy - 1] = true;

  const int q = sxy - 1;
  const libdivide::divider<int> fast_sx(sx);
  int i, k, next_k;
  T tmp1, tmp2;
  
  for (int i = 1; i < q; i++) {
    if (visited[i]) {
      continue;
    }

    k = i;
    tmp1 = arr[k];
    next_k = sy * k - q * (k / fast_sx); // libdivide version

    while (!visited[next_k]) {
      tmp2 = arr[next_k];
      arr[next_k] = tmp1;
      tmp1 = tmp2;
      visited[next_k] = true;
      k = next_k;
      next_k = sy * k - q * (k / fast_sx); // libdivide version
    }
  }
}

// note: sx == sy == sz... find better convention?
template <typename T>
void square_ipt_3d(
    T* arr, 
    const int sx, const int sy, const int sz
  ) {

  T tmp = 0;

  const int sxy = sx * sy;
  const int syz = sy * sz;

  int k = 0;
  int kprime = 0;
  for (int z = 0; z < sz; z++) {
    for (int y = 0; y < sy; y++) {
      k = sx * y + sxy * z;
      kprime = z + sz * y;
      for (int x = z; x < sx; x++) {
        k += x;
        kprime += syz * x;

        tmp = arr[kprime];
        arr[kprime] = arr[k];
        arr[k] = tmp;
      }
    }
  }
}


};

#endif
