/*
 * Tencent is pleased to support the open source community by making Angel available.
 *
 * Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 */

package com.tencent.angel.ml.lda.algo.structures;

import it.unimi.dsi.fastutil.HashCommon;

import java.util.Arrays;

public class S2ITraverseMap extends TraverseHashMap {

  public int[] value;
  boolean[] used;
  public short[] idx;
  short[] poss;

  public S2ITraverseMap(int expected) {
    super(expected);
    value = new int[n];
    used = new boolean[n];
    idx = new short[n];
    poss = new short[n];
  }

  @Override
  public short get(short k) {
    // The starting point
    int pos = (HashCommon.murmurHash3(k)) & mask;

    // There's always an unused entry.
    while (used[pos]) {
      if (key[pos] == k) {
        return (short) value[pos];
      }
      pos = (pos + 1) & mask;
    }
    return 0;
  }

  @Override
  public short get(int k) {
    return 0;
  }

  @Override
  public void put(short k, short v) {

  }

  public void put(short k, int v) {
    if (v == 0)
      return;

    // The starting point
    int pos = (HashCommon.murmurHash3(k)) & mask;

    // There's always an unused entry.
    while (used[pos]) {
      if (key[pos] == k) {
        value[pos] = v;
        return;
      }
      pos = (pos + 1) & mask;
    }

    used[pos] = true;
    key[pos] = k;
    value[pos] = v;
    idx[size] = (short) pos;
    poss[(short) pos] = size;
    size++;
  }

  @Override
  public void rehash() {
    short[] kkey = key;
    int[] vvalue = value;

    // print();

    key = new short[n];
    value = new int[n];

    Arrays.fill(used, false);

    int temp = size;
    size = 0;

    for (int i = 0; i < temp; i++) {
      short k = kkey[idx[i]];
      int v = vvalue[idx[i]];
      put(k, v);
    }
  }

  @Override
  public short dec(short k) {
    int pos = (HashCommon.murmurHash3(k)) & mask;

    while (used[pos]) {
      if (key[pos] == k) {
        value[pos]--;
        if (value[pos] == 0) {
          size--;
          idx[poss[pos]] = idx[size];
          poss[idx[size]] = poss[pos];
        }
        return (short) value[pos];
      }

      pos = (pos + 1) & mask;
    }
    return 0;
  }

  @Override
  public short dec(int k) {
    return dec((short) k);
  }

  @Override
  public short inc(short k) {
    int pos = (HashCommon.murmurHash3(k)) & mask;

    int cnt = 0;
    while (used[pos]) {
      if (key[pos] == k) {
        value[pos]++;
        if (value[pos] == 1) {
          idx[size] = (short) pos;
          poss[(short) pos] = size;
          size++;
        }

        return (short) value[pos];
      }

      cnt++;
      if (cnt > n) {
        rehash();
        return inc(k);
      }
      pos = (pos + 1) & mask;
    }

    key[pos] = k;
    value[pos] = 1;
    used[pos] = true;
    idx[size] = (short) pos;
    poss[(short) pos] = size;
    size++;
    return 1;
  }

  @Override
  public short inc(int k) {
    return inc((short) k);
  }

  @Override
  public int bytes() {
    int sum = 0;
    sum += key.length * 2;
    sum += value.length * 4;
    sum += used.length;
    sum += idx.length * 2;
    sum += poss.length * 2;
    return sum;
  }

  @Override
  public short getKey(int idx) {
    return key[this.idx[idx]];
  }

  @Override
  public int getVal(int idx) {
    return  value[this.idx[idx]];
  }
}
