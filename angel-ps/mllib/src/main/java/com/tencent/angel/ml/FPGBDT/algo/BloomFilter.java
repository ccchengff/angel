package com.tencent.angel.ml.FPGBDT.algo;

import com.tencent.angel.ml.FPGBDT.algo.hash.*;

import java.util.BitSet;

public class BloomFilter {
  private static final int DEFAULT_HASH_NUM = 3;

  int size;
  BitSet table;
  Int2IntHash[] hash;

  public BloomFilter(int size, int hashNum) {
    this.size = size;
    this.table = new BitSet(size);
    this.hash = new Int2IntHash[hashNum];
    hash[0] = new Mix64Hash(size);
    if (hashNum > 1) hash[1] = new TWHash(size);
    if (hashNum > 2) hash[2] = new BJHash(size);
    if (hashNum > 3) hash[3] = new BKDRHash(size, 31);
    if (hashNum > 4) hash[4] = new BKDRHash(size, 131);
  }

  public BloomFilter(int size) {
    this(size, DEFAULT_HASH_NUM);
  }

  public void insert(int key) {
    for (Int2IntHash h: hash) {
      int code = h.encode(key);
      table.set(code);
    }
  }

  public boolean query(int key) {
    for (Int2IntHash h: hash) {
      int code = h.encode(key);
      if (table.get(code))
        return true;
    }
    return false;
  }

}
