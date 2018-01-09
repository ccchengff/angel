package com.tencent.angel.ml.FPGBDT.algo.hash;

public class TWHash extends Int2IntHash {
  public TWHash(int length) {
    super(length);
  }

  @Override
  public int encode(int key) {
    key = ~key + (key << 15);
    key = key ^ (key >> 12);
    key = key + (key << 2);
    key = key ^ (key >> 4);
    key = key * 2057;
    key = key ^ (key >> 16);
    key = key % length;
    return key >= 0 ? key : key + length;
  }
}
