package com.tencent.angel.ml.FPGBDT.algo.hash;

public class Mix64Hash extends Int2IntHash {
  public Mix64Hash(int length) {
    super(length);
  }

  @Override
  public int encode(int key) {
    key = (~key) + (key << 21); // code = (code << 21) - code - 1;
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8); // code * 265
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4); // code * 21
    key = key ^ (key >> 28);
    key = key + (key << 31);
    key = key % length;
    return key >= 0 ? key : key + length;
  }
}
