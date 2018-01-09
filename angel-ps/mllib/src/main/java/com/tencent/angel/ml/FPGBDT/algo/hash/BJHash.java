package com.tencent.angel.ml.FPGBDT.algo.hash;

public class BJHash extends Int2IntHash {
  public BJHash(int length) {
    super(length);
  }

  @Override
  public int encode(int key) {
    key = (key + 0x7ed55d16) + (key << 12);
    key = (key ^ 0xc761c23c) ^ (key >> 19);
    key = (key + 0x165667b1) + (key << 5);
    key = (key + 0xd3a2646c) ^ (key << 9);
    key = (key + 0xfd7046c5) + (key << 3);
    key = (key ^ 0xb55a4f09) ^ (key >> 16);
    key = key % length;
    return key >= 0 ? key : key + length;
  }
}
