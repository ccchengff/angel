package com.tencent.angel.ml.FPGBDT.algo.hash;

public class BKDRHash extends Int2IntHash {
  private int seed;
  private static final int DEFAULT_SEED = 31;

  public BKDRHash(int length, int seed) {
    super(length);
    this.seed = seed;
  }

  public BKDRHash(int length) {
    this(length, DEFAULT_SEED);
  }

  @Override
  public int encode(int key) {
    int code = 0;
    while (key != 0) {
      code = seed * code + (key % 10);
      key /= 10;
    }
    key = key % length;
    return key >= 0 ? key : key + length;
  }
}
