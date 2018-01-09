package com.tencent.angel.ml.FPGBDT.algo.hash;

public abstract class Int2IntHash {
  protected int length;

  public Int2IntHash(int length) {
    this.length = length;
  }

  public abstract int encode(int key);
}
