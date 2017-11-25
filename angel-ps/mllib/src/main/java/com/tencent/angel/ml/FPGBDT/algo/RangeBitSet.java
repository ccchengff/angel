package com.tencent.angel.ml.FPGBDT.algo;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.io.Serializable;
import java.util.Arrays;
import java.util.BitSet;

/**
 * Created by ccchengff on 2017/11/20.
 */
public class MyBitSet implements Serializable {
  private static final Log LOG = LogFactory.getLog(MyBitSet.class);

  private byte[] bits;
  private int from;
  private int to;
  private int offset;

  public MyBitSet(int from, int to) {
    this.from = from;
    this.to = to;
    this.offset = (int)(from & 0b111);
    this.bits = new byte[needNumBytes(from, to)];
  }

  private int needNumBytes(long from, long to) {
    int first = (int)(from >> 3);
    int last = (int)(to >> 3);
    return last - first + 1;
  }

  public void set(int index) {
    index = index - from + offset;
    int x = index >> 3;
    int y = index & 0b111;
    bits[x] = (byte)(bits[x] | (1 << y));
  }

  public boolean get(int index) {
    index = index - from + offset;
    int x = index >> 3;
    int y = index & 0b111;
    return ((bits[x] >> y) & 0x1) == 1;
  }

  public byte[] toByteArray() {
    //return bits.clone();
    return bits;
  }

}
