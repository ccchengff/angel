package com.tencent.angel.ml.FPGBDT.algo;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.util.BitSet;

/**
 * Created by ccchengff on 2017/11/20.
 */
public class MyBitSet extends BitSet {
  private static final Log LOG = LogFactory.getLog(MyBitSet.class);

  public MyBitSet(int size) {
    super(size);
  }

}
