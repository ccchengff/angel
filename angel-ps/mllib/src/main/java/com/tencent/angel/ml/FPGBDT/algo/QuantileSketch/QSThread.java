package com.tencent.angel.ml.FPGBDT.algo.QuantileSketch;

/**
 * Created by ccchengff on 2017/11/30.
 */
public class QSThread implements Runnable {
  int threadIdx;
  HeapQuantileSketch qs;
  boolean[] isFinished;
  float[] vals;
  int begin;
  int length;

  public QSThread(int threadIdx, HeapQuantileSketch qs, boolean[] isFinished,
                  float[] vals, int begin, int length) {
    this.threadIdx = threadIdx;
    this.qs = qs;
    this.isFinished = isFinished;
    this.vals = vals;
    this.begin = begin;
    this.length = length;
  }

  @Override
  public void run() {
    int end = begin + length;
    for (int i = begin; i < end; i++) {
      qs.update(vals[i]);
    }
    isFinished[threadIdx] = true;
  }
}
