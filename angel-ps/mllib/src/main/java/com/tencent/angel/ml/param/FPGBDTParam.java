package com.tencent.angel.ml.param;

/**
 * Created by ccchengff on 2017/11/17.
 */
public class FPGBDTParam extends RegTParam {
  public int numTree = 10;
  public int numThread = 20;
  //public int numBatch = 10000;

  public int numWorker = 1;
  public int featLo;
  public int featHi;

}
