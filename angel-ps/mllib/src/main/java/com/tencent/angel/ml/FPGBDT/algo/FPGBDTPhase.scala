package com.tencent.angel.ml.FPGBDT.algo

/**
  * Created by ccchengff on 2017/11/17.
  */
object FPGBDTPhase {
  val NEW_TREE: Int = 0
  val CHOOSE_ACTIVE: Int = 1
  val RUN_ACTIVE: Int = 2
  val FIND_SPLIT: Int = 3
  val AFTER_SPLIT: Int = 4
  val FINISH_TREE: Int = 5
  val FINISHED: Int = 6
  val CREATE_SKETCH: Int = 7 // should abandon
}
