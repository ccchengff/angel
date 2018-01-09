package com.tencent.angel.ml.FPGBDT.algo.storage

import com.tencent.angel.ml.param.FPGBDTParam

abstract class FPRegTDataStore(param: FPGBDTParam, _numInstance: Int) {
  val numInstance: Int = _numInstance

  var labels: Array[Float] = _
  var preds: Array[Float] = _
  var weights: Array[Float] = _

  def getNumInstances: Int = numInstance

  def getLabel(i: Int) = labels(i)

  def getLabels = labels

  def setLabels(labels: Array[Float]): Unit = {
    this.labels = labels
  }

  def getPred(i: Int) = preds(i)

  def getPreds = preds

  def setPreds(preds: Array[Float]): Unit = {
    this.preds = preds
  }

  def getWeight(i: Int) = weights(i)

  def getWeights = weights

  def setWeights(weights: Array[Float]): Unit = {
    this.weights = weights
  }
}
