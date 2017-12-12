package com.tencent.angel.ml.FPGBDT.algo.FPRegTreeDataStore

import java.util

import com.tencent.angel.ml.math.vector.SparseDoubleSortedVector
import com.tencent.angel.ml.param.FPGBDTParam
import org.apache.commons.logging.LogFactory

/**
  * Created by ccchengff on 2017/12/1.
  */
class TestDataStore(param: FPGBDTParam, _numInstance: Int) {
  val LOG = LogFactory.getLog(classOf[TestDataStore])

  val numInstance: Int = _numInstance
  var instances: util.List[SparseDoubleSortedVector] = _
  var labels: Array[Float] = _
  var preds: Array[Float] = _
  var weights: Array[Float] = _

  def getNumInstances: Int = numInstance

  def getInstance(index: Int): SparseDoubleSortedVector = instances.get(index)

  def getInstances: util.List[SparseDoubleSortedVector] = instances

  def setInstances(instances: util.List[SparseDoubleSortedVector]): Unit = {
    this.instances = instances
  }

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
