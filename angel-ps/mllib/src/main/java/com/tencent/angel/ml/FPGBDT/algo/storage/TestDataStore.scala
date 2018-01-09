package com.tencent.angel.ml.FPGBDT.algo.storage

import java.util

import com.tencent.angel.ml.math.vector.SparseDoubleSortedVector
import com.tencent.angel.ml.param.FPGBDTParam
import org.apache.commons.logging.LogFactory

/**
  * Created by ccchengff on 2017/12/1.
  */
class TestDataStore(param: FPGBDTParam, numInstance: Int) extends FPRegTDataStore(param, numInstance) {
  val LOG = LogFactory.getLog(classOf[TestDataStore])

  var instances: util.List[SparseDoubleSortedVector] = _

  def getInstance(index: Int): SparseDoubleSortedVector = instances.get(index)

  def getInstances: util.List[SparseDoubleSortedVector] = instances

  def setInstances(instances: util.List[SparseDoubleSortedVector]): Unit = {
    this.instances = instances
  }

}
