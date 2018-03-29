package com.spark.mlib.optimizer

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD

/**
  * Created by qinzhiheng on 2017/7/18.
  */
trait Optimizer extends Serializable {
  def optimize(data: RDD[(Vector, Array[Double])], initialWeights: Vector): Vector
}
