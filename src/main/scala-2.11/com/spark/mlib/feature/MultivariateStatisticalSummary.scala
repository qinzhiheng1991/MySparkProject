package com.spark.mlib.feature

import org.apache.spark.mllib.linalg.Vector

/**
  * Created by qinzhiheng on 2017/11/21.
  */
trait MultivariateStatisticalSummary {

  /**
    * Sample mean vector.
    */
  def mean: Vector

  /**
    * Sample variance vector. Should return a zero vector if the sample size is 1.
    */
  def variance: Vector

  /**
    * Sample size.
    */
  def count: Long

  /**
    * Number of nonzero elements (including explicitly presented zero values) in each column.
    */
  def numNonzeros: Vector

  /**
    * Maximum value of each column.
    */
  def max: Vector

  /**
    * Minimum value of each column.
    */
  def min: Vector

  /**
    * Euclidean magnitude of each column
    */
  def normL2: Vector

  /**
    * L1 norm of each column
    */
  def normL1: Vector
}

