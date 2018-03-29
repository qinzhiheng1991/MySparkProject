package com.spark.mlib.feature

import org.apache.spark.SparkException
import org.apache.spark.mllib.linalg.{Vector, Vectors}


/**
  * Created by qinzhiheng on 2017/11/23.
  */

/**
  * Class that represents the features, labels and counts of a data point.
  *
  * @param features List of features for this data point.
  * @param num Numbers of this feature and labels, the length represents the class number.
  */
case class MyLabeledPoint(features: Vector, num: Array[Double]) {
  override def toString: String = s"($features, [${num.mkString(", ").trim}])"
}

object MyLabeledPoint {
  def parse(s: String): MyLabeledPoint = {
    if (s.startsWith("(")) {
      NumericParser.parse(s) match {
        case Seq(features: Any, num: Array[Double]) =>
          features match {
            case Seq(size: Double, indices: Array[Double], values: Array[Double]) =>
              MyLabeledPoint(Vectors.sparse(size.toInt, indices.map(_.toInt), values), num)
            case values: Array[Double] =>
              MyLabeledPoint(Vectors.dense(values), num)
            case other =>
              throw new SparkException(s"can not parse ${other.getClass.getName}")
          }
        case other =>
          throw new SparkException(s"can not parse ${other.getClass.getName}")
      }
    }
    else {
      val num_start = s.indexOf("[")
      val num_end = s.lastIndexOf("]")
      if (num_start <= 1 || num_end <= 3 || (num_end - num_start) < 4)
        throw new SparkException(s"can not parse ${s.trim}")
      val feature = s.substring(0, num_start - 1).split(",").map(_.trim.toDouble)
      val num = s.substring(num_start + 1, num_end).split(",").map(_.trim.toDouble)
      MyLabeledPoint(Vectors.dense(feature), num)
    }
  }
}
