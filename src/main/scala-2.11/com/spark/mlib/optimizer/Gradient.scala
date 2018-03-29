package com.spark.mlib.optimizer

import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector, Vectors}
import com.spark.mlib.linear.BLAS._
import org.apache.spark.SparkException


/**
  * Class Compute loss gradient exclude L1 or L2, L1 and L2 gradient computed by updater
  * by w - (w - regular_loss) which can be found in LBFGS
  *
  * Created by qinzhiheng on 2017/7/17.
  */
abstract class Gradient extends Serializable {
  def compute(data: Vector,
              num: Array[Double],
              dividends: Array[Vector],
              cumGradient: Vector): Double = {
    0.0
  }

  def compute(data: Vector,
              num: Array[Double],
              dividends: Array[Vector],
              cumGradient: Vector,
              breakpoint: Int): Double = {
    0.0
  }

  def compute(data: Vector,
              num: Array[Double],
              dividends: (Array[Vector], Array[Vector], Array[Vector]),
              cumGradient: Vector): Double = {
    0.0
  }
}

class LogisticGradient extends Gradient {
  /**
    *
    * @param data feature
    * @param dividends the gradient dividend, the process of weight is on the driver
    * @param cumGradient the total gradient
    * @return the loss
    *
    * Notice: because of Nan, we prefer to put EXP operator to dividend
    */
  override def compute(data: Vector,
                       num: Array[Double],
                       dividends: Array[Vector],
                       cumGradient: Vector): Double = {
    require(num.length * data.size == dividends(0).size,
      s"in ${this.getClass.getName}, class num: ${num.length}, data size: ${data.size}, dividend size: ${dividends(0).size}")
    require((num.length - 1) * data.size == cumGradient.size,
      s"in ${this.getClass.getName}, gradient length should match weights size}")

    val classNum = num.length
    val featureNum = data.size
    val instanceNum = num.sum
    val cumGradientValues = cumGradient.toDense.values
    var kModel = Vectors.zeros(classNum)
    var loss = 0.0
    var probability = 0.0
    var sum = 0.0
    var index = 0
    while (index < classNum) {
      sum = 0.0
      kModel = fDotExp(data, dividends(index))
      kModel.foreachActive((_, value) => {sum += value})
      probability = 1.0 / sum
      if (index > 0) {
        val gradientAtIndex = (instanceNum - num(index)) * probability + num(index) * (probability - 1.0)
        data.foreachActive((i, value) => cumGradientValues((index - 1) * featureNum + i) += gradientAtIndex * value)
      }
      loss += -1.0 * Math.log(probability + 1e-9) * num(index)
      index += 1
    }
    loss
  }
}

class TwoStageModelGradient extends Gradient {
  /**
    *
    * @param data feature
    * @param dividends the model
    * @param cumGradient the total gradient
    * @param breakpoint the breakpoint of two-stage model
    * @return the loss
    *
    * Notice: because of Nan, we prefer to put EXP operator to dividend
    */
  override def compute(data: Vector,
                       num: Array[Double],
                       dividends: Array[Vector],
                       cumGradient: Vector,
                       breakpoint: Int): Double = {
    require(data.size == dividends(0).size,
      s"in ${this.getClass.getName}, class num: ${num.length}, data size: ${data.size}, dividend size: ${dividends(0).size}")
    require((num.length - 1) * data.size == cumGradient.size,
      s"in ${this.getClass.getName}, gradient length should match weights size")
    require(breakpoint >= 1 && breakpoint < data.size - 1,
      s"in ${this.getClass.getName}, breakpoint is wrong, do not match")
    require(num.length == 2,
      s"n ${this.getClass.getName}, multi-stage only supports binary classification")
    val numFeature = data.size
    val weightsCopy = dividends(0).copy
    val weightsCopyValues = weightsCopy.toDense.values
    var sum1 = 0.0
    var sum2 = 0.0
    data.foreachActive((i, value) => {
      if (i < breakpoint) {
        sum1 += weightsCopyValues(i) * value
      } else {
        sum2 += weightsCopyValues(i) * value
      }
    })
    sum1 = 1.0 / (1.0 + math.exp(-1.0 * sum1))
    sum2 = 1.0 / (1.0 + math.exp(-1.0 * sum2))
    val grad1 = (num(1) * (sum1 - 1) + (num(0) + num(1)) * sum1 * sum2 * (1 - sum1)) / (1 - sum1 * sum2)
    val grad2 = (num(1) * (sum2 - 1) + (num(0) + num(1)) * sum1 * sum2 * (1 - sum2)) / (1 - sum1 * sum2)
    val cumGradientValue = cumGradient.toDense.values
    var index = 0
    while (index < numFeature) {
      if (index < breakpoint) {
        cumGradientValue(index) += grad1 * data(index)
      } else {
        cumGradientValue(index) += grad2 * data(index)
      }
      index += 1
    }
    val loss = -1.0 * num(1) * math.log(sum1 * sum2) - num(0) * math.log(1 - sum1 * sum2)
    loss
  }
}

class FMGradient extends Gradient {
  /**
    *
    * @param data feature
    * @param dividends the model transform vector
    * @param cumGradient the total gradient
    * @return the loss
    *
    * Notice: because of Nan, we prefer to put EXP operator to dividend
    */
  override def compute(data: Vector,
                       num: Array[Double],
                       dividends: (Array[Vector], Array[Vector], Array[Vector]),
                       cumGradient: Vector): Double = {
    require(dividends._1(0).size == (data.size + 1) * num.length,
      s"in ${this.getClass.getName}, class num: ${num.length}, data size: ${data.size}, dividend._1 size: ${dividends._1(0).size}")
    require(dividends._1.length == num.length && dividends._2.length == num.length && dividends._3.length == num.length,
      s"in ${this.getClass.getName}, dividends length: 1.${dividends._1.length},2.${dividends._2.length},3.${dividends._3.length},"
        + s"num class:${num.length}")
    val embedding_dim = dividends._2(0).size / (data.size * num.length)
    val numFeature = data.size
    val classNum = num.length
    require(cumGradient.size == (1 + data.size + embedding_dim * data.size) * (num.length - 1),
      s"gradient length: ${cumGradient.size} not match data size:${data.size}, embedding:$embedding_dim, classNum: ${num.length}")

    val dataExtend = data match {
      case value: DenseVector =>
        Vectors.dense(Array(1.0) ++ value.values)
      case value: SparseVector =>
        val indices = Array(0) ++ value.indices.map(_ + 1)
        val values = Array(1.0) ++ value.values
        Vectors.sparse(value.size + 1, indices, values)
      case _ =>
        throw new SparkException(s"cant not parse ${data.getClass.getName}")
    }
    val instanceNum = num.sum
    val cumGradientValues = cumGradient.toDense.values // cumGradientValues used to update the value of cumGradient
    val eachModelLength = 1 + numFeature + embedding_dim * numFeature // each class model parameter length
    val featureMinusPartArray = dividends._1 // weight model minus, the 0th is raw feature parameter
    val embeddingMinusPartArray = dividends._2 // embedding parameter minus, the 0th is raw embedding
    val embeddingPlusPartArray = dividends._3 // embedding parameter plus, the 0th is raw embedding
    var kModel = Vectors.zeros(classNum) // store each exponent parameter value in the dividend
    val rawEmbeddingPara = embeddingMinusPartArray(0) // rawEmbedding parameter, numFeautres * embedding_size * classNum
    var loss = 0.0
    var index = 0
    // in order to compute gradient for each class model in soft max, firstly we need to compute each class
    // probability, secondly we need to compute dy/dw,
    // inorder to accelerate computing, we use 1/2*((A+B)^2 -(A^2+B^2)) = AB to compute the cross item in FM
    while (index < classNum) {
      val featureVector = featureMinusPartArray(index)
      val embeddingMinusVector = embeddingMinusPartArray(index)
      val embeddingPlusVector = embeddingPlusPartArray(index)
      kModel = fDot(dataExtend, featureVector) // linear part
      val kModelValues = kModel.toDense.values
      var feature_index = 0
      var embedding_index = 0
      var inner_index = 0

      while (inner_index < classNum) {
        var embeddingCrossItem = 0.0
        embedding_index = 0
        while (embedding_index < embedding_dim) {
          //compute (sum((vik + vik')*xi) * sum((vik-vik')*xi)) - sum((vik + vik')(vik - vik')xi^2), i is feature index
          var crossPlusItem = 0.0
          var crossMinusItem = 0.0
          var squareSumItem = 0.0
          feature_index = 0
          while (feature_index < numFeature) {
            val embeddingPlusMinusItem = inner_index * numFeature * embedding_dim + feature_index * embedding_dim + embedding_index
            val embeddingPlusItem = embeddingPlusVector(embeddingPlusMinusItem)
            val embeddingMinusItem = embeddingMinusVector(embeddingPlusMinusItem)
            crossPlusItem += embeddingPlusItem * data(feature_index)
            crossMinusItem += embeddingMinusItem * data(feature_index)
            squareSumItem += embeddingPlusItem * embeddingMinusItem * data(feature_index) * data(feature_index)
            feature_index += 1
          }
          embeddingCrossItem += crossPlusItem * crossMinusItem - squareSumItem
          embedding_index += 1
        }
        kModelValues(inner_index) += 1.0 / 2 * embeddingCrossItem
        inner_index += 1
      }
      var sum = 0.0
      kModel.foreachActive((_, value) => sum += math.exp(value))
      val probability = 1.0 / sum
      //compute dy/dw, w = w0, dy/dw0 = 1; w = wi, dy/dwi = xi; w = vjk, dy/dvjk = xj * sum(vik * xi) - vjk * xj^2
      if (index > 0) {
        val dyw = Vectors.zeros(eachModelLength)
        val dywValue = dyw.toDense.values
        dataExtend.foreachActive((i, value) => {
          dywValue(i) = value
        })
        var embed_index = 0
        var dywIndex = 0
        feature_index = 0
        while (feature_index < numFeature) {
          embedding_index = 0
          while (embedding_index < embedding_dim) {
            var sum = 0.0
            var i = 0
            while (i < numFeature && i != feature_index) {
              embed_index = (index - 1) * embedding_dim * numFeature + i * embedding_dim + embedding_index
              sum += rawEmbeddingPara(embed_index) * data(i)
              i += 1
            }
            dywIndex = numFeature + 1 + feature_index * embedding_dim + embedding_index
            dywValue(dywIndex) = data(feature_index) * sum
            embedding_index += 1
          }
          feature_index += 1
        }
        //update this instance corresponding gradient
        val gradientAtIndex = (instanceNum - num(index)) * probability + num(index) * (probability - 1.0)

        dyw.foreachActive((i, value) => {
          cumGradientValues((index - 1) * eachModelLength + i) += gradientAtIndex * value
        })
      }
      loss += -1.0 * Math.log(probability + 1e-9) * num(index)
      index += 1
    }
    loss
  }
}