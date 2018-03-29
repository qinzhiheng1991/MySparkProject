package com.spark.mlib.optimizer

import com.spark.mlib.linear.BLAS._
import org.apache.spark.mllib.linalg.Vector

/**
  * Created by qinzhiheng on 2017/7/17.
  */
abstract class Updater extends Serializable {
  def compute(weightsOld: Vector,
              gradient: Vector,
              stepSize: Double,
              iter: Int,
              regParam: Double): (Vector, Double) = {
    null
  }

  def adagradCompute(weightsOld: Vector,
                     gradient: Vector,
                     cumSquareGradient: Vector,
                     stepSize: Double,
                     epsilon: Double,
                     regParam: Double): (Vector, Double) = {
    null
  }

  def adamCompute(weightsOld: Vector,
                     gradient: Vector,
                     cumSquareGradient: Vector,
                     stepSize: Double,
                     epsilon: Double,
                     regParam: Double): (Vector, Double) = {
    null
  }
}

class L1Updater extends Updater {
}

class L2Updater extends Updater {
}


class LogisticL1Updater extends L1Updater {
  /**
    *
    * @param weightsOld old model para
    * @param gradient gradient
    * @param stepSize step size
    * @param iter iter num
    * @param regParam reg para
    * @return (updated para, cum gradient, new norm value)
    */
  override def compute(weightsOld: Vector,
                       gradient: Vector,
                       stepSize: Double,
                       iter: Int,
                       regParam: Double): (Vector, Double) = {
    require(weightsOld.size == gradient.size)
    val thisStepSize = stepSize / math.sqrt(iter)
    axpy(-1.0 * thisStepSize, gradient, weightsOld)
    val shrinkage = thisStepSize * regParam
    val size = weightsOld.size
    val weights = weightsOld.copy
    val weightsValues = weights.toDense.values
    var i = 0
    var norm = 0.0
    while (i < size) {
      weightsValues(i) = math.signum(weightsValues(i)) * math.max(0.0, math.abs(weightsValues(i)) - shrinkage)
      norm += math.abs(weightsValues(i))
      i += 1
    }
    //val norm = regParam * math.sqrt(dot(weightsVector, weightsVector))
    (weights, regParam * norm)
  }
}

class LogisticL2Updater extends L2Updater {
  /**
    *
    * @param weightsOld old model para
    * @param gradient gradient
    * @param stepSize step size
    * @param iter the iter num
    * @param regParam reg para
    * @return (updated para, new norm value)
    */
  override def compute(weightsOld: Vector,
                       gradient: Vector,
                       stepSize: Double,
                       iter: Int,
                       regParam: Double): (Vector, Double) = {
    require(weightsOld.size == gradient.size)
    val thisStepSize = stepSize / math.sqrt(iter)
    val weights = weightsOld.copy
    axpy(-1.0 * thisStepSize * regParam, weightsOld, weights)
    axpy(-1.0 * thisStepSize, gradient, weights)
    val norm = 1.0 / 2 * regParam * dot(weights, weights)
    (weights, norm)
  }
}

class AdagradL1Updater extends L1Updater {
  /**
    *
    * @param weightsOld old model para
    * @param gradient gradient
    * @param cumSquareGradient each feature's accumulate square of gradient
    * @param stepSize step size
    * @param epsilon smooth parameter
    * @param regParam reg para
    * @return (updated para, cum gradient, new norm value)
    */
  override def adagradCompute(weightsOld: Vector,
                              gradient: Vector,
                              cumSquareGradient: Vector,
                              stepSize: Double,
                              epsilon: Double,
                              regParam: Double): (Vector, Double) = {
    require(weightsOld.size == gradient.size)
    val size = weightsOld.size
    val weightsOldCopy = weightsOld.copy
    val weightsOldValues = weightsOldCopy.toDense.values
    val learning_rates = Array.ofDim[Double](size)
    var i = 0
    while (i < size) {
      learning_rates(i) = stepSize / math.sqrt(cumSquareGradient(i) + epsilon)
      weightsOldValues(i) -= learning_rates(i) * gradient(i)
      i += 1
    }

    val weights = weightsOld.copy
    val weightsValues = weights.toDense.values
    i = 0
    var norm = 0.0
    while (i < size) {
      weightsValues(i) = math.signum(weightsValues(i)) * math.max(0.0, math.abs(weightsValues(i)) - learning_rates(i) * regParam)
      norm += math.abs(weightsValues(i))
      i += 1
    }
    //val norm = regParam * math.sqrt(dot(weightsVector, weightsVector))
    (weights, regParam * norm)
  }
}

class AdagradL2Updater extends L2Updater {
  /**
    *
    * @param weightsOld old model para
    * @param gradient gradient
    * @param cumSquareGradient each feature's accumulate square of gradient
    * @param stepSize step size
    * @param epsilon smooth parameter
    * @param regParam reg para
    * @return (updated para, cum gradient, new norm value)
    */
  override def adagradCompute(weightsOld: Vector,
                              gradient: Vector,
                              cumSquareGradient: Vector,
                              stepSize: Double,
                              epsilon: Double,
                              regParam: Double): (Vector, Double) = {
    require(weightsOld.size == gradient.size)

    val size = weightsOld.size
    val weightsOldCopy = weightsOld.copy //we should not change the value of weightsOld
    val weightsOldValues = weightsOldCopy.toDense.values
    var i = 0
    while (i < size) {
      weightsOldValues(i) -= stepSize / math.sqrt(cumSquareGradient(i) + epsilon) * (gradient(i) + regParam * weightsOldValues(i))
      i += 1
    }
    val norm = 1.0 / 2 * regParam * dot(weightsOld, weightsOld)
    (weightsOld, norm)
  }
}

class AdamL1Updater extends L1Updater {
  /**
    *
    * @param weightsOld old model para
    * @param gradient gradient
    * @param cumSquareGradient each feature's accumulate square of gradient
    * @param stepSize step size
    * @param epsilon smooth parameter
    * @param regParam reg para
    * @return (updated para, cum gradient, new norm value)
    */
  override def adamCompute(weightsOld: Vector,
                              gradient: Vector,
                              cumSquareGradient: Vector,
                              stepSize: Double,
                              epsilon: Double,
                              regParam: Double): (Vector, Double) = {
    require(weightsOld.size == gradient.size)
    val size = weightsOld.size
    val weightsOldCopy = weightsOld.copy
    val weightsOldValues = weightsOldCopy.toDense.values
    val learning_rates = Array.ofDim[Double](size)
    var i = 0
    while (i < size) {
      learning_rates(i) = stepSize / (math.sqrt(cumSquareGradient(i)) + epsilon)
      weightsOldValues(i) -= learning_rates(i) * gradient(i)
      i += 1
    }

    val weights = weightsOld.copy
    val weightsValues = weights.toDense.values
    i = 0
    var norm = 0.0
    while (i < size) {
      weightsValues(i) = math.signum(weightsValues(i)) * math.max(0.0, math.abs(weightsValues(i)) - learning_rates(i) * regParam)
      norm += math.abs(weightsValues(i))
      i += 1
    }
    //val norm = regParam * math.sqrt(dot(weightsVector, weightsVector))
    (weights, regParam * norm)
  }
}

class AdamL2Updater extends L2Updater {
  /**
    *
    * @param weightsOld old model para
    * @param gradient gradient
    * @param cumSquareGradient each feature's accumulate square of gradient
    * @param stepSize step size
    * @param epsilon smooth parameter
    * @param regParam reg para
    * @return (updated para, cum gradient, new norm value)
    */
  override def adamCompute(weightsOld: Vector,
                           gradient: Vector,
                           cumSquareGradient: Vector,
                           stepSize: Double,
                           epsilon: Double,
                           regParam: Double): (Vector, Double) = {
    require(weightsOld.size == gradient.size)

    val size = weightsOld.size
    val weightsOldCopy = weightsOld.copy //we should not change the value of weightsOld
    val weightsOldValues = weightsOldCopy.toDense.values
    var i = 0
    while (i < size) {
      weightsOldValues(i) -= stepSize / (math.sqrt(cumSquareGradient(i)) + epsilon) * (gradient(i) + regParam * weightsOldValues(i))
      i += 1
    }
    val norm = 1.0 / 2 * regParam * dot(weightsOld, weightsOld)
    (weightsOld, norm)
  }
}
