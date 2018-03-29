package com.spark.mlib.optimizer

import breeze.linalg.{norm, DenseVector => BDV}
import com.spark.mlib.linear.BLAS
import com.spark.mlib.log.Logging
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer


/**
  * Created by qinzhiheng on 2017/7/17.
  *
  * Class used to solve an optimization problem using Momentum or NesterovMomentum Gradient Descent.
  *
  * @param gradient Gradient function to be used.
  * @param updater Updater to be used to update weights after every iteration.
  *
  */
class MomentumGradientDescent(private var gradient: Gradient, private var updater: Updater)
  extends Optimizer with Logging {

  private var learning_rate: Double = 0.01
  private var numIterations: Int = 1000
  private var regParam: Double = 0.0
  private var miniBatchFraction: Double = 0.001
  private var convergenceTol: Double = 0.001
  private var convergenceTime: Int = 0
  private var numFeatures: Int = 0
  private var classNum: Int = 0
  private var momentum: Double = 0.9
  private var isNesterov: Boolean = false

  def setLearningRate(step: Double): this.type = {
    require(learning_rate > 0 && learning_rate < 1,
      s"learning_rate must be in range (0, 1) but got $learning_rate")
    this.learning_rate = learning_rate
    this
  }

  /**
    * Set fraction of data to be used for each SGD iteration.
    * Default 1.0 (corresponding to deterministic/classical gradient descent)
    */
  def setMiniBatchFraction(fraction: Double): this.type = {
    require(fraction > 0 && fraction <= 1.0,
      s"Fraction for mini-batch SGD must be in range (0, 1] but got $fraction")
    this.miniBatchFraction = fraction
    this
  }

  def setMomentum(momentum: Double): this.type = {
    require(momentum >= 0.5 && momentum < 1,
      s"momentum must be in range [0.5, 1) but got $momentum")
    this.momentum = momentum
    this
  }

  def setNumIterations(iters: Int): this.type = {
    require(iters > 0,
      s"Number of iterations must be non-negative but got $iters")
    this.numIterations = iters
    this
  }

  def setRegParam(regParam: Double): this.type = {
    require(regParam >= 0,
      s"Regularization parameter must be non-negative but got $regParam")
    this.regParam = regParam
    this
  }

  def setConvergenceTol(tolerance: Double): this.type = {
    require(tolerance >= 0.0 && tolerance <= 1.0,
      s"Convergence tolerance must be in range [0, 1] but got $tolerance")
    this.convergenceTol = tolerance
    this
  }

  def setConvergenceTime(convergenceTime: Int): this.type = {
    require(convergenceTime >= 1,
      s"convergenceTime must be at least one but got $convergenceTime")
    this.convergenceTime = convergenceTime
    this
  }

  def setClassNum(classNum: Int): this.type = {
    require(classNum > 1,
      s"classNum must be in larger than 1 but got $classNum")
    this.classNum = classNum
    this
  }

  def setNumFeatures(numFeatures: Int): this.type = {
    require(numFeatures > 1,
      s"classNum must be in larger than 1 but got $numFeatures")
    this.numFeatures = numFeatures
    this
  }

  def setNesterov(isNesterov: Boolean): this.type = {
    this.isNesterov = isNesterov
    this
  }

  def setGradient(gradient: Gradient): this.type = {
    this.gradient = gradient
    this
  }

  def setUpdater(updater: Updater): this.type = {
    this.updater = updater
    this
  }

  def optimize(data: RDD[(Vector, Array[Double])], initialWeights: Vector): Vector = {
    val (weights, _) = MomentumGradientDescent.runMiniBatchMomentumSGD(
      data,
      gradient,
      updater,
      learning_rate,
      momentum,
      numIterations,
      regParam,
      miniBatchFraction,
      initialWeights,
      convergenceTol,
      numFeatures,
      classNum,
      convergenceTime,
      isNesterov)
    weights
  }
}

object MomentumGradientDescent extends Logging {

  def runMiniBatchMomentumSGD(data: RDD[(Vector, Array[Double])],
                      gradient: Gradient,
                      updater: Updater,
                      learning_rate: Double,
                      momentum: Double,
                      numIterations: Int,
                      regParam: Double,
                      miniBatchFraction: Double,
                      initialWeights: Vector,
                      convergenceTol: Double,
                      numFeatures: Int,
                      classNum: Int,
                      convergenceTime: Int,
                      isNesterov: Boolean): (Vector, Array[Double]) = {

    if (miniBatchFraction < 1.0 && convergenceTol > 0.0) {
      logWarning("Testing against a convergenceTol when using miniBatchFraction " +
        "< 1.0 can be unstable because of the stochasticity in sampling.")
    }

    if (numIterations * miniBatchFraction < 1.0) {
      logWarning("Not all examples will be used if numIterations * miniBatchFraction < 1.0: " +
        s"numIterations=$numIterations and miniBatchFraction=$miniBatchFraction")
    }

    val stochasticLossHistory = new ArrayBuffer[Double](numIterations)
    // Record previous weight and current one to calculate solution vector difference

    var previousWeights: Option[Vector] = None
    var currentWeights: Option[Vector] = None

    val numExamples = data.treeAggregate(0.0)(
      seqOp = (c, q) => (c, q) match {
        case (count, (features, num)) =>
          count + num.sum
      },
      combOp = (c, q) => (c, q) match {
        case (count1, count2) =>
          count1 + count2
      }
    )

    if (numExamples == 0) {
      logWarning("GradientDescent.runMiniBatchSGD returning initial weights, no data found")
      return (initialWeights, stochasticLossHistory.toArray)
    }

    if (numExamples * miniBatchFraction < 1) {
      logWarning("The miniBatchFraction is too small")
    }

    var weights = Vectors.dense(initialWeights.toArray)
    val n = weights.size

    /**
      * For the first iteration, the regVal will be initialized as sum of weight squares
      * if it's L2 updater; for L1 updater, the same logic is followed.
      */
    var regVal = updater.compute(weights, Vectors.zeros(weights.size), 0, 1, regParam)._2
    var previousGradientSum = Vectors.zeros(n)
    var curGradientSum = Vectors.zeros(n)
    var gradientTemp = Vectors.zeros(n)
    var previousModel = weights.copy
    var converged = false
    var i = 1
    var time = 0
    while (time < convergenceTime && i <= numIterations) {
      logInfo(s"runMiniBatchMomentumSGD $i epoch starting...")
      if (isNesterov && i > 1) {
        previousModel = weights.copy
        BLAS.axpy(-1.0 * momentum * learning_rate, previousGradientSum, weights)
      }
      val dividends = BLAS.convertModel(weights, classNum, numFeatures)
      val bcWeights = data.context.broadcast(dividends)
      val (gradientSum, lossSum, miniBatchSize) = data.sample(withReplacement = false, miniBatchFraction)
          .treeAggregate((Vectors.zeros(n), 0.0, 0L))(
            seqOp = (c, v) => {
              // c: (grad, loss, count), v: (features, counts)
              val l = gradient.compute(v._1, v._2, bcWeights.value, c._1)
              (c._1, c._2 + l, c._3 + v._2.sum.toLong)
            },
            combOp = (c1, c2) => {
              // c: (grad, loss, count)
              BLAS.axpy(1.0, c2._1, c1._1)
              (c1._1, c1._2 + c2._2, c1._3 + c2._3)
            }
          )
      bcWeights.destroy()

      if (miniBatchSize > 0) {
        /**
          * lossSum is computed using the weights from the previous iteration
          * and regVal is the regularization value computed in the previous iteration as well.
          */
        stochasticLossHistory += lossSum / miniBatchSize + regVal

        curGradientSum = gradientSum.copy
        BLAS.scal(1.0 / miniBatchSize, curGradientSum)
        /**
          * compute regular loss, gradientTotal = w - (w - regular_loss)
          */
        gradientTemp = weights.copy
        BLAS.axpy(-1.0, updater.compute(weights, Vectors.zeros(n), 1, 1, regParam)._1, gradientTemp)
        BLAS.axpy(1.0, gradientTemp, curGradientSum)
        // Momentum gradient
        BLAS.axpy(momentum, previousGradientSum, curGradientSum)
        previousGradientSum = curGradientSum

        val update = if (isNesterov) {
          updater.compute(previousModel, curGradientSum, learning_rate, 1, regParam)
        } else {
          updater.compute(weights, curGradientSum, learning_rate, 1, regParam)
        }
        weights = update._1
        regVal = update._2
        previousWeights = currentWeights
        currentWeights = Some(weights)
        if (previousWeights.isDefined && currentWeights.isDefined) {
          converged = isConverged(previousWeights.get, currentWeights.get, convergenceTol)
          if (!converged) {
            time = 0
          }
          else {
            time += 1
          }
        }
      } else {
        logWarning(s"Iteration ($i/$numIterations). The size of sampled batch is zero")
      }
      i += 1
    }
    logInfo("MomentumGradientDescent.runMiniBatchMomentumSGD finished. Last 10 stochastic losses %s".format(
      stochasticLossHistory.takeRight(10).mkString(", ").trim))

    (weights, stochasticLossHistory.toArray)
  }

  /**
    * Alias of [[runMiniBatchMomentumSGD]] with convergenceTol set to default value of 0.001.
    */
  def runMiniBatchMomentumSGD(data: RDD[(Vector, Array[Double])],
                              gradient: Gradient,
                              updater: Updater,
                              learning_rate: Double,
                              momentum: Double,
                              numIterations: Int,
                              regParam: Double,
                              miniBatchFraction: Double,
                              initialWeights: Vector,
                              numFeatures: Int,
                              classNum: Int,
                              convergenceTime: Int,
                              isNesterov: Boolean): (Vector, Array[Double]) = {
    MomentumGradientDescent.runMiniBatchMomentumSGD(data, gradient, updater, learning_rate, momentum, numIterations,
      regParam, miniBatchFraction, initialWeights, 0.0001, numFeatures, classNum, convergenceTime, isNesterov)
  }


  private def isConverged(previousWeights: Vector,
                          currentWeights: Vector,
                          convergenceTol: Double): Boolean = {
    val previousBDV = new BDV[Double](previousWeights.toArray)
    val currentBDV = new BDV[Double](currentWeights.toArray)
    val solutionVecDiff: Double = norm(previousBDV - currentBDV)
    solutionVecDiff < convergenceTol * Math.max(norm(currentBDV), 1.0)
  }
}
