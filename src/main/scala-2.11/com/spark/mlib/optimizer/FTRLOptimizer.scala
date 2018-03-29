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
  * Class used to solve an optimization problem using FTRL_proximal Gradient Descent.
  *
  * @param gradient Gradient function to be used.
  *
  */
class FTRLOptimizer(private var gradient: Gradient) extends Optimizer with Logging {

  private var alpha: Double = 0.5
  private var beta: Double = 1
  private var lambda1: Double = 0.0
  private var lambda2: Double = 0.0
  private var numIterations: Int = 1000
  private var miniBatchFraction: Double = 0.001
  private var convergenceTol: Double = 0.001
  private var convergenceTime: Int = 0
  private var numFeatures: Int = 0
  private var classNum: Int = 0

  def setAlpha(alpha: Double): this.type = {
    require(alpha > 0 && alpha < 1,
      s"alpha must be in range (0, 1) but got $alpha")
    this.alpha = alpha
    this
  }

  def setBeta(beta: Double): this.type = {
    require(beta > 0 && beta < 1,
      s"beta must be in range (0, 1) but got $beta")
    this.beta = beta
    this
  }

  //L1 regular parameter
  def setLambda1(lambda1: Double): this.type = {
    require(lambda1 >= 0 && lambda1 < 1,
      s"lambda1 must be in range [0, 1) but got $lambda1")
    this.lambda1 = lambda1
    this
  }

  //L2 regular parameter
  def setLambda2(lambda2: Double): this.type = {
    require(lambda2 >= 0 && lambda2 < 1,
      s"lambda1 must be in range [0, 1) but got $lambda2")
    this.lambda2 = lambda2
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

  def setNumIterations(iters: Int): this.type = {
    require(iters > 0,
      s"Number of iterations must be larger than 10 but got $iters")
    this.numIterations = iters
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

  def setGradient(gradient: Gradient): this.type = {
    this.gradient = gradient
    this
  }

  def optimize(data: RDD[(Vector, Array[Double])], initialWeights: Vector): Vector = {
    val (weights, _) = FTRLOptimizer.runMiniBatchFTRLSGD(
      data,
      initialWeights,
      gradient,
      alpha,
      beta,
      lambda1,
      lambda2,
      numIterations,
      miniBatchFraction,
      convergenceTol,
      convergenceTime,
      numFeatures,
      classNum)
    weights
  }
}

object FTRLOptimizer extends Logging {

  def runMiniBatchFTRLSGD(data: RDD[(Vector, Array[Double])],
                          initialWeights: Vector,
                          gradient: Gradient,
                          alpha: Double,
                          beta: Double,
                          lambda1: Double,
                          lambda2: Double,
                          numIterations: Int,
                          miniBatchFraction: Double,
                          convergenceTol: Double,
                          convergenceTime: Int,
                          numFeatures: Int,
                          classNum: Int): (Vector, Array[Double]) = {

    if (miniBatchFraction < 1.0 && convergenceTol > 0.0) {
      logWarning("Testing against a convergenceTol when using miniBatchFraction " +
        "< 1.0 can be unstable because of the stochasticity in sampling.")
    }

    if (numIterations * miniBatchFraction < 1.0) {
      logWarning("Not all examples will be used if numIterations * miniBatchFraction < 1.0: " +
        s"numIterations=$numIterations and miniBatchFraction=$miniBatchFraction")
    }
    //we need to store the true loss history
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

    val weights = Vectors.dense(initialWeights.toArray)
    val weightsValue = weights.toDense.values
    val n = weights.size

    val z = Vectors.zeros(n)
    val zValues = z.toDense.values

    val cumSquareGradient = Vectors.zeros(n)
    val cumSquareGradientValues = cumSquareGradient.toDense.values


    var converged = false
    var i = 1
    var time = 0

    while (i <= numIterations) {
      logInfo(s"runMiniBatchFTRLSGD $i epoch starting...")
      //copy old model parameter
      previousWeights = Some(weights)
      //update new parameter
      var index = 0
      var multiplier = 0.0
      while (index < n) {
        if (math.abs(z(index)) <= lambda1) {
          weightsValue(index) = 0
        }
        else {
          multiplier = (beta + math.sqrt(cumSquareGradient(index))) / alpha + lambda2
          weightsValue(index) = -1.0 / multiplier * (zValues(index) - math.signum(zValues(index)) * lambda1)
        }
        index += 1
      }

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
        //compute history loss, although FTRL_proximal optimizes surrogate loss, we still record true loss
        val lossCurrent = lossSum / numExamples + lambda1 * BLAS.norm(weights, 1) + lambda2 * BLAS.norm(weights, 2)
        stochasticLossHistory += lossCurrent
        //compute average gradient
        BLAS.scal(1.0 / miniBatchSize, gradientSum)

        var sigma = 0.0
        index = 0
        while (index < n) {

          sigma = 1.0 / alpha * (math.sqrt(cumSquareGradientValues(index) + math.pow(gradientSum(index), 2)) -
                                 math.sqrt(cumSquareGradientValues(index)))
          zValues(index) += gradientSum(index) - sigma * weights(index)
          cumSquareGradientValues(index) += math.pow(gradientSum(index), 2)
          index += 1
        }
      } else {
        logWarning(s"Iteration ($i/$numIterations). The size of sampled batch is zero")
      }
      i += 1
    }
    logInfo("FTRLOptimizer.runMiniBatchFTRLSGD finished. Last 10 stochastic losses %s".format(
      stochasticLossHistory.takeRight(10).mkString(", ").trim))

    (weights, stochasticLossHistory.toArray)
  }

  /**
    * Alias of [[runMiniBatchFTRLSGD]] with convergenceTol set to default value of 0.001.
    */
  def runMiniBatchFTRLSGD(data: RDD[(Vector, Array[Double])],
                          initialWeights: Vector,
                          gradient: Gradient,
                          alpha: Double,
                          beta: Double,
                          lambda1: Double,
                          lambda2: Double,
                          numIterations: Int,
                          miniBatchFraction: Double,
                          convergenceTime: Int,
                          numFeatures: Int,
                          classNum: Int): (Vector, Array[Double]) = {
    FTRLOptimizer.runMiniBatchFTRLSGD(data: RDD[(Vector, Array[Double])],
      initialWeights: Vector,
      gradient: Gradient,
      alpha: Double,
      beta: Double,
      lambda1: Double,
      lambda2: Double,
      numIterations: Int,
      miniBatchFraction: Double,
      1e-6,
      convergenceTime: Int,
      numFeatures: Int,
      classNum: Int)
  }

  //TODO: how to efficiently evaluate convergence for stochastic gradient descent
  private def isConverged(previousWeights: Vector,
                          currentWeights: Vector,
                          convergenceTol: Double): Boolean = {
    val previousBDV = new BDV[Double](previousWeights.toArray)
    val currentBDV = new BDV[Double](currentWeights.toArray)
    val solutionVecDiff: Double = norm(previousBDV - currentBDV)
    solutionVecDiff < convergenceTol * Math.max(norm(currentBDV), 1.0)
  }
}
