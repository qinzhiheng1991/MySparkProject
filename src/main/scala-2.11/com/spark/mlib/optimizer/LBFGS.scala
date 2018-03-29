package com.spark.mlib.optimizer

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import breeze.optimize.{CachedDiffFunction, DiffFunction, OWLQN}
import com.spark.mlib.linear.BLAS._
import com.spark.mlib.log.Logging
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD

import scala.collection.mutable

/**
  * Created by qinzhiheng on 2017/7/17.
  *
  * class invoke spark inherent Breeze LBFGS and OWLQN interface
  *
  * @param gradient Gradient function to be used.
  * @param updater Updater to be used to update weights after every iteration.
  *
  */

class LBFGS(private var gradient: Gradient,
            private var updater: Updater) extends Optimizer with Logging {
  private var numCorrections: Int = 10
  private var convergenceToVal: Double = 1e-9
  private var maxNumIterations: Int = 100
  private var regParams: Double = 0.0
  private var numFeatures: Int = 0
  private var classNum: Int = 0
  private var breakpoint: Int = 0 //for multi-stage model
  private var embedding_dim: Int = 0
  private var forceLBFGS: Boolean = false

  def setNumCorrections(numCorrections: Int): this.type = {
    require(numCorrections > 0,
      s"num corrections must be positive but got $numCorrections")
    this.numCorrections = numCorrections
    this
  }

  def setConvergenceToVal(convergenceToVal: Double): this.type = {
    require(convergenceToVal > 0,
      s"convergenceToVal must be positive but got $convergenceToVal")
    this.convergenceToVal = convergenceToVal
    this
  }

  def setMaxNumIterations(maxNumIterations: Int): this.type = {
    require(maxNumIterations > 0,
      s"maxNumIterations must be larger than 1 but got $maxNumIterations")
    this.maxNumIterations = maxNumIterations
    this
  }

  def setRegParams(regParams: Double): this.type = {
    require(regParams >= 0,
      s"regParams must be positive but got $regParams")
    this.regParams = regParams
    this
  }

  def setGradient(gradient: Gradient): this.type = {
    this.gradient = gradient
    this
  }

  def setUpdater(updater: Updater): this.type  = {
    this.updater = updater
    this
  }

  def setClassNum(classNum: Int): this.type = {
    require(classNum > 1,
      s"classNum must be larger than 1 but got $classNum")
    this.classNum = classNum
    this
  }

  def setNumFeatures(numFeatures: Int): this.type = {
    require(numFeatures > 1,
      s"numFeatures must be larger than 1 but got $numFeatures")
    this.numFeatures = numFeatures
    this
  }

  def setBreakpoint(breakpoint: Int): this.type = {
    require(breakpoint >= 1 && breakpoint < numFeatures - 1,
      s"breakpoint must be [1, numFeatures - 1) but got $breakpoint")
    this.breakpoint = breakpoint
    this
  }

  def setEmbeddingDim(embedding_dim: Int): this.type = {
    require(embedding_dim >= 1,
      s"embedding_dim must be [1, inf) but got $embedding_dim")
    this.embedding_dim = embedding_dim
    this
  }

  // use forced LBFGS with L1 regular
  def setForceLBFGS(forceLBFGS: Boolean): this.type = {
    this.forceLBFGS = true
    this
  }

  def getNumCorrections: Int = this.numCorrections
  def getConvergenceToVal: Double = this.convergenceToVal
  def getMaxNumIterations: Int = this.maxNumIterations
  def getRegParams: Double = this.regParams
  def getClassNum: Int = this.classNum
  def getNumFeatures: Int = this.numFeatures
  def getGradient: Gradient = this.gradient
  def getUpdater: Updater = this.updater

  override def optimize(data: RDD[(Vector, Array[Double])], initialWeights: Vector): Vector = {
    val (weights, _) = LBFGS.runLBFGS(data,
                                      updater,
                                      gradient,
                                      numCorrections,
                                      convergenceToVal,
                                      maxNumIterations,
                                      regParams,
                                      classNum,
                                      numFeatures,
                                      initialWeights,
                                      breakpoint,
                                      embedding_dim,
                                      forceLBFGS)
    weights
  }
}


object LBFGS extends Logging {
  def runLBFGS(data: RDD[(Vector, Array[Double])],
               updater: Updater,
               gradient: Gradient,
               numCorrections: Int,
               convergenceToVal: Double,
               maxNumIterations: Int,
               regParam: Double,
               classNum: Int,
               numFeatures: Int,
               initialWeights: Vector,
               breakpoint: Int,
               embedding_dim: Int,
               forceLBFGS: Boolean): (Vector, Array[Double]) = {
    val lossHistory = mutable.ArrayBuilder.make[Double]
    //val numExamples = data.map(iter=>(1, iter._3)).reduceByKey(_+_).collect()(0)._2
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
      return (initialWeights, lossHistory.result())
    }
    val costFun = new CostFun(data,
                              gradient,
                              updater,
                              regParam,
                              classNum,
                              numFeatures,
                              numExamples,
                              breakpoint,
                              embedding_dim,
                              forceLBFGS)
    //if L1Updater we use OWLQN, OWLQN is a little different from LBFGS
    //1. direction: when lbfgs search direction is not compatible with gradient descend, the corresponding direction is 0
    //              , this may hurt the convergence.
    //2. takeStep: compute the orthant according to the raw weight direction, if weight is zero, adjust direction based on
    //             adjusted grad, if the updated weight direction is no compatible with the orthant, set zero
    //3. update gradient: according to L1 regular, compute cum-gradient, the same with SGD in this project
    //
    val lbfgs = if (!updater.isInstanceOf[L1Updater] || forceLBFGS) {
      logInfo(".....................LBFGS is starting.......................")
      new BreezeLBFGS[BDV[Double]](maxNumIterations, numCorrections, convergenceToVal)
    } else {
      logInfo(".....................OWLQN is starting.......................")
      new OWLQN[Int, BDV[Double]](maxNumIterations, numCorrections, convergenceToVal, regParam)
    }
    val states = lbfgs.iterations(new CachedDiffFunction(costFun), new BDV[Double](initialWeights.toArray).toDenseVector)
    var state = states.next()
    while (states.hasNext) {
      lossHistory += state.value
      state = states.next()
    }
    lossHistory += state.value
    val lossHistoryArray = lossHistory.result()
    logInfo("last ten 10 values: " + lossHistoryArray.takeRight(10).mkString(", ").trim)
    val weights = fromBreeze(state.x)
    (weights, lossHistoryArray)
  }

  private class CostFun(data: RDD[(Vector, Array[Double])],
                        gradient: Gradient,
                        updater: Updater,
                        regParam: Double,
                        classNum: Int,
                        numFeatures: Int,
                        numExamples: Double,
                        breakpoint: Int,
                        embedding_dim: Int,
                        forceLBFGS: Boolean) extends DiffFunction[BDV[Double]] with Serializable {
    override def calculate(weights: BDV[Double]): (Double, BDV[Double]) = {
      val w = fromBreeze(weights)
      val n = w.size
      val useMultiStage = breakpoint >= 1 && breakpoint < numFeatures - 1
      val isFM = embedding_dim >= 1
      require(!isFM || !useMultiStage, "FM and multi-stage can not be supported simultaneously!")
      val localGradient = gradient
      val (gradientSum, lossSum) = if (isFM) {
        val dividends = fmConvertModel(w, classNum, numFeatures, embedding_dim)
        //logInfo("..............dividends.............." + dividends._2(0).size)
        val bcw = data.context.broadcast(dividends)
        data.treeAggregate((Vectors.zeros(n), 0.0))(
          seqOp = (c, q) => (c, q) match {
            case ((grad, loss), (features, num)) =>
              val loss1 = localGradient.compute(features, num, bcw.value, grad)
              (grad, loss + loss1)
          },
          combOp = (c1, c2) => (c1, c2) match {
            case ((grad1, loss1), (grad2, loss2)) =>
              axpy(1.0, grad2, grad1)
              (grad1, loss1 + loss2)
          }
        )
      } else {
        val bcw = if (useMultiStage) {
          data.context.broadcast(Array(w))
        } else {
          val dividends = convertModel(w, classNum, numFeatures)
          data.context.broadcast(dividends)
        }
        data.treeAggregate((Vectors.zeros(n), 0.0))(
          seqOp = (c, q) => (c, q) match {
            case ((grad, loss), (features, num)) =>
              val loss1 = if (useMultiStage) {
                localGradient.compute(features, num, bcw.value, grad, breakpoint)
              } else {
                localGradient.compute(features, num, bcw.value, grad)
              }
              (grad, loss + loss1)
          },
          combOp = (c1, c2) => (c1, c2) match {
            case ((grad1, loss1), (grad2, loss2)) =>
              axpy(1.0, grad2, grad1)
              (grad1, loss1 + loss2)
          }
        )
      }

      val avgLoss = lossSum / numExamples + updater.compute(w, Vectors.zeros(n), 0, 1, regParam)._2
      val gradientTotal = w.copy
      //weights - (weights - regular_gradient)
      if (!updater.isInstanceOf[L1Updater] || forceLBFGS) {
        axpy(-1.0, updater.compute(w, Vectors.zeros(n), 1, 1, regParam)._1, gradientTotal)
      }
      axpy(1.0 / numExamples, gradientSum, gradientTotal)
      (avgLoss, new BDV[Double](gradientTotal.toArray))
    }
  }

  private def fromBreeze(breezeVector: BV[Double]): Vector = {
    breezeVector match {
      case v: BDV[Double] =>
        if (v.offset == 0 && v.stride == 1 && v.length == v.data.length) {
          new DenseVector(v.data)
        } else {
          new DenseVector(v.toArray)
        }
      case v: BSV[Double] =>
        if (v.index.length == v.used) {
          new SparseVector(v.length, v.index, v.data)
        } else {
          new SparseVector(v.length, v.index.slice(0, v.used), v.data.slice(0, v.used))
        }
      case v: BV[_] =>
        sys.error(s"Unsupported Breeze vector type: ${v.getClass.getName}")
    }
  }
}