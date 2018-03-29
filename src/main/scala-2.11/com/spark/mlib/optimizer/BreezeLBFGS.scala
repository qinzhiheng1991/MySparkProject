package com.spark.mlib.optimizer

import breeze.linalg._
import breeze.linalg.operators.OpMulMatrix
import breeze.math.MutableInnerProductModule
import breeze.optimize._
import breeze.optimize.FirstOrderMinimizer.ConvergenceCheck
import breeze.util.SerializableLogging

/**
  * This is an Spark inherent class, we take it out to adjust the line search iterations
  * Port of LBFGS to Scala.
  *
  * Special note for LBFGS:
  *  If you use it in published work, you must cite one of:
  *     * J. Nocedal. Updating  Quasi-Newton  Matrices  with  Limited  Storage
  *    (1980), Mathematics of Computation 35, pp. 773-782.
  *  * D.C. Liu and J. Nocedal. On the  Limited  mem  Method  for  Large
  *    Scale  Optimization  (1989),  Mathematical  Programming  B,  45,  3,
  *    pp. 503-528.
  *  *
  *
  * @param m: The memory of the search. 3 to 7 is usually sufficient.
  */
class BreezeLBFGS[T](convergenceCheck: ConvergenceCheck[T], m: Int)(implicit space: MutableInnerProductModule[T, Double]) extends FirstOrderMinimizer[T, DiffFunction[T]](convergenceCheck) with SerializableLogging {

  def this(maxIter: Int = -1, m: Int=7, tolerance: Double=1E-9)
          (implicit space: MutableInnerProductModule[T, Double]) = this(FirstOrderMinimizer.defaultConvergenceCheck(maxIter, tolerance), m )
  import space._
  require(m > 0)

  type History = BreezeLBFGS.ApproximateInverseHessian[T]


  override protected def adjustFunction(f: DiffFunction[T]): DiffFunction[T] = f.cached

  protected def takeStep(state: State, dir: T, stepSize: Double) = state.x + dir * stepSize
  protected def initialHistory(f: DiffFunction[T], x: T):History = new BreezeLBFGS.ApproximateInverseHessian(m)
  protected def chooseDescentDirection(state: State, fn: DiffFunction[T]):T = {
    state.history * state.grad
  }

  protected def updateHistory(newX: T, newGrad: T, newVal: Double,  f: DiffFunction[T], oldState: State): History = {
    oldState.history.updated(newX - oldState.x, newGrad :- oldState.grad)
  }

  /**
    * Given a direction, perform a line search to find
    * a direction to descend. At the moment, this just executes
    * backtracking, so it does not fulfill the wolfe conditions.
    *
    * @param state the current state
    * @param f The objective
    * @param dir The step direction
    * @return stepSize
    */
  protected def determineStepSize(state: State, f: DiffFunction[T], dir: T) = {
    val x = state.x
    val grad = state.grad

    val ff = LineSearch.functionFromSearchDirection(f, x, dir)
    // inherent parameter is 10, 10  we tune it bigger both line search and room
    val search = new StrongWolfeLineSearch(maxZoomIter = 20, maxLineSearchIter = 20) // TODO: Need good default values here.
    val alpha = search.minimize(ff, if(state.iter == 0.0) 1.0/norm(dir) else 1.0)

    if(alpha * norm(grad) < 1E-10)
      throw new StepSizeUnderflow
    alpha
  }
}

object BreezeLBFGS {
  case class ApproximateInverseHessian[T](m: Int,
                                          private val memStep: IndexedSeq[T] = IndexedSeq.empty,
                                          private val memGradDelta: IndexedSeq[T] = IndexedSeq.empty)
                                         (implicit space: MutableInnerProductModule[T, Double]) extends NumericOps[ApproximateInverseHessian[T]] {

    import space._

    def repr: ApproximateInverseHessian[T] = this

    def updated(step: T, gradDelta: T) = {
      val memStep = (step +: this.memStep) take m
      val memGradDelta = (gradDelta +: this.memGradDelta) take m

      new ApproximateInverseHessian(m, memStep,memGradDelta)
    }


    def historyLength = memStep.length

    def *(grad: T) = {
      val diag = if(historyLength > 0) {
        val prevStep = memStep.head
        val prevGradStep = memGradDelta.head
        val sy = prevStep dot prevGradStep
        val yy = prevGradStep dot prevGradStep
        if(sy < 0 || sy.isNaN) throw new NaNHistory
        sy/yy
      } else {
        1.0
      }

      val dir = space.copy(grad)
      val as = new Array[Double](m)
      val rho = new Array[Double](m)

      for(i <- 0 until historyLength) {
        rho(i) = memStep(i) dot memGradDelta(i)
        as(i) = (memStep(i) dot dir)/rho(i)
        if(as(i).isNaN) {
          throw new NaNHistory
        }
        axpy(-as(i), memGradDelta(i), dir)
      }

      dir *= diag

      for(i <- (historyLength - 1) to 0 by (-1)) {
        val beta = (memGradDelta(i) dot dir)/rho(i)
        axpy(as(i) - beta, memStep(i), dir)
      }

      dir *= -1.0
      dir
    }
  }

  implicit def multiplyInverseHessian[T](implicit vspace: MutableInnerProductModule[T, Double]):OpMulMatrix.Impl2[ApproximateInverseHessian[T], T, T] = {
    new OpMulMatrix.Impl2[ApproximateInverseHessian[T], T, T] {
      def apply(a: ApproximateInverseHessian[T], b: T): T = a * b
    }
  }
}

