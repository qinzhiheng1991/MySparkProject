package com.spark.mlib.linear

import com.github.fommil.netlib.BLAS.{getInstance => NativeBLAS}
import com.github.fommil.netlib.{F2jBLAS, BLAS => NetlibBLAS}
import org.apache.spark.SparkException
import org.apache.spark.mllib.linalg._
import scala.collection.mutable

object BLAS extends Serializable {

  @transient private var _f2jBLAS: NetlibBLAS = _
  @transient private var _nativeBLAS: NetlibBLAS = _

  // For level-1 routines, we use Java implementation.
  private def f2jBLAS: NetlibBLAS = {
    if (_f2jBLAS == null) {
      _f2jBLAS = new F2jBLAS
    }
    _f2jBLAS
  }

  /**
    * y += a * x
    */
  def axpy(a: Double, x: Vector, y: Vector): Unit = {
    require(x.size == y.size)
    y match {
      case dy: DenseVector =>
        x match {
          case sx: SparseVector =>
            axpy(a, sx, dy)
          case dx: DenseVector =>
            axpy(a, dx, dy)
          case _ =>
            throw new UnsupportedOperationException(
              s"axpy doesn't support x type ${x.getClass}.")
        }
      case _ =>
        throw new IllegalArgumentException(
          s"axpy only supports adding to a dense vector but got type ${y.getClass}.")
    }
  }

  /**
    * y += a * x
    */
  private def axpy(a: Double, x: DenseVector, y: DenseVector): Unit = {
    val n = x.size
    f2jBLAS.daxpy(n, a, x.values, 1, y.values, 1)
  }

  def axpyNM(a: Double, x: DenseVector, y: DenseVector, n: Int, m: Int) = {
    val size = x.size
    f2jBLAS.daxpy(size, a, x.values, n, y.values, m)
  }

  //TODO: support axpy for SparseVector
  def axpyCopy(a: Double, x: Vector, y: SparseVector): Vector = {
    require(x.size == y.size)
    val yMap = scala.collection.mutable.Map[Int, Double]()
    y.foreachActive((i, value) => yMap(i) = value)
    val size = x.size
    val xSparse = x.toSparse
    val xIndices = xSparse.indices
    val xValues = xSparse.values
    val yIndices = mutable.ArrayBuilder.make[Int]()
    val yValues = mutable.ArrayBuilder.make[Double]()
    val len = xIndices.length
    var index = 0
    var value = 0.0

    while (index < len) {
      if (yMap.contains(xIndices(index))) {
        value = yMap(xIndices(index)) + xValues(index) * a
        if (value == 0.0) {
          yMap.remove(xIndices(index))
        }
        else {
          yMap(xIndices(index)) = value
        }
      }
      else {
        yMap(xIndices(index)) = xValues(index) * a
      }
      index += 1
    }
    yMap.foreach(s => {yIndices += s._1; yValues += s._2})
    Vectors.sparse(size, yIndices.result(), yValues.result())
  }

  /**
    * y += a * x
    */
  private def axpy(a: Double, x: SparseVector, y: DenseVector): Unit = {
    val xValues = x.values
    val xIndices = x.indices
    val yValues = y.values
    val nnz = xIndices.length

    if (a == 1.0) {
      var k = 0
      while (k < nnz) {
        yValues(xIndices(k)) += xValues(k)
        k += 1
      }
    } else {
      var k = 0
      while (k < nnz) {
        yValues(xIndices(k)) += a * xValues(k)
        k += 1
      }
    }
  }

  /** Y += a * x */
  private def axpy(a: Double, X: DenseMatrix, Y: DenseMatrix): Unit = {
    require(X.numRows == Y.numRows && X.numCols == Y.numCols, "Dimension mismatch: " +
      s"size(X) = ${(X.numRows, X.numCols)} but size(Y) = ${(Y.numRows, Y.numCols)}.")
    f2jBLAS.daxpy(X.numRows * X.numCols, a, X.values, 1, Y.values, 1)
  }

  /**
    * dot(x, y)
    */
  def dot(x: Vector, y: Vector): Double = {
    require(x.size == y.size,
      "BLAS.dot(x: Vector, y:Vector) was given Vectors with non-matching sizes:" +
        " x.size = " + x.size + ", y.size = " + y.size)
    (x, y) match {
      case (dx: DenseVector, dy: DenseVector) =>
        dot(dx, dy)
      case (sx: SparseVector, dy: DenseVector) =>
        dot(sx, dy)
      case (dx: DenseVector, sy: SparseVector) =>
        dot(sy, dx)
      case (sx: SparseVector, sy: SparseVector) =>
        dot(sx, sy)
      case _ =>
        throw new IllegalArgumentException(s"dot doesn't support (${x.getClass}, ${y.getClass}).")
    }
  }

  def fDot(x: Vector, y: Vector): Vector = {
    require(y.size % x.size == 0,
      "BLAS.dot(x: Vector, y:Vector) was given Vectors with non-matching sizes:" +
        " x.size = " + x.size + ", y.size = " + y.size)
    (x, y) match {
      case (dx: DenseVector, dy: DenseVector) =>
        fDot(dx, dy)
      case (sx: SparseVector, dy: DenseVector) =>
        fDot(sx, dy)
      case (dx: DenseVector, sy: SparseVector) =>
        fDot(dx, sy)
      case (sx: SparseVector, sy: SparseVector) =>
        fDot(sx, sy)
      case _ =>
        throw new IllegalArgumentException(s"dot doesn't support (${x.getClass}, ${y.getClass}).")
    }
  }

  def fDotExp(x: Vector, y: Vector): Vector = {
    require(y.size % x.size == 0,
      "BLAS.dot(x: Vector, y:Vector) was given Vectors with non-matching sizes:" +
        " x.size = " + x.size + ", y.size = " + y.size)
    (x, y) match {
      case (dx: DenseVector, dy: DenseVector) =>
        fDotExp(dx, dy)
      case (sx: SparseVector, dy: DenseVector) =>
        fDotExp(sx, dy)
      case (dx: DenseVector, sy: SparseVector) =>
        fDotExp(dx, sy)
      case (sx: SparseVector, sy: SparseVector) =>
        fDotExp(sx, sy)
      case _ =>
        throw new IllegalArgumentException(s"dot doesn't support (${x.getClass}, ${y.getClass}).")
    }
  }

  /**
    * dot(x, y)
    */
  private def dot(x: DenseVector, y: DenseVector): Double = {
    val n = x.size
    f2jBLAS.ddot(n, x.values, 1, y.values, 1)
  }

  private def fDot(x: DenseVector, y: DenseVector): Vector = {
    val n = x.size
    val dim = y.size / n
    val results = new Array[Double](dim)
    val yValues = y.values
    var sum = 0.0
    var index = 0
    while (index < dim) {
      sum = f2jBLAS.ddot(n, x.values, 1, yValues.slice(index * n, (index + 1) * n), 1)
      results(index) = sum
      index += 1
    }
    Vectors.dense(results)
  }

  private def fDotExp(x: DenseVector, y: DenseVector): Vector = {
    val n = x.size
    val dim = y.size / n
    val results = new Array[Double](dim)
    val yValues = y.values
    var sum = 0.0
    var index = 0
    while (index < dim) {
      sum = f2jBLAS.ddot(n, x.values, 1, yValues.slice(index * n, (index + 1) * n), 1)
      results(index) = Math.exp(sum)
      index += 1
    }
    Vectors.dense(results)
  }

  /**
    * dot(x, y)
    */
  private def dot(x: SparseVector, y: DenseVector): Double = {
    val xValues = x.values
    val xIndices = x.indices
    val yValues = y.values
    val nnz = xIndices.length

    var sum = 0.0
    var k = 0
    while (k < nnz) {
      sum += xValues(k) * yValues(xIndices(k))
      k += 1
    }
    sum
  }

  private def fDot(x: SparseVector, y: DenseVector): Vector = {
    val xValues = x.values
    val xIndices = x.indices
    val nnz = xIndices.length
    val yValues = y.values
    val dim = y.size / x.size
    val results = new Array[Double](dim)
    var index = 0
    while (index < dim) {
      var k = 0
      var sum = 0.0
      while (k < nnz) {
        sum += xValues(k) * yValues(xIndices(k) + index * x.size)
        k += 1
      }
      results(index) = sum
      index += 1
    }
    Vectors.dense(results)
  }

  private def fDotExp(x: SparseVector, y: DenseVector): Vector = {
    val xValues = x.values
    val xIndices = x.indices
    val nnz = xIndices.length
    val yValues = y.values
    val dim = y.size / x.size
    val results = new Array[Double](dim)
    var index = 0
    while (index < dim) {
      var k = 0
      var sum = 0.0
      while (k < nnz) {
        sum += xValues(k) * yValues(xIndices(k) + index * x.size)
        k += 1
      }
      results(index) = Math.exp(sum)
      index += 1
    }
    Vectors.dense(results)
  }

  private def fDot(x: DenseVector, y: SparseVector): Vector = {
    val yValues = y.values
    val yIndices = y.indices
    val nnz = yIndices.length
    val xValues = x.values
    val dim = y.size / x.size
    val results = new Array[Double](dim)

    var index = 0
    var oldSeg = 0
    var newSeg = 0
    var sum = 0.0
    while (index < nnz) {
      newSeg = yIndices(index) / x.size
      if (newSeg != oldSeg) {
        results(oldSeg) = sum
        sum = 0.0
        oldSeg = newSeg
      }
      sum += yValues(index) * xValues(yIndices(index) % x.size)
      index += 1
    }
    results(newSeg) = sum
    Vectors.dense(results)
  }

  private def fDotExp(x: DenseVector, y: SparseVector): Vector = {
    val yValues = y.values
    val yIndices = y.indices
    val nnz = yIndices.length
    val xValues = x.values
    val dim = y.size / x.size
    val results = new Array[Double](dim)

    var index = 0
    var oldSeg = 0
    var newSeg = 0
    var sum = 0.0
    while (index < nnz) {
      newSeg = yIndices(index) / x.size
      if (newSeg != oldSeg) {
        results(oldSeg) = Math.exp(sum)
        sum = 0.0
        oldSeg = newSeg
      }
      sum += yValues(index) * xValues(yIndices(index) % x.size)
      index += 1
    }
    results(newSeg) = Math.exp(sum)
    Vectors.dense(results)
  }

  /**
    * dot(x, y)
    */
  private def dot(x: SparseVector, y: SparseVector): Double = {
    val xValues = x.values
    val xIndices = x.indices
    val yValues = y.values
    val yIndices = y.indices
    val nnzx = xIndices.length
    val nnzy = yIndices.length

    var kx = 0
    var ky = 0
    var sum = 0.0
    // y catching x
    while (kx < nnzx && ky < nnzy) {
      val ix = xIndices(kx)
      while (ky < nnzy && yIndices(ky) < ix) {
        ky += 1
      }
      if (ky < nnzy && yIndices(ky) == ix) {
        sum += xValues(kx) * yValues(ky)
        ky += 1
      }
      kx += 1
    }
    sum
  }

  private def fDot(x: SparseVector, y: SparseVector): Vector = {
    val xValues = x.values
    val xIndicesSet = x.indices.toSet
    val yValues = y.values
    val yIndices = y.indices
    val nnz = yIndices.length
    val dim = y.size / x.size
    val results = new Array[Double](dim)

    var index = 0
    var oldSeg = 0
    var newSeg = 0
    var sum = 0.0
    while (index < nnz) {
      newSeg = yIndices(index) / x.size
      if (newSeg != oldSeg) {
        results(oldSeg) = sum
        sum = 0.0
        oldSeg = newSeg
      }
      if (xIndicesSet.contains(yIndices(index) % x.size)) {
        sum += yValues(index) * xValues(yIndices(index) % x.size)
      }
      index += 1
    }
    results(newSeg) = sum
    Vectors.dense(results)
  }

  private def fDotExp(x: SparseVector, y: SparseVector): Vector = {
    val xValues = x.values
    val xIndicesSet = x.indices.toSet
    val yValues = y.values
    val yIndices = y.indices
    val nnz = yIndices.length
    val dim = y.size / x.size
    val results = new Array[Double](dim)

    var index = 0
    var oldSeg = 0
    var newSeg = 0
    var sum = 0.0
    while (index < nnz) {
      newSeg = yIndices(index) / x.size
      if (newSeg != oldSeg) {
        results(oldSeg) = Math.exp(sum)
        sum = 0.0
        oldSeg = newSeg
      }
      if (xIndicesSet.contains(yIndices(index) % x.size)) {
        sum += yValues(index) * xValues(yIndices(index) % x.size)
      }
      index += 1
    }
    results(newSeg) = Math.exp(sum)
    Vectors.dense(results)
  }

  /**
    * norm x
    */
  def norm(x: Vector, k: Double): Double = {
    var result = 0.0
    x.foreachActive((_, value) => result += math.pow(math.abs(result), k))
    result
  }

  /**
    * y = x
    */
  def copy(x: Vector, y: Vector): Unit = {
    val n = y.size
    require(x.size == n)
    y match {
      case dy: DenseVector =>
        x match {
          case sx: SparseVector =>
            val sxIndices = sx.indices
            val sxValues = sx.values
            val dyValues = dy.values
            val nnz = sxIndices.length

            var i = 0
            var k = 0
            while (k < nnz) {
              val j = sxIndices(k)
              while (i < j) {
                dyValues(i) = 0.0
                i += 1
              }
              dyValues(i) = sxValues(k)
              i += 1
              k += 1
            }
            while (i < n) {
              dyValues(i) = 0.0
              i += 1
            }
          case dx: DenseVector =>
            Array.copy(dx.values, 0, dy.values, 0, n)
        }
      case _ =>
        throw new IllegalArgumentException(s"y must be dense in copy but got ${y.getClass}")
    }
  }

  /**
    * x = a * x
    */
  def scal(a: Double, x: Vector): Unit = {
    x match {
      case sx: SparseVector =>
        f2jBLAS.dscal(sx.values.length, a, sx.values, 1)
      case dx: DenseVector =>
        f2jBLAS.dscal(dx.values.length, a, dx.values, 1)
      case _ =>
        throw new IllegalArgumentException(s"scal doesn't support vector type ${x.getClass}.")
    }
  }

  // For level-3 routines, we use the native BLAS.
  private def nativeBLAS: NetlibBLAS = {
    if (_nativeBLAS == null) {
      _nativeBLAS = NativeBLAS
    }
    _nativeBLAS
  }

  /**
    * Adds alpha * v * v.t to a matrix in-place. This is the same as BLAS's ?SPR.
    *
    * @param U the upper triangular part of the matrix in a [[DenseVector]](column major)
    */
  def spr(alpha: Double, v: Vector, U: DenseVector): Unit = {
    spr(alpha, v, U.values)
  }

  /**
    * Adds alpha * v * v.t to a matrix in-place. This is the same as BLAS's ?SPR.
    *
    * @param U the upper triangular part of the matrix packed in an array (column major)
    */
  def spr(alpha: Double, v: Vector, U: Array[Double]): Unit = {
    val n = v.size
    v match {
      case DenseVector(values) =>
        NativeBLAS.dspr("U", n, alpha, values, 1, U)
      case SparseVector(size, indices, values) =>
        val nnz = indices.length
        var colStartIdx = 0
        var prevCol = 0
        var col = 0
        var j = 0
        var i = 0
        var av = 0.0
        while (j < nnz) {
          col = indices(j)
          // Skip empty columns.
          colStartIdx += (col - prevCol) * (col + prevCol + 1) / 2
          av = alpha * values(j)
          i = 0
          while (i <= j) {
            U(colStartIdx + indices(i)) += av * values(i)
            i += 1
          }
          j += 1
          prevCol = col
        }
    }
  }

  /**
    * A := alpha * x * x^T^ + A
    *
    * @param alpha a real scalar that will be multiplied to x * x^T^.
    * @param x the vector x that contains the n elements.
    * @param A the symmetric matrix A. Size of n x n.
    */
  def syr(alpha: Double, x: Vector, A: DenseMatrix) {
    val mA = A.numRows
    val nA = A.numCols
    require(mA == nA, s"A is not a square matrix (and hence is not symmetric). A: $mA x $nA")
    require(mA == x.size, s"The size of x doesn't match the rank of A. A: $mA x $nA, x: ${x.size}")

    x match {
      case dv: DenseVector => syr(alpha, dv, A)
      case sv: SparseVector => syr(alpha, sv, A)
      case _ =>
        throw new IllegalArgumentException(s"syr doesn't support vector type ${x.getClass}.")
    }
  }

  private def syr(alpha: Double, x: SparseVector, A: DenseMatrix) {
    val mA = A.numCols
    val xIndices = x.indices
    val xValues = x.values
    val nnz = xValues.length
    val Avalues = A.values

    var i = 0
    while (i < nnz) {
      val multiplier = alpha * xValues(i)
      val offset = xIndices(i) * mA
      var j = 0
      while (j < nnz) {
        Avalues(xIndices(j) + offset) += multiplier * xValues(j)
        j += 1
      }
      i += 1
    }
  }

  def convertModel(model: Vector, classNum: Int, numFeatures: Int): Array[Vector] = {
    model match {
      case value: DenseVector =>
        convertDenseModel(model, classNum, numFeatures)
      case value: SparseVector =>
        convertSparseModel(model, classNum, numFeatures)
      case other =>
        throw new SparkException(s"cant not parse ${other.getClass.getName}")
    }
  }


  /**
    * For applying numerator normalization for each class probability, this function
    * only keep the exponent difference value, actually a matrix we use Array to store it
    *
    * @param model the SparseVector of Model, size: (classNum - 1) * numFeatures
    * @param classNum the number of class
    * @param numFeatures the number of features
    * @return Arrays of Vector, 3 class for example, two model w1, w2, for data x each class probability:
    *                          (1/sum(exp[0 w1 w2]'*x, 1/sum(exp[-w1 0 w2-w1]*x), 1/sum(exp[-w2 w1-w2 0]*x)))
    *
    */
  private def convertSparseModel(model: Vector, classNum: Int, numFeatures: Int): Array[Vector] = {
    val modeCopy = model.copy // avoid alter raw model

    val modelIndices = modeCopy.toSparse.indices.map(_ + numFeatures)
    val modelValues = modeCopy.toSparse.values
    val modelSize = modeCopy.size + numFeatures
    val modelVector = Vectors.sparse(modelSize, modelIndices, modelValues)
    val modelArrayBuilder = mutable.ArrayBuilder.make[Vector]()
    val indexBuilder = mutable.ArrayBuilder.make[Int]()
    val valueBuilder = mutable.ArrayBuilder.make[Double]()

    modelArrayBuilder += modelVector
    var index = 1
    while (index < classNum) {
      val indexLowerBound = index * numFeatures
      val indexUpperBound = (index + 1) * numFeatures
      var i = 0
      while (i < modelIndices.length) {
        if (modelIndices(i) >= indexLowerBound && modelIndices(i) < indexUpperBound) {
          indexBuilder += modelIndices(i)
          valueBuilder += modelValues(i)
        }
        i += 1
      }

      val indexArray = indexBuilder.result()
      val valueArray = valueBuilder.result()
      val kModelLength = indexArray.length
      val modelVectorCopy = modelVector.copy
      val indexesArray = Array.tabulate(kModelLength * classNum)(t =>
        (t / kModelLength) * numFeatures + indexArray(t % kModelLength) % numFeatures
      )
      val valuesArray = Array.tabulate(kModelLength * classNum)(t => valueArray(t % kModelLength))
      //println("indexesArray: " + indexesArray.mkString(","))
      //println("valuesArray: " + valuesArray.mkString(","))
      modelVectorCopy match {
        case value: SparseVector =>
          modelArrayBuilder += axpyCopy(-1.0, Vectors.sparse(modelSize, indexesArray, valuesArray), value)
        case value: DenseVector =>
          axpy(-1.0, Vectors.sparse(modelSize, indexesArray, valuesArray), value)
          modelArrayBuilder += value
      }
      indexBuilder.clear()
      valueBuilder.clear()
      index += 1
    }
    indexBuilder.clear()
    valueBuilder.clear()
    modelArrayBuilder.result()
  }

  /**
    * For applying numerator normalization for each class probability, this function
    * only keep the exponent difference value, actually a matrix we use Array to store it
    *
    * @param model the DenseVector of Model, size: (classNum - 1) * numFeatures
    * @param classNum the number of class
    * @param numFeatures the number of features
    * @return Arrays of Vector, 3 class for example, two model w1, w2, for data x each class probability:
    *                          (1/sum(exp[0 w1 w2]'*x, 1/sum(exp[-w1 0 w2-w1]'*x), 1/sum(exp[-w2 w1-w2 0]'*x)))
    *
    */
  private def convertDenseModel(model: Vector, classNum: Int, numFeatures: Int): Array[Vector] = {
    val modelArray = Array.fill(numFeatures)(0.0) ++ model.toArray
    val modelVector = Vectors.dense(modelArray)
    val modelArrayBuilder = mutable.ArrayBuilder.make[Vector]()
    modelArrayBuilder += modelVector
    var index = 1
    while (index < classNum) {
      val kModel = modelArray.slice(index * numFeatures, (index + 1) * numFeatures)
      val kModelVector = Vectors.dense(
        Array.tabulate(numFeatures * classNum)(
          t => kModel(t % numFeatures)
        )
      )
      val modelVectorCopy = modelVector.copy
      axpy(-1.0, kModelVector, modelVectorCopy)
      modelArrayBuilder += modelVectorCopy
      index += 1
    }
    modelArrayBuilder.result()
  }

  private def convertDenseModel(model: Vector, classNum: Int, numFeatures: Int, weight: Double): Array[Vector] = {
    val modelArray = Array.fill(numFeatures)(0.0) ++ model.toArray
    val modelVector = Vectors.dense(modelArray)
    val modelArrayBuilder = mutable.ArrayBuilder.make[Vector]()
    modelArrayBuilder += modelVector
    var index = 1
    while (index < classNum) {
      val kModel = modelArray.slice(index * numFeatures, (index + 1) * numFeatures)
      val kModelVector = Vectors.dense(
        Array.tabulate(numFeatures * classNum)(
          t => kModel(t % numFeatures)
        )
      )
      val modelVectorCopy = modelVector.copy
      axpy(weight, kModelVector, modelVectorCopy)
      modelArrayBuilder += modelVectorCopy
      index += 1
    }
    modelArrayBuilder.result()
  }

  /**
    * For applying numerator normalization for each class probability in FM, this function
    * only keep the exponent difference value, actually a matrix we use Array to store it
    *
    * @param model the Vector of Model
    * @param classNum the number of class
    * @param numFeatures the number of features
    * @param embedding_dim the embedding dim of each feature
    * @return (model, featurePartMinus, embedding minus part, embedding plus part)
    *         store the embedding minus and plus part is to reduce the computation
    *         refer to the FM embedding part
    */

  def fmConvertModel(model: Vector,
                     classNum: Int,
                     numFeatures: Int,
                     embedding_dim: Int): (Array[Vector], Array[Vector], Array[Vector]) = {
    model match {
      case value: DenseVector =>
        fmConvertDenseModel(value, classNum, numFeatures, embedding_dim)
      case value: SparseVector =>
        fmConvertSparseModel(value, classNum, numFeatures, embedding_dim)
      case _ =>
        throw new SparkException(s"unsupported class ${model.getClass.getName}")
    }
  }

  private def fmConvertSparseModel(model: Vector,
                                   classNum: Int,
                                   numFeatures: Int,
                                   embedding_dim: Int): (Array[Vector], Array[Vector], Array[Vector]) = {
    require((numFeatures + 1 + numFeatures * embedding_dim) * (classNum - 1) == model.size,
      s"model.size:${model.size} does not match classNum:$classNum, numFeatures:$numFeatures, embedding_dim:$embedding_dim")
    //embedding parameters are always not sparse, we convert features part use sparse convert and embeddings part dense convert
    //separate feature and embedding part
    val eachModelLength = model.size / (classNum - 1)
    val modelCopy = model.copy.toSparse
    val modelCopyIndices = modelCopy.indices
    val modelCopyValues = modelCopy.values
    val featurePartSize = (numFeatures + 1) * (classNum - 1)
    val featurePartIndices = collection.mutable.ArrayBuilder.make[Int]()
    val featurePartValues = collection.mutable.ArrayBuilder.make[Double]()
    val embeddingPart = Array.tabulate[Double]((embedding_dim * numFeatures) * (classNum - 1))(
      i => model(i / (embedding_dim * numFeatures) * eachModelLength + (i + numFeatures + 1))
    )

    var eachModelIndex = 0
    var index = 0
    while (index < modelCopyIndices.length) {
      eachModelIndex = modelCopyIndices(index) % eachModelLength
      if (eachModelIndex >= 0 && eachModelIndex <= numFeatures) {
        featurePartIndices += (modelCopyIndices(index) / eachModelLength) * (numFeatures + 1) + eachModelIndex
        featurePartValues += modelCopyValues(index)
      }
      index += 1
    }
    val featurePart = Vectors.sparse(featurePartSize, featurePartIndices.result, featurePartValues.result)
    featurePartIndices.clear()
    featurePartValues.clear()
    val featuresPartMinus = convertSparseModel(featurePart, classNum, numFeatures + 1)
    val embeddingPartMinus = convertDenseModel(Vectors.dense(embeddingPart), classNum, embedding_dim * numFeatures)
    val embeddingPartPlus = convertDenseModel(Vectors.dense(embeddingPart), classNum, embedding_dim * numFeatures, 1.0)
    (featuresPartMinus, embeddingPartMinus, embeddingPartPlus)
  }

  private def fmConvertDenseModel(model: Vector,
                                  classNum: Int,
                                  numFeatures: Int,
                                  embedding_dim: Int): (Array[Vector], Array[Vector], Array[Vector]) = {
    require((numFeatures + 1 + numFeatures * embedding_dim) * (classNum - 1) == model.size,
      s"model.size:${model.size} does not match classNum:$classNum, numFeatures:$numFeatures, embedding_dim:$embedding_dim")
    // because in FM Model we have bias parameter, so feature part length is numFeatures + 1
    val eachModelLength = model.size / (classNum - 1)
    val featurePart = Array.tabulate[Double]((numFeatures + 1) * (classNum - 1))(
      i => model(i / (numFeatures + 1) * eachModelLength + i % (numFeatures + 1))
    )
    val embeddingPart = Array.tabulate[Double]((embedding_dim * numFeatures) * (classNum - 1))(
      i => model(i / (embedding_dim * numFeatures) * eachModelLength + (i % (embedding_dim * numFeatures) + numFeatures + 1))
    )
    /*println("featurePart:" + featurePart.mkString(","))
    println("embeddingPart:" + embeddingPart.mkString(","))*/
    val featuresPartMinus = convertDenseModel(Vectors.dense(featurePart), classNum, numFeatures + 1)
    val embeddingPartMinus = convertDenseModel(Vectors.dense(embeddingPart), classNum, embedding_dim * numFeatures)
    val embeddingPartPlus = convertDenseModel(Vectors.dense(embeddingPart), classNum, embedding_dim * numFeatures, 1.0)
    (featuresPartMinus, embeddingPartMinus, embeddingPartPlus)
  }
}
