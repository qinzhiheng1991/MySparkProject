package com.spark.mlib.feature

import com.spark.mlib.log.Logging
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.rdd.RDD

/**
  * Created by qinzhiheng on 2017/11/21.
  */
class MyStandardScaler(withMean: Boolean, withStd: Boolean) extends Logging{
  def this() = this(false, true)
  if (!(withMean || withStd)) {
    logWarning("both withMean and withStd is false, do nothing!")
  }

  def fit(data: RDD[MyLabeledPoint]): StandardScalerModel = {
    val summary = data.treeAggregate(new MultivariateOnlineSummarizer)(
      (aggregator, data) => aggregator.add(data.features, data.num.sum),
      (aggregator1, aggregator2) => aggregator1.merge(aggregator2))

    new StandardScalerModel(
      Vectors.dense(summary.variance.toArray.map(v => math.sqrt(v))),
      summary.mean,
      withStd,
      withMean)
  }
}

class StandardScalerModel(val std: Vector,
                          val mean: Vector,
                          var withStd: Boolean,
                          var withMean: Boolean) extends Serializable {
  def this(std: Vector, mean: Vector) {
    this(std, mean, withStd = std != null, withMean = mean != null)
    require(this.withStd || this.withMean,
      "at least one of std or mean vectors must be provided")
    if (this.withStd && this.withMean) {
      require(mean.size == std.size,
        "mean and std vectors must have equal size if both are provided")
    }
  }

  def this(std: Vector) = this(std, null)

  def setWithMean(withMean: Boolean): this.type = {
    require(!(withMean && this.mean == null), "cannot set withMean to true while mean is null")
    this.withMean = withMean
    this
  }

  def setWithStd(withStd: Boolean): this.type = {
    require(!(withStd && this.std == null),
      "cannot set withStd to true while std is null")
    this.withStd = withStd
    this
  }

  // lazy value, only use it
  private lazy val shift: Array[Double] = mean.toArray

  /**
    * Applies standardization transformation on a vector.
    *
    * @param vector Vector to be standardized.
    * @return Standardized vector. If the std of a column is zero, it will return default `0.0`
    *         for the column with zero std.
    */
  def transform(vector: Vector): Vector = {
    require(vector.size % std.size == 0)
    val stdSize = std.size
    if (withMean) {
      val localShift = shift
      val values = vector match {
        case d: DenseVector => d.values.clone()
        case v: Vector => v.toArray
      }
      val size = values.length
      if (withStd) {
        var i = 0
        var stdIndex = 0
        while (i < size) {
          stdIndex = i % stdSize
          values(i) = if (std(stdIndex) != 0.0) (values(i) - localShift(stdIndex)) * (1.0 / std(stdIndex)) else 0.0
          i += 1
        }
      } else {
        var i = 0
        var stdIndex = 0
        while (i < size) {
          stdIndex = i % stdSize
          values(i) -= localShift(stdIndex)
          i += 1
        }
      }
      Vectors.dense(values)
    } else if (withStd) {
      vector match {
        case DenseVector(vs) =>
          val values = vs.clone()
          val size = values.length
          var i = 0
          var stdIndex = 0
          while(i < size) {
            stdIndex = i % stdSize
            values(i) *= (if (std(stdIndex) != 0.0) 1.0 / std(stdIndex) else 0.0)
            i += 1
          }
          Vectors.dense(values)
        case SparseVector(size, indices, vs) =>
          val values = vs.clone()
          val nnz = values.length
          var i = 0
          var stdIndex = 0
          while (i < nnz) {
            stdIndex = indices(i) % stdSize
            values(i) *= (if (std(stdIndex) != 0.0) 1.0 / std(stdIndex) else 0.0)
            i += 1
          }
          Vectors.sparse(size, indices, values)
        case v => throw new IllegalArgumentException("Do not support vector type " + v.getClass)
      }
    } else {
      // Note that it's safe since we always assume that the data in RDD should be immutable.
      vector
    }
  }

  /**
    * Applies standardization transformation for a model.
    *
    * @param vector Vector to be standardized.
    * @return Standardized model
    */
  def transformModel(vector: Vector): Vector = {
    val rawWithMean = withMean
    setWithMean(false)
    val newModel = transform(vector)
    setWithMean(rawWithMean)
    newModel
  }

  /**
    * Applies standardization transformation for fm model.
    *
    * @param vector Vector to be standardized.
    * @return Standardized model
    */
  def fmTransformModel(vector: Vector, numFeature: Int, embedding_dim: Int): Vector = {
    val modelCopy = vector.copy
    val modelCopyValue = modelCopy.toDense.values
    val eachModelLength = numFeature + 1 + numFeature * embedding_dim
    var stdIndex = 0
    var index = 0
    vector.foreachActive((i, value) => {
        index = i % eachModelLength
        if (index >= 1) {
          if (index < numFeature + 1) {
            modelCopyValue(i) *= (if (std(index - 1) != 0.0) 1.0 / std(index - 1) else 0.0)
          }
          else {
            stdIndex = (index - numFeature - 1) / embedding_dim
            modelCopyValue(i) *= (if (std(stdIndex) != 0.0) 1.0 / std(stdIndex) else 0.0)
          }
        }
      }
    )
    modelCopy
  }
}

