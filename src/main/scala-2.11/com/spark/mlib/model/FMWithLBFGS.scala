package com.spark.mlib.model

import com.spark.mlib.feature.{MyLabeledPoint, MyStandardScaler, StandardScalerModel}
import com.spark.mlib.linear.BLAS._
import com.spark.mlib.log.Logging
import com.spark.mlib.optimizer.{FMGradient, LBFGS, LogisticL1Updater, LogisticL2Updater}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkException
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

import scala.util.Random

/**
  * Created by qinzhiheng on 2017/7/16.
  *
  * FM model supports soft-max include binary classification use LBFGS
  * TODO: FFM is a little more complexity, every embedding dim's length is feature field times of FM
  */

object FMWithLBFGS {
  def train(data: RDD[MyLabeledPoint],
            numIterations:Int = 100,
            scaleWithStd: Boolean = true,
            scaleWithMean: Boolean = false,
            numCorrections: Int = 10,
            regParam: Double = 0.0,
            convergenceToVal: Double = 1e-6,
            updater_strategy: String = "L2",
            init_std: Double = 0.0,
            init_mean: Double = 0.0,
            classNum: Int = 2,
            forceLBFGS: Boolean): FMWithLBFGS = {
    new FMWithLBFGS(numIterations, scaleWithStd, scaleWithMean, convergenceToVal, classNum)
      .setUpdaterStrategy(updater_strategy)
      .setInitStd(init_std)
      .setInitMean(init_mean)
      .setNumCorrections(numCorrections)
      .setRegParam(regParam)
      .setForceLBFGS(forceLBFGS)
      .run(data)
  }
}

class FMWithLBFGS(private var numIterations: Int = 100,
                  private var scaleWithStd: Boolean = true,
                  private var scaleWithMean: Boolean = false,
                  private var convergenceToVal: Double = 0.0,
                  private var classNum: Int = 2) extends Serializable with Logging {
  private var numFeatures: Int = 0
  private var embedding_dim: Int = 0
  private var init_std: Double = 1.0
  private var init_mean: Double = 0.0
  private var model: Vector = _
  private var regParam: Double = 0.0
  private var updater_strategy = "L2"
  private var numCorrections: Int = 10
  private var standardScalerModel: StandardScalerModel = _
  private var dataMeanValue = Vectors.zeros(numFeatures)
  private var forceLBFGS = false


  def setNumClass(classNum: Int): this.type = {
    require(classNum >= 2,
      s"classNum should be in range [2, inf), but classNum:$classNum")
    this.classNum = classNum
    this
  }

  def setNumIter(numIterations: Int): this.type = {
    require(numIterations >= 1,
      s"numIterations should be in range [1, inf), but numIterations:$numIterations")
    this.numIterations = numIterations
    this
  }

  def setScaleWithStd(scaleWithStd: Boolean): this.type = {
    this.scaleWithStd = scaleWithStd
    this
  }

  def setScaleWithMean(scaleWithMean: Boolean): this.type = {
    this.scaleWithMean = scaleWithMean
    this
  }

  def setForceLBFGS(forceLBFGS: Boolean): this.type = {
    this.forceLBFGS = forceLBFGS
    this
  }

  def setNumCorrections(numCorrections: Int): this.type = {
    require(numCorrections >= 1,
      s"numCorrections should be in range [1, inf), but numCorrections:$numCorrections")
    this.numCorrections = numCorrections
    this
  }

  def setRegParam(regParam: Double): this.type = {
    require(regParam >= 0,
      s"regParam should be in range [0, inf), but regParam:$regParam")
    this.regParam = regParam
    this
  }

  def setConvergenceToVal(convergenceToVal: Double): this.type = {
    require(convergenceToVal > 0,
      s"convergenceToVal should be in range [0, inf), but convergenceToVal:$convergenceToVal")
    this.convergenceToVal = convergenceToVal
    this
  }

  def setUpdaterStrategy(updater_strategy: String): this.type = {
    this.updater_strategy = updater_strategy
    this
  }

  def setNumFeatures(numFeatures: Int): this.type = {
    require(numFeatures >= 1,
      s"numFeatures should be in range [1, inf), but numFeatures:$numFeatures")
    this.numFeatures = numFeatures
    dataMeanValue = Vectors.zeros(numFeatures)
    this
  }

  def setEmbeddingDim(embedding_dim: Int): this.type = {
    require(embedding_dim >= 1,
      s"embedding_dim should be in range [1, inf), but numFeatures:$embedding_dim")
    this.embedding_dim = embedding_dim
    this
  }

  def setInitStd(init_std: Double): this.type = {
    require(init_std >= 0,
      s"init_std should be in range [0, inf), but init_std:$init_std")
    this.init_std = init_std
    this
  }

  def setInitMean(init_mean: Double): this.type = {
    require(init_mean >= 0,
      s"init_mean should be in range [0, inf), but init_mean:$init_mean")
    this.init_mean = init_mean
    this
  }

  def setDataMeanValue(dataMeanValue: Vector): this.type = {
    this.dataMeanValue = dataMeanValue
    this
  }
  // FM initialization init_std and init_mean can not be zero simultaneously
  // init_std suggest sqrt(2/modelLength), mean suggest 0.
  //TODO: sparse model initialization ?
  def generateInitialWeights(input: RDD[MyLabeledPoint]): Vector = {
    if (numFeatures <= 0) {
      val sample = input.take(1)(0)
      numFeatures = sample.features.size
      require(sample.num.length == classNum,
        s"data class number should match classNUm!")
    }
    require(embedding_dim > 0,
      s"embedding_dim should be in range (0, inf), but embedding_dim:$embedding_dim")
    require(init_std != 0 || init_mean != 0,
      s"init_std and init_mean cannot all be 0, otherwise the embedding gradient will be always 0")
    val modelLength = (1 + numFeatures + embedding_dim * numFeatures) * (classNum - 1) //bias + numFeatures + embedding_num
    model = Vectors.dense(Array.fill(modelLength)(Random.nextGaussian() * init_std + init_mean))
    model
  }

  def setInitialWeights(weights: Any): this.type = { //support for delta learning
    model = weights match {
      case values: Vector =>
        values
      case values: Array[Double] =>
        Vectors.dense(values)
      case Seq(size: Double, indices: Array[Double], values: Array[Double]) =>
        Vectors.sparse(size.toInt, indices.map(_.toInt), values)
      case other =>
        throw new SparkException(s"unable to parse ${weights.getClass.getName}")
    }
    this
  }

  def getClassNum: Int = this.classNum
  def getNumIter: Int = this.numIterations
  def getScaleWithStd: Boolean = this.scaleWithStd
  def getScaleWithMean: Boolean = this.scaleWithMean
  def getNumCorrections: Int = this.numCorrections
  def getRegParams: Double = this.regParam
  def getConvergenceToVal: Double = this.convergenceToVal
  def getNumFeatures: Int = this.numFeatures
  def getInitStd: Double = this.init_std
  def getInitMean: Double = this.init_mean
  def getModel: Vector = this.model

  def predictPoint(dataPoint: Vector): Array[Double] = {
    require(model.size == (dataPoint.size + 1 + numFeatures * embedding_dim) * (classNum - 1),
      s"model is not trained or initialized, model size: ${model.size}, dataPointSize:!")
    val dividends = fmConvertModel(model, classNum, numFeatures, embedding_dim)
    predictPoint(dataPoint, dividends, dataMeanValue)
  }

  def predictPoint(dataPoint: Vector,
                   dividends: (Array[Vector], Array[Vector], Array[Vector]),
                   dataMeanValue: Vector): Array[Double] = {
    require(dividends._1(0).size == classNum * (dataPoint.size + 1),
      s"dividends._1(0) length ${dividends._1(0).size} should match classNum: $classNum, dataPoint size: ${dataPoint.size}")
    axpy(-1.0, dataMeanValue, dataPoint)
    val probabilities = collection.mutable.ArrayBuilder.make[Double]()
    val dataExtend = dataPoint match {
      case value: DenseVector =>
        Vectors.dense(Array(1.0) ++ value.values)
      case value: SparseVector =>
        val indices = Array(0) ++ value.indices.map(_ + 1)
        val values = Array(1.0) ++ value.values
        Vectors.sparse(value.size + 1, indices, values)
      case _ =>
        throw new SparkException(s"cant not parse ${dataPoint.getClass.getName}")
    }
    val (featureMinusPartArray, embeddingMinusPartArray, embeddingPlusPartArray) = dividends
    var kModel = Vectors.zeros(classNum)
    var index = 0
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
          while (feature_index < numFeatures) {
            val itemIndex = inner_index * numFeatures * embedding_dim + feature_index * embedding_dim + embedding_index
            val embeddingPlusItem = embeddingPlusVector(itemIndex)
            val embeddingMinusItem = embeddingMinusVector(itemIndex)
            crossPlusItem += embeddingPlusItem * dataPoint(feature_index)
            crossMinusItem += embeddingMinusItem * dataPoint(feature_index)
            squareSumItem += embeddingPlusItem * embeddingMinusItem * dataPoint(feature_index) * dataPoint(feature_index)
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
      probabilities += 1.0 / sum
      index += 1
    }
    probabilities.result()
  }

  def predict(data: RDD[Vector]): RDD[Array[Double]] = {
    val localWeights = model
    val dividends = fmConvertModel(localWeights, classNum, numFeatures, embedding_dim)
    val brc = data.context.broadcast(dividends)
    val meanBrc = data.context.broadcast(dataMeanValue)
    val result = data.mapPartitions(iter => { // broadcast only to the partition, also each executor
      val w = brc.value
      val meanBrcValue = meanBrc.value
      iter.map(v => predictPoint(v, w, meanBrcValue))
    })
    brc.destroy()
    result
  }

  override def toString: String = {
    model match {
      case DenseVector(value) =>
        s"${this.getClass.getName}, model weights=${model.toArray.mkString("\t").trim}, numFeatures=$numFeatures, embedding_dim=$embedding_dim"
      case SparseVector(size, indices, values) =>
        s"${this.getClass.getName}, model paras: size=$numFeatures, index=[${indices.mkString(",")}], value=[${values.mkString(",")}], " +
          s"numFeatures=$numFeatures, embedding_dim=$embedding_dim"
      case _ =>
        throw new SparkException(s"unable to parse model ${model.getClass.getName}")
    }
  }

  def toDebugString: String = {
    model match {
      case DenseVector(value) =>
        s"LRModel:w=[${model.toArray.mkString(",").trim}], numFeatures=$numFeatures, embedding_dim=$embedding_dim"
      case SparseVector(size, indices, values) =>
        s"LRModel:size=$size, indices=[${indices.mkString(",").trim}, values=[${values.mkString(",").trim}]], numFeatures=$numFeatures, embedding_dim=$embedding_dim"
      case _ =>
        throw new SparkException(s"unable to parse model ${model.getClass.getName}")
    }
  }

  def run(data: RDD[MyLabeledPoint]): this.type = {
    run(data, generateInitialWeights(data))
  }

  def run(data: RDD[MyLabeledPoint], initialWeights: Vector): this.type = {
    if (numFeatures < 0) {
      numFeatures = data.take(1)(0).features.size
      dataMeanValue = Vectors.zeros(numFeatures)
    }
    require(initialWeights.size == (numFeatures + 1 + numFeatures * embedding_dim) * (classNum - 1))

    if (data.getStorageLevel == StorageLevel.NONE) {
      logWarning("the input data is not cached, which may hurt the performance")
    }
    standardScalerModel = new MyStandardScaler(scaleWithMean, scaleWithStd).fit(data)
    dataMeanValue = standardScalerModel.mean.copy
    val input = data.map(labelPoint => (standardScalerModel.transform(labelPoint.features), labelPoint.num)).cache()
    val updater = updater_strategy.toUpperCase() match {
      case "L1" =>
        new LogisticL1Updater
      case "L2" =>
        new LogisticL2Updater
      case other =>
        throw new SparkException(s"wrong update strategy ${other.toUpperCase()}")
    }
    val optimizer = new LBFGS(new FMGradient, updater).setNumFeatures(numFeatures).setClassNum(classNum)
    optimizer.setConvergenceToVal(convergenceToVal)
      .setMaxNumIterations(numIterations)
      .setNumCorrections(numCorrections)
      .setRegParams(regParam)
      .setEmbeddingDim(embedding_dim)
      .setForceLBFGS(forceLBFGS)
    val weights = optimizer.optimize(input, initialWeights)
    //logWarning(weights.size + "\t" + numFeatures + "\t" + classNum)
    model = standardScalerModel.fmTransformModel(weights, numFeatures, embedding_dim)
    this
  }

  case class Data(weights: Option[Vector],
                  dataMeanValue: Option[Vector],
                  numFeatures: Int,
                  classNum: Int,
                  embedding_dim: Int)

  protected def formatVersion: String = "1.0"

  private def save(ss: SparkSession, path: String, data: Data): Unit = {
    val modelClassName = this.getClass.getName
    val metadata = compact(render(
      ("class" -> modelClassName) ~ ("version" -> formatVersion) ~ ("numFeatures" -> numFeatures)
        ~ ("classNum" -> classNum) ~ ("embedding_dim" -> embedding_dim)
    ))
    val hadoopConfig = ss.sparkContext.hadoopConfiguration
    val fs = FileSystem.get(hadoopConfig)
    val dir = new Path(path).toUri.toString
    val metaDataPath = new Path(path, "metadata").toUri.toString
    val dataPath = new Path(path, "data").toUri.toString
    if (fs.exists(new Path(dir))) {
      logError(s"$dir has been existed! Please delete it first")
      sys.exit(1)
    }
    val sc = ss.sparkContext
    sc.parallelize(Seq(metadata), 1).saveAsTextFile(metaDataPath)
    ss.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
  }

  def save(ss: SparkSession, path: String = "FM-LBFGS"): Unit = {
    save(ss, path, Data(Option(model), Option(dataMeanValue), numFeatures, classNum, embedding_dim))
  }

  def load(ss: SparkSession, path: String, modelClass: String): this.type = {
    val validPath = new Path(path, "data").toUri.toString.trim
    val dataRDD = ss.read.parquet(validPath)
    val dataArray = dataRDD.select("weights").take(1)
    val dataMeanValueArray = dataRDD.select("dataMeanValue").take(1)
    val numFeaturesArray = dataRDD.select("numFeatures").take(1)
    val classNumArray = dataRDD.select("classNum").take(1)
    val embeddingDimArray = dataRDD.select("embedding_dim").take(1)
    assert(dataArray.length == 1 && numFeaturesArray.length == 1 && classNumArray.length == 1,
      s"unable to load $modelClass data from: $validPath")
    val data = dataArray(0)
    val dataMeanValues = dataMeanValueArray(0)
    val numFeatures = numFeaturesArray(0)
    val classNums = classNumArray(0)
    val embeddingDims = embeddingDimArray(0)
    val weights = data.getAs[Vector](0)
    val dataMeanValue = dataMeanValues.getAs[Vector](0)
    val numFeature = numFeatures.getAs[Int](0)
    val classNum = classNums.getAs[Int](0)
    val embedding_dim = embeddingDims.getAs[Int](0)
    if (weights == null || weights.size == 0 || dataMeanValue == null || dataMeanValue.size == 0
        || embedding_dim <= 0) {
      throw new SparkException("model loaded false!")
    }
    setInitialWeights(weights)
    setDataMeanValue(dataMeanValue)
    setNumFeatures(numFeature)
    setNumClass(classNum)
    setEmbeddingDim(embedding_dim)
    this
  }
}
