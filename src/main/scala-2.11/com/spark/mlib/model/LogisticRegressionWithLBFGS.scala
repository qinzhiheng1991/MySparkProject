package com.spark.mlib.model

import com.spark.mlib.feature.{MyLabeledPoint, MyStandardScaler, StandardScalerModel}
import com.spark.mlib.linear.BLAS._
import com.spark.mlib.log.Logging
import com.spark.mlib.optimizer.{LBFGS, LogisticGradient, LogisticL1Updater, LogisticL2Updater}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkException
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._
import scala.collection.mutable
import scala.util.Random

/**
  * Created by qinzhiheng on 2017/7/16.
  *
  * a linear model supports soft-max include binary classification
  */

object LogisticRegressionWithLBFGS {
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
            forceLBFGS: Boolean = false): LogisticRegressionWithLBFGS = {
    new LogisticRegressionWithLBFGS(numIterations, scaleWithStd, scaleWithMean, numCorrections, regParam, convergenceToVal, classNum)
        .setUpdaterStrategy(updater_strategy).setInitStd(init_std).setInitMean(init_mean).run(data)
  }
}

class LogisticRegressionWithLBFGS(private var numIterations: Int = 100,
                                  private var scaleWithStd: Boolean = true,
                                  private var scaleWithMean: Boolean = false,
                                  private var numCorrections: Int = 10,
                                  private var regParam: Double = 0.0,
                                  private var convergenceToVal: Double = 0.0,
                                  private var classNum: Int = 2) extends Serializable with Logging {
  private var numFeatures: Int = 0
  private var init_std: Double = 1.0
  private var init_mean: Double = 0.0
  private var model: Vector = _
  private var updater_strategy = "L2"
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

  def setForceLBFGS(forceLBFGS: Boolean): this.type = {
    this.forceLBFGS = forceLBFGS
    this
  }

  //TODO: sparse model initialization ?
  def generateInitialWeights(input: RDD[MyLabeledPoint]): Vector = {
    if (numFeatures <= 0) {
      val sample = input.take(1)(0)
      numFeatures = sample.features.size
      require(sample.num.length == classNum, s"data class number should match classNUm!")
    }
    model = Vectors.dense(Array.fill(numFeatures * (classNum - 1))(Random.nextGaussian() * init_std + init_mean))
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
    require(model.size == dataPoint.size * (classNum - 1), "model is not trained or initialized, please check!")
    val dividends = convertModel(model, classNum, numFeatures)
    predictPoint(dataPoint, dividends, dataMeanValue)
  }

  def predictPoint(dataPoint: Vector, dividends: Array[Vector], dataMeanValue: Vector): Array[Double] = {
    require(dividends.length == classNum,
      s"dividends length ${dividends.length} should be equal to classNum: $classNum")
    require(dividends(0).size == dataPoint.size * classNum,
      s"model's size ${dividends(0).size} does not equal feature' size $numFeatures")
    axpy(-1.0, dataMeanValue, dataPoint)
    var kModel = Vectors.zeros(classNum)
    val probabilities = mutable.ArrayBuilder.make[Double]()
    var index = 0
    while (index < classNum) {
      kModel = Vectors.zeros(classNum)
      axpy(1.0, fDotExp(dataPoint, dividends(index)), kModel)
      var sum = 0.0
      kModel.foreachActive((i, value) => sum += value)
      probabilities += 1.0 / sum
      index += 1
    }
    probabilities.result()
  }

  def predict(data: RDD[Vector]): RDD[Array[Double]] = {
    val localWeights = model
    val dividends = convertModel(localWeights, classNum, numFeatures)
    val brc = data.context.broadcast(dividends)
    val meanBrc = data.context.broadcast(dataMeanValue)
    val result = data.mapPartitions(iter => { // broadcast only to the partition, also each executor
      val w = brc.value
      val menBrcValue = meanBrc.value
      iter.map(v => predictPoint(v, w, menBrcValue))
    })
    brc.destroy()
    meanBrc.destroy()
    result
  }

  override def toString: String = {
    model match {
      case DenseVector(value) =>
        s"${this.getClass.getName}, model weights: ${model.toArray.mkString("\t").trim}"
      case SparseVector(size, indices, values) =>
        s"${this.getClass.getName}, model paras: size=$numFeatures, index=[${indices.mkString(",")}], value=[${values.mkString(",")}]"
      case _ =>
        throw new SparkException(s"unable to parse model ${model.getClass.getName}")
    }
  }

  def toDebugString: String = {
    model match {
      case DenseVector(value) =>
        s"LRModel:w=[${model.toArray.mkString(",").trim}]"
      case SparseVector(size, indices, values) =>
        s"LRModel:size=$size, indices=[${indices.mkString(",").trim}, values=[${values.mkString(",").trim}]]"
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
    require(numFeatures * (classNum - 1) == initialWeights.size)

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
    val optimizer = new LBFGS(new LogisticGradient, updater).setNumFeatures(numFeatures).setClassNum(classNum)
    optimizer.setConvergenceToVal(convergenceToVal)
             .setMaxNumIterations(numIterations)
             .setNumCorrections(numCorrections)
             .setForceLBFGS(forceLBFGS)
             .setRegParams(regParam)
    val weights = optimizer.optimize(input, initialWeights)
    model = standardScalerModel.transformModel(weights)
    this
  }

  case class Data(weights: Option[Vector], dataMeanValue: Option[Vector], numFeatures: Int, classNum: Int)

  protected def formatVersion: String = "1.0"

  private def save(ss: SparkSession, path: String, data: Data): Unit = {
    val modelClassName = this.getClass.getName
    val metadata = compact(render(
      ("class" -> modelClassName) ~ ("version" -> formatVersion) ~ ("numFeatures" -> numFeatures)
      ~ ("classNum" -> classNum)
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

  def save(ss: SparkSession, path: String = "LR-LBFGS"): Unit = {
    save(ss, path, Data(Option(model), Option(dataMeanValue), numFeatures, classNum))
  }

  def load(ss: SparkSession, path: String, modelClass: String): this.type = {
    val validPath = new Path(path, "data").toUri.toString.trim
    val dataRDD = ss.read.parquet(validPath)
    val dataArray = dataRDD.select("weights").take(1)
    val dataMeanValueArray = dataRDD.select("dataMeanValue").take(1)
    val numFeaturesArray = dataRDD.select("numFeatures").take(1)
    val classNumArray = dataRDD.select("classNum").take(1)
    assert(dataArray.length == 1 && numFeaturesArray.length == 1 && classNumArray.length == 1,
      s"unable to load $modelClass data from: $validPath")
    val data = dataArray(0)
    val dataMeanValues = dataMeanValueArray(0)
    val numFeatures = numFeaturesArray(0)
    val classNums = classNumArray(0)
    val weights = data.getAs[Vector](0)
    val dataMeanValue = dataMeanValues.getAs[Vector](0)
    val numFeature = numFeatures.getAs[Int](0)
    val classNum = classNums.getAs[Int](0)
    if (weights == null || weights.size == 0 || dataMeanValue == null || dataMeanValue.size == 0) {
      throw new SparkException("model loaded false!")
    }
    setInitialWeights(weights)
    setDataMeanValue(dataMeanValue)
    setNumFeatures(numFeature)
    setNumClass(classNum)
    this
  }
}
