package com.spark.mlib.model

import com.spark.mlib.feature.MyLabeledPoint
import com.spark.mlib.linear.BLAS._
import com.spark.mlib.log.Logging
import com.spark.mlib.optimizer.{LogisticGradient, LogisticL1Updater, LogisticL2Updater, RMSPropGradientDescent}
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
  * support for softmax include binary classification gradient descend
  * optimization include BGD, SBGD, Momentum, Adagrad, RMSProp, Adam, FTRL
  *
  * For batch or mini-batch, standard-scaler is not suitable
  * As our features are most continuous, mini-batch optimization may be sensitive to noise,
  * suggest a little bigger mini-batch size
  *
  * TODO: scattered continuous features: like GBDT-LR, or interval
  */

object LogisticRegressionWithRMSProp {
  def train(data: RDD[MyLabeledPoint],
            numIterations: Int = 1000,
            miniBatchFraction: Double = 0.001,
            stepSize: Double = 1.0,
            epsilon: Double = 1e-8,
            decayRate: Double = 1e-8,
            regParam: Double = 0.0,
            convergenceToVal: Double = 1e-6,
            convergenceTime: Int = 3,
            updater_strategy: String = "L2",
            init_std: Double = 0.0,
            init_mean: Double = 0.0,
            momentum: Double = 0.8,
            classNum: Int = 2): LogisticRegressionWithRMSProp = {
    new LogisticRegressionWithRMSProp(numIterations, miniBatchFraction, stepSize, regParam, convergenceToVal, convergenceTime, classNum)
      .setUpdaterStrategy(updater_strategy)
      .setInitStd(init_std)
      .setInitMean(init_mean)
      .setStepSize(stepSize)
      .setEpsilon(epsilon)
      .setDecayRate(decayRate)
      .setMomentum(momentum)
      .run(data)
  }
}

class LogisticRegressionWithRMSProp(private var numIterations: Int = 1000,
                                    private var miniBatchFraction: Double = 0.001,
                                    private var stepSize: Double = 1.0,
                                    private var regParam: Double = 0.0,
                                    private var convergenceToVal: Double = 0.0,
                                    private var convergenceTime: Int = 3,
                                    private var classNum: Int = 2) extends Serializable with Logging {
  private var numFeatures: Int = 0
  private var init_std: Double = 1.0
  private var init_mean: Double = 0.0
  private var model: Vector = _
  private var updater_strategy = "L2"
  private var epsilon = 1e-8
  private var decayRate = 1e-8
  private var momentum = 0.8

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

  def setMiniBatchFraction(miniBatchFraction: Double): this.type = {
    require(miniBatchFraction > 0,
      s"miniBatchFraction should be in range (0, inf), but miniBatchFraction:$miniBatchFraction")
    this.miniBatchFraction = miniBatchFraction
    this
  }

  def setStepSize(stepSize: Double): this.type = {
    require(stepSize > 0,
      s"stepSize should be in range (0, inf), but stepSize:$stepSize")
    this.stepSize = stepSize
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

  def setConvergenceTime(convergenceTime: Int): this.type = {
    require(convergenceTime >= 1,
      s"convergenceTime should be in range [1, inf), but convergenceTime:$convergenceTime")
    this.convergenceTime = convergenceTime
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

  def setEpsilon(epsilon: Double): this.type = {
    require(epsilon > 0,
      s"epsilon should be in range (0, inf), but epsilon:$epsilon")
    this.epsilon = epsilon
    this
  }

  def setDecayRate(decayRate: Double): this.type = {
    require(decayRate > 0,
      s"decayRate should be in range (0, inf), but decayRate:$decayRate")
    this.decayRate = decayRate
    this
  }

  def setMomentum(momentum: Double): this.type = {
    require(momentum > 0 && momentum < 1,
      s"momentum must be in range (0, 1) but got $momentum")
    this.momentum = momentum
    this
  }

  //TODO: sparse model initialization ?
  def generateInitialWeights(input: RDD[MyLabeledPoint]): Vector = {
    if (numFeatures <= 0) {
      numFeatures = input.map(_.features.size).first()
    }
    model = Vectors.dense(Array.fill(numFeatures * (classNum - 1))(Random.nextDouble() * init_std + init_mean))
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
  def getMiniBatchFraction: Double = this.miniBatchFraction
  def getRegParams: Double = this.regParam
  def getConvergenceToVal: Double = this.convergenceToVal
  def getNumFeatures: Int = this.numFeatures
  def getInitStd: Double = this.init_std
  def getInitMean: Double = this.init_mean
  def getModel: Vector = this.model

  def predictPoint(dataPoint: Vector): Array[Double] = {
    require(model.size == dataPoint.size * (classNum - 1), "model is not trained or initialized, please check!")
    val dividends = convertModel(model, classNum, numFeatures)
    predictPoint(dataPoint, dividends)
  }

  def predictPoint(dataPoint: Vector, dividends: Array[Vector]): Array[Double] = {
    require(dividends.length == classNum,
      s"dividends length ${dividends.length} should be equal to classNum: $classNum")
    require(dividends(0).size == dataPoint.size * classNum,
      s"model's size ${dividends(0).size} does not equal feature' size $numFeatures")
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
    val result = data.mapPartitions(iter => { // broadcast only to the partition, also each executor
    val w = brc.value
      iter.map(v => predictPoint(v, w))
    })
    brc.destroy()
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
      numFeatures = data.map(_.features.size).first()
    }
    require(numFeatures * (classNum - 1) == initialWeights.size)

    if (data.getStorageLevel == StorageLevel.NONE) {
      logWarning("the input data is not cached, which may hurt the performance")
    }
    val input = data.map(labelPoint => (labelPoint.features, labelPoint.num)).cache()
    val updater = updater_strategy.toUpperCase() match {
      case "L1" =>
        new LogisticL1Updater
      case "L2" =>
        new LogisticL2Updater
      case other =>
        throw new SparkException(s"wrong update strategy ${other.toUpperCase()}")
    }
    val optimizer = new RMSPropGradientDescent(new LogisticGradient, updater)

    optimizer.setConvergenceTime(convergenceTime)
      .setClassNum(classNum)
      .setNumFeatures(numFeatures)
      .setConvergenceTol(convergenceToVal)
      .setNumIterations(numIterations)
      .setRegParam(regParam)
      .setLearningRate(stepSize)
      .setDecayRate(decayRate)
      .setMiniBatchFraction(miniBatchFraction)
      .setMomentum(momentum)
    val weights = optimizer.optimize(input, initialWeights)
    setInitialWeights(weights)
    this
  }

  case class Data(weights: Option[Vector], numFeatures: Int, classNum: Int)

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
    save(ss, path, Data(Option(model), numFeatures, classNum))
  }

  def load(ss: SparkSession, path: String, modelClass: String): this.type = {
    val validPath = new Path(path, "data").toUri.toString.trim
    val dataRDD = ss.read.parquet(validPath)
    val dataArray = dataRDD.select("weights").take(1)
    val numFeaturesArray = dataRDD.select("numFeatures").take(1)
    val classNumArray = dataRDD.select("classNum").take(1)
    assert(dataArray.length == 1 && numFeaturesArray.length == 1 && classNumArray.length == 1,
      s"unable to load $modelClass data from: $validPath")
    val data = dataArray(0)
    val numFeatures = numFeaturesArray(0)
    val classNums = classNumArray(0)
    val weights = data.getAs[Vector](0)
    val numFeature = numFeatures.getAs[Int](0)
    val classNum = classNums.getAs[Int](0)
    if (weights == null || weights.size == 0) {
      throw new SparkException("model loaded false!")
    }
    setInitialWeights(weights)
    setNumFeatures(numFeature)
    setNumClass(classNum)
    this
  }
}
