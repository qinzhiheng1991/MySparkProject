package com.spark.mlib.model

import com.spark.mlib.feature.{MyLabeledPoint, MyStandardScaler, StandardScalerModel}
import com.spark.mlib.linear.BLAS._
import com.spark.mlib.log.Logging
import com.spark.mlib.optimizer.{LBFGS, LogisticL1Updater, LogisticL2Updater, TwoStageModelGradient}
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
  * ClickModel in search engine estimates doc true relevance by solving the bias problem,
  * which is infected by position, picture, display height and red in title etc.
  *
  *This class is a two-stage binary-classification model. In click model examine assumption, click probability satisfies:
  *      P(c|q,u) = P(e|p,d) * P(a|q,u)
  * c is click, q is query, p is position, d is the distance from the click above, u is doc, a is relevance, e is examine
  * a doc is clicked only if it is examined and relevant
  *
  * The examine and relevance variables are unobservable, the paper below estimates them through observable
  * variable "c" by EM Algorithm. However, examine hypothesis considers just only the position and click distance.
  * refer: http://www.bpiwowar.net/wp-content/papercite-data/pdf/dupret2008a-user-browsing.pdf
  *
  * We apply two stage linear models for P(e|p,d) and P(a|q,u) separately.
  * For ranking, we only use the P(a|q,u) part to rank documents in click_part of search engine.
  *
  * The relevant stage, we use no-bias click features(based on BBM), text features and doc category
  * The bias part, we use position(on-hot 10 length), doc category, red in title etc.
  *
  * Two-stage Model just be suitable for binary classification, so classNum variable is 2.
  * And also, the regular loss is based on the overall Gaussian or Laplace prior distribution not separately for each stage.
  *
  * Two-stage Model gradient computation refers to optimizer.Gradient.compute method
  *
  * Created by qinzhiheng on 2017/7/11.
  */

object TwoStageNobiasClickModelWithLBFGS {
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
            numFeatures: Int = 0,
            breakpoint: Int = 1): TwoStageNobiasClickModelWithLBFGS = {
    new TwoStageNobiasClickModelWithLBFGS(
      numIterations,
      scaleWithStd,
      scaleWithMean,
      numCorrections,
      regParam,
      convergenceToVal,
      numFeatures,
      breakpoint).setUpdaterStrategy(updater_strategy).setInitStd(init_std).setInitMean(init_mean).run(data)
  }
}

class TwoStageNobiasClickModelWithLBFGS(private var numIterations: Int = 100,
                                        private var scaleWithStd: Boolean = true,
                                        private var scaleWithMean: Boolean = false,
                                        private var numCorrections: Int = 10,
                                        private var regParam: Double = 0.0,
                                        private var convergenceToVal: Double = 0.0,
                                        private var numFeatures: Int = 0,
                                        private var breakpoint: Int = 1) extends Serializable with Logging {

  private var init_std: Double = 1.0
  private var init_mean: Double = 0.0
  private var model: Vector = _
  private var updater_strategy = "L2"
  private var standardScalerModel: StandardScalerModel = _
  private var dataMeanValue = Vectors.zeros(numFeatures)

  def setBreakPoint(breakpoint: Int): this.type = {
    require(breakpoint >= 1 && breakpoint < numFeatures - 1,
      s"breakpoint should be in [1, ${numFeatures - 1}), but is $breakpoint")
    this.breakpoint = breakpoint
    this
  }

  def setNumIter(numIterations: Int): this.type = {
    require(numIterations > 0,
      s"numIterations should be [0, inf], but is $numIterations")
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

  //TODO: sparse model initialization ?
  def generateInitialWeights(input: RDD[MyLabeledPoint]): Vector = {
    if (numFeatures <= 0) {
      val sample = input.take(1)(0)
      numFeatures = sample.features.size
      require(sample.num.length == 2, s"two-stage model class number should be 2, but ${sample.num.length}")
    }
    model = Vectors.dense(Array.fill(numFeatures)(Random.nextDouble() * init_std + init_mean))
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
    require(model.size == dataPoint.size, "model is not trained or initialized, please check!")
    predictPoint(dataPoint, model, dataMeanValue)
  }

  def predictPoint(dataPoint: Vector, model: Vector, dataMeanValue: Vector): Array[Double] = {
    require(model.size == dataPoint.size , "model's size does not equal features' size")
    //val dataPointCopy = dataPoint.copy
    axpy(-1.0, dataMeanValue, dataPoint)
    //we only use stage-1
    var sum = 0.0
    dataPoint.foreachActive((i, value) => {
      if (i < breakpoint) {
        sum += model(i) * value
      }
    })
    val prob = 1.0 / (1.0 + math.exp(-1.0 * sum))
    Array(1 - prob, prob)
  }

  def predict(data: RDD[Vector]): RDD[Array[Double]] = {
    val localWeights = model
    val brc = data.context.broadcast(localWeights)
    val meanBrc = data.context.broadcast(dataMeanValue)
    val result = data.mapPartitions(iter => { // broadcast only to the partition, also each executor
      val w = brc.value
      val d = meanBrc.value
      iter.map(v => predictPoint(v, w, d))
    })
    brc.destroy()
    meanBrc.destroy()
    result
  }

  override def toString: String = {
    model match {
      case DenseVector(value) =>
        s"${this.getClass.getName}, model weights: ${model.toArray.mkString("\t").trim}, breakpoint: $breakpoint"
      case SparseVector(size, indices, values) =>
        s"${this.getClass.getName}, model paras: size=$numFeatures, index=[${indices.mkString(",")}], value=[${values.mkString(",")}], breakpoint: $breakpoint"
      case _ =>
        throw new SparkException(s"unable to parse model ${model.getClass.getName}")
    }
  }

  def toDebugString: String = {
    model match {
      case DenseVector(value) =>
        s"LRModel1:w=[${model.toArray.slice(0, breakpoint).mkString(",").trim}]; LRModel2:w=[${model.toArray.slice(breakpoint, numFeatures).mkString(",").trim}]"
      case SparseVector(size, indices, values) =>
        val index1 = collection.mutable.ArrayBuilder.make[Int]()
        val index2 = collection.mutable.ArrayBuilder.make[Int]()
        val value1 = collection.mutable.ArrayBuilder.make[Double]()
        val value2 = collection.mutable.ArrayBuilder.make[Double]()
        val valueLen = indices.length
        var index = 0
        while (index < valueLen) {
          if (indices(index) < breakpoint) {
            index1 += indices(index)
            value1 += values(index)
          } else {
            index2 += indices(index) - breakpoint
            value2 += values(index)
          }
          index += 1
        }
        s"LRModel1:size=$breakpoint,index=[${index1.result().mkString(",")},value=[${value1.result().mkString(",")}]];" +
        s"LRModel2:size=${numFeatures-breakpoint},index=[${index2.result().mkString(",")}],value=[${value2.result().mkString(",")}]"
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
    require(numFeatures == initialWeights.size)
    require(breakpoint >= 1 && breakpoint < numFeatures - 1, s"breakpoint is $breakpoint not match")
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
    val optimizer = new LBFGS(new TwoStageModelGradient, updater).setNumFeatures(numFeatures).setClassNum(2)
    optimizer.setConvergenceToVal(convergenceToVal)
      .setMaxNumIterations(numIterations)
      .setNumCorrections(numCorrections)
      .setRegParams(regParam)
      .setBreakpoint(breakpoint)
    val weights = optimizer.optimize(input, initialWeights)
    model = standardScalerModel.transformModel(weights)
    this
  }

  case class Data(weights: Option[Vector], dataMeanValue: Option[Vector], numFeatures: Int, breakpoint: Int)

  protected def formatVersion: String = "1.0"

  private def save(ss: SparkSession, path: String, data: Data): Unit = {
    val modelClassName = this.getClass.getName
    val metadata = compact(render(
      ("class" -> modelClassName) ~ ("version" -> formatVersion) ~ ("numFeatures" -> numFeatures)
        ~ ("breakpoint" -> breakpoint)
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

  def save(ss: SparkSession, path: String = "TwoStageNobiasClickModelWithLBFGS"): Unit = {
    save(ss, path, Data(Option(model), Option(dataMeanValue), numFeatures, breakpoint))
  }

  def load(ss: SparkSession, path: String, modelClass: String): this.type = {
    val validPath = new Path(path, "data").toUri.toString.trim
    val dataRDD = ss.read.parquet(validPath)
    val dataArray = dataRDD.select("weights").take(1)
    val dataMeanValueArray = dataRDD.select("dataMeanValue").take(1)
    val numFeaturesArray = dataRDD.select("numFeatures").take(1)
    val breakpointsArray = dataRDD.select("breakpoint").take(1)
    assert(dataArray.length == 1 && numFeaturesArray.length == 1 && breakpointsArray.length == 1,
      s"unable to load $modelClass data from: $validPath")
    val data = dataArray(0)
    val dataMeanValues = dataMeanValueArray(0)
    val numFeatures = numFeaturesArray(0)
    val breakpoints = breakpointsArray(0)
    val weights = data.getAs[Vector](0)
    val dataMeanValue = dataMeanValues.getAs[Vector](0)
    val numFeature = numFeatures.getAs[Int](0)
    val breakpoint = breakpoints.getAs[Int](0)
    if (weights == null || weights.size == 0 || dataMeanValue == null || dataMeanValue.size == 0) {
      throw new SparkException(s"model loaded failed! weights:$weights;weightsSize:${weights.size};" +
        s"dataMeanValue:$dataMeanValue, dataMeanValueSize:${dataMeanValue.size}")
    }
    setInitialWeights(weights)
    setDataMeanValue(dataMeanValue)
    setNumFeatures(numFeature)
    setBreakPoint(breakpoint)
    this
  }
}
