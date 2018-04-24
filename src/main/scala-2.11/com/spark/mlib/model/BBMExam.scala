package com.spark.mlib.model

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ArrayBuffer

/**
  * ClickModel attempts to solve bias problem such as position bias.
  * Bias affects mostly the examine probability of doc.
  * BBM(Bayesian Browsing Model) make two assumptions:
  * 1. Every(q, d) relevance is iid of Uniform(0, 1). (prior distribution)
  * 2. Exam probability is influenced by position and distance
  * refer: https://www.researchgate.net/profile/Fan_Guo3/publication/221654382_BBM_Bayesian_browsing_model_from_petabyte-scale_data/links/55abc15f08aea3d086853ba1/BBM-Bayesian-browsing-model-from-petabyte-scale-data.pdf
  *
  * And also, BBM is a generalization of COEC
  *
  * We generalize the assumption 1 to any distribution.
  * And the maximum likelihood in the paper, can be write as the expectation of relevance, and we use CTR to approximate it.
  * Created by qinzhiheng on 2018/3/24.
  */

object BBMExam {
  def main(args: Array[String]) {
    if (args.length < 2) {
      sys.error("error: wrong parameter number!")
      sys.exit(1)
    }
    val inputPath = args(0).trim
    val savepath = args(1).trim
    val conf = new SparkConf().setAppName("BBM Exam Estimate")
    val ss = SparkSession.builder().config(conf).getOrCreate()
    val sc = ss.sparkContext
    val hadoopConfig = sc.hadoopConfiguration
    val fs = FileSystem.get(hadoopConfig)
    if (fs.exists(new Path(savepath))) {
      sys.error("error: output path has been existed!")
      sys.exit(2)
    }
    val rdd = sc.textFile(inputPath).flatMap(line=>transform(line))
    rdd.reduceByKey(_+_).map(line=> {
      val segments = line._1.split("\t")
      val key = segments(0) + "\t" + segments(1)
      val value = segments(2) + "\t" + line._2
      (key, value)}
    ).repartition(1).saveAsTextFile(savepath)
  }

  def transform(line: String): ArrayBuffer[(String, Double)] = {
    val arrBuf = ArrayBuffer[(String, Double)]()
    val segs = line.split("\t")
    if (segs.length >= 42) {
      var preced_click_pos = 0
      for (i <- 2.until(42, 4)) {
        val pos = (i - 2)/4 + 1
        val distance = pos - preced_click_pos
        if (segs(i + 1).trim.equals("1")) {
          if (preced_click_pos > 0) {
            arrBuf += Tuple2(preced_click_pos + "\t" + distance + "\t" + 1, 1)
          }
          preced_click_pos = pos
        }
        else {
          if (preced_click_pos > 0) {
            arrBuf += Tuple2(preced_click_pos + "\t" + distance + "\t" + 0, 1)
          }
        }
      }
    }
    arrBuf
  }
}

