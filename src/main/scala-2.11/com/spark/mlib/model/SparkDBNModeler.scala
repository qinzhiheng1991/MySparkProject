package com.spark.mlib.model

import com.spark.mlib.log.Logging
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
/**
  * refer: http://olivier.chapelle.cc/pub/DBN_www2009.pdf
  *
  * A Dynamic Bayesian Network Click Model for Web Search Ranking
  *
  * Created by qinzhiheng on 2017/7/16.
  */

object SparkDBNModeler extends Logging {

  case class Conf(max_queries: Long,
                  max_url_per_query: Int,
                  withIntent: Boolean,
                  gamma: Array[Double], //hyper-parameters, in dbn probabilistic graphical model, P(Ei+1 = 1|Si = 0) = gamma
                  maxIter: Int,
                  minDelta: Double)

  /**
    * @param train_data: RDD[((q),Seq[(u, isClick, pos)])]
    */
  def train(train_data: RDD[(Long, Array[(Long, Boolean, Int)])],
            conf: Conf) = {
    logInfo("train data: " + train_data.count())
    val Conf(max_queries, max_url_per_query, withIntent, gammas, maxIter, minDelta) = conf
    val sc = train_data.sparkContext
    val intents = if (withIntent) 1 else 2
    //initialize gamma
    val element = train_data.flatMap(ele => {
      val buf = ArrayBuffer[(Long, Long)]()
      val q = ele._1
      for (s <- ele._2) {
        buf += Tuple2(q, s._1)
      }
      buf
    }).distinct().collect().map(ele => {
      ((ele._1, ele._2), 0.5)
    }).toMap

    // initialize a_u and s_u
    val new_a_u_array = collection.mutable.ArrayBuilder.make[Map[(Long, Long), Double]]()
    val new_s_u_array = collection.mutable.ArrayBuilder.make[Map[(Long, Long), Double]]()
    for (i <- 0 until intents) {
      new_a_u_array += element
      new_s_u_array += element
    }
    //val a_u = Array.tabulate[Map[(Long, Long), Double]](intents)(i => element)
    //val s_u = Array.tabulate[Map[(Long, Long), Double]](intents)(i => element)
    val a_u = new_a_u_array.result()
    val s_u = new_s_u_array.result()
    // broadcast it

    var a_u_br = sc.broadcast(a_u)
    var s_u_br = sc.broadcast(s_u)
    val gammas_br = sc.broadcast(gammas)
    var delta = Double.PositiveInfinity
    for (i <- 0 until maxIter if delta > minDelta) {
      // (query, url),((a_u_0, a_u_1), (s_u_0, s_u_1))
      val newParas = train_data.flatMap(session => {
        val results = ArrayBuffer[((Long, Long),((Double, Double), (Double, Double)))]()
        val a_u_values = a_u_br.value
        val s_u_values = s_u_br.value
        val gamma_values = gammas_br.value
        val query = session._1
        val clicks = Array.tabulate[Int](session._2.length)(i => 0)
        val paras = Array.tabulate[Double](intents, 2, session._2.length)((i, j, k) => 0) // 0 indicate au, 1 indicate su
        for ((url, isClick, pos) <- session._2) {
          clicks(pos) = if (isClick) 1 else 0
          for (intent <- 0 until intents) {
            paras(intent)(0)(pos) = a_u_values(intent)((query, url))
            paras(intent)(1)(pos) = s_u_values(intent)((query, url))
          }
        }
        val sessionEstimates = collection.mutable.ArrayBuilder.make[Array[Array[Double]]]()
        val observers = collection.mutable.ArrayBuilder.make[Double]()
        for (i <- 0 until intents) {
          val sessionEstimate = getSessionEstimates(paras, clicks, gamma_values, i)
          sessionEstimates += sessionEstimate._1
          observers += sessionEstimate._2
        }
        val sessionEstimatesResult = sessionEstimates.result()
        val observersResult = observers.result()
        var a_u_0 = 0.0
        var a_u_1 = 0.0
        var s_u_0 = 0.0
        var s_u_1 = 0.0
        for ((url, isClick, pos) <- session._2) {
          a_u_0 = sessionEstimatesResult(0)(0)(pos) * observersResult(0) / (observersResult(0) + observersResult(1))
          a_u_1 = sessionEstimatesResult(1)(0)(pos) * observersResult(1) / (observersResult(0) + observersResult(1))
          if (isClick) {
            s_u_0 = sessionEstimatesResult(0)(1)(pos) * observersResult(0) / (observersResult(0) + observersResult(1))
            s_u_1 = sessionEstimatesResult(1)(1)(pos) * observersResult(1) / (observersResult(0) + observersResult(1))
          }
          results += Tuple2((query, url),((a_u_0, a_u_1), (s_u_0, s_u_1)))
          a_u_0 = 0.0//set zero
          a_u_1 = 0.0
          s_u_0 = 0.0
          s_u_1 = 0.0
        }
        results
      }).reduceByKey((value1, value2) => {
        ((value1._1._1 + value2._1._1, value1._1._2 + value2._1._2),
          (value1._2._1 + value2._2._1, value1._2._2 + value2._2._2))
      }).collectAsMap()

      a_u_br.destroy()
      s_u_br.destroy()

      delta = 0.0
      var count = 0
      var q_functional = 0.0
      var log_loss = 0.0
      val a_u_map = collection.mutable.Map[(Long, Long), Double]()
      val s_u_map = collection.mutable.Map[(Long, Long), Double]()
      new_a_u_array.clear()
      new_s_u_array.clear()
      for (i <- 0 until intents) {
        newParas.foreach(item => {
          val query = item._1._1
          val url = item._1._2
          val a_u_0 = item._2._1._1
          val a_u_1 = item._2._1._2
          val s_u_0 = item._2._2._1
          val s_u_1 = item._2._2._2
          val new_a_u = if ((a_u_1 + a_u_0) > 0.0) a_u_1/(a_u_0 + a_u_1) else 0.0
          val new_s_u = if ((s_u_1 + s_u_0) > 0.0) s_u_1/(s_u_0 + s_u_1) else 0.0
          delta += math.pow(a_u(i)((query, url)) - new_a_u, 2)
          q_functional += -1.0 * a_u_1 * math.log(new_a_u) - a_u_0 * math.log(1 - new_a_u)
          log_loss += -1.0 * new_a_u * math.log(a_u(i)((query, url))) - (1 - new_a_u) * math.log(1 - a_u(i)((query, url)))
          count += 1
          if (new_s_u > 0 && new_s_u < 1) {
            delta += math.pow(s_u(i)((query, url)) - new_s_u, 2)
            q_functional += -1.0 * s_u_1 * math.log(new_s_u) - s_u_0 * math.log(1 - new_s_u)
            log_loss += -1.0 * new_s_u * math.log(s_u(i)((query, url))) - (1 - new_s_u) * math.log(1 - s_u(i)((query, url)))
            count += 1
          }
          a_u_map((query, url)) = new_a_u
          s_u_map((query, url)) = new_s_u
        })
        new_a_u_array += a_u_map.toMap
        new_s_u_array += s_u_map.toMap
        a_u_map.clear()
        s_u_map.clear()
      }

      val model = new DBNModel(max_queries, max_url_per_query, withIntent)
      model.new_a_u_array = Option(new_a_u_array.result())
      model.new_s_u_array = Option(new_s_u_array.result())

      delta /= count
      q_functional /= count
      log_loss /= count
      logInfo(s"$i iterations, rms:$delta, q_functional: $q_functional, log_loss: $log_loss")
      a_u_br = sc.broadcast(new_a_u_array.result())
      s_u_br = sc.broadcast(new_s_u_array.result())
    }

    // compute posterior probability
    def getSessionEstimates(paras: Array[Array[Array[Double]]],
                            clicks: Array[Int],
                            gammas:Array[Double],
                            intent: Int) = {
      val N = clicks.length
      val gamma = gammas(intent)
      val sessionEstimate = Array.tabulate[Double](2, N)((i, j) => 0.0)
      val (alpha, beta) = getForwardBackwardEstimates(paras, clicks, gammas, intent)
      val varphi = Array.tabulate[Double](N, 2)((i, j) => {
        alpha(i)(j) * beta(i)(j)/(alpha(i)(0) * beta(i)(0) + alpha(i)(1) * beta(i)(1))
      })
      val observe = alpha(0)(0) * beta(0)(0) + alpha(0)(1) * beta(0)(1)
      var index = 0
      while (index < N) {
        val a_u = paras(intent)(0)(index)
        val s_u = paras(intent)(1)(index)
        if (clicks(index) == 0) {
          sessionEstimate(0)(index) = a_u * varphi(index)(0)
          sessionEstimate(1)(index) = 0.0
        } else {
          sessionEstimate(0)(index) = 1.0
          sessionEstimate(1)(index) = varphi(index + 1)(0) * s_u / (s_u + (1 - gamma) * (1 - s_u))
        }
        index += 1
      }
      (sessionEstimate, observe)
    }

    def getForwardBackwardEstimates(paras: Array[Array[Array[Double]]],
                                    clicks: Array[Int],
                                    gammas: Array[Double],
                                    intent: Int) = {
      val N = clicks.length
      val alpha = Array.tabulate[Double](N + 1, 2)((i, j) => 0.0)
      val beta = Array.tabulate[Double](N + 1, 2)((i, j) => 0.0)
      // alpha(i) = P(C1,...,Ci-1,Ei=e), beta = P(Ci,...,C10|Ei)
      // alpha(i + 1) = sum{e'=0,1}(alpha(i) * P(Ei+1=e',Ci|Ei = e))
      alpha(0)(1) = 1.0  // boundary conditions, P(E0 = 1) = 1, P(E0 = 0) = 0, examine exists
      alpha(0)(0) = 0.0
      beta(N)(0) = 1.0 // boundary conditions, P(|E10 = 1) = 1, P(|E10 = 0) = 1
      beta(N)(1) = 1.0
      // transition probability P(Ei+1=e',Ci|Ei = e)
      val updateMatrix = Array.tabulate[Double](N, 2, 2)((i, j, k) => 0.0)
      val gamma = gammas(intent)
      var index = 0
      var click = 0
      //compute posterior probability
      while (index < N) {
        click = clicks(index)
        val a_u = paras(intent)(0)(index)
        val s_u = paras(intent)(1)(index)
        if (click == 0) {
          updateMatrix(index)(0)(0) = 1
          updateMatrix(index)(0)(1) = (1 - gamma) * (1 - a_u)
          updateMatrix(index)(1)(0) = 0
          updateMatrix(index)(1)(1) = gamma * (1 - a_u)
        } else {
          updateMatrix(index)(0)(0) = 0
          updateMatrix(index)(0)(1) = (s_u + (1 - gamma) * (1 - s_u)) * a_u
          updateMatrix(index)(1)(0) = 0
          updateMatrix(index)(1)(1) = gamma * (1 - s_u) * a_u
        }
        index += 1
      }
      index = 0
      while (index < N) {
        for (i <- 0 until 2) {
          alpha(index + 1)(i) = alpha(index)(0) * updateMatrix(index)(i)(0) + alpha(index)(1) * updateMatrix(index)(i)(1)
          beta(N-index-1)(i) = beta(N-index)(0) * updateMatrix(N-index-1)(i)(0) + beta(N-index)(1) * updateMatrix(N-index-1)(i)(1)
        }
      }
      (alpha, beta)
    }
  }

  class DBNModel(val max_queries: Long,
                 val max_url_per_query: Int,
                 val withIntention: Boolean) {
    var new_a_u_array: Option[Array[Map[(Long, Long), Double]]] = None
    var new_s_u_array: Option[Array[Map[(Long, Long), Double]]] = None
  }
}