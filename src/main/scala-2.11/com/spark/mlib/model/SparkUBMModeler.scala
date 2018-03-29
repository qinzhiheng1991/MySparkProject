package com.spark.mlib.model

import com.spark.mlib.log.Logging
import org.apache.spark.rdd.RDD
/**
  * refer: http://www.bpiwowar.net/wp-content/papercite-data/pdf/dupret2008a-user-browsing.pdf
  */

object SparkUBMModeler extends Logging {

  case class Conf(max_queries: Long,
                  max_url_per_query: Int,
                  browsingModes: Int,
                  maxIter: Int,
                  minDelta: Double,
                  numPartitions: Int)

  /**
    * @param train_data: RDD[((q,u,r,d),Seq[(isClick, count)])]
    */
  def train(train_data: RDD[((Long, Long, Int, Int), Seq[(Boolean, Int)])],
            conf: Conf) = {
    logInfo("train data: " + train_data.count())
    val Conf(max_queries, max_url_per_query, browsingModes, maxIter, minDelta, numPartitions) = conf
    val sc = train_data.sparkContext
    //initialize gamma
    val gamma = train_data.map{
      case ((q,u,r,d), _) => (r,d)
    }.distinct().collect().map { case (r, d) =>
      ((r, d), Array.fill(browsingModes)(0.5))
    }.toMap

    var gamma_br = sc.broadcast(gamma)
    //initialize alpha
    var alpha = train_data.map { case ((q, u, r, d), _) =>
      (q, u)
    }.distinct(numPartitions).map {
      (_, 0.5)
    }.cache()
    //initialize mu
    var mu = train_data.map {
      _._1._1
    }.distinct().map { q =>
      (q, Array.tabulate(browsingModes) { _ => 1.0 / browsingModes})
    }.cache()

    var delta = Double.PositiveInfinity
    var joined_data = train_data.map { case ((q, u, r, d), cnts) =>
      ((q, u), (r, d, cnts))
    }.join(alpha).map { case ((q, u), ((r, d, cnts), alpha_qu)) =>
      (q, (u, r, d, cnts, alpha_qu))
    }.join(mu)

    for (i <- 0 until maxIter if delta > minDelta) {
      val updates = joined_data.flatMap { case (q, ((u, r, d, cnts, alpha_qu), mu_q)) =>
        val gamma_rd = gamma_br.value(r,d)

        val mu_gamma = mu_q zip gamma_rd map { case (x, y) => x * y}
        val dot_prod_mu_gamma = mu_gamma.sum
        val Q_m_a1_e1_c1 = mu_gamma.map {
          _ / dot_prod_mu_gamma
        }
        val Q_m_e1_c1 = Q_m_a1_e1_c1
        val Q_m_c1 = Q_m_a1_e1_c1
        val Q_a1_c1 = 1.0
        val Q_a1_c0 = alpha_qu * (1 - dot_prod_mu_gamma) / (1 - alpha_qu * dot_prod_mu_gamma)
        val Q_m_e1_c0 = mu_gamma.map {
          _ * (1 - alpha_qu) / (1 - alpha_qu * dot_prod_mu_gamma)
        }
        val Q_m_c0 = gamma_rd.map { gamma_rdm =>
          1 - alpha_qu * gamma_rdm
        }.zip(mu_q).map {
          case (x, y) => x * y / (1 - alpha_qu * dot_prod_mu_gamma)
        }

        val fractions = cnts.map { case (c, cnt) =>
          val alpha_fraction = if (c) {
            (Q_a1_c1 * cnt, cnt)
          } else {
            (Q_a1_c0 * cnt, cnt)
          }

          val gamma_fraction = if (c) {
            Q_m_e1_c1.map {_ * cnt}.zip(Q_m_c1.map {_ * cnt})
          } else {
            Q_m_e1_c0.map {_ * cnt}.zip(Q_m_c0.map {_ * cnt})
          }
          val mu_fraction = if (c) {
            Q_m_c1.map { q_m_c => (q_m_c * cnt, cnt)}
          } else {
            Q_m_c0.map { q_m_c => (q_m_c * cnt, cnt)}
          }
          (alpha_fraction, gamma_fraction, mu_fraction)
        }

        fractions.map{ case fs => ((q,u,r,d), fs)}
      }.cache()

      // update alpha
      val new_alpha = updates.map { case ((q, u, r, d), fractions) =>
        ((q, u), fractions._1)
      }.reduceByKey { case (lhs, rhs) =>
        (lhs._1 + rhs._1, lhs._2 + rhs._2)
      }.mapValues { case (num, den) =>
        num / den
      }.cache()

      val delta_alpha = alpha.join(new_alpha).values.map{
        case (x, y) => math.abs(x - y)
      }.max()

      // update mu
      val new_mu = updates.map { case ((q, u, r, d), fractions) =>
        (q, fractions._3)
      }.reduceByKey { case (x, y) =>
        x zip y map { case (lhs, rhs) =>
          (lhs._1 + rhs._1, lhs._2 + rhs._2)
        }
      }.mapValues {
        _.map { case (num, den) =>
          num / den
        }
      }.cache()

      val delta_mu = mu.join(new_mu).values.map{
        case (lhs, rhs) => lhs.zip(rhs).map{
          case (x, y) => math.abs(x - y)
        }.max
      }.max()

      delta = math.max(delta_alpha, delta_mu)

      // update gamma
      updates.map { case ((q, u, r, d), fractions) =>
        ((r, d), fractions._2)
      }.reduceByKey { case (x, y) =>
        x zip y map { case (lhs, rhs) =>
          (lhs._1 + rhs._1, lhs._2 + rhs._2)
        }
      }.mapValues {
        _.map { case (num, den) =>
          num / den
        }
      }.collect().foreach { case ((r, d), gamma_rd) =>
        gamma_rd.zipWithIndex.foreach {
          case (gamma_rdm, m) =>
            delta = math.max(delta, math.abs(gamma(r,d)(m) - gamma_rdm))
            gamma(r,d)(m) = gamma_rdm
        }
      }
      gamma_br = sc.broadcast(gamma)

      updates.unpersist()
      alpha.unpersist()
      mu.unpersist()
      joined_data.unpersist()

      alpha = new_alpha
      mu = new_mu

      joined_data = train_data.map { case ((q, u, r, d), cnts) =>
        ((q, u), (r, d, cnts))
      }.join(alpha).map { case ((q, u), ((r, d, cnts), alpha_qu)) =>
        (q, (u, r, d, cnts, alpha_qu))
      }.join(mu).cache()

      val perplexity = joined_data.flatMap{
        case (q, ((u, r, d, cnts, alpha_qu), mu_q)) =>
          val gamma_rd = gamma_br.value(r, d)

          cnts.map{ case (c, cnt) =>
            val p_c1 = alpha_qu * gamma_rd.zip(mu_q).map{ case (x, y) => x * y}.sum
            (if (c) - cnt * log2(p_c1) else - cnt * log2(1-p_c1), cnt)
          }
      }.reduce{
        (x, y) => (x._1 + y._1, x._2 + y._2)
      }

      logInfo(f"iteration $i: delta = $delta%.6f, " +
        f"perplexity = ${math.pow(2, perplexity._1 / perplexity._2)}%.6f")
    }

    val model = new UBMModel(max_queries, max_url_per_query, browsingModes)
    model.gamma = Some(gamma)
    model.alpha = Some(alpha)
    model.mu = Some(mu)

    model
  }

  def log2(x: Double): Double = math.log(x) / math.log(2)

  class UBMModel(val max_queries: Long,
                 val max_url_per_query: Int,
                 val browsingModes: Int = 1) {

    var gamma: Option[Map[(Int, Int), Array[Double]]] = None
    var alpha: Option[RDD[((Long, Long), Double)]] = None
    var mu: Option[RDD[(Long, Array[Double])]] = None
  }
}