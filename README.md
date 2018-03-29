# Spark-Tools
This is a spark tool for training LR, FM and some click models. It supports FM Softmax, LR Softmax using SGD, Momentum,
Nesterov Momentum, Adagrad, RMSProp, Adam, FTRL, LBFGS and OWLQN. ClickModels such as no-bias two-stage LR, BBM, UBM
and DBN are also implemented. I wrote this simple tool because of the poor performance of spark inherent LR. The gradient
and label point is defined not reasonable. In our click train sets, LR-LBFGS is 30 times faster than the source code
without any result loss. But in online learning, may be same.


# Highlight
BBM click model:
https://www.researchgate.net/profile/Fan_Guo3/publication/221654382_BBM_Bayesian_browsing_model_from_petabyte-scale_data/links/55abc15f08aea3d086853ba1/BBM-Bayesian-browsing-model-from-petabyte-scale-data.pdf

UBM click model:
http://www.bpiwowar.net/wp-content/papercite-data/pdf/dupret2008a-user-browsing.pdf

DBN click model:
http://olivier.chapelle.cc/pub/DBN_www2009.pdf


# Examples
## Scala API
```scala
val input_path = ""
val save_path = ""
val iter = 1000
val classNum = 3
val conf = new SparkConf().setAppName("FWWithLBFGS")
conf.setExecutorEnv("spark.executor.memory", "1g")
val ss = SparkSession.builder().config(conf).getOrCreate()
val sc = ss.sparkContext
val train = sc.textFile(input_path).flatMap(f: String => MyLabelPoint).cache()
val cur = System.currentTimeMillis()
val FM = new FMWithLBFGS(iter, true, false, 1e-13, classNum).setUpdaterStrategy("l1").
  setInitStd(0.01).setEmbeddingDim(5).setRegParam(1e-7).setForceLBFGS(true).run(train)
val cur1 = System.currentTimeMillis()
println(FM.toDebugString)
println("time consumed: " + (cur1 - cur))
ss.stop()
```

# Requirements
Spark-tool is built against Spark 2.1.1.

# Build From Source
```scala
 sbt package
```

# Contact & Feedback
If you encounter bugs, feel free to submit an issue or pull request. Also you can mail to:
+ qinzhiheng (309796496@qq.com).

#TODO
1. spark inherent boost-trees algorithm is not well implemented.
2. new stop criterion.
3. decoupling the model and optimization.
