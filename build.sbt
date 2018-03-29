name := "MySparkProject"
version := "1.0"
scalaVersion := "2.11.8"
val sparkVersion = "2.1.1"
val xgboostVersion = "0.7"


libraryDependencies += "junit" % "junit" % "4.8" % "test"
libraryDependencies ++= Seq("org.slf4j" % "slf4j-api" % "1.7.5",
  "org.slf4j" % "slf4j-simple" % "1.7.5",
  "org.clapper" %% "grizzled-slf4j" % "1.0.2")
libraryDependencies += "org.apache.spark"%%"spark-core"   % sparkVersion
libraryDependencies += "org.apache.spark"%%"spark-streaming"   % sparkVersion
libraryDependencies += "org.apache.spark"%%"spark-mllib"   % sparkVersion
libraryDependencies += "org.apache.spark"%%"spark-streaming-flume-sink" % sparkVersion
libraryDependencies += "org.apache.spark"%%"spark-sql"   % sparkVersion
libraryDependencies += "org.apache.spark"%%"spark-graphx"   % sparkVersion
libraryDependencies += "org.apache.spark"%%"spark-hive"   % sparkVersion

resolvers += "MavenRepository" at "https://mvnrepository.com/"
