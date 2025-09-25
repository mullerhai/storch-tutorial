ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "3.6.3"
def javaMajorVersion: Int = System.getProperty("java.version").split("\\.").head.toInt
lazy val javaDeepLearningSettings = Seq(
  // JDK17+ https://github.com/apache/spark/blob/v3.5.1/pom.xml#L299-L317
  javaOptions ++= {
    if (javaMajorVersion >= 17)
      Seq(
        "-XX:+IgnoreUnrecognizedVMOptions",
        "--add-modules=jdk.incubator.vector",
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
        "--add-opens=java.base/java.io=ALL-UNNAMED",
        "--add-opens=java.base/java.net=ALL-UNNAMED",
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
        "--add-opens=java.base/java.util=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
        "--add-opens=java.base/jdk.internal.ref=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
        "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
        "-Djdk.reflect.useDirectMethodHandle=false",
        "-Dio.netty.tryReflectionSetAccessible=true"
      )
    else Nil
  }
)
lazy val root = (project in file("."))
  .settings(
    name := "storch-tutorial",
    javaOptions ++= Seq(
      "--add-modules=jdk.incubator.vector",
      "-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false"
    )
  )
libraryDependencies +=   "io.github.mullerhai" % "storch_core_3" % "0.6.1-1.15.2"
//libraryDependencies += "io.github.mullerhai" % "core_3" %   "0.2.6-1.15.1"
//libraryDependencies += "io.github.mullerhai" % "storch_vision_3" % "0.3.0-1.15.1"
//libraryDependencies += "io.github.mullerhai" % "storch-tqdm_3" % "0.0.2"
//libraryDependencies +=   "dev.storch" % "vision_3" % "0.2.3-1.15.1"
// https://mvnrepository.com/artifact/org.bytedeco/openblas-platform
libraryDependencies += "org.bytedeco" % "openblas-platform" % "0.3.28-1.5.11"
// https://mvnrepository.com/artifact/org.bytedeco/mkl-platform
libraryDependencies += "org.bytedeco" % "mkl-platform" % "2025.0-1.5.11"
libraryDependencies += "org.apache.commons" % "commons-compress" % "1.21"