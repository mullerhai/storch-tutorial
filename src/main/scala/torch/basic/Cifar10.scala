///*
// * Copyright 2022 storch.dev
// *
// * Licensed under the Apache License, Version 2.0 (the "License");
// * you may not use this file except in compliance with the License.
// * You may obtain a copy of the License at
// *
// *     http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// */
//
//package torch.basic
//
//import org.bytedeco.pytorch
//import torch.Tensor.fromNative
//import torch.data.TensorDataset
//import torch.{CIFARBase, Float32, Int64}
//
//import java.io.{FileInputStream, ObjectInputStream}
//import java.net.URL
//import java.nio.file.{Files, Path, Paths, StandardCopyOption}
//import java.util.zip.GZIPInputStream
//import scala.util.{Failure, Success, Try, Using}
//
//trait CIFARBase(
//    val baseFolder: String,
//    val url: String,
//    val filename: String,
//    val tgzMd5: String,
//    val trainList: Seq[(String, String)],
//    val testList: Seq[(String, String)],
//    val meta: Map[String, String],
//    val root: Path,
//    val train: Boolean,
//    val download: Boolean
//) extends TensorDataset[Float32, Int64] {
//
//  private def downloadAndExtractArchive(url: URL, targetDir: Path): Unit = {
//    println(s"Downloading from $url  to dist dir ${targetDir}")
//    val tempFile = Files.createTempFile("cifar", ".tar.gz")
//    println(s"generate file ${tempFile}")
//    Using.resource(url.openStream()) { inputStream =>
//      println(s"read gzip inputstream ...")
//      val _ = Files.copy(GZIPInputStream(inputStream), targetDir)
//    }
////    Using.resource(url.openStream()) { inputStream =>
////      println(s"read normal inputstream ...")
////      Files.copy(inputStream, tempFile, StandardCopyOption.REPLACE_EXISTING)
////    }
//    // 这里需要添加解压 tar.gz 文件的逻辑
//    // 简化处理，假设已经有工具类处理解压
//    // extractTarGz(tempFile, targetDir)
//  }
//
//  private def checkIntegrity(filePath: Path, md5: String): Boolean = {
//    // 简化处理，假设已经有 MD5 校验工具类
//    true
//  }
//
//  if (download) {
//    Files.createDirectories(root)
//    val targetDir = root.resolve(baseFolder)
//    if (!Files.exists(targetDir)) {
//      Try(downloadAndExtractArchive(new URL(url), targetDir)) match {
//        case Failure(exception) => println(exception)
//        case Success(_) =>
//      }
//    }
//  }
//
//  val downloadedList = if (train) trainList else testList
//
//  var data: Array[Array[Byte]] = Array.empty
//  var target: Array[Int] = Array.empty
//
//  for ((fileName, checksum) <- downloadedList) {
//    val filePath = root.resolve(baseFolder).resolve(fileName)
//    if (!checkIntegrity(filePath, checksum)) {
//      throw new RuntimeException("Dataset not found or corrupted. You can use download=true to download it")
//    }
//    Using.resource(new ObjectInputStream(new FileInputStream(filePath.toFile))) { ois =>
//      val entry = ois.readObject().asInstanceOf[java.util.HashMap[String, Any]]
//      data ++= entry.get("data").asInstanceOf[Array[Array[Byte]]]
//      if (entry.containsKey("labels")) {
//        target ++= entry.get("labels").asInstanceOf[java.util.ArrayList[Int]].toArray.map(_.asInstanceOf[Int])
//      } else {
//        target ++= entry.get("fine_labels").asInstanceOf[java.util.ArrayList[Int]].toArray.map(_.asInstanceOf[Int])
//      }
//    }
//  }
//
//  // 这里需要处理数据转换，将 data 转换为合适的张量格式
//  // 简化处理，假设已经有转换工具类
//  // val tensorData = convertData(data)
//  private val mode =
//    if train then pytorch.MNIST.Mode.kTrain.intern().value
//    else pytorch.MNIST.Mode.kTest.intern().value
//  private val native = pytorch.MNIST(root.toString(), mode)
//
////  private val native = pytorch.MNIST(root.toString(), if (train) pytorch.MNIST.Mode.kTrain else pytorch.MNIST.Mode.kTest)
//  private val ds = TensorDataset(
//    fromNative[Float32](native.images().clone()),
//    fromNative[Int64](native.targets().clone())
//  )
//  export ds.{apply, length, features, targets}
//
//  private def loadMeta(): Unit = {
//    val path = root.resolve(baseFolder).resolve(meta("filename"))
//    if (!checkIntegrity(path, meta("md5"))) {
//      throw new RuntimeException("Dataset metadata file not found or corrupted. You can use download=true to download it")
//    }
//    Using.resource(new ObjectInputStream(new FileInputStream(path.toFile))) { ois =>
//      val data = ois.readObject().asInstanceOf[java.util.HashMap[String, Any]]
//      val classes = data.get(meta("key")).asInstanceOf[java.util.ArrayList[String]].toArray.map(_.asInstanceOf[String])
//      // 这里可以保存类名和索引的映射
//    }
//  }
//
//  loadMeta()
//
////  override def toString(): String = ds.toString()
//}
//
///** The [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.
//  *
//  * @param root
//  *   Root directory of dataset where directory `cifar-10-batches-py` exists or will be saved to if download is set to true.
//  * @param train
//  *   If true, creates dataset from training set, otherwise creates from test set.
//  * @param download
//  *   If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
//  */
//class CIFAR10(root: Path, train: Boolean = true, download: Boolean = false)
//    extends CIFARBase(
//      baseFolder = "cifar-10-batches-py",
//      url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
//      filename = "cifar-10-python.tar.gz",
//      tgzMd5 = "c58f30108f718f92721af3b95e74349a",
//      trainList = Seq(
//        ("data_batch_1", "c99cafc152244af753f735de768cd75f"),
//        ("data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"),
//        ("data_batch_3", "54ebc095f3ab1f0389bbae665268c751"),
//        ("data_batch_4", "634d18415352ddfa80567beed471001a"),
//        ("data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb")
//      ),
//      testList = Seq(
//        ("test_batch", "40351d587109b95175f43aff81a1287e")
//      ),
//      meta = Map(
//        "filename" -> "batches.meta",
//        "key" -> "label_names",
//        "md5" -> "5ff9c542aee3614f3951f8cda6e48888"
//      ),
//      root,
//      train,
//      download
//    ) {
////  override def features: Tensor[Float32] = ???
////
////  override def apply(i: Int): (Tensor[Float32], Tensor[Int64]) = ???
////
////  override def length: Int = ???
//}
//
///** The [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.
//  *
//  * @param root
//  *   Root directory of dataset where directory `cifar-100-python` exists or will be saved to if download is set to true.
//  * @param train
//  *   If true, creates dataset from training set, otherwise creates from test set.
//  * @param download
//  *   If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
//  */
//class CIFAR100(root: Path, train: Boolean = true, download: Boolean = false)
//    extends CIFARBase(
//      baseFolder = "cifar-100-python",
//      url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
//      filename = "cifar-100-python.tar.gz",
//      tgzMd5 = "eb9058c3a382ffc7106e4002c42a8d85",
//      trainList = Seq(
//        ("train", "16019d7e3df5f24257cddd939b257f8d")
//      ),
//      testList = Seq(
//        ("test", "f0ef6b0ae62326f3e7ffdfab6717acfc")
//      ),
//      meta = Map(
//        "filename" -> "meta",
//        "key" -> "fine_label_names",
//        "md5" -> "7973b15100ade9c7d40fb424638fde48"
//      ),
//      root,
//      train,
//      download
//    ) {
//  override val downloadedList: Seq[(String, String)] = ???
//
////  override def features: Tensor[Float32] = ???
////
////  override def apply(i: Int): (Tensor[Float32], Tensor[Int64]) = ???
////
////  override def length: Int = ???
//}