package torch.basic

import java.io._
import java.util.zip.GZIPInputStream
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import scala.collection.mutable.ArrayBuffer

object CIFAR10Loader {
  // 定义 CIFAR - 10 数据集的类别标签
  val LABEL_NAMES = Array(
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
  )

  // 解压 tar.gz 文件
  def extractTarGz(filePath: String, outputDir: String): Unit = {
    val gis = new GZIPInputStream(new FileInputStream(filePath))
    val tarIn = new TarArchiveInputStream(gis)
    var entry = tarIn.getNextTarEntry

    while (entry != null) {
      if (entry.isDirectory) {
        new File(outputDir + entry.getName).mkdirs()
      } else {
        val fos = new FileOutputStream(outputDir + entry.getName)
        val buffer = new Array[Byte](4096)
        var bytesRead = tarIn.read(buffer)
        while (bytesRead != -1) {
          fos.write(buffer, 0, bytesRead)
          bytesRead = tarIn.read(buffer)
        }
        fos.close()
      }
      entry = tarIn.getNextTarEntry
    }
    tarIn.close()
  }

  // 读取单个数据文件
  def readDataFile(filePath: String): (Array[Int], Array[Array[Byte]]) = {
    val fis = new FileInputStream(filePath)
    val dis = new DataInputStream(fis)
    val numImages = 10000
    val labels = new Array[Int](numImages)
    val images = new Array[Array[Byte]](numImages)

    for (i <- 0 until numImages) {
      labels(i) = dis.readByte().toInt
      val image = new Array[Byte](3072)
      dis.readFully(image)
      images(i) = image
    }

    dis.close()
    fis.close()
    (labels, images)
  }

  // 主方法，用于测试
  def main(args: Array[String]): Unit = {
    val tarGzPath = "D:\\data\\CIFAR101\\cifar-10-python.tar.gz"
    val outputDir = "D:\\data\\CIFAR101\\"
    extractTarGz(tarGzPath, outputDir)
    val dataBatch1Path = outputDir + "cifar-10-batches-py\\data_batch_1"
    val (labels, images) = readDataFile(dataBatch1Path)
    println(s"Read ${labels.length} images from $dataBatch1Path")
    images.length
    for (i <- 0 until 5) {
      println(s"Image $i: Label = ${LABEL_NAMES(labels(i))}")
    }
  }
}