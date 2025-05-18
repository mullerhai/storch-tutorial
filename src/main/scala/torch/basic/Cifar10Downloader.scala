package torch.basic

import java.io.InputStream
import java.net.URL
import java.nio.file.{Files, Path, Paths, StandardCopyOption}
import scala.util.{Try, Using}

object Cifar10Downloader {
  def main(args: Array[String]): Unit = {
    val downloadUrl = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    //    val targetDir = Paths.get("/opt/cifar10")
    val targetDir = Paths.get("D:\\data\\CIFAR10")
    val targetFile = targetDir.resolve("cifar-10-python.tar.gz")

    try {
      // 创建目标目录
      Files.createDirectories(targetDir)

      // 下载文件
      val result = Try {
        Using.resource(new URL(downloadUrl).openStream()) { inputStream =>
          Files.copy(inputStream, targetFile, StandardCopyOption.REPLACE_EXISTING)
        }
      }

      result match {
        case scala.util.Success(_) =>
          println(s"文件下载成功，保存路径: $targetFile")
        case scala.util.Failure(exception) =>
          println(s"文件下载失败: ${exception.getMessage}")
      }
    } catch {
      case e: Exception =>
        println(s"发生错误: ${e.getMessage}")
    }
  }
}