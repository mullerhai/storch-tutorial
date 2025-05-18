package torch.basic

import java.io.*
import java.net.{HttpURLConnection, URL}
import java.nio.file.{Files, Paths}
import scala.util.Using

object Flickr8kSingleDownloader {
  private val DOWNLOAD_URL = "https://github.com/sorohere/flickr-dataset/releases/download/v0.1.0/flickr8k-dataset.zip"
  private val TARGET_DIR = "D:\\data\\Flickr8k"
  private val TARGET_FILE = s"$TARGET_DIR\\flickr8k-dataset.zip"
  private val BUFFER_SIZE = 819200 // 8KB 缓冲区大小

  def main(args: Array[String]): Unit = {
    try {
      // 创建目标目录
      Files.createDirectories(Paths.get(TARGET_DIR))

      val url = new URL(DOWNLOAD_URL)
      val conn = url.openConnection().asInstanceOf[HttpURLConnection]

      // 检查是否支持断点续传
      conn.setRequestMethod("HEAD")
      val fileSize = conn.getContentLengthLong()
      conn.disconnect()

      val outputFile = new File(TARGET_FILE)
      var startByte: Long = 0
      if (outputFile.exists()) {
        startByte = outputFile.length()
        if (startByte == fileSize) {
          println("文件已下载完成")
          return
        }
      }

      val newConn = url.openConnection().asInstanceOf[HttpURLConnection]
      newConn.setRequestMethod("GET")
      if (startByte > 0) {
        newConn.setRequestProperty("Range", s"bytes=$startByte-")
      }

      Using.resource(newConn.getInputStream()) { inputStream =>
        Using.resource(new RandomAccessFile(outputFile, "rw")) { output =>
          output.seek(startByte)
          val buffer = new Array[Byte](BUFFER_SIZE)
          var bytesRead: Int = inputStream.read(buffer)
          while (bytesRead != -1) {
            output.write(buffer, 0, bytesRead)
            println(s"正在读取... ${bytesRead}")
            bytesRead = inputStream.read(buffer)
          }
        }
      }

      println("文件下载成功，保存路径: " + TARGET_FILE)
    } catch {
      case e: Exception =>
        println(s"发生错误: ${e.getMessage}")
    }
  }
}