package torch.basic

import java.io.*
import java.net.{HttpURLConnection, URL}
import java.nio.file.{Files, Paths}
import java.util.concurrent.{Executors, Future}
import scala.collection.mutable.ListBuffer

object Cifar10DownloaderMu {
  private val THREAD_COUNT = 8
  private val BUFFER_SIZE = 8192 // 8KB 缓冲区大小
  private val DOWNLOAD_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
  private val TARGET_DIR = "D:\\data\\CIFAR101"
  private val TARGET_FILE = s"$TARGET_DIR/cifar-10-python.tar.gz"
  private val TEMP_DIR = s"$TARGET_DIR/temp"

  def main(args: Array[String]): Unit = {
    try {
      // 创建目标目录和临时目录
      Files.createDirectories(Paths.get(TARGET_DIR))
      Files.createDirectories(Paths.get(TEMP_DIR))

      val url = new URL(DOWNLOAD_URL)
      val conn = url.openConnection().asInstanceOf[HttpURLConnection]
      conn.setRequestMethod("HEAD")
      val fileSize = conn.getContentLengthLong()
      conn.disconnect()

      if (fileSize == -1) {
        println("无法获取文件大小")
        return
      }

      val threadPool = Executors.newFixedThreadPool(THREAD_COUNT)
      val futures = new ListBuffer[Future[_]]()

      val blockSize = fileSize / THREAD_COUNT
      for (i <- 0 until THREAD_COUNT) {
        val start = i * blockSize
        val end = if (i == THREAD_COUNT - 1) fileSize - 1 else start + blockSize - 1
        futures += threadPool.submit(new DownloadTask(url, start, end, i))
      }

      // 等待所有线程完成
      futures.foreach(_.get())
      threadPool.shutdown()

      // 合并临时文件
      mergeTempFiles()

      println("文件下载成功，保存路径: " + TARGET_FILE)
    } catch {
      case e: Exception =>
        println(s"发生错误: ${e.getMessage}")
    }
  }

  private def mergeTempFiles(): Unit = {
    val output = new RandomAccessFile(TARGET_FILE, "rw")
    try {
      for (i <- 0 until THREAD_COUNT) {
        val tempFile = new File(s"$TEMP_DIR/part$i")
        val input = new FileInputStream(tempFile)
        try {
          val buffer = new Array[Byte](BUFFER_SIZE)
          var bytesRead: Int = input.read(buffer)
          while (bytesRead != -1) {
            output.write(buffer, 0, bytesRead)
            bytesRead = input.read(buffer)
          }
        } finally {
          input.close()
          tempFile.delete()
        }
      }
    } finally {
      output.close()
    }
    // 删除临时目录
    new File(TEMP_DIR).delete()
  }

  private class DownloadTask(url: URL, start: Long, end: Long, threadId: Int) extends Runnable {
    override def run(): Unit = {
      val tempFile = new File(s"$TEMP_DIR/part$threadId")
      var input: InputStream = null
      var output: RandomAccessFile = null
      try {
        val conn = url.openConnection().asInstanceOf[HttpURLConnection]
        conn.setRequestMethod("GET")
        conn.setRequestProperty("Range", s"bytes=$start-$end")

        input = conn.getInputStream()
        output = new RandomAccessFile(tempFile, "rw")
        output.seek(0)

        val buffer = new Array[Byte](BUFFER_SIZE)
        var bytesRead: Int = input.read(buffer)
        while (bytesRead != -1) {
          output.write(buffer, 0, bytesRead)
          bytesRead = input.read(buffer)
        }
      } catch {
        case e: Exception =>
          println(s"线程 $threadId 下载出错: ${e.getMessage}")
      } finally {
        if (input != null) input.close()
        if (output != null) output.close()
      }
    }
  }
}