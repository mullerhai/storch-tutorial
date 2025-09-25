package torch
package basic

import jdk.incubator.vector
import org.bytedeco.pytorch.ByteVector

import java.nio.file.{Files, Paths}
//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
@main
def main(): Unit =
  //TIP Press <shortcut actionId="ShowIntentionActions"/> with your caret at the highlighted text
  // to see how IntelliJ IDEA suggests fixing it.
  (1 to 5).map(println)
  val dataPath = Paths.get("D:\\data\\CIFAR101\\cifar\\")
  val bytes =Files.readAllBytes(dataPath)
  val byteVector = new ByteVector(bytes*)

//  println(s"bytes ${bytes.length}")
  val data = torch.pickleLoad(byteVector)
  println(data.size)
  for (i <- 1 to 5) do
    //TIP Press <shortcut actionId="Debug"/> to start debugging your code. We have set one <icon src="AllIcons.Debugger.Db_set_breakpoint"/> breakpoint
    // for you, but you can always add more by pressing <shortcut actionId="ToggleLineBreakpoint"/>.
    println(s"i = $i")

