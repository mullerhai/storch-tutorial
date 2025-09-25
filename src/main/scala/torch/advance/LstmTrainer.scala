package torch.advance

import org.bytedeco.javacpp.{FloatPointer, PointerScope}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.global.torch as torchNative
import org.bytedeco.pytorch.*
import torch.Device.{CPU, CUDA}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.nn.functional as F
import torch.nn.modules.HasParams
import torch.numpy.TorchNumpy
import torch.numpy.TorchNumpy.loadNDArrayFromCSV
import torch.numpy.enums.DType.Float32 as NPFloat32
import torch.numpy.matrix.NDArray
import torch.optim.Adam
import torch.pandas.DataFrame
import torch.utils.data.dataloader.*
import torch.utils.data.datareader.ChunkDataReader
import torch.utils.data.dataset.*
import torch.utils.data.dataset.custom.{FashionMNIST, MNIST}
import torch.utils.data.sampler.RandomSampler
import torch.utils.data.*
import torch.utils.trainer.LstmNet
import torch.*

import java.net.URL
import java.nio.file.{Files, Path, Paths}
import java.util.zip.GZIPInputStream
import scala.collection.{mutable, Set as KeySet}
import scala.util.*
//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
@main
def main(): Unit =
  val num_epochs = 10
  val batchSize = 1000
  val timeout = 10.0f
  val device = if torch.cuda.isAvailable then CUDA else CPU
  println(s"Using device: $device")
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  //  val model  = GruNet().to(device)
  //  val model = LstmNet().to(device)
  //  val model = RnnNet().to(device)
  val modelPahth = "D:\\data\\git\\storch-tutorial\\lstm-netss.pt"
  val dataPath = Paths.get("D:\\data\\FashionMNIST")
  val mnistTrain = FashionMNIST(dataPath, train = true, download = true)
  val mnistEval = FashionMNIST(dataPath, train = false)
//  val mnistTrain = MNIST(dataPath, train = true, download = true)
//  val mnistEval = MNIST(dataPath, train = false)
  val trainFeatures = mnistTrain.features.to(device)
  val trainTargets = mnistTrain.targets.to(device)
  val evalFeatures = mnistEval.features.to(device)
  val evalTargets = mnistEval.targets.to(device)
  val r = Random(seed = 0)
//  val exampleSeq = mnistTrain.map(x => new Example(x._1.native, x._2.native))
//  val exampleVector = new ExampleVector(exampleSeq *)
  val model = new LstmNet[Float32]().to(device)
  val criterion = torch.nn.loss.CrossEntropyLoss()
  val optimizer = Adam(model.parameters, lr = 1e-3, amsgrad = true)
  val optimizerCopy = Adam(model.parameters, lr = 1e-3, amsgrad = true)
  val sampler = new RandomSampler(mnistTrain.length)
  val batchSampler = new RandomSampler(mnistTrain.length)
  val trainLoader = new DataLoader(dataset = mnistTrain, batch_size = batchSize, shuffle = true, sampler = sampler, batch_sampler = batchSampler, timeout = timeout)
  val evalLoader = new DataLoader(dataset = mnistEval, batch_size = batchSize, shuffle = false, sampler = sampler, batch_sampler = batchSampler, timeout = timeout)
  for (epoch <- 1 to num_epochs) {
    model.train()
    var totalLoss = 0l
    // val exampleIter = trainLoader.iterator
    var batchIndex = 0
//    println(s"try to get all dataset iter step length = ${trainLoader}")
//    for ((inputs, targets) <- trainLoader.iteratorSeq) {
    for ((inputs, targets) <- trainLoader) {
      println(s"hhh epoch = $epoch,begin batchIndex = $batchIndex, inputs = ${inputs.shape}, targets = ${targets.shape}")
      Using.resource(new PointerScope()) { _ =>
        // 将数据移到目标设备
        val inputsDevice = inputs.to(device)
        val targetsDevice = targets.to(device)

        println(s"hhh epoch = $epoch,scope batchIndex = $batchIndex, inputs = ${inputs.shape}, targets = ${targets.shape}")
        // 前向传播
        val outputs = model(inputsDevice.reshape(-1, 28, 28).to(torch.float32))
                // val loss = criterion(outputs, targetsDevice)
                // 反向传播和优化
        println(s"model_output epoch = $epoch,scope batchIndex = $batchIndex, outputs = ${outputs.shape}")
        optimizer.zeroGrad()
                // loss.backward()
        optimizer.step()

        // totalLoss = totalLoss + loss.item.asInstanceOf[Long]

        println(s"hhh epoch = $epoch,end  batchIndex = $batchIndex ..")
        batchIndex = batchIndex + 1
        // 定期评估
        // if (batchIndex % eval_interval == 0) {
        //   val (evalLoss, accuracy) = evaluate()
        //   println(
        //     s"Epoch: $epoch, Iteration: ${batchIndex}, | Training loss: ${loss.item}%.4f " +
        //       s"Train Loss: ${totalLoss / eval_interval}, Eval Loss: $evalLoss, Accuracy: $accuracy"
        //   )
        //   totalLoss = 0.0f
        //   model.train()
        // }
      }

    }
    println(s"complete epoch = $epoch, batchIndex = $batchIndex ..")

    // 每个epoch结束后评估
    // val (evalLoss, accuracy) = evaluate()
    // println(s"Epoch $epoch completed. Eval Loss: $evalLoss, Accuracy: $accuracy")
  }

//  //TIP Press <shortcut actionId="ShowIntentionActions"/> with your caret at the highlighted text
//  // to see how IntelliJ IDEA suggests fixing it.
//  (1 to 5).map(println)
//
//  for (i <- 1 to 5) do
//    //TIP Press <shortcut actionId="Debug"/> to start debugging your code. We have set one <icon src="AllIcons.Debugger.Db_set_breakpoint"/> breakpoint
//    // for you, but you can always add more by pressing <shortcut actionId="ToggleLineBreakpoint"/>.
//    println(s"i = $i")

