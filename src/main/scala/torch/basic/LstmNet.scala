//package torch.basic
//
////> using scala "3.3"
////> using repository "sonatype:snapshots"
////> using repository "sonatype-s01:snapshots"
////> using lib "dev.storch::vision:0.0-2fff591-SNAPSHOT"
//// replace with pytorch-platform-gpu if you have a CUDA capable GPU
////> using lib "org.bytedeco:pytorch-platform:2.1.2-1.5.10"
//// enable for CUDA support
//////> using lib "org.bytedeco:cuda-platform-redist:12.3-8.9-1.5.10"
//// enable for native Apple Silicon support
//// will not be needed with newer versions of pytorch-platform
//////> using lib "org.bytedeco:pytorch:2.1.2-1.5.10,classifier=macosx-arm64"
//
//import org.bytedeco.javacpp.{FloatPointer, PointerScope}
//import org.bytedeco.pytorch.{Example, InputArchive, OutputArchive, TensorExampleVectorIterator}
//import torch.Device.{CPU, CUDA}
//import torch.data.dataset.ChunkSharedBatchDataset
//import torch.nn.functional as F
//import torch.nn.modules.HasParams
//import torch.optim.Adam
//import torch.*
//import torch.tqdm.Tqdm.tqdm
//
//import java.nio.file.Paths
//import scala.collection.immutable.::
//import torch.internal.NativeConverters.{fromNative, toNative}
//
//import scala.util.{Random, Using}
//
//
///** Shows how to train a simple LstmNet on the MNIST dataset */
//object LstmNetApp extends App {
//  val device = if torch.cuda.isAvailable then CUDA else CPU
//  println(s"Using device: $device")
////  val model  = GruNet().to(device)
//  //  val model = LstmNet().to(device)
////  val model = RnnNet().to(device)
//  val modelPahth ="D:\\data\\git\\storch-tutorial\\lstm-net.pt"
//  val model = LstmNet().to(device)
////  val input = new InputArchive()
////  val mo = input.load_from(modelPahth)
////  var modelz = new org.bytedeco.pytorch.Module()
////  val model = modelz.load(input) //.asInstanceOf[LstmNet[Float32]]
//  //  val model  = TransformerClassifier(embedding_dim = 128 , num_heads= 6, num_layers=6, hidden_dim = 30, num_classes=10, dropout_rate=0.1).to(device)
//  // prepare data FashionMNIST
//  //  val dataPath = Paths.get("data/mnist")
//  //  val mnistTrain = MNIST(dataPath, train = true, download = true)
//  //  val mnistEval = MNIST(dataPath, train = false)
//  // "D:\\code\\data\\FashionMNIST"
//  //  val dataPath = Paths.get("data/FashionMNIST")
//  val dataPath = Paths.get("D:\\data\\FashionMNIST")
//  val mnistTrain = FashionMNIST(dataPath, train = true, download = true)
//  val mnistEval = FashionMNIST(dataPath, train = false)
//  println(s"model ${model.modules.toSeq.mkString(" \n")}")
//  println(s"model ${model.summarize}")
//  reader(exampleVector)
//  //  val ds = new ChunkSharedBatchDataset(new ChunkDataset(reader, new RandomSampler(exampleSeq.size), new RandomSampler(exampleSeq.size), new ChunkDatasetOptions(prefetch_count, batch_size))).map(new ExampleStack)
//  //  val ds  = new ChunkSharedTensorBatchDataset(new ChunkTensorDataset(reader,new RS(exampleTensorSeq.size),new ChunkDatasetOptions(prefetch_count, batch_size))).map(new TensorExampleStack)
//  val ds = new ChunkSharedBatchDataset(
//    new ChunkDataset(
//      reader,
//      new RandomSampler(exampleSeq.size),
//      new RandomSampler(exampleSeq.size),
//      new ChunkDatasetOptions(prefetch_count, batch_size)
//    )
//  ).map(new ExampleStack)
//  //  val ds = new TensorDataset(reader)
//  //  val ds = new StreamDataset(reader)
//  val opts = new DataLoaderOptions(32)
//  //  opts.enforce_ordering.put(true)
//  //  opts.drop_last.put(false)
//  val data_loader = new ChunkRandomDataLoader(ds, opts)
//
//  def dataLoader: Iterator[(torch.Tensor[Float32], torch.Tensor[Int64])] =
//    r.shuffle(mnistTrain).grouped(8).map { batch =>
//      val (features, targets) = batch.unzip
//      (torch.stack(features).to(device), torch.stack(targets).to(device))
//    }
//  //  opts.workers.put(5)
//  opts.batch_size.put(32)
//
//  def exampleVectorToExample(exVec: ExampleVector): Example = {
//    val example = new Example(exVec.get(0).data(), exVec.get(0).target())
//    example
//  }
//  val evalFeatures = mnistEval.features.to(device)
//  val evalTargets = mnistEval.targets.to(device)
//  val r = Random(seed = 0)
//  val exampleSeq = mnistTrain.map(x => new Example(x._1.native, x._2.native))
//
//  import org.bytedeco.pytorch.{ChunkDatasetOptions, Example, ExampleIterator, ExampleStack, ExampleVector, RandomSampler}
//  import torch.data.DataLoaderOptions
//  import torch.data.dataloader.*
//  import torch.data.datareader.ChunkDataReader
//  import torch.data.dataset.*
//  //  val ex1 = new Example(mnistTrain.features.native ,mnistTrain.targets.native)
//  val exampleVector = new ExampleVector(exampleSeq *)
//  //  val exampleTensorSeq = mnistTrain.map(x => new TensorExample(x._1.native))
//  //  val tensorExampleVector = new TensorExampleVector(exampleTensorSeq*)
//  //  val reader = new ChunkTensorDataReader()// new TensorExampleVectorReader()
//  //  reader(tensorExampleVector)
//  val reader = new ChunkDataReader()
//  //
//  //  val ds = new JavaDataset() {
//  //     val exampleVector = new ExampleVector(exampleSeq.toArray:_*)
//  //    override def get(index: Long): Example = exampleVector.get(index)
//  //
//  //    override def size = new SizeTOptional(exampleVector.size)
//  //
//  //  }
//  //  val ds = new JD(reader)//.map(new ExampleStack())
//  //val ds = new JSD() {
//  //  val exampleVector = reader.exampleVec
//  //
//  //  override def get_batch(size: Long): ExampleVector = exampleVector
//  //
//  //  override def size = new SizeTOptional(exampleVector.size)
//  //}
//  //val ds = new TD() {
//  //  val tex = reader.tensorExampleVec //new TensorExampleVector(new TensorExample(Tensor.create(10.0, 20.0, 50.0, 80.0, 100.0)), new TensorExample(Tensor.create(15.0, 30.0, 50.0, 80.0, 300.0)), new TensorExample(Tensor.create(20.0, 20.0, 50.0, 80.0, 100.0)), new TensorExample(Tensor.create(35.0, 30.0, 50.0, 80.0, 300.0)), new TensorExample(Tensor.create(40.0, 20.0, 50.0, 80.0, 100.0)), new TensorExample(Tensor.create(55.0, 30.0, 50.0, 80.0, 300.0)), new TensorExample(Tensor.create(60.0, 20.0, 50.0, 80.0, 100.0)), new TensorExample(Tensor.create(75.0, 30.0, 50.0, 80.0, 300.0)))
//  //
//  //  override def get(index: Long): TensorExample = {
//  //    tex.get(index)
//  //    //                    return super.get(index);
//  //  }
//  //
//  //  override def get_batch(indices: SizeTArrayRef): TensorExampleVector = tex //.get_batch(indices) // ds.get_batch(indices) // exampleVector
//  //
//  //  override def size = new SizeTOptional(tex.size)
//  //}
//  val batch_size = 32
//  val prefetch_count = 1
//  //  {
//  //    override def read_chunk(chunk_index: Long) = exampleVector //  new ExampleVector(new Example(Tensor.create(10.0, 20.0, 50.0, 80.0, 100.0), Tensor.create(200.0)), new Example(Tensor.create(15.0, 30.0, 50.0, 80.0, 300.0), Tensor.create(400.0)), new Example(Tensor.create(20.0, 20.0, 50.0, 80.0, 100.0), Tensor.create(500.0)), new Example(Tensor.create(35.0, 30.0, 50.0, 80.0, 300.0), Tensor.create(600.0)), new Example(Tensor.create(40.0, 20.0, 50.0, 80.0, 100.0), Tensor.create(700.0)), new Example(Tensor.create(55.0, 30.0, 50.0, 80.0, 300.0), Tensor.create(800.0)), new Example(Tensor.create(60.0, 20.0, 50.0, 80.0, 100.0), Tensor.create(900.0)), new Example(Tensor.create(75.0, 30.0, 50.0, 80.0, 300.0), Tensor.create(300.0)))
//  //
//  //    override def chunk_count:Long = 1
//  //
//  //    override def reset(): Unit = {
//  //    }
//  //  }
//  reader(exampleVector)
//  //  val ds = new ChunkSharedBatchDataset(new ChunkDataset(reader, new RandomSampler(exampleSeq.size), new RandomSampler(exampleSeq.size), new ChunkDatasetOptions(prefetch_count, batch_size))).map(new ExampleStack)
//  //  val ds  = new ChunkSharedTensorBatchDataset(new ChunkTensorDataset(reader,new RS(exampleTensorSeq.size),new ChunkDatasetOptions(prefetch_count, batch_size))).map(new TensorExampleStack)
//  val ds = new ChunkSharedBatchDataset(
//    new ChunkDataset(
//      reader,
//      new RandomSampler(exampleSeq.size),
//      new RandomSampler(exampleSeq.size),
//      new ChunkDatasetOptions(prefetch_count, batch_size)
//    )
//  ).map(new ExampleStack)
//  //  val ds = new TensorDataset(reader)
//  //  val ds = new StreamDataset(reader)
//  val opts = new DataLoaderOptions(32)
//  //  opts.enforce_ordering.put(true)
//  //  opts.drop_last.put(false)
//  val data_loader = new ChunkRandomDataLoader(ds, opts)
//
//  def dataLoader: Iterator[(torch.Tensor[Float32], torch.Tensor[Int64])] =
//    r.shuffle(mnistTrain).grouped(8).map { batch =>
//      val (features, targets) = batch.unzip
//      (torch.stack(features).to(device), torch.stack(targets).to(device))
//    }
//  //  opts.workers.put(5)
//  opts.batch_size.put(32)
//
//  def exampleVectorToExample(exVec: ExampleVector): Example = {
//    val example = new Example(exVec.get(0).data(), exVec.get(0).target())
//    example
//  }
//  //  val data_loader = new ChunkRandomTensorDataLoader(ds, opts)
//  //  val data_loader = new JavaDistributedSequentialTensorDataLoader(ds, new DSS(ds.size.get), opts)
//  //  val data_loader = new JavaDistributedRandomTensorDataLoader(ds, new DRS(ds.size.get), opts)
//  //  val data_loader = new JavaSequentialTensorDataLoader(ds, new SS(ds.size.get), opts)
//  //  val data_loader = new JavaStreamDataLoader(ds, new STS(ds.size.get), opts)
//  //  val data_loader = new JavaStreamDataLoader(ds, new STS(ds.size.get), opts)
//  //  val data_loader = new JavaStreamDataLoader(ds, new StreamSampler(ds.size.get), opts)
//  //  val data_loader = new RandomDataLoader(ds, new RS(ds.size.get), opts)
//  //  val data_loader = new SequentialDataLoader(ds, new SS(ds.size.get), opts)
//  //  val data_loader = new DistributedSequentialDataLoader(ds, new DistributedSequentialSampler(ds.size.get), opts)
//  //  val data_loader = new DistributedRandomDataLoader(ds, new DistributedRandomSampler(ds.size.get), opts)
//  //  val data_loader = new JavaRandomDataLoader(ds, new RandomSampler(ds.size.get), opts)
//  println(s"ds.size.get {ds.size.get} data_loader option ${data_loader.options.batch_size()}")
//  for (epoch <- tqdm(List(1, 2, 3, 4, 5, 6,7,8,9,10,11,12), "iterating",color =None,sleepSpeed = Some(50),colorRandom = false)) {
//    //    var it: ExampleVectorIterator = data_loader.begin
//    //    var it :TensorExampleVectorIterator = data_loader.begin
//    //    var it :TensorExampleIterator = data_loader.begin
//    var it: ExampleIterator = data_loader.begin
//    var batchIndex = 0
//    println("coming in for loop")
//    while (!it.equals(data_loader.end)) {
//      Using.resource(new PointerScope()) { p =>
//        val batch = it.access
//        optimizer.zeroGrad()
//        val trainDataTensor = fromNative(batch.data())
//        val prediction = model(fromNative(batch.data()).reshape(-1, 28, 28))
//        val loss = lossFn(prediction, fromNative(batch.target()))
//        loss.backward()
//        optimizer.step()
//        it = it.increment
//        batchIndex += 1
//        if batchIndex % 200 == 0 then
//          // run evaluation
//          val predictions = model(evalFeatures.reshape(-1, 28, 28))
//          val evalLoss = lossFn(predictions, evalTargets)
//          val featuresData = new Array[Float](1000)
//          val fp4 = new FloatPointer(predictions.native.data_ptr_float())
//          fp4.get(featuresData)
//          println(s"\n ffff size ${featuresData.size} shape ${
//            evalFeatures.shape
//              .mkString(", ")
//          }a data ${featuresData.mkString(" ")}")
//          println(s"predictions : ${predictions} \n")
//          println(s"loss grad_fn: ${evalLoss.grad_fn()}")
//          val accuracy =
//            (predictions.argmax(dim = 1).eq(evalTargets).sum / mnistEval.length).item
//          println(
//            f"Epoch: $epoch | Batch: $batchIndex%4d | Training loss: ${loss.item}%.4f | Eval loss: ${evalLoss.item}%.4f | Eval accuracy: $accuracy%.4f"
//          )
//        //        it = it.increment
//
//      }
//    }
//    optimizerCopy.add_parameters(model.namedParameters()) //
//    println(s"optimizerCopy ${optimizerCopy}")
//    println(s"optimizer ${optimizer}")
//    println(s"judge optimizer ${optimizer == optimizerCopy}")
//    println(s"model parameters dict ${model.namedParameters()}")
//  }
//
//  val archive = new OutputArchive
//  model.save(archive)
//  archive.save_to("lstm-net.pkl")
//  //  //a.index(indexArrayRefA).add_(torch.mul(x.index(indexArrayRefX), a_prev.index(indexArrayRefA_prev)))
//  class PositionalEncoding[D <: BFloat16 | Float32 : Default](d_model: Long, max_len: Long = 28 * 28)
//    extends HasParams[D] {
//
//    import torch.{---, Slice}
//
//    val arr = Seq(max_len, d_model)
//    println(s"First row: ${tensor(0)}")
//    // First row: tensor dtype=float32, shape=[4], device=CPU
//    // [1.0000, 1.0000, 1.0000, 1.0000]
//    println(s"First column: ${tensor(Slice(), 0)}")
//    // First column: tensor dtype=float32, shape=[4], device=CPU
//    // [1.0000, 1.0000, 1.0000, 1.0000]
//    println(s"Last column: ${tensor(---, -1)}")
//    val position = torch.arange(0, max_len, dtype = this.paramType).unsqueeze(1)
//    val div_term =
//      torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.Tensor(10000.0)) / d_model))
//    val sinPosition = torch.sin(position * div_term).to(dtype = this.paramType)
//    val cosPosition = torch.cos(position * div_term).to(dtype = this.paramType)
//    val indexSin = torch.Tensor(Seq(0L, 1L))
//    val indexCos = torch.Tensor(Seq(1L, 1L))
//    var tensor = torch.ones(Seq(4, 4))
//    var encoding = torch.zeros(size = arr.map(_.toInt), dtype = this.paramType)
//    encoding.index(::, 1.::(13)).add(sinPosition)
//    encoding.index(::, Seq[Long](2, 1, 13)).add(sinPosition)
//    encoding.index(::, 13).equal(sinPosition)
//    encoding.update(indices = Seq(2.::(21), 1.::(13)), values = sinPosition)
//    encoding.update(indices = Seq(---, 2.::(21), 1.::(13)), values = sinPosition)
//    encoding.update(indices = Seq(---, ::(21), 1.::(13)), values = sinPosition)
//    encoding.update(indices = Seq(---, 1.::, 1.::(13)), values = sinPosition)
//    encoding.update(indices = Seq(---, ::, 1.::(13)), values = sinPosition)
//    encoding = encoding.to(dtype = this.paramType)
//    encoding = torch.indexCopy(encoding, 0, indexSin, sinPosition)
//    encoding = torch.indexCopy(encoding, 0, indexCos, cosPosition)
//    encoding = encoding.unsqueeze(0)
//
//    // return x + self.encoding[: ,: x.size(1)].to(x.device)
//    def apply(x: torch.Tensor[D]): torch.Tensor[D] =
//      x.add(encoding).to(x.device)
//  }
//
//
//  //
//  //   run training
//  //  for (epoch <- 1 to 50) do
//  //    for (batch <- dataLoader.zipWithIndex) do
//  //      // make sure we deallocate intermediate tensors in time shape [32,1,28,28]
//  //      Using.resource(new PointerScope()) { p =>
//  //        val ((feature, target), batchIndex) = batch
//  //        optimizer.zeroGrad()
//  //        val prediction = model(feature.reshape(-1,28,28))
//  //        val loss = lossFn(prediction, target)
//  //        loss.backward()
//  //        optimizer.step()
//  //        if batchIndex % 200 == 0 then
//  //          // run evaluation
//  //          val predictions= model(evalFeatures.reshape(-1,28,28))
//  //          val evalLoss = lossFn(predictions, evalTargets)
//  //          val featuresData  = new Array[Float](1000)
//  //          val fp4 = new FloatPointer(predictions.native.data_ptr_float())
//  //          fp4.get(featuresData)
//  //          println(s"\n ffff size ${featuresData.size} shape ${evalFeatures.shape.mkString(", ")}a data ${featuresData.mkString(" " )}")
//  //          println(s"predictions : ${predictions} \n")
//  //          val accuracy =
//  //            (predictions.argmax(dim = 1).eq(evalTargets).sum / mnistEval.length).item
//  //          println(
//  //            f"Epoch: $epoch | Batch: $batchIndex%4d | Training loss: ${loss.item}%.4f | Eval loss: ${evalLoss.item}%.4f | Eval accuracy: $accuracy%.4f"
//  //          )
//  //      }
//
//  //}
//}