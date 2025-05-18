package torch
package advance

import torch.FloatNN
import torch.nn.modules.HasParams
import org.bytedeco.javacpp.{FloatPointer, PointerScope}
import org.bytedeco.pytorch.{OutputArchive, TensorExampleVectorIterator}
import torch.Device.{CPU, CUDA}
import torch.data.dataset.ChunkSharedBatchDataset
import torch.nn.functional as F
import torch.nn.modules.HasParams
import torch.optim.Adam
import torch.*
import torchvision.datasets.FashionMNIST

import java.nio.file.Paths
import scala.util.{Random, Using}
import torch.data.DataLoaderOptions
import torch.data.dataloader.*
import torch.data.datareader.{ChunkDataReader, ChunkTensorDataReader, ExampleVectorReader, TensorExampleVectorReader}
import torch.data.dataset.*
import torch.data.dataset.java.{StatefulDataset, StatefulTensorDataset, StreamDataset, StreamTensorDataset, TensorDataset, JavaDataset as JD}
import torch.data.sampler.{DistributedRandomSampler, DistributedSequentialSampler, StreamSampler, RandomSampler as RS, SequentialSampler as SS}
import torch.internal.NativeConverters.fromNative

class ConvNet[D <: Float16 | Float32 | Float64  : Default](input_size: Int, hidden_size: Int, num_layers: Int, num_classes: Int = 10 ) extends HasParams[D]{

  val sequentialFirst = nn.Sequential(
      nn.Conv2d(1, 16, kernel_size = 5, stride = 1, padding = 2),
      nn.BatchNorm2d(16),
      nn.ReLU(true),
      nn.MaxPool2d(kernel_size = 2, stride = 2))
  val demoCheck = nn.Sequential(
    nn.Flatten(),
    nn.Unflatten(),
    nn.Dropout(),
    nn.Dropout3d(),
    nn.LogSigmoid(),
    nn.CosineSimilarity(),
    nn.AlphaDropout(),
    nn.Identity(),
    nn.Transformer(),
    nn.TransformerDecoder(nn.TransformerDecoderLayer(3,4),34),
    nn.TransformerDecoderLayer(3,4),
    nn.TransformerEncoder(nn.TransformerEncoderLayer(4,4),3),
    nn.TransformerEncoderLayer(4,4),
    nn.MultiheadAttention(1,1),
    nn.Fold(1,1),
    nn.Unfold(1,1),
    nn.GroupNorm(1,2),
    nn.LayerNorm(Seq(2),1),
    nn.LocalResponseNorm(1),
    nn.RMSNorm(Seq(9)),
    nn.Conv3d(1,1,1),
    nn.ConvTranspose2d(1,1,1),
    nn.Embedding(1,1),
    nn.EmbeddingBag(1,1),
    nn.InstanceNorm2d(1),
    nn.SELU(1,true),
    nn.CELU(1,0.1f, true),
    nn.Bilinear(1,1,1),
    nn.Hardtanh(1,1f,1f),
    nn.GRU(1,1,1),
    nn.RNN(1,1,1),
    nn.LSTM(1,1,1),
    nn.ConstantPad2d(1,2f),
    nn.ReflectionPad2d(1),
    nn.LPPool2d(2f,1,1),
    nn.AdaptiveAvgPool2d(1),
    nn.AvgPool2d(1,1),
    nn.MaxPool2d(kernel_size = 2, stride = 2),
    nn.MaxUnpool2d(1,1),
    nn.PositionalEncoding(6),
    nn.PixelShuffle(1),
    nn.PixelUnshuffle(1),
    nn.Upsample(None,1f))

//    nn.CrossMapLRN2d(1,2f,2f,2),
//    nn.MarginRankingLoss(),
//    nn.BCEWithLogitsLoss(),
//    nn.KLDivLoss(),
//    nn.HuberLoss(),



  
    val sequentialSecond = nn.Sequential[D](
      nn.Conv2d(16,32,kernel_size = 5, stride =1, padding =2),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size =2,stride =2))
  val fc = nn.Linear(7*7*32, num_classes)

  def apply(input: Tensor[D]): Tensor[D] ={
    var out = sequentialFirst(input)
    out = sequentialSecond(out)
    out = out.reshape(out.size.head, -1)
    fc(out)

  }




}
object ConvNet02 {

}
