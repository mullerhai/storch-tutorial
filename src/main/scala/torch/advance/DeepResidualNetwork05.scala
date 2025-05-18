package torch
package advance

import torch.FloatNN
import torch.nn.modules.{HasParams, TensorModule}
import org.bytedeco.pytorch.{OutputArchive, TensorExampleVectorIterator}
import torch.Device.{CPU, CUDA}
import torch.data.dataset.ChunkSharedBatchDataset
import torch.nn.{Sequential, functional as F}
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
import torch.nn.modules.conv.Conv2d

import scala.collection.mutable.ListBuffer
class DeepResidualNetwork[D <: FloatNN :Default](block: ResidualBlock[D], layers: Seq[Int], num_classes: Int =100) extends HasParams[D] {

  var in_channels = 16
  val conv = register(conv3x3(3,16))
  val bn  =  register(nn.BatchNorm2d(16))
  val relu = register(nn.ReLU())
  val layer1 = register(make_layer(block,16,layers(0)))
  val layer2 = register(make_layer(block,32,layers(1),2))
  val layer3 = register(make_layer(block,64,layers(2),2))
  val avg_pool = register(nn.AvgPool2d[D](8,8))
  val fc = register(nn.Linear(64,num_classes))
  
  //    def make_layer(self, block, out_channels, blocks, stride=1):
  //        downsample = None
  //        if (stride != 1) or (self.in_channels != out_channels):
  //            downsample = nn.Sequential(
  //                conv3x3(self.in_channels, out_channels, stride=stride),
  //                nn.BatchNorm2d(out_channels))
  //        layers = []
  //        layers.append(block(self.in_channels, out_channels, stride, downsample))
  //        self.in_channels = out_channels
  //        for i in range(1, blocks):
  //            layers.append(block(out_channels, out_channels))
  //        return nn.Sequential(*layers)
  def make_layer(block:ResidualBlock[D], out_channels: Int, blocks: Int,stride: Int = 1):Sequential[D]={
    var  downsample = null
    if( stride != 1 || in_channels != out_channels){
      val  ds  = nn.Sequential(conv3x3(in_channels, out_channels, stride),
        nn.BatchNorm2d(out_channels)
      )
    }
    val layers = new ListBuffer[ResidualBlock[D]]()
    layers.append(ResidualBlock[D](in_channels, out_channels, stride, downsample))
    this.in_channels = out_channels
    (1 to blocks).foreach( i =>{
      layers.append(ResidualBlock[D](out_channels, out_channels))
    })
    nn.Sequential(layers.toSeq*)
    
  }

  def conv3x3(in_channels: Int, out_channels: Int, stride: Int = 1): Conv2d[D] = {
    nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = false)
  }
  def apply(input :Tensor[D]):Tensor[D]={
    var out = conv(input)
    out = bn(out)
    out = relu(out)
    out = layer1(out)
    out = layer2(out)
    out = layer3(out)
    out = avg_pool(out)
    out = out.view(out.size.head, -1)
    out = fc(out)
    out
  }
}
class ResidualBlock[D <: FloatNN :Default](in_channels: Int, out_channels: Int, stride:Int =1, downsample: Option[TensorModule[D]] =None) extends HasParams[D] with TensorModule[D] {

  def conv3x3(in_channels: Int, out_channels: Int, stride: Int = 1): Conv2d[D] = {
    nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = false)
  }

  val conv1 = register(conv3x3(in_channels, out_channels, stride))
  val bn1 =  register(nn.BatchNorm2d(out_channels))
  val relu = register(nn.ReLU()) //inplace=True
  val conv2 = register(conv3x3(in_channels, out_channels))
  val bn2 = register(nn.BatchNorm2d(out_channels))
//  val downsample = downsample
  def apply(input: Tensor[D]): Tensor[D] = {
    var residual = input
    var out = conv1(input)
    out = bn1(out)
    out = relu(out)
    out = conv2(out)
    out = bn2(out)
    if downsample.isDefined then {
      val layer = downsample.get
      residual = layer(input)
    }
    out  = out.add(residual)
    out  = relu(out)
    out

  }
}

object DeepResidualNetwork05 {
  val num_epochs = 80
  val batch_size = 100
  val learning_rate = 0.001


}