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
//import ai.djl.nn.core.Linear;
//import org.bytedeco.javacpp.FloatPointer;
//import org.bytedeco.javacpp.PointerScope;
//import org.bytedeco.pytorch.*;
//import torch.Device;
//import torch.nn.modules.HasParams;
//import torch.optim.Adam;
//import java.nio.file.Paths;
//import java.util.ArrayList;
//import java.util.Arrays;
//import java.util.List;
//import java.util.Random;
//import java.util.stream.Collectors;
//
//import static ai.djl.engine.rust.RustLibrary.logSoftmax;
//
//// 定义 LSTM 网络模型
//public class LstmNet<D extends torch.TensorType> extends HasParams<D> {
//    private torch.nn.LSTM lstm;
//    private Linear fc;
//    private int inputSize;
//    private int hiddenSize;
//    private int numLayers;
//    private int numClasses;
//
//    public LstmNet(int inputSize, int hiddenSize, int numLayers, int numClasses) {
//        this.inputSize = inputSize;
//        this.hiddenSize = hiddenSize;
//        this.numLayers = numLayers;
//        this.numClasses = numClasses;
//        this.lstm = register(torch.nn.LSTM(inputSize, hiddenSize, numLayers, true));
//        this.fc = register(torch.nn.Linear(hiddenSize, numClasses));
//    }
//
//    // 无参构造函数，使用默认参数
//    public LstmNet() {
//        this(28, 128, 2, 10);
//    }
//
//    public torch.Tensor<D> forward(torch.Tensor<D> i) {
//        List<Long> arr = Arrays.asList((long)numLayers, i.size(0), (long)hiddenSize);
//        torch.Tensor<D> h0 = torch.zeros(arr, i.dtype());
//        torch.Tensor<D> c0 = torch.zeros(arr, i.dtype());
//        torch.TensorTuple3<D, D, D> outTuple3 = lstm.forward(i, h0, c0);
//        torch.Tensor<D> out = outTuple3.get0();
//        out = out.index(torch.indexing.all(), -1L, torch.indexing.all());
//        return logSoftmax(fc.forward(out), 1);
//    }
//}
//
//// 定义 RNN 网络模型
//class RnnNet<D extends torch.TensorType> extends HasParams<D> {
//    private torch.nn.RNN rnn;
//    private torch.nn.Linear fc;
//    private int inputSize;
//    private int hiddenSize;
//    private int numLayers;
//    private int numClasses;
//
//    public RnnNet(int inputSize, int hiddenSize, int numLayers, int numClasses) {
//        this.inputSize = inputSize;
//        this.hiddenSize = hiddenSize;
//        this.numLayers = numLayers;
//        this.numClasses = numClasses;
//        this.rnn = register(torch.nn.RNN(inputSize, hiddenSize, numLayers, true));
//        this.fc = register(torch.nn.Linear(hiddenSize, numClasses));
//    }
//
//    // 无参构造函数，使用默认参数
//    public RnnNet() {
//        this(28, 128, 2, 10);
//    }
//
//    public torch.Tensor<D> forward(torch.Tensor<D> i) {
//        List<Long> arr = Arrays.asList((long)numLayers, i.size(0), (long)hiddenSize);
//        torch.Tensor<D> h0 = torch.zeros(arr, i.dtype());
//        torch.Tensor<D> c0 = torch.zeros(arr, i.dtype());
//        torch.TensorTuple2<D, D> outTuple2 = rnn.forward(i, h0);
//        torch.Tensor<D> out = outTuple2.get0();
//        out = out.index(torch.indexing.all(), -1L, torch.indexing.all());
//        return torch.nn.functional.logSoftmax(fc.forward(out), 1);
//    }
//}
//
//// 定义 GRU 网络模型
//class GruNet<D extends torch.TensorType> extends HasParams<D> {
//    private torch.nn.GRU gru;
//    private torch.nn.Linear fc;
//    private int inputSize;
//    private int hiddenSize;
//    private int numLayers;
//    private int numClasses;
//
//    public GruNet(int inputSize, int hiddenSize, int numLayers, int numClasses) {
//        this.inputSize = inputSize;
//        this.hiddenSize = hiddenSize;
//        this.numLayers = numLayers;
//        this.numClasses = numClasses;
//        this.gru = register(torch.nn.GRU(inputSize, hiddenSize, numLayers, true));
//        this.fc = register(torch.nn.Linear(hiddenSize, numClasses));
//    }
//
//    // 无参构造函数，使用默认参数
//    public GruNet() {
//        this(28, 128, 2, 10);
//    }
//
//    public torch.Tensor<D> forward(torch.Tensor<D> i) {
//        List<Long> arr = Arrays.asList((long)numLayers, i.size(0), (long)hiddenSize);
//        torch.Tensor<D> h0 = torch.zeros(arr, i.dtype());
//        torch.Tensor<D> c0 = torch.zeros(arr, i.dtype());
//        torch.TensorTuple2<D, D> outTuple2 = gru.forward(i, h0);
//        torch.Tensor<D> out = outTuple2.get0();
//        out = out.index(torch.indexing.all(), -1L, torch.indexing.all());
//        return torch.nn.functional.logSoftmax(fc.forward(out), 1);
//    }
//}
//
//// 定义位置编码类
//class PositionalEncoding<D extends torch.TensorType> extends HasParams<D> {
//    private long d_model;
//    private long max_len;
//    private torch.Tensor<D> encoding;
//
//    public PositionalEncoding(long d_model, long max_len) {
//        this.d_model = d_model;
//        this.max_len = max_len;
//        initializeEncoding();
//    }
//
//    public PositionalEncoding(long d_model) {
//        this(d_model, 28 * 28);
//    }
//
//    private void initializeEncoding() {
//        List<Long> arr = Arrays.asList(max_len, d_model);
//        D paramType = paramType();
//        torch.Tensor<D> position = torch.arange(0, max_len, paramType).unsqueeze(1);
//        torch.Tensor<torch.Float32> arangeFloat = torch.arange(0, d_model, 2).to(torch.Float32.class);
//        torch.Tensor<torch.Float32> div_term = torch.exp(
//            arangeFloat.mul(torch.log(torch.tensor(10000.0)).neg().div((float)d_model))
//        );
//        torch.Tensor<D> sinPosition = torch.sin(position.mul(div_term.to(paramType))).to(paramType);
//        torch.Tensor<D> cosPosition = torch.cos(position.mul(div_term.to(paramType))).to(paramType);
//        torch.Tensor<torch.Int64> indexSin = torch.tensor(Arrays.asList(0L, 1L));
//        torch.Tensor<torch.Int64> indexCos = torch.tensor(Arrays.asList(1L, 1L));
//        
//        encoding = torch.zeros(arr.stream().mapToInt(Long::intValue).boxed().collect(Collectors.toList()), paramType);
//        encoding = torch.indexCopy(encoding, 0, indexSin.to(paramType), sinPosition);
//        encoding = torch.indexCopy(encoding, 0, indexCos.to(paramType), cosPosition);
//        encoding = encoding.unsqueeze(0);
//    }
//
//    public torch.Tensor<D> forward(torch.Tensor<D> x) {
//        return x.add(encoding).to(x.device());
//    }
//
//    @SuppressWarnings("unchecked")
//    private D paramType() {
//        return (D)torch.Float32.class; // 默认返回 Float32 类型
//    }
//}
//
///**
// * 主应用类，展示如何在 MNIST 数据集上训练简单的 LstmNet
// */
//public class LstmNetApp {
//    public static void main(String[] args) {
//        Device device = torch.cuda.isAvailable() ? Device.CUDA : Device.CPU;
//        System.out.println("Using device: " + device);
//        
//        // 创建模型并移至指定设备
//        LstmNet<torch.Float32> model = new LstmNet<>();
//        model.to(device);
//        
//        // 准备 FashionMNIST 数据
//        String dataPath = "D:\\data\\FashionMNIST";
//        FashionMNIST mnistTrain = new FashionMNIST(Paths.get(dataPath), true, true);
//        FashionMNIST mnistEval = new FashionMNIST(Paths.get(dataPath), false, true);
//        
//        // 打印模型信息
//        System.out.println("model " + String.join("\n", model.modules().toList()));
//        System.out.println("model " + model.summarize());
//        
//        // 设置损失函数和优化器
//        torch.nn.loss.CrossEntropyLoss lossFn = new torch.nn.loss.CrossEntropyLoss();
//        Adam optimizer = new Adam(model.parameters(), 1e-3, true);
//        Adam optimizerCopy = new Adam(model.parameters(), 1e-3, true);
//        
//        // 准备评估数据
//        torch.Tensor<torch.Float32> evalFeatures = mnistEval.features().to(device);
//        torch.Tensor<torch.Int64> evalTargets = mnistEval.targets().to(device);
//        
//        // 准备训练数据
//        Random r = new Random(0);
//        List<Example> exampleSeq = new ArrayList<>();
//        for (int i = 0; i < mnistTrain.length(); i++) {
//            torch.Tensor<torch.Float32> feature = mnistTrain.get(i).get0();
//            torch.Tensor<torch.Int64> target = mnistTrain.get(i).get1();
//            exampleSeq.add(new Example(feature.nativeTensor(), target.nativeTensor()));
//        }
//        
//        // 创建数据加载器
//        ExampleVector exampleVector = new ExampleVector();
//        for (Example example : exampleSeq) {
//            exampleVector.put(example);
//        }
//        
//        ChunkDataReader reader = new ChunkDataReader();
//        reader.put(exampleVector);
//        
//        int batch_size = 32;
//        int prefetch_count = 1;
//        
//        ChunkDatasetOptions chunkOptions = new ChunkDatasetOptions(prefetch_count, batch_size);
//        RandomSampler sampler = new RandomSampler(exampleSeq.size());
//        
//        ChunkDataset chunkDataset = new ChunkDataset(
//            reader, sampler, sampler, chunkOptions
//        );
//        
//        ChunkSharedBatchDataset ds = new ChunkSharedBatchDataset(chunkDataset).map(new ExampleStack());
//        
//        DataLoaderOptions opts = new DataLoaderOptions(32);
//        opts.batch_size().put(batch_size);
//        
//        ChunkRandomDataLoader data_loader = new ChunkRandomDataLoader(ds, opts);
//        
//        // 打印数据加载器信息
//        System.out.println("ds.size.get " + ds.size().get() + " data_loader option " + data_loader.options().batch_size().get());
//        
//        // 训练循环
//        List<Integer> epochs = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
//        for (int epoch : epochs) {
//            System.out.println("Epoch: " + epoch);
//            ExampleIterator it = data_loader.begin();
//            int batchIndex = 0;
//            System.out.println("coming in for loop");
//            
//            while (!it.equals(data_loader.end())) {
//                try (PointerScope p = new PointerScope()) {
//                    Example batch = it.access();
//                    optimizer.zeroGrad();
//                    
//                    // 转换为 torch.Tensor 并进行前向传播
//                    torch.Tensor<torch.Float32> trainDataTensor = torch.internal.NativeConverters.fromNative(batch.data());
//                    torch.Tensor<torch.Float32> input = torch.internal.NativeConverters.fromNative(batch.data()).reshape(-1, 28, 28);
//                    torch.Tensor<torch.Float32> prediction = model.forward(input);
//                    
//                    // 计算损失并进行反向传播
//                    torch.Tensor<torch.Float32> target = torch.internal.NativeConverters.fromNative(batch.target());
//                    torch.Tensor<torch.Float32> loss = lossFn.forward(prediction, target);
//                    loss.backward();
//                    optimizer.step();
//                    
//                    // 移动到下一个批次
//                    it = it.increment();
//                    batchIndex++;
//                    
//                    // 每 200 个批次进行一次评估
//                    if (batchIndex % 200 == 0) {
//                        // 运行评估
//                        torch.Tensor<torch.Float32> predictions = model.forward(evalFeatures.reshape(-1, 28, 28));
//                        torch.Tensor<torch.Float32> evalLoss = lossFn.forward(predictions, evalTargets);
//                        
//                        // 获取预测数据
//                        float[] featuresData = new float[1000];
//                        FloatPointer fp4 = new FloatPointer(predictions.nativeTensor().data_ptr_float());
//                        fp4.get(featuresData);
//                        
//                        System.out.println("\n ffff size " + featuresData.length + " shape " + 
//                            String.join(", ", Arrays.stream(evalFeatures.shape()).mapToObj(Long::toString).toArray(String[]::new)) + 
//                            "a data " + Arrays.toString(featuresData).replaceAll("\\[|\\]", ""));
//                        
//                        System.out.println("predictions : " + predictions + "\n");
//                        System.out.println("loss grad_fn: " + evalLoss.grad_fn());
//                        
//                        // 计算准确率
//                        torch.Tensor<torch.Bool> correct = predictions.argmax(1).eq(evalTargets);
//                        torch.Tensor<torch.Float32> accuracy = correct.sum().div((float)mnistEval.length());
//                        
//                        System.out.printf(
//                            "Epoch: %d | Batch: %4d | Training loss: %.4f | Eval loss: %.4f | Eval accuracy: %.4f\n",
//                            epoch, batchIndex, loss.item(), evalLoss.item(), accuracy.item()
//                        );
//                    }
//                }
//            }
//            
//            // 添加参数到优化器副本
//            optimizerCopy.add_parameters(model.namedParameters());
//            System.out.println("optimizerCopy " + optimizerCopy);
//            System.out.println("optimizer " + optimizer);
//            System.out.println("judge optimizer " + optimizer.equals(optimizerCopy));
//            System.out.println("model parameters dict " + model.namedParameters());
//        }
//        
//        // 保存模型
//        OutputArchive archive = new OutputArchive();
//        model.save(archive);
//        archive.save_to("lstm-net.pkl");
//    }
//}
