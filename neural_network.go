package cnn

import (
	"fmt"
	"math"
	"math/rand"
)

type NeuralNetwork struct {
	opt       *networkOption
	StudyRate float64

	lossFn ILoss

	inputLayer  *NeuronLayer
	outputLayer *NeuronLayer
	softmaxOuts []float64
}

type WeightMap map[string][2][]float64

type networkOption struct {
	Softmax       bool
	DefaultWeight float64
	optimizer     Optimizer
}

func defaultNetworkOption() *networkOption {
	return &networkOption{
		Softmax: false,
	}
}

type NetWorkOption func(opt *networkOption)

func WithSoftmax() NetWorkOption {
	return func(opt *networkOption) {
		opt.Softmax = true
	}
}

func WithDefaultWeight(weight float64) NetWorkOption {
	return func(opt *networkOption) {
		opt.DefaultWeight = weight
	}
}

func WithOptimizer(optimizer Optimizer) NetWorkOption {
	return func(opt *networkOption) {
		opt.optimizer = optimizer
	}
}

func NewNeuralNetwork(data []int64, actives []IActive, lossFn ILoss, opts ...NetWorkOption) *NeuralNetwork {
	n := new(NeuralNetwork)
	n.opt = defaultNetworkOption()
	for _, item := range opts {
		item(n.opt)
	}

	n.lossFn = lossFn

	var cur *NeuronLayer
	for i, num := range data {
		opts := make([]NeuronLayerOption, 0)
		if i == 0 {
			opts = append(opts, WithNLBias(0))
		} else if n.opt.DefaultWeight != 0 {
			opts = append(opts, WithNLBias(n.opt.DefaultWeight))
		}

		if cur == nil {
			cur = NewNeuronLayer(int64(i), num, nil, opts...)
		} else {
			opts = append(opts, WithNLActive(actives[i-1]))
			cur = NewNeuronLayer(int64(i), num, cur, opts...)
		}

		if n.inputLayer == nil {
			n.inputLayer = cur
		}
	}
	n.outputLayer = cur

	n.initWeight()

	return n
}

func (n *NeuralNetwork) initWeight() {
	cur := n.inputLayer
	for {
		if cur == nil {
			break
		}

		// 输入层
		if cur.Prev == nil {
			cur = cur.Next
			continue
		}

		// 隐藏层
		wg := n.getWeightGenerator(cur.opt.active)
		weights := wg.GetWeight(int(cur.Prev.num), int(cur.num))

		for i := range weights {
			cur.neurons[i].Weights = weights[i]
		}

		cur = cur.Next
	}
}

func (n *NeuralNetwork) Calculate(inputs [][]float64) [][]float64 {
	outs := make([][]float64, 0)

	for _, input := range inputs {
		out := n.Compute(input...)
		outs = append(outs, out)
		// fmt.Printf("input:%v -> out:%v\n", input, out)
	}

	return outs
}

func (n *NeuralNetwork) Train(inputs [][]float64, expects [][]float64, sprint, batchSize int, lrScheduler LRScheduler, threshold float64) []float64 {
	result := make([]float64, 0)
	times := 0

	for epoch := 0; epoch < sprint; epoch++ {
		var avgLoss float64
		var success bool
		var count int64
		lr := lrScheduler.GetLR(epoch)

		// inputs, expects = shuffleData(inputs, expects)
		batchInputs, batchExpects := inputs, expects
		if batchSize != 0 {
			batchInputs, batchExpects = n.sampleData(inputs, expects, batchSize)
		}

		for i, input := range batchInputs {
			count += 1
			out := n.Compute(input...)
			for _, o := range out {
				if math.IsNaN(o) {
					fmt.Printf("WARN: loss -> %v\n", o)
					return result
				}
			}
			// fmt.Printf("input:%v -> out:%v[expect:%v]\n", input, out, expects[i])
			loss := n.lossFn.Loss(out, batchExpects[i])
			avgLoss += loss

			if loss < threshold {
				times += 1
			} else {
				times = 0
			}
			if times > len(inputs)/2 {
				success = true
				break
			}

			n.BP(lr, batchExpects[i]...)
		}

		avgLoss = avgLoss / float64(count)
		fmt.Printf("epoch: %d, lr:%.06f, avg loss: %.06f\n", epoch, lr, avgLoss)

		result = append(result, avgLoss)

		if success {
			break
		}
	}

	return result
}

func (n *NeuralNetwork) TrainWithTest(inputs, expects, testInputs, testExpects [][]float64, sprint, batchSize int, lrScheduler LRScheduler, threshold float64) ([]float64, []float64) {
	result := make([]float64, 0)
	testResult := make([]float64, 0)
	times := 0

	for epoch := 0; epoch < sprint; epoch++ {
		var avgLoss float64
		var success bool
		var count int64
		lr := lrScheduler.GetLR(epoch)

		// Train

		// inputs, expects = shuffleData(inputs, expects)
		batchInputs, batchExpects := n.sampleData(inputs, expects, batchSize)
		for i, input := range batchInputs {
			count += 1
			out := n.Compute(input...)
			for _, o := range out {
				if math.IsNaN(o) {
					fmt.Printf("WARN: loss -> %v\n", o)
					return result, testResult
				}
			}
			// fmt.Printf("input:%v -> out:%v[expect:%v]\n", input, out, batchExpects[i])
			loss := n.lossFn.Loss(out, batchExpects[i])
			avgLoss += loss

			n.BP(lr, batchExpects[i]...)
		}

		avgLoss = avgLoss / float64(count)
		result = append(result, avgLoss)

		// Test
		testAvgLoss := 0.0
		for i, input := range testInputs {
			count += 1
			out := n.Compute(input...)
			for _, o := range out {
				if math.IsNaN(o) {
					fmt.Printf("WARN: test loss -> %v\n", o)
					return result, testResult
				}
			}
			loss := n.lossFn.Loss(out, testExpects[i])
			testAvgLoss += loss

			if loss < threshold {
				times += 1
			} else {
				times = 0
			}

			if times > len(inputs)/2 {
				success = true
				break
			}
		}

		testAvgLoss = testAvgLoss / float64(len(testInputs))
		testResult = append(testResult, avgLoss)

		fmt.Printf("epoch: %d, lr:%.06f, train loss: %.06f, test loss: %.06f \n", epoch, lr, avgLoss, testAvgLoss)

		if success {
			break
		}
	}

	return result, testResult
}

func (n *NeuralNetwork) Compute(inputs ...float64) []float64 {
	if len(inputs) != int(n.inputLayer.num) {
		panic(fmt.Sprintf("error input num: expect %d got %d", n.inputLayer.num, len(inputs)))
	}
	// fmt.Println("inputs: ", inputs)

	var cur *NeuronLayer
	var outs []float64

	cur = n.inputLayer

	for {
		// fmt.Println("l -> ", cur.no, cur.num)
		// fmt.Println("input -> ", inputs)
		outs = cur.Compute(inputs...)
		// fmt.Println("out -> ", outs)

		if cur.Next == nil {
			break
		}

		cur = cur.Next
		inputs = outs
	}

	if n.opt.Softmax {
		outs = n.Softmax(outs)
	}

	return outs
}

func (n *NeuralNetwork) BP(step float64, expects ...float64) {
	optimizer := n.opt.optimizer
	if optimizer == nil {
		optimizer = NewNoneOptimizer(step)
	}
	weights, grads := make(map[string][]float64), make(map[string][]float64)

	cur := n.outputLayer
	for {
		if cur.Prev == nil {
			break
		}

		for i, neuron := range cur.neurons {
			// 权重
			curWeights := make([]float64, len(neuron.Weights))
			copy(curWeights, neuron.Weights)
			curGrads := make([]float64, len(neuron.Weights))
			// 偏差
			biasKey := neuron.BiasId()
			weights[biasKey] = []float64{neuron.Bias}

			out := neuron.Out

			var pd float64
			if cur.Next == nil {
				if n.opt.Softmax && len(n.softmaxOuts) > 0 {
					// fmt.Printf("(%.06f - %.06f) * %.06f * (1 - %.06f)\n", n.softmaxOuts[i], expects[i], out, out)
					pd = (n.softmaxOuts[i] - expects[i]) * out * (1 - out)
				} else {
					// pd = n.lossFn.BP(out, expects[i]) * out * (1 - out)
					// fmt.Printf("%.06f * %.06f * (1 - %.06f) => %.06f\n", n.lossFn.BP(out, expects[i]), out, out, pd)
					pd = neuron.BP(n.lossFn.BP(out, expects[i]), out)
					if math.IsNaN(pd) {
						fmt.Printf("%.06f * %.06f * (1 - %.06f) => %.06f\n", n.lossFn.BP(out, expects[i]), out, out, pd)
					}
				}
			} else {
				var sum float64
				for _, nextNeuron := range cur.Next.neurons {
					sum += grads[nextNeuron.Id][i] * neuron.Out * weights[nextNeuron.Id][i]
				}
				// pd = sum * out * (1 - out)
				// fmt.Printf("sum -> %.06f * %.06f * (1 - %.06f) => %.06f\n", sum, out, out, pd)
				pd = neuron.BP(sum, out)
				if math.IsNaN(pd) {
					fmt.Printf("sum -> %.06f * %.06f * (1 - %.06f) => %.06f\n", sum, out, out, pd)
				}
			}
			if math.IsNaN(pd) {
				panic("NaN")
			}

			for idx := range neuron.Weights {
				curGrads[idx] = pd * cur.Prev.neurons[idx].Out // 权重偏导
			}

			// 权重
			weights[neuron.Id] = curWeights    // 权重
			grads[neuron.Id] = curGrads        // 权重偏导
			grads[biasKey] = []float64{pd * 1} // 偏差偏导
		}

		cur = cur.Prev
	}

	// 优化器
	newWeight := optimizer.Update(weights, grads)
	n.applyOptimizerWeights(newWeight)
}

func (n *NeuralNetwork) applyOptimizerWeights(weights map[string][]float64) {
	cur := n.outputLayer

	for {
		if cur.Prev == nil {
			break
		}

		for i := range cur.neurons {
			cur.neurons[i].Weights = weights[cur.neurons[i].Id]
			cur.neurons[i].Bias = weights[cur.neurons[i].BiasId()][0]
		}

		cur = cur.Prev
	}
}

func (n *NeuralNetwork) Softmax(inputs []float64) []float64 {
	n.softmaxOuts = make([]float64, 0)

	var max float64
	for _, item := range inputs {
		if item > max {
			max = item
		}
	}

	var sum float64
	outs := make([]float64, 0, len(inputs))
	for _, item := range inputs {
		out := math.Exp(item - max)
		sum += out
		outs = append(outs, out)
	}

	for i, _ := range outs {
		outs[i] = outs[i] / sum
		n.softmaxOuts = append(n.softmaxOuts, outs[i])
	}

	return outs
}

func (n *NeuralNetwork) SoftmaxBP(outs []float64, expects []float64) []float64 {
	var expectIdx int
	for i, item := range expects {
		if item == 1 {
			expectIdx = i
			break
		}
	}

	douts := make([]float64, 0, len(outs))
	for i, item := range outs {
		if i == expectIdx {
			douts = append(douts, -1*math.Log(item))
		} else {
			douts = append(douts, item)
		}
	}

	return douts
}

func (n *NeuralNetwork) ApplyWeight(data WeightMap) {
	cur := n.inputLayer

	for {
		for _, neuron := range cur.neurons {
			if _, ok := data[neuron.Id]; !ok {
				continue
			}

			neuron.Weights = data[neuron.Id][0]
			neuron.Bias = data[neuron.Id][1][0]
		}

		if cur.Next == nil {
			break
		}

		cur = cur.Next
	}
}

func (n *NeuralNetwork) getWeightGenerator(active IActive) WeightGenerator {
	switch active.(type) {
	case *reLU, *leakyReLU:
		return &heGenerator{}
	case *sigmoid:
		return &xavierGenerator{}
	default:
		return &xavierGenerator{}
	}
}

func (n *NeuralNetwork) ExportWeight() WeightMap {
	data := make(map[string][2][]float64)

	cur := n.inputLayer
	for {
		for i, neuron := range cur.neurons {
			data[neuron.Id] = [2][]float64{neuron.Weights, {cur.neurons[i].Bias}}
		}

		if cur.Next == nil {
			break
		}

		cur = cur.Next
	}

	return data
}

func (n *NeuralNetwork) sampleData(inputs [][]float64, expects [][]float64, batchSize int) ([][]float64, [][]float64) {
	batchInputs := make([][]float64, 0, batchSize)
	batchExpects := make([][]float64, 0, batchSize)

	for i := 0; i < len(inputs); i++ {
		if len(batchInputs) >= batchSize {
			break
		}

		cur := len(inputs) - 1 - i
		idx := rand.Intn(len(inputs) - i)

		inputs[idx], inputs[cur] = inputs[cur], inputs[idx]
		expects[idx], expects[cur] = expects[cur], expects[idx]

		batchInputs = append(batchInputs, inputs[cur])
		batchExpects = append(batchExpects, expects[cur])
	}

	// for {
	// 	if len(batchInputs) >= batchSize {
	// 		break
	// 	}

	// 	idx := rand.Intn(len(inputs))
	// 	batchInputs = append(batchInputs, inputs[idx])
	// 	batchExpects = append(batchExpects, expects[idx])
	// }

	return batchInputs, batchExpects
}

func shuffleData(inputs [][]float64, expects [][]float64) ([][]float64, [][]float64) {
	for i := len(inputs) - 1; i > 0; i-- {
		idx := rand.Intn(i + 1)
		inputs[i], inputs[idx] = inputs[idx], inputs[i]
		expects[i], expects[idx] = expects[idx], expects[i]
	}
	return inputs, expects
}
