package cnn

import (
	"fmt"
	"math"
)

type NeuralNetwork struct {
	opt       *networkOption
	StudyRate float64

	lossFn ILoss

	inputLayer  *NeuronLayer
	outputLayer *NeuronLayer
	softmaxOuts []float64
}

type WeightMap [][]map[string]float64

type networkOption struct {
	Softmax bool
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

func NewNeuralNetwork(data []int64, actives []IActive, lossFn ILoss, opts ...NetWorkOption) *NeuralNetwork {
	n := new(NeuralNetwork)
	n.opt = defaultNetworkOption()
	for _, item := range opts {
		item(n.opt)
	}

	n.lossFn = lossFn

	var cur *NeuronLayer
	for i, num := range data {
		if cur == nil {
			cur = NewNeuronLayer(int64(i), num, nil, WithNLRandomWeight())
		} else {
			cur = NewNeuronLayer(int64(i), num, cur, WithNLActive(actives[i-1]), WithNLRandomWeight())
		}

		if n.inputLayer == nil {
			n.inputLayer = cur
		}
	}
	n.outputLayer = cur

	return n
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

func (n *NeuralNetwork) Train(inputs [][]float64, expects [][]float64, sprint int, studyRate, threshold float64) []float64 {
	result := make([]float64, 0)
	times := 0

	for i := 0; i < sprint; i++ {
		var avgLoss float64
		var success bool
		var count int64

		for i, input := range inputs {
			count += 1
			out := n.Compute(input...)
			for _, o := range out {
				if math.IsNaN(o) {
					fmt.Printf("WARN: loss -> %v\n", o)
					return result
				}
			}
			fmt.Printf("input:%v -> out:%v[expect:%v]\n", input, out, expects[i])
			loss := n.lossFn.Loss(out, expects[i])
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

			n.BP(studyRate, expects[i]...)
		}

		avgLoss = avgLoss / float64(count)
		fmt.Printf("avg loss : %v\n", avgLoss)

		result = append(result, avgLoss)

		if success {
			break
		}
	}

	return result
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
	cur := n.outputLayer

	for {
		if cur.Prev == nil {
			break
		}

		for i, neuron := range cur.neurons {
			cur.neurons[i].OldWeight = cur.neurons[i].InputWeight
			cur.neurons[i].InputWeight = make(map[string]float64)
			dm := make(map[string]float64)

			j := 0
			for k, v := range cur.neurons[i].OldWeight {
				out := neuron.Out
				outW := cur.Prev.neurons[j].Out

				var pd float64
				if cur.Next == nil {
					if n.opt.Softmax && len(n.softmaxOuts) > 0 {
						// fmt.Printf("(%.06f - %.06f) * %.06f * (1 - %.06f)\n", n.softmaxOuts[i], expects[i], out, out)
						pd = (n.softmaxOuts[i] - expects[i]) * out * (1 - out)
					} else {
						// pd = n.lossFn.BP(out, expects[i]) * out * (1 - out)
						// fmt.Printf("%.06f * %.06f * (1 - %.06f) => %.06f\n", n.lossFn.BP(out, expects[i]), out, out, pd)
						pd = cur.neurons[i].BP(n.lossFn.BP(out, expects[i]), out)
					}
				} else {
					var sum float64
					for m, _ := range cur.Next.neurons {
						sum += cur.Next.neurons[m].DM[cur.neurons[i].Id] * cur.Next.neurons[m].OldWeight[cur.neurons[i].Id]
					}
					// pd = sum * out * (1 - out)
					// fmt.Printf("sum -> %.06f * %.06f * (1 - %.06f) => %.06f\n", sum, out, out, pd)
					pd = cur.neurons[j].BP(sum, out)
				}
				if math.IsNaN(pd) {
					panic("NaN")
				}
				dm[k] = pd

				d := v - step*pd*outW
				cur.neurons[i].InputWeight[k] = d
				// fmt.Printf("weight -> %v - %v * %v => ", cur.neurons[i].Weight, step, pd)
				cur.neurons[i].Weight = cur.neurons[i].Weight - step*pd*1
				// fmt.Printf("%v\n", cur.neurons[i].Weight)

				j += 1
			}

			cur.neurons[i].DM = dm

			// fmt.Printf("n[%s]\n", cur.neurons[i].Id)
			// fmt.Printf("old => ")
			// for k, v := range cur.neurons[i].OldWeight {
			// 	fmt.Printf("%s:%.06f ", k, v)
			// }
			// fmt.Println()
			// fmt.Printf("new => ")
			// for k, v := range cur.neurons[i].InputWeight {
			// 	fmt.Printf("%s:%.06f ", k, v)
			// }
			// fmt.Println()

			// fmt.Printf("dm => ")
			// for k, v := range cur.neurons[i].DM {
			// 	fmt.Printf("%s:%.06f ", k, v)
			// }
			// fmt.Println()
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
	i := 0
	cur := n.inputLayer

	for {
		for j, _ := range cur.neurons {
			m := make(map[string]float64, len(data[i][j]))
			for k, v := range data[i][j] {
				if k == "weight" {
					cur.neurons[j].Weight = v
					continue
				}
				m[k] = v
			}

			cur.neurons[j].InputWeight = m
		}

		if cur.Next == nil {
			break
		}

		cur = cur.Next
		i += 1
	}
}

func (n *NeuralNetwork) ExportWeight() WeightMap {
	data := make(WeightMap, 0)

	cur := n.inputLayer
	for {
		m := make([]map[string]float64, 0)
		for i, _ := range cur.neurons {
			mm := make(map[string]float64, len(cur.neurons[i].InputWeight))
			for k, v := range cur.neurons[i].InputWeight {
				mm[k] = v
			}
			mm["weight"] = cur.neurons[i].Weight
			m = append(m, mm)
		}
		data = append(data, m)

		if cur.Next == nil {
			break
		}

		cur = cur.Next
	}

	return data
}
