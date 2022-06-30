package cnn

import (
	"fmt"
)

func ApplyWeight(l *NeuronLayer, data [][]map[string]float64) {
	i := 0
	cur := l

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

func ExportWeight(l *NeuronLayer) [][]map[string]float64 {
	data := make([][]map[string]float64, 0)

	cur := l
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

func Compute(l *NeuronLayer, inputs ...float64) []float64 {
	if len(inputs) != int(l.num) {
		panic(fmt.Sprintf("error input num: expect %d got %d", l.num, len(inputs)))
	}

	var cur *NeuronLayer
	var outs []float64

	cur = l

	for {
		// fmt.Println("l -> ", cur.no, cur.num)
		// fmt.Println("input -> ", inputs)
		outs = cur.Compute(inputs...)
		// fmt.Println("out -> ", outs)

		if cur.Next == nil {
			return outs
		}

		cur = cur.Next
		inputs = outs
	}
}

func BP(outLayer *NeuronLayer, step float64, loss ILoss, expects ...float64) {
	cur := outLayer

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
					pd = loss.BP(out, expects[i]) * out * (1 - out)
				} else {
					var sum float64
					for m, _ := range cur.Next.neurons {
						sum += cur.Next.neurons[m].DM[cur.neurons[i].Id] * cur.Next.neurons[m].OldWeight[cur.neurons[i].Id]
					}
					pd = sum * out * (1 - out)
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
