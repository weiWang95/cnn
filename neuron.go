package cnn

import "fmt"

type Neuron struct {
	IActive

	Id      string
	Weights []float64
	Bias    float64

	Net float64
	Out float64
	PD  float64
}

func (n *Neuron) Compute(input []float64) float64 {
	// fmt.Println("n -> ", n.Id, input, n.InputWeight, n.Weight)
	var total float64

	// fmt.Printf("[%s]fn = ", n.Id)

	for idx, v := range input {
		// fmt.Printf("%f + %f * %f + ", total, n.InputWeight[k], v)
		total = total + n.Weights[idx]*v
	}

	// fmt.Printf("%f => %f", n.Weight, total+n.Weight)

	n.Net = total + n.Bias
	n.Out = n.Active(n.Net)

	// fmt.Printf("  [active] => %f\n", n.Out)
	return n.Out
}

func (n *Neuron) BiasId() string {
	return fmt.Sprintf("%s-bias", n.Id)
}
