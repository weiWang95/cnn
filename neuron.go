package cnn

type Neuron struct {
	IActive

	Id          string
	InputWeight map[string]float64
	OldWeight   map[string]float64
	Weight      float64

	Out float64
	DM  map[string]float64
}

func (n *Neuron) Compute(input map[string]float64) float64 {
	// fmt.Println("n -> ", n.Id, input, n.InputWeight, n.Weight)
	var total float64

	// fmt.Printf("[%s]fn = ", n.Id)

	for k, v := range input {
		// fmt.Printf("%f + %f * %f + ", total, n.InputWeight[k], v)
		total = total + n.InputWeight[k]*v
	}

	// fmt.Printf("%f => %f", n.Weight, total+n.Weight)

	n.Out = n.Active(total + n.Weight)

	// fmt.Printf("  [active] => %f\n", n.Out)
	return n.Out
}
