package cnn

import (
	"fmt"
	"math/rand"
)

const FirstNLId = "0-0"

type neuronLayerOption struct {
	active             IActive
	randomWeight       bool
	defaultInputWeight float64
	defaultWeight      float64
}

type NeuronLayerOption func(opt *neuronLayerOption)

func WithNLActive(active IActive) NeuronLayerOption {
	return func(opt *neuronLayerOption) {
		opt.active = active
	}
}

func WithNLWeigth(weight float64) NeuronLayerOption {
	return func(opt *neuronLayerOption) {
		opt.defaultWeight = weight
	}
}

func WithNLInputWeigth(weight float64) NeuronLayerOption {
	return func(opt *neuronLayerOption) {
		opt.defaultInputWeight = weight
	}
}
func WithNLRandomWeight(enable bool) NeuronLayerOption {
	return func(opt *neuronLayerOption) {
		opt.randomWeight = enable
	}
}

func getDefaultNLOption() *neuronLayerOption {
	return &neuronLayerOption{
		active:             ReLU,
		defaultInputWeight: 1,
		defaultWeight:      0,
	}
}

type NeuronLayer struct {
	Prev *NeuronLayer
	Next *NeuronLayer

	no  int64
	num int64

	neurons []Neuron
	opt     *neuronLayerOption
}

func NewNeuronLayer(no, num int64, prev *NeuronLayer, opts ...NeuronLayerOption) *NeuronLayer {
	l := new(NeuronLayer)

	opt := getDefaultNLOption()
	for _, fn := range opts {
		fn(opt)
	}
	l.opt = opt

	l.no = no
	l.num = num

	if prev != nil {
		l.Prev = prev
		prev.Next = l
	}

	l.initNeurons()

	return l
}

func (l *NeuronLayer) initNeurons() {
	l.neurons = make([]Neuron, 0, l.num)
	preVNeurons := []Neuron{{Id: FirstNLId}}

	if l.Prev != nil {
		preVNeurons = l.Prev.neurons
	}

	for i := 0; i < int(l.num); i++ {
		iw := make(map[string]float64, len(preVNeurons))
		for _, item := range preVNeurons {
			iw[item.Id] = l.opt.defaultInputWeight
			if l.opt.randomWeight {
				iw[item.Id] = rand.Float64()
			}
		}

		l.neurons = append(l.neurons, Neuron{
			IActive:     l.opt.active,
			Id:          fmt.Sprintf("%d-%d", l.no, i),
			InputWeight: iw,
			Weight:      l.opt.defaultWeight,
		})
	}
}

func (l *NeuronLayer) Compute(inputs ...float64) []float64 {
	if l.Prev == nil && len(inputs) != int(l.num) {
		panic(fmt.Sprintf("error input num: expect %d got %d", l.num, len(inputs)))
	} else if l.Prev != nil && len(inputs) != int(l.Prev.num) {
		panic(fmt.Sprintf("error input num: expect %d got %d", l.Prev.num, len(inputs)))
	}

	outs := make([]float64, 0)

	if l.Prev == nil {
		for i, input := range inputs {
			out := l.neurons[i].Compute(map[string]float64{FirstNLId: input})
			outs = append(outs, out)
		}
	} else {
		m := make(map[string]float64, l.Prev.num)
		for i, item := range l.Prev.neurons {
			m[item.Id] = inputs[i]
		}

		for i, _ := range l.neurons {
			out := l.neurons[i].Compute(m)
			outs = append(outs, out)
		}
	}

	return outs
}

func (l *NeuronLayer) ActiveType() string {
	return fmt.Sprintf("%T", l.opt.active)
}
