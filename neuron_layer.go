package cnn

import (
	"fmt"
)

const FirstNLId = "0-0"

type neuronLayerOption struct {
	active      IActive
	defaultBias float64
}

type NeuronLayerOption func(opt *neuronLayerOption)

func WithNLActive(active IActive) NeuronLayerOption {
	return func(opt *neuronLayerOption) {
		opt.active = active
	}
}

func WithNLBias(bias float64) NeuronLayerOption {
	return func(opt *neuronLayerOption) {
		opt.defaultBias = bias
	}
}

func getDefaultNLOption() *neuronLayerOption {
	return &neuronLayerOption{
		active:      ReLU,
		defaultBias: 0.1,
	}
}

type NeuronLayer struct {
	Prev *NeuronLayer
	Next *NeuronLayer

	no  int64
	num int64

	neurons   []Neuron
	neuronMap map[string]*Neuron
	opt       *neuronLayerOption
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
	l.neuronMap = make(map[string]*Neuron, l.num)

	for i := 0; i < int(l.num); i++ {
		n := Neuron{
			IActive: l.opt.active,
			Id:      fmt.Sprintf("%d-%d", l.no, i),
		}
		if l.Prev != nil {
			n.Weights = make([]float64, 0, l.Prev.num)
			n.Bias = l.opt.defaultBias
		}
		l.neurons = append(l.neurons, n)
		l.neuronMap[n.Id] = &l.neurons[i]
	}
}

func (l *NeuronLayer) Compute(inputs ...float64) []float64 {
	if l.Prev == nil && len(inputs) != int(l.num) {
		panic(fmt.Sprintf("error input num: expect %d got %d", l.num, len(inputs)))
	} else if l.Prev != nil && len(inputs) != int(l.Prev.num) {
		panic(fmt.Sprintf("error input num: expect %d got %d", l.Prev.num, len(inputs)))
	}

	outs := make([]float64, 0)

	// 输入层
	if l.Prev == nil {
		for i := range l.neurons {
			l.neurons[i].Out = inputs[i]
		}

		outs = inputs
	} else {
		for i := range l.neurons {
			out := l.neurons[i].Compute(inputs)
			outs = append(outs, out)
		}
	}

	return outs
}

func (l *NeuronLayer) ActiveType() string {
	return fmt.Sprintf("%T", l.opt.active)
}
