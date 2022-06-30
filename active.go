package cnn

import (
	"math"
)

type IActive interface {
	Active(v float64) float64
}

type ReLU struct {
}

func (a *ReLU) Active(v float64) float64 {
	if v > 0 {
		return v
	}

	return 0
}

type Sigmoid struct {
}

func (a *Sigmoid) Active(v float64) float64 {
	return 1 / (1 + math.Pow(math.E, -v))
}
