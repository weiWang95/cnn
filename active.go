package cnn

import (
	"math"
)

type IActive interface {
	Active(v float64) float64
}

var (
	ReLU    IActive = &reLU{}
	Sigmoid IActive = &sigmoid{}
)

type reLU struct {
}

func (a *reLU) Active(v float64) float64 {
	if v > 0 {
		return v
	}

	return 0
}

type sigmoid struct {
}

func (a *sigmoid) Active(v float64) float64 {
	return 1 / (1 + math.Pow(math.E, -v))
}
