package cnn

import (
	"math"
)

type IActive interface {
	Name() string
	Active(v float64) float64
	BP(dout, out float64) float64
}

var (
	ReLU      IActive = &reLU{}
	LeakyReLU IActive = &leakyReLU{}
	Sigmoid   IActive = &sigmoid{}
)

type reLU struct {
}

func (a *reLU) Name() string {
	return "ReLU"
}

func (a *reLU) Active(v float64) float64 {
	if v > 0 {
		return v
	}

	return 0
}

func (a *reLU) BP(dout, out float64) float64 {
	if out < 0 {
		return 0
	}
	return dout
}

type leakyReLU struct {
}

func (a *leakyReLU) Name() string {
	return "LeakyReLU"
}

func (a *leakyReLU) Active(v float64) float64 {
	if v > 0 {
		return v
	}

	return 0.01 * v
}

func (a *leakyReLU) BP(dout, out float64) float64 {
	if out < 0 {
		return 0.01 * dout
	}
	return dout
}

type sigmoid struct {
}

func (a *sigmoid) Name() string {
	return "Sigmoid"
}

func (a *sigmoid) Active(v float64) float64 {
	if v < -45 {
		return 0
	} else if v > 45 {
		return 1
	}

	return 1 / (1 + math.Pow(math.E, -v))
}

func (a *sigmoid) BP(dout, out float64) float64 {
	pd := dout * out * (1 - out)
	// fmt.Printf("n -> %.06f * %.06f * (1 - %.06f) => %.06f \n", dout, out, out, pd)
	return pd
}
