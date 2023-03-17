package cnn

import (
	"fmt"
	"math"
)

type ILoss interface {
	Loss(outs []float64, expects []float64) float64
	BP(out float64, expect float64) float64
}

var (
	SquareDiff   ILoss = &squareDiff{}
	LogisticDiff ILoss = &logisticDiff{}
	SoftmaxDiff  ILoss = &softmaxDiff{}
)

type squareDiff struct{}

func (d *squareDiff) Loss(outs []float64, expects []float64) float64 {
	var loss float64

	for i, _ := range outs {
		r := 0.5 * math.Pow(expects[i]-outs[i], 2)
		if math.IsNaN(r) || math.IsInf(r, 0) {
			fmt.Printf("0.5 * (%.06f-%.06f)^2 = %.06f\n", expects[i], outs[i], r)
		}
		loss += r
	}

	return loss
}
func (d *squareDiff) BP(out float64, expect float64) float64 {
	return -(expect - out)
}

type logisticDiff struct{}

func (d *logisticDiff) Loss(outs []float64, expects []float64) float64 {
	var loss float64

	for i, _ := range outs {
		out := outs[i]
		expect := expects[i]
		loss += -1 * (expect*math.Log(out) + (1-expect)*math.Log(1-out))
	}

	return loss
}

func (d *logisticDiff) BP(out float64, expect float64) float64 {
	return (out - expect) / (out * (1 - out))
}

type softmaxDiff struct{}

func (d *softmaxDiff) Loss(outs []float64, expects []float64) float64 {
	return 0
}

func (d *softmaxDiff) BP(out float64, expect float64) float64 {
	return out - expect
}
